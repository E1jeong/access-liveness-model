"""MobileNetV2 기반 안티스푸핑 모델 정의.

세 가지 형태를 만든다.
  dual     : RGB 백본 + IR 백본 → 특징 concat(late fusion) → 분류 헤드
  crop_rgb : RGB 백본 하나
  crop_ir  : IR(1채널) 백본 하나

출력은 항상 softmax 없는 logits (len(CLASS_NAMES)=10차원)이다.
`rgb_input_mobilenet_range` / `average_pool_op` / `fixed_batch_size` / `classifier_as_conv`
인자는 학습이 아니라 TFLite·NPU 변환 경로에서 켜는 스위치라 기본값은 모두 학습용이다.
"""
import sys
import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from classes import CLASS_NAMES
from keras_pipeline.spec import MODEL_INPUT_SIGNATURES, RGB_MEAN, RGB_STD


# IR(1채널) 백본에 RGB(3채널) ImageNet 가중치를 이식한다. Conv1만 채널축을 접고 나머지는 그대로 복사.
#
# 왜 필요한가: keras.applications.MobileNetV2는 1채널 입력에 대한 사전학습 가중치를
# 제공하지 않는다(weights="imagenet"은 3채널 전용). IR 백본을 랜덤으로 두면
# 엣지·질감 필터도 처음부터 학습해야 한다. Conv1(첫 층)만 3채널을
# 1채널로 접어 넣으면 나머지 층은 shape이 완전히 동일하므로 그대로 복사할 수 있다.
def _transfer_imagenet_weights_to_gray_backbone(source_backbone, gray_backbone, label, reduction="mean"):
    """3채널 MobileNetV2의 가중치로 1채널 MobileNetV2를 초기화한다."""
    if reduction not in ("mean", "sum"):
        raise ValueError(f"Unknown reduction: {reduction}")
    copied = 0
    for source_layer in source_backbone.layers:
        # 두 백본은 같은 아키텍처라 층 이름이 일치한다. 이름으로 짝을 찾고,
        # 없으면(InputLayer 등 이름이 백본별로 다른 층) 조용히 건너뛴다.
        try:
            target_layer = gray_backbone.get_layer(source_layer.name)
        except ValueError:
            continue

        # 가중치가 없는 층(ReLU, Add, Pad 등)은 복사할 것이 없다.
        source_weights = source_layer.get_weights()
        if not source_weights:
            continue

        if source_layer.name == "Conv1":
            # Conv1 커널 shape: (3, 3, in_ch, 32). 여기서만 in_ch가 3 vs 1로 다르다.
            kernel = source_weights[0]
            if reduction == "mean":
                # 평균: RGB 세 채널에 같은 회색 값을 넣은 원본 응답의 1/3이 되어
                # 첫 층의 출력 크기가 작아진다.
                transferred_kernel = kernel.mean(axis=2, keepdims=True)
            else:  # 합산 방식(sum)
                # 합산: RGB 세 채널에 같은 회색 값이 들어왔을 때의 응답과 정확히 같아진다.
                #   원본: k_R·g + k_G·g + k_B·g = (k_R+k_G+k_B)·g
                # 즉 그레이스케일 입력에 대해 사전학습 필터의 응답 크기를 그대로 보존하므로
                # 뒤따르는 BatchNorm 통계(ImageNet에서 학습된 값)와도 스케일이 맞는다.
                # → 이 저장소는 sum을 채택했고 mean은 기각된 변형이다(AGENTS.md).
                transferred_kernel = kernel.sum(axis=2, keepdims=True)
            # Conv1은 use_bias=False라 가중치 목록이 커널 하나뿐이다.
            target_layer.set_weights([transferred_kernel])
            copied += 1
            continue

        # Conv1 외 층은 shape이 완전히 같을 때만 그대로 복사한다.
        # (개수나 shape이 다르면 조용히 건너뛴다 — 잘못 끼워 넣느니 랜덤 초기화를 남긴다.)
        target_weights = target_layer.get_weights()
        if len(source_weights) != len(target_weights):
            continue
        if all(sw.shape == tw.shape for sw, tw in zip(source_weights, target_weights)):
            target_layer.set_weights(source_weights)
            copied += 1

    # 복사된 층 수를 반드시 로그로 남긴다. 이 숫자가 갑자기 줄면 이식이 조용히 실패한 것이다.
    print(f"[{label} backbone] copied ImageNet weights (reduction={reduction}) into {copied} MobileNetV2 layers")


# 백본 특징 벡터를 클래스 logits으로 변환한다. classifier_as_conv=True면 Dense 대신 1x1 Conv2D를 쓴다(NPU 호환용).
#
# 구조: [Dense/Conv1x1 + ReLU] → [Dropout] → [Dense/Conv1x1] → logits
# 1x1 Conv2D는 (1,1) 공간 위치에서 채널 방향 내적을 하므로 Dense와 수학적으로 동등하다.
# 일부 NPU가 FullyConnected를 지원하지 않아 Conv 형태를 선택할 수 있게 열어 둔 것이며,
# 학습(tf_train.py)에서는 항상 기본값 False = Dense 경로를 쓴다.
def _build_classifier_head(x, classifier_units, dropout, num_classes, classifier_as_conv, dtype=None):
    if classifier_as_conv:
        # Conv2D는 4D 입력이 필요하다. pooling="avg"로 이미 (batch, C)가 된 특징을
        # (batch, 1, 1, C)로 되돌려 준다.
        if len(x.shape) == 2:
            x = layers.Reshape((1, 1, x.shape[-1]), name="fused_reshape_4d")(x)
        if classifier_units > 0:
            x = layers.Conv2D(classifier_units, kernel_size=(1, 1), activation="relu", name="classifier_dense_conv")(x)
        if dropout > 0:
            x = layers.Dropout(dropout, name="classifier_dropout")(x)
        # 마지막 층에는 활성화가 없다 → raw logits (손실 쪽 from_logits=True와 짝을 이룬다).
        logits_4d = layers.Conv2D(num_classes, kernel_size=(1, 1), name="logits_conv", dtype=dtype)(x)
        # 앱이 기대하는 출력 shape은 (batch, num_classes)이므로 다시 2D로 편다.
        return layers.Reshape((num_classes,), name="logits", dtype=dtype)(logits_4d)
    else:
        # classifier_units=0이면 은닉층 없이 백본 특징에서 곧바로 logits을 뽑는다.
        if classifier_units > 0:
            x = layers.Dense(classifier_units, activation="relu", name="classifier_dense")(x)
        # Dropout은 학습 시에만 동작하고 추론에서는 항등 함수다(Keras가 자동 처리).
        if dropout > 0:
            x = layers.Dropout(dropout, name="classifier_dropout")(x)
        # 출력 층 이름 "logits"는 TFLite 서명/검증(export_validator)에서 참조하므로 바꾸지 말 것.
        return layers.Dense(num_classes, name="logits", dtype=dtype)(x)


# ImageNet 정규화로 들어온 RGB를 MobileNetV2가 기대하는 [-1,1]로 되돌린다(앱 전처리 계약 유지용 보정 레이어).
#
# 왜 이런 층이 필요한가: 데이터셋/PyTorch/안드로이드 앱은 모두 ImageNet 표준화
# ((x-mean)/std)로 입력을 만든다. 반면 keras.applications.MobileNetV2의 사전학습
# 가중치는 [-1,1] 입력을 전제로 학습됐다. 앱 전처리를 바꿀 수 없으므로,
# 모델 안에서 되돌린 뒤 다시 [-1,1]로 보내 양쪽 계약을 모두 지킨다.
def _rgb_current_norm_to_mobilenet_range(x):
    # 입력은 기존 Android/PyTorch 계약을 따른다.
    # rgb = (raw_0_1 - ImageNet 평균) / ImageNet 표준편차
    mean = tf.constant(RGB_MEAN, dtype=tf.float32)
    std = tf.constant(RGB_STD, dtype=tf.float32)
    raw_0_1 = x * std + mean   # 표준화를 역산해 원래 0~1 값 복원
    return raw_0_1 * 2.0 - 1.0  # 0~1 → -1~1 선형 매핑


# RGB/IR 2입력 모델: 백본 2개로 각각 특징을 뽑아 concat한 뒤(late fusion) 분류 헤드에 통과시킨다.
def build_dual_mobilenetv2(
    rgb_weights="imagenet",
    dropout=0.2,
    classifier_units=1024,
    gray_imagenet_init=True,
    rgb_input_mobilenet_range=False,  # True면 입력이 이미 [-1,1]이라고 보고 보정 층을 넣지 않는다(변환용)
    average_pool_op=False,            # True면 pooling 대신 명시적 AveragePooling2D 층 사용(NPU 호환용)
    fixed_batch_size=None,            # None이면 동적 배치. 변환 시에만 1로 고정한다
    classifier_as_conv=False,         # True면 헤드를 1x1 Conv로 구성(NPU 호환용)
    conv1_reduction="sum",
):
    # 접두사 이름으로 TFLite 서명과 입력 목록의 순서를 RGB, IR 순으로 고정한다.
    # (TFLite는 입력 순서를 이름 사전순으로 정렬하므로 a_/b_ 접두사로 순서를 못박는다.
    #  안드로이드 앱이 인덱스 0=RGB, 1=IR로 값을 넣기 때문에 이 순서가 계약이다.)
    rgb_name, rgb_shape = MODEL_INPUT_SIGNATURES["dual"][0]
    ir_name, ir_shape = MODEL_INPUT_SIGNATURES["dual"][1]
    rgb_input = keras.Input(batch_size=fixed_batch_size, shape=rgb_shape, name=rgb_name)
    ir_input = keras.Input(batch_size=fixed_batch_size, shape=ir_shape, name=ir_name)

    if rgb_input_mobilenet_range:
        rgb_preprocessed = rgb_input
    else:
        # 학습 경로는 항상 이쪽. 데이터셋이 ImageNet 표준화로 주므로 [-1,1]로 되돌린다.
        rgb_preprocessed = layers.Lambda(
            _rgb_current_norm_to_mobilenet_range,
            name="rgb_to_mobilenet_range",
        )(rgb_input)

    # include_top=False: 1000-클래스 ImageNet 분류기를 떼고 특징 추출기만 쓴다.
    # pooling="avg": 마지막 (7,7,1280) 특징맵을 전역 평균 풀링해 (1280,) 벡터로 만든다.
    rgb_backbone = keras.applications.MobileNetV2(
        input_shape=rgb_shape,
        include_top=False,
        weights=rgb_weights,
        pooling=None if average_pool_op else "avg",
        name="rgb_mobilenetv2",
    )
    # IR 백본은 1채널 입력이라 ImageNet 가중치를 직접 받을 수 없다(weights=None).
    # 사전학습 값은 바로 아래에서 수동 이식한다.
    ir_backbone = keras.applications.MobileNetV2(
        input_shape=ir_shape,
        include_top=False,
        weights=None,
        pooling=None if average_pool_op else "avg",
        name="ir_mobilenetv2",
    )

    # RGB 쪽이 사전학습 가중치를 갖고 있을 때만 이식이 의미가 있다
    # (rgb_weights=None이면 소스가 랜덤이라 복사해도 얻는 게 없다).
    if rgb_weights is not None and gray_imagenet_init:
        _transfer_imagenet_weights_to_gray_backbone(rgb_backbone, ir_backbone, "IR", reduction=conv1_reduction)

    # RGB와 IR은 서로 다른 모달리티이므로 두 백본은 가중치를 공유하지 않는다.
    rgb_features = rgb_backbone(rgb_preprocessed)
    ir_features = ir_backbone(ir_input)
    if average_pool_op:
        rgb_features = layers.AveragePooling2D(pool_size=(7, 7), name="rgb_average_pool")(rgb_features)
        rgb_features = layers.Reshape((1280,), name="rgb_reshape")(rgb_features)
        ir_features = layers.AveragePooling2D(pool_size=(7, 7), name="ir_average_pool")(ir_features)
        ir_features = layers.Reshape((1280,), name="ir_reshape")(ir_features)
    # Late fusion: 픽셀이 아니라 '특징 벡터' 단계에서 합친다. (1280) + (1280) → (2560)
    # 초기 융합(입력 채널 concat) 대비 장점은 각 모달리티가 자기 통계에 맞는 필터를
    # 따로 학습할 수 있다는 점이다.
    fused = layers.Concatenate(name="fused_features")([rgb_features, ir_features])
    logits = _build_classifier_head(fused, classifier_units, dropout, len(CLASS_NAMES), classifier_as_conv)
    # 입력 리스트 순서 = TFLite 입력 순서 = 앱이 값을 넣는 순서.
    return keras.Model(inputs=[rgb_input, ir_input], outputs=logits, name="dual_mobilenetv2")


# RGB 또는 IR 단일 입력 모델. crop_ir일 때는 가중치 이식용 RGB 백본을 임시로 만들어 쓰고 버린다.
def build_single_mobilenetv2(
    input_type="crop_rgb",
    rgb_weights="imagenet",
    dropout=0.2,
    classifier_units=1024,
    gray_imagenet_init=True,
    rgb_input_mobilenet_range=False,
    average_pool_op=False,
    fixed_batch_size=None,
    classifier_as_conv=False,
    conv1_reduction="sum",
):
    if input_type not in ("crop_rgb", "crop_ir"):
        raise ValueError(f"Unknown input_type: {input_type}")

    # 단일 입력이므로 서명 튜플의 첫 원소 하나만 쓴다.
    input_name, input_shape = MODEL_INPUT_SIGNATURES[input_type][0]

    if input_type == "crop_rgb":
        rgb_input = keras.Input(batch_size=fixed_batch_size, shape=input_shape, name=input_name)
        if rgb_input_mobilenet_range:
            rgb_preprocessed = rgb_input
        else:
            rgb_preprocessed = layers.Lambda(
                _rgb_current_norm_to_mobilenet_range,
                name="rgb_to_mobilenet_range",
            )(rgb_input)

        rgb_backbone = keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=rgb_weights,
            pooling=None if average_pool_op else "avg",
            name="crop_rgb_mobilenetv2",
        )

        features = rgb_backbone(rgb_preprocessed)
        if average_pool_op:
            features = layers.AveragePooling2D(pool_size=(7, 7), name="crop_rgb_average_pool")(features)
            features = layers.Reshape((1280,), name="crop_rgb_reshape")(features)

        inputs = rgb_input

    else:  # crop_ir 입력 경로
        ir_input = keras.Input(batch_size=fixed_batch_size, shape=input_shape, name=input_name)
        ir_backbone = keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=None,
            pooling=None if average_pool_op else "avg",
            name="crop_ir_mobilenetv2",
        )

        if rgb_weights is not None and gray_imagenet_init:
            # crop_ir 모델에는 RGB 백본이 없으므로, 가중치를 꺼내 올 3채널 백본을
            # 임시로 만든다. 이식이 끝나면 최종 Model에 포함되지 않아 그대로 버려진다
            # (ImageNet 가중치를 로드하는 시간·메모리만 잠깐 쓰는 셈).
            temp_rgb_backbone = keras.applications.MobileNetV2(
                input_shape=(input_shape[0], input_shape[1], 3),
                include_top=False,
                weights=rgb_weights,
                pooling=None,
                name="temp_rgb_mobilenetv2",
            )
            _transfer_imagenet_weights_to_gray_backbone(temp_rgb_backbone, ir_backbone, "IR", reduction=conv1_reduction)

        features = ir_backbone(ir_input)
        if average_pool_op:
            features = layers.AveragePooling2D(pool_size=(7, 7), name="crop_ir_average_pool")(features)
            features = layers.Reshape((1280,), name="crop_ir_reshape")(features)

        inputs = ir_input

    logits = _build_classifier_head(features, classifier_units, dropout, len(CLASS_NAMES), classifier_as_conv)

    return keras.Model(inputs=inputs, outputs=logits, name=f"single_{input_type}_mobilenetv2")


# MobileNetV2 백본 생성 래퍼. 현재 호출부 없음(dead code).
def _make_backbone(input_shape, weights, pooling, name):
    return keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
        pooling=pooling,
        name=name,
    )


# 7x7 feature map을 평균 풀링해 1280차원 벡터로 만든다. 현재 호출부 없음(dead code).
def _pool_backbone_output(features, prefix):
    features = layers.AveragePooling2D(pool_size=(7, 7), name=f"{prefix}_average_pool")(features)
    return layers.Reshape((1280,), name=f"{prefix}_reshape")(features)


# 이 파일을 직접 실행해 모델 구조만 확인할 때 쓰는 CLI 인자 파서(학습 경로는 tf_train.py가 따로 정의).
def parse_args():
    parser = argparse.ArgumentParser(description="Keras MobileNetV2 모델을 만들고 구조를 출력합니다.")
    parser.add_argument("--rgb-weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--classifier-units", type=int, default=1024)
    parser.add_argument("--no-gray-imagenet-init", action="store_true")
    parser.add_argument(
        "--model-type",
        choices=["dual", "crop_rgb", "crop_ir"],
        default="dual",
        help="학습할 모델 종류 (dual: 2입력, crop_rgb: 단일 RGB, crop_ir: 단일 IR)"
    )
    parser.add_argument(
        "--conv1-reduction",
        choices=["mean", "sum"],
        default="sum",
        help="1채널 Conv1 가중치 이식 시 축소 방식 (mean: 평균, sum: 합산)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rgb_weights = None if args.rgb_weights == "none" else args.rgb_weights
    if args.model_type == "dual":
        model = build_dual_mobilenetv2(
            rgb_weights=rgb_weights,
            dropout=args.dropout,
            classifier_units=args.classifier_units,
            gray_imagenet_init=not args.no_gray_imagenet_init,
            conv1_reduction=args.conv1_reduction,
        )
        dummy_inputs = [
            tf.zeros((1, 224, 224, 3), dtype=tf.float32),
            tf.zeros((1, 224, 224, 1), dtype=tf.float32),
        ]
    else:
        model = build_single_mobilenetv2(
            input_type=args.model_type,
            rgb_weights=rgb_weights,
            dropout=args.dropout,
            classifier_units=args.classifier_units,
            gray_imagenet_init=not args.no_gray_imagenet_init,
            conv1_reduction=args.conv1_reduction,
        )
        if args.model_type == "crop_rgb":
            dummy_inputs = tf.zeros((1, 224, 224, 3), dtype=tf.float32)
        else:
            dummy_inputs = tf.zeros((1, 224, 224, 1), dtype=tf.float32)
    model.summary()
    out = model(dummy_inputs, training=False)
    print("output shape:", out.shape)
