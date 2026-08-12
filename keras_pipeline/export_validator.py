"""NPU용 모델 재조립 + 동등성 검증 + 산출물 매니페스트 작성.

이름은 "validator"지만 실제로는 네 가지 일을 한다.

  1) build_npu_export_model      — 학습 모델과 같은 계산을 하지만 NPU가 소화할 수 있는
                                   구조로 된 모델을 새로 만들고 가중치를 옮겨 담는다
  2) validate_npu_export_parity  — 그 갈아끼우기가 옳았는지 logits 비교로 증명한다
  3) inspect_tflite              — 변환된 .tflite의 실제 텐서 정보를 출력·검증한다
  4) write_tflite_sidecar_manifest — 안드로이드 앱이 읽는 계약서 JSON을 쓴다

1번이 필요한 이유: 학습 그래프를 그대로 NPU(NNAPI)에 올리면 지원하지 않는 연산 때문에
준비 단계에서 실패한다. 현재 안드로이드 master는 NNAPI 준비/워밍업 실패 시 CPU로
폴백하지 않고 모델 슬롯을 거부하므로, 구조를 미리 맞춰 내보내야 한다.
"""
import json
import os
import numpy as np
import tensorflow as tf

from classes import CLASS_NAMES
from keras_pipeline.model_signature import validate_tflite_model_signature
from keras_pipeline.tf_dataset import RGB_MEAN, RGB_STD


# tf_model._rgb_current_norm_to_mobilenet_range와 같은 식의 numpy 판.
# export 모델에는 이 변환을 하던 Lambda 층이 없으므로(앱이 직접 [-1,1]로 넣는 계약),
# 검증·캘리브레이션 입력을 만들 때 파이썬 쪽에서 대신 적용해 준다.
def _rgb_imagenet_norm_to_mobilenet_range(rgb):
    raw_0_1 = rgb * RGB_STD + RGB_MEAN
    return raw_0_1 * 2.0 - 1.0


# 학습 모델의 백본 가중치를 export 모델의 같은 이름 백본으로 통째로 옮긴다.
# 백본은 Model 안에 Model이 들어 있는 중첩 구조라, 바깥 층 이름(rgb_mobilenetv2)으로
# 찾은 뒤 그 안의 하위 층을 이름으로 하나씩 짝지어 복사해야 한다.
#
# tf_model의 ImageNet 이식과 달리 여기서는 shape이 안 맞으면 조용히 건너뛰지 않고
# 전부 예외로 던진다. 이식은 "안 되면 랜덤으로 두면 되는" 최적화지만, 이쪽은
# 한 층이라도 빠지면 학습 결과가 반영되지 않은 모델이 배포되기 때문이다.
def _copy_nested_weights(source_model, target_model, layer_name):
    source_layer = source_model.get_layer(layer_name)
    target_layer = target_model.get_layer(layer_name)

    source_sub_layers = {layer.name: layer for layer in source_layer.layers}
    # 가중치를 가진 층만 비교 대상이다(ReLU·Add 같은 층은 옮길 값이 없다).
    target_weighted_layers = [layer for layer in target_layer.layers if layer.get_weights()]
    source_weighted_names = {
        layer.name for layer in source_layer.layers if layer.get_weights()
    }
    target_weighted_names = {layer.name for layer in target_weighted_layers}
    # 학습 모델에는 있는데 export 모델에 없는 층 = 옮길 곳이 없는 가중치.
    # 아래 루프는 target 기준으로 도니 이 경우를 못 잡는다 → 여기서 미리 확인한다.
    missing_in_target = source_weighted_names - target_weighted_names
    if missing_in_target:
        raise ValueError(
            f"{layer_name}: export backbone에 없는 source weighted layer: "
            f"{sorted(missing_in_target)}"
        )

    copied = []
    for target_sub_layer in target_layer.layers:
        target_weights = target_sub_layer.get_weights()
        if not target_weights:
            continue

        # 반대 방향 누락: export 모델에는 있는데 학습 모델에 없는 층.
        source_sub_layer = source_sub_layers.get(target_sub_layer.name)
        if source_sub_layer is None:
            raise ValueError(
                f"{layer_name}: source backbone에 없는 export weighted layer: "
                f"{target_sub_layer.name}"
            )
        # 이름이 같아도 shape이 다르면 다른 층이다. 그대로 넣으면 조용히 망가지므로 실패시킨다.
        source_weights = source_sub_layer.get_weights()
        source_shapes = [tuple(weight.shape) for weight in source_weights]
        target_shapes = [tuple(weight.shape) for weight in target_weights]
        if source_shapes != target_shapes:
            raise ValueError(
                f"{layer_name}/{target_sub_layer.name}: weight tensor shape mismatch "
                f"(source={source_shapes}, target={target_shapes})"
            )
        target_sub_layer.set_weights(source_weights)
        copied.append((target_sub_layer.name, source_shapes))

    print(f"[weight copy] {layer_name}: {copied}")


# 동등성 비교용 입력 한 쌍을 만든다. 두 모델이 기대하는 RGB 범위가 다르기 때문에
# 같은 이미지라도 넣는 값이 달라야 한다.
#   학습 모델  : ImageNet 표준화 값 그대로 (모델 안의 Lambda가 [-1,1]로 바꿔 준다)
#   export 모델: 이미 [-1,1]로 바꾼 값 (Lambda가 없으므로 밖에서 해 준다)
# IR은 양쪽 모두 [-1,1]이라 변환 없이 그대로 쓴다.
def _npu_parity_inputs(sample, model_type):
    # 모델은 배치 축을 요구하므로 (224,224,3) → (1,224,224,3)로 한 겹 씌운다.
    def batch(value):
        return np.expand_dims(value, axis=0).astype(np.float32)

    if model_type == "dual":
        return [batch(sample[0]), batch(sample[1])], [
            batch(_rgb_imagenet_norm_to_mobilenet_range(sample[0])),
            batch(sample[1]),
        ]
    if model_type == "crop_rgb":
        return [batch(sample[0])], [batch(_rgb_imagenet_norm_to_mobilenet_range(sample[0]))]
    if model_type == "crop_ir":
        return [batch(sample[1])], [batch(sample[1])]

    # 아래는 제거된 multimodal 5-입력 모델용 경로다. --model-type의 choices가
    # dual/crop_rgb/crop_ir 셋뿐이라 현재는 도달할 수 없다(dead code).
    source_inputs = [batch(value) for value in sample]
    export_inputs = [
        batch(_rgb_imagenet_norm_to_mobilenet_range(sample[0])),
        batch(sample[1]),
        batch(_rgb_imagenet_norm_to_mobilenet_range(sample[2])),
        batch(sample[3]),
        batch(sample[4]),
    ]
    return source_inputs, export_inputs


# 재조립한 export 모델이 학습 모델과 '같은 계산'을 하는지 실제 값으로 확인한다.
# 구조를 바꾸고 가중치를 손으로 옮겼기 때문에, 한 층이라도 어긋나면 조용히 다른
# 모델이 나간다. 그것을 막는 마지막 관문이다. (양자화 오차는 아직 없는 단계 —
# Keras 대 Keras, float 대 float 비교라 결과가 거의 정확히 같아야 정상이다.)
def validate_npu_export_parity(trained_model, export_model, sample, model_type):
    """NPU export 그래프가 Keras logits을 보존하지 못하면 실패시킨다."""
    source_inputs, export_inputs = _npu_parity_inputs(sample, model_type)
    # 단일 입력 모델은 리스트가 아니라 텐서 하나를 그대로 넘겨야 한다.
    source_call_inputs = source_inputs[0] if len(source_inputs) == 1 else source_inputs
    export_call_inputs = export_inputs[0] if len(export_inputs) == 1 else export_inputs
    # training=False로 Dropout·BatchNorm을 추론 모드로 고정한다(안 그러면 매번 값이 달라진다).
    trained_logits = trained_model(source_call_inputs, training=False).numpy()
    export_logits = export_model(export_call_inputs, training=False).numpy()
    error = np.abs(trained_logits - export_logits)
    result = {
        "max_error": float(error.max()),
        "mean_error": float(error.mean()),
        "argmax_agreement": bool(np.array_equal(
            np.argmax(trained_logits, axis=-1), np.argmax(export_logits, axis=-1)
        )),
    }
    # 두 조건을 모두 요구한다.
    #  - argmax 일치: 예측 클래스가 하나라도 다르면 실격 (실사용에 직결되는 기준)
    #  - allclose  : 예측이 같아도 logits 값 자체가 1e-5 넘게 벌어지면 어딘가 틀린 것
    if not result["argmax_agreement"] or not np.allclose(
        trained_logits, export_logits, rtol=1e-5, atol=1e-5
    ):
        raise ValueError(f"NPU export Keras logits parity failed: {result}")
    print(f"[NPU export Keras logits parity] {result}")
    return result


# 학습 모델과 같은 계산을 하되 NPU가 소화 가능한 구조로 된 모델을 만들어 가중치를 옮긴다.
#
#   학습 모델                    → export 모델              (이유)
#   Dense                        → Conv2D 1x1               FullyConnected 미지원
#   pooling="avg"                → AveragePooling2D 층      명시적 연산 필요
#   배치 크기 동적(None)         → 1로 고정                  NPU는 고정 크기 요구
#   입력 보정 Lambda 층          → 없음(앱이 [-1,1]로 넣음)  그래프 단순화
#   Dropout(학습 설정값)         → 0.0                      추론에는 불필요
#
# Dense (2560,1024)를 Conv2D (1,1,2560,1024)로 reshape하는 것은 모양만 바꾸는 일이라
# 숫자는 하나도 변하지 않는다. 1x1 Conv는 (1,1) 위치에서 채널 방향 내적을 하므로
# Dense와 수학적으로 동일한 연산이다.
def build_npu_export_model(trained_model, model_type):
    # 학습 체크포인트에서 classifier 구성을 읽는다 (--classifier-units 0 지원).
    # 인자로 다시 받지 않고 모델에서 직접 읽는 이유는, 변환 시점에 학습 때와 다른
    # 값을 넘겨 구조가 어긋나는 사고를 원천 차단하기 위해서다.
    try:
        trained_dense = trained_model.get_layer("classifier_dense")
        classifier_units = trained_dense.units
    except ValueError:
        trained_dense = None
        classifier_units = 0

    if model_type == "dual":
        # 함수 안에서 import하는 이유: tf_model이 이 모듈을 다시 참조하는 순환 import를 피한다.
        from keras_pipeline.tf_model import build_dual_mobilenetv2
        export_model = build_dual_mobilenetv2(
            # ImageNet을 다시 받을 필요가 없다 — 어차피 학습 가중치로 전부 덮어쓴다.
            rgb_weights=None,
            dropout=0.0,
            classifier_units=classifier_units,
            gray_imagenet_init=False,
            rgb_input_mobilenet_range=True,
            average_pool_op=True,
            fixed_batch_size=1,
            classifier_as_conv=True,
        )
        backbones = ["rgb_mobilenetv2", "ir_mobilenetv2"]
    else:
        from keras_pipeline.tf_model import build_single_mobilenetv2
        export_model = build_single_mobilenetv2(
            input_type=model_type,
            rgb_weights=None,
            dropout=0.0,
            classifier_units=classifier_units,
            gray_imagenet_init=False,
            rgb_input_mobilenet_range=True,
            average_pool_op=True,
            fixed_batch_size=1,
            classifier_as_conv=True,
        )
        backbones = [f"{model_type}_mobilenetv2"]

    # ① 백본(구조가 동일) — 이름으로 짝지어 통째로 복사.
    for layer_name in backbones:
        _copy_nested_weights(trained_model, export_model, layer_name)

    # ② classifier_dense(Dense)를 classifier_dense_conv(1x1 Conv2D)로 이식한다.
    #    (2560, 1024) → (1, 1, 2560, 1024). bias는 모양이 같아 그대로 쓴다.
    if trained_dense is not None:
        export_dense_conv = export_model.get_layer("classifier_dense_conv")
        dense_w, dense_b = trained_dense.get_weights()
        conv_w = np.reshape(dense_w, (1, 1, dense_w.shape[0], dense_w.shape[1]))
        export_dense_conv.set_weights([conv_w, dense_b])

    # ③ logits(Dense)를 logits_conv(1x1 Conv2D)로 이식한다.
    #    (1024, 10) → (1, 1, 1024, 10). 이 층은 classifier_units=0이어도 항상 존재한다.
    trained_logits = trained_model.get_layer("logits")
    export_logits_conv = export_model.get_layer("logits_conv")
    logits_w, logits_b = trained_logits.get_weights()
    logits_conv_w = np.reshape(logits_w, (1, 1, logits_w.shape[0], logits_w.shape[1]))
    export_logits_conv.set_weights([logits_conv_w, logits_b])

    return export_model


# 변환이 끝난 .tflite 파일을 실제로 열어 서명을 검사하고 텐서 정보를 로그에 남긴다.
# 파일로 저장된 실물 기준으로 이름·shape·양자화 파라미터를 확인하고 로그에 남기는 단계다.
def inspect_tflite(path, model_type):
    interpreter = tf.lite.Interpreter(model_path=path)
    # allocate_tensors()를 불러야 텐서 정보(details)를 읽을 수 있다.
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    validate_tflite_model_signature(input_details, output_details, model_type)
    print("[tflite tensors]")
    for idx, detail in enumerate(input_details):
        print(
            f" input {idx}: name={detail['name']} shape={detail['shape'].tolist()} "
            f"dtype={detail['dtype']} quant={detail['quantization']}"
        )
    for idx, detail in enumerate(output_details):
        print(
            f" output {idx}: name={detail['name']} shape={detail['shape'].tolist()} "
            f"dtype={detail['dtype']} quant={detail['quantization']}"
        )


# .tflite 옆에 나란히 두는 계약서 JSON을 만든다.
#
# .tflite 파일만으로는 "이 입력에 어떤 정규화를 적용해야 하는지", "출력 인덱스 3이
# 어떤 클래스인지"를 알 수 없다. 앱이 그걸 추측하면 조용히 틀린 예측을 하게 되므로,
# 변환기가 직접 읽어낸 사실(이름·shape·dtype·양자화 파라미터)과 이 저장소가 아는
# 계약(정규화·클래스 순서)을 한 파일에 적어 함께 배포한다.
# 안드로이드 배포 시 .tflite와 이 매니페스트를 반드시 쌍으로 복사해야 한다.
def write_tflite_sidecar_manifest(tflite_path, model_type):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 변형 구분을 파일명으로 한다. npu_int8만 정규화 계약과 delegate가 다르기 때문이다.
    # artifact_paths의 표준 이름을 전제로 하므로, 파일을 임의로 바꾸면 잘못 판별될 수 있다.
    is_npu_int8 = "npu_int8" in os.path.basename(tflite_path)

    inputs_info = []
    for idx, detail in enumerate(input_details):
        name = detail['name']
        shape = detail['shape'].tolist()
        dtype = detail['dtype'].__name__
        channels = shape[-1]

        # 앱이 이 입력 자리에 무엇을 넣어야 하는지 알려주는 표시.
        # dual은 입력이 둘이라 채널 수(3=RGB, 1=IR)로 구분한다.
        input_kind = "unknown"
        if model_type == "crop_rgb":
            input_kind = "rgb"
        elif model_type == "crop_ir":
            input_kind = "ir"
        elif model_type == "dual":
            input_kind = "rgb" if channels == 3 else "ir"

        # 양자화 파라미터. 앱은 실수값을 (값/scale + zero_point)로 int8에 담아 넣어야 한다.
        # float 모델은 (0.0, 0)으로 나오므로 그 경우 null로 두어 "양자화 없음"을 표현한다.
        scale, zero_point = detail['quantization']
        quant = None
        if scale != 0.0 or zero_point != 0:
            quant = {
                "scale": float(scale),
                "zero_point": int(zero_point)
            }

        # RGB 정규화 계약은 변형마다 다르다 — 여기가 앱과의 가장 중요한 약속이다.
        #   npu_int8 : 보정 Lambda 층을 뺐으므로 앱이 직접 [-1,1]로 만들어 넣어야 한다
        #   그 외    : 모델 안 Lambda가 변환해 주므로 ImageNet 표준화 값을 그대로 넣는다
        # 이 둘을 뒤바꾸면 예외 없이 조용히 오답만 나온다.
        if channels == 3:
            if is_npu_int8:
                norm = {
                    "mean": [0.5, 0.5, 0.5],
                    "std": [0.5, 0.5, 0.5],
                    "range": "[-1, 1]"
                }
            else:
                norm = {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "range": "imagenet"
                }
        else:
            # IR은 변형과 무관하게 항상 [-1,1] (mean=std=0.5).
            norm = {
                "mean": [0.5],
                "std": [0.5],
                "range": "[-1, 1]"
            }

        inputs_info.append({
            "name": name,
            "index": idx,
            "shape": shape,
            "dtype": dtype,
            "input_kind": input_kind,
            "quantization": quant,
            "normalization": norm
        })

    outputs_info = []
    for idx, detail in enumerate(output_details):
        name = detail['name']
        shape = detail['shape'].tolist()
        dtype = detail['dtype'].__name__

        scale, zero_point = detail['quantization']
        quant = None
        if scale != 0.0 or zero_point != 0:
            quant = {
                "scale": float(scale),
                "zero_point": int(zero_point)
            }

        outputs_info.append({
            "name": name,
            "index": idx,
            "shape": shape,
            "dtype": dtype,
            "quantization": quant,
            # softmax를 통과하지 않은 raw logits이라는 표시.
            # 앱이 확률이 필요하면 직접 softmax를 씌워야 한다.
            "output_is_logits": True
        })

    manifest = {
        "model_type": model_type,
        "file_name": os.path.basename(tflite_path),
        # 앱이 이 모델을 어느 백엔드에 올릴지. npu_int8만 NNAPI 대상이고,
        # 나머지는 CPU다("Backend CPU" = NNAPI 가속 없음).
        "delegate": "nnapi" if is_npu_int8 else "cpu",
        "inputs": inputs_info,
        "outputs": outputs_info,
        # 출력 인덱스 → 클래스 이름 대응. 앱이 결과를 해석하는 유일한 근거다.
        "class_order": CLASS_NAMES,
        # 얼굴 검출 박스를 10% 넓혀 크롭하라는 현재 앱 계약이다.
        # 학습 crop과 이 값의 일치 여부를 자동으로 검증하는 코드는 아직 없다.
        "crop_margin_ratio": 0.10
    }

    # 산출물과 같은 이름에 _manifest.json만 붙인다 (artifact_paths.sidecar_manifest_path와 동일 규칙).
    manifest_path = tflite_path.replace(".tflite", "_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[tflite sidecar manifest saved] {manifest_path}")
