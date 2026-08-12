"""tf.data 입력 파이프라인.

이미지 디코딩·증강·정규화를 OpenCV로 수행하고 tf.py_function으로 감싸 tf.data에 태운다.
공간 증강 방식과 정규화 계약은 PyTorch 파이프라인(pytorch_pipeline/dataset.py)에 맞췄지만,
resize 구현과 ColorJitter 연산 순서가 달라 픽셀 단위 결과까지 같지는 않다.

정규화 계약(앱/PyTorch와 동일해야 함):
  RGB: BGR→RGB → 0~1 → (x - ImageNet_mean) / ImageNet_std
  IR : 그레이스케일 → 0~1 → (x - 0.5) / 0.5   즉 [-1, 1]
"""
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from keras_pipeline.spec import IMAGE_SIZE, RGB_MEAN, RGB_STD, IR_MEAN, IR_STD



def load_sample(rgb_path, ir_path, augment=False, flip=0, angle=0.0, brightness_f=1.0, contrast_f=1.0, sat_f=1.0):
    """이미지를 불러와 정규화한다. augment=True이면 학습용 데이터 증강을 적용한다.

    dual 모델용 — RGB와 IR을 한 쌍으로 읽는다.
    증강 파라미터(flip/angle/brightness_f/...)는 이 함수가 직접 뽑지 않고 인자로 받는다.
    난수 생성을 밖(_sample_augmentation_params)으로 빼야 시드 기반 재현이 가능하기 때문이다.

    반환: (rgb (224,224,3) float32, ir (224,224,1) float32)
    """
    # cv2.imread는 BGR 순서로 읽고, 실패해도 예외 대신 None을 준다 → 명시적으로 검사.
    rgb = cv2.imread(rgb_path)
    if rgb is None:
        raise ValueError(f"Failed to read RGB image: {rgb_path}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)  # 모델/ImageNet 통계는 RGB 순서 기준

    # IR은 1채널로 강제 로드 (파일이 3채널로 저장돼 있어도 그레이스케일로 변환된다).
    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    if ir is None:
        raise ValueError(f"Failed to read IR image: {ir_path}")

    if augment:
        # 공간 변환: RGB/IR 동일하게 적용해 두 채널 정렬 유지
        # (한쪽만 뒤집거나 다른 각도로 돌리면 두 백본이 서로 다른 얼굴 위치를 보게 되어
        #  late fusion이 무의미해진다. 그래서 flip/angle은 반드시 공유한다.)
        if flip == 1:
            rgb = cv2.flip(rgb, 1)  # 1 = 좌우 반전 (얼굴은 좌우 대칭이라 라벨이 보존된다)
            ir = cv2.flip(ir, 1)
        h, w = rgb.shape[:2]
        # 이미지 중심을 축으로 angle도 회전, 배율 1.0(확대/축소 없음).
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rgb = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR)
        ir = cv2.warpAffine(ir, M, (w, h), flags=cv2.INTER_LINEAR)

    # PyTorch 경로와 같은 처리 순서로 원본 해상도에서 회전한 뒤 224로 resize한다.
    # 축소에는 OpenCV가 권장하는 INTER_AREA 보간을 사용한다.
    rgb = cv2.resize(rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    ir = cv2.resize(ir, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    if augment:
        # PyTorch 경로처럼 resize 뒤 RGB에만 ColorJitter를 적용한다. 다만 torchvision은
        # 밝기·대비·채도 적용 순서를 무작위화하고, 이 구현은 그 순서를 고정한다.
        # IR에는 공간 증강만 적용하며, 밝기·대비 같은 광도 증강의 효과는 아직 검증되지 않았다.
        rgb_f = rgb.astype(np.float32)
        # 1) 밝기: 전 픽셀에 상수배
        rgb_f = np.clip(rgb_f * brightness_f, 0, 255)
        # 2) 대비: 이미지 평균을 기준점으로 두고 편차를 확대/축소
        mean_val = rgb_f.mean()
        rgb_f = np.clip((rgb_f - mean_val) * contrast_f + mean_val, 0, 255)
        # 3) 채도: ITU-R BT.601 가중치로 만든 흑백 버전과 원본을 sat_f로 선형 보간
        #    sat_f=0이면 완전 흑백, 1이면 원본, 1보다 크면 색이 진해진다.
        gray = (0.299 * rgb_f[:, :, 0] + 0.587 * rgb_f[:, :, 1] + 0.114 * rgb_f[:, :, 2])[:, :, np.newaxis]
        rgb_f = np.clip(gray + sat_f * (rgb_f - gray), 0, 255)
        rgb = rgb_f.astype(np.uint8)  # 이후 0~1 변환을 위해 uint8로 되돌린다.

    # 정규화: 0~255 → 0~1 → ImageNet 표준화. 파일 상단 계약 참고.
    rgb = rgb.astype(np.float32) / 255.0
    rgb = (rgb - RGB_MEAN) / RGB_STD

    ir = ir.astype(np.float32) / 255.0
    ir = np.expand_dims(ir, axis=-1)  # (224,224) → (224,224,1), Conv2D는 채널축을 요구한다
    ir = (ir - IR_MEAN) / IR_STD      # mean=std=0.5 → 결과 범위 [-1, 1]

    return rgb.astype(np.float32), ir.astype(np.float32)


def load_single_sample(path, input_type="crop_rgb", augment=False, flip=0, angle=0.0, brightness_f=1.0, contrast_f=1.0, sat_f=1.0):
    """단일 이미지(RGB 혹은 IR)를 불러와 정규화하고 필요시 증강한다.

    crop_rgb / crop_ir 단일 입력 모델용. 처리 내용은 load_sample의 해당 모달리티 절반과
    동일하다(로직을 공유하지 않고 복제해 둔 상태 — 한쪽만 고치지 않도록 주의).
    IR 분기는 색 증강 인자를 받기만 하고 쓰지 않는다(IR에 ColorJitter를 걸지 않는 정책).
    """
    if input_type == "crop_rgb":
        rgb = cv2.imread(path)
        if rgb is None:
            raise ValueError(f"Failed to read RGB image: {path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        if augment:
            if flip == 1:
                rgb = cv2.flip(rgb, 1)
            h, w = rgb.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            rgb = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR)

        rgb = cv2.resize(rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

        if augment:
            # RGB에만 ColorJitter 적용
            rgb_f = rgb.astype(np.float32)
            rgb_f = np.clip(rgb_f * brightness_f, 0, 255)
            mean_val = rgb_f.mean()
            rgb_f = np.clip((rgb_f - mean_val) * contrast_f + mean_val, 0, 255)
            gray = (0.299 * rgb_f[:, :, 0] + 0.587 * rgb_f[:, :, 1] + 0.114 * rgb_f[:, :, 2])[:, :, np.newaxis]
            rgb_f = np.clip(gray + sat_f * (rgb_f - gray), 0, 255)
            rgb = rgb_f.astype(np.uint8)

        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - RGB_MEAN) / RGB_STD
        return rgb.astype(np.float32)
    else:  # crop_ir 입력 경로
        ir = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if ir is None:
            raise ValueError(f"Failed to read IR image: {path}")

        if augment:
            if flip == 1:
                ir = cv2.flip(ir, 1)
            h, w = ir.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            ir = cv2.warpAffine(ir, M, (w, h), flags=cv2.INTER_LINEAR)

        ir = cv2.resize(ir, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

        ir = ir.astype(np.float32) / 255.0
        ir = np.expand_dims(ir, axis=-1)
        ir = (ir - IR_MEAN) / IR_STD
        return ir.astype(np.float32)


def _sample_augmentation_params(index, seed, augment):
    """tf.data map 함수 내에서 index와 seed 기반의 결정론적 증강 파라미터를 추출한다.

    핵심: tf.random.stateless_* 는 전역 난수 상태가 아니라 넘겨준 seed 텐서만으로 값이
    정해진다. seed를 (샘플 index, 학습 시드)로 만들었기 때문에
      - 같은 시드 + 같은 index → 항상 같은 증강 (재현 가능, tests/dataset/test_deterministic_augmentation.py가 검증)
      - index가 다르면 → 서로 다른 난수 입력을 사용한다(결과 값이 우연히 같을 수는 있다)
    이 두 가지가 동시에 성립한다. stateless 난수를 쓰면 병렬 map의 실행 순서와 난수 결과를
    분리할 수 있어 실행 간 재현성이 유지된다.

    index는 파일 목록 내 위치가 아니라 repeat 뒤에 붙인 enumerate가 매기는 '스트림 전역
    카운터'다(make_dataset 참고). 데이터 스트림을 계속 소비할수록 index가 증가하므로 같은
    이미지가 나중에 다시 등장하면 다른 난수 입력을 받는다. 이 반복 주기는 batch 경계 때문에
    Keras 학습 에폭과 정확히 일치하지 않을 수 있으며, 난수 수열 자체는 seed로 고정된다.
    """
    if augment:
        # 2원소 int64 텐서가 stateless 계열이 요구하는 seed 형식이다.
        seed_tensor = tf.stack([index, tf.cast(seed, tf.int64)])
        # 파라미터마다 seed의 두 번째 성분을 +1씩 밀어 별도의 난수 입력을 사용한다.
        flip_val = tf.random.stateless_uniform([], seed=seed_tensor, minval=0, maxval=2, dtype=tf.int32)          # 0 또는 1
        angle_val = tf.random.stateless_uniform([], seed=seed_tensor + [0, 1], minval=-10.0, maxval=10.0, dtype=tf.float32)   # ±10도
        brightness_val = tf.random.stateless_uniform([], seed=seed_tensor + [0, 2], minval=0.7, maxval=1.3, dtype=tf.float32) # ±30%
        contrast_val = tf.random.stateless_uniform([], seed=seed_tensor + [0, 3], minval=0.7, maxval=1.3, dtype=tf.float32)   # ±30%
        sat_val = tf.random.stateless_uniform([], seed=seed_tensor + [0, 4], minval=0.8, maxval=1.2, dtype=tf.float32)        # ±20%
    else:
        # 검증/변환 경로: 전부 항등원(뒤집지 않음, 0도, 배율 1.0)이라 원본이 그대로 통과한다.
        flip_val = tf.constant(0, dtype=tf.int32)
        angle_val = tf.constant(0.0, dtype=tf.float32)
        brightness_val = tf.constant(1.0, dtype=tf.float32)
        contrast_val = tf.constant(1.0, dtype=tf.float32)
        sat_val = tf.constant(1.0, dtype=tf.float32)
    return flip_val, angle_val, brightness_val, contrast_val, sat_val


def make_dataset(items, batch_size=8, shuffle=False, seed=42, augment=False, repeat=False):
    """dual(RGB+IR) 모델용 tf.data 데이터셋을 만든다.

    items: [(rgb_path, ir_path, label), ...]
    반환 원소 형태: ((rgb, ir), label) — 모델 입력이 2개이므로 x가 튜플이다.
    repeat=True면 무한 반복 데이터셋이 된다(학습용, fit이 steps_per_epoch로 끊는다).
    """
    items = list(items)  # 제너레이터로 들어와도 두 번 순회할 수 있게 실체화
    if not items:
        raise ValueError("Dataset items list cannot be empty.")
    if shuffle:
        # 파이썬 레벨 1차 셔플. collect_split_items가 클래스별로 정렬해 모아 오므로
        # 이 단계가 없으면 아래 index(= 증강 시드)가 클래스 순서와 붙어 버린다.
        random.Random(seed).shuffle(items)

    # 튜플 리스트를 컬럼별 리스트로 전치한다(from_tensor_slices가 요구하는 형태).
    rgb_paths = [item[0] for item in items]
    ir_paths = [item[1] for item in items]
    labels = [item[2] for item in items]

    # 이미지가 아니라 '경로 문자열'만 텐서로 올린다 → 메모리에 데이터셋 전체를 올리지 않는다.
    ds = tf.data.Dataset.from_tensor_slices((rgb_paths, ir_paths, labels))

    if shuffle:
        # 2차 셔플: 버퍼가 전체 크기라 완전 셔플이 되고,
        # reshuffle_each_iteration=True로 에폭마다 배치 구성이 달라진다.
        ds = ds.shuffle(buffer_size=len(items), seed=seed, reshuffle_each_iteration=True)

    if repeat:
        ds = ds.repeat()

    # enumerate가 붙이는 번호를 증강 시드로 쓴다. repeat '뒤'에 놓아야 카운터가 에폭
    # 경계에서 0으로 되돌아가지 않고 계속 증가한다 → 같은 이미지도 에폭마다 다른 증강.
    # (repeat 앞에 두면 매 에폭 같은 번호가 다시 나와 증강이 통째로 반복된다.)
    ds = ds.enumerate()

    def map_fn(index, element):
        rgb_path, ir_path, label = element
        # 증강 파라미터는 그래프 모드에서 결정론적으로 뽑고, 실제 픽셀 작업만 파이썬으로 넘긴다.
        flip_val, angle_val, brightness_val, contrast_val, sat_val = _sample_augmentation_params(index, seed, augment)

        def _py_fn(r_path, i_path, lbl, flp, ang, brt, cnt, sat):
            # py_function 안에서는 eager 텐서라 .numpy()로 파이썬 값을 꺼낼 수 있다.
            r_path_str = r_path.numpy().decode('utf-8')  # tf 문자열 텐서는 bytes다
            i_path_str = i_path.numpy().decode('utf-8')
            lbl_val = int(lbl.numpy())
            rgb, ir = load_sample(
                r_path_str, i_path_str, augment=augment,
                flip=int(flp.numpy()), angle=float(ang.numpy()),
                brightness_f=float(brt.numpy()), contrast_f=float(cnt.numpy()), sat_f=float(sat.numpy())
            )
            return rgb, ir, np.int32(lbl_val)

        # OpenCV는 TF 그래프 연산이 아니므로 py_function으로 감싸야 tf.data에 태울 수 있다.
        # 이 방식은 Python GIL 때문에 병렬 처리량이 제한될 수 있다.
        outputs = tf.py_function(
            _py_fn,
            inp=[rgb_path, ir_path, label, flip_val, angle_val, brightness_val, contrast_val, sat_val],
            Tout=[tf.float32, tf.float32, tf.int32]
        )

        # py_function은 출력 shape을 모른 채 <unknown>으로 두기 때문에 직접 알려줘야 한다.
        # 이걸 빼면 batch 이후 shape이 (None, None, None, None)이 되어 모델이 빌드되지 않는다.
        outputs[0].set_shape((224, 224, 3))
        outputs[1].set_shape((224, 224, 1))
        outputs[2].set_shape(())  # 스칼라 라벨

        return (outputs[0], outputs[1]), outputs[2]

    # AUTOTUNE: 병렬 map 스레드 수를 런타임이 알아서 조절.
    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    # prefetch로 GPU가 N번째 배치를 계산하는 동안 CPU가 N+1번째를 준비하게 겹친다.
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_single_dataset(items, input_type="crop_rgb", batch_size=8, shuffle=False, seed=42, augment=False, repeat=False):
    """crop_rgb / crop_ir 단일 입력 모델용 데이터셋. 구조는 make_dataset과 동일하다.

    반환 원소 형태: (image, label) — 입력이 하나이므로 x가 튜플이 아니다.
    repeat=True면 무한 반복 데이터셋이 된다.
    """
    items = list(items)
    if not items:
        raise ValueError("Dataset items list cannot be empty.")
    if shuffle:
        random.Random(seed).shuffle(items)

    # items 튜플은 항상 (rgb, ir, label) 3원소다. 여기서 필요한 모달리티 하나만 골라낸다
    # → dual과 단일 모델이 완전히 같은 파일 목록/순서를 공유하게 된다.
    if input_type == "crop_rgb":
        paths = [item[0] for item in items]
    else:
        paths = [item[1] for item in items]
    labels = [item[2] for item in items]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(items), seed=seed, reshuffle_each_iteration=True)

    if repeat:
        ds = ds.repeat()

    ds = ds.enumerate()  # make_dataset과 동일 — repeat 뒤여야 카운터가 누적된다.

    def map_fn(index, element):
        path, label = element
        flip_val, angle_val, brightness_val, contrast_val, sat_val = _sample_augmentation_params(index, seed, augment)

        def _py_fn(p, lbl, flp, ang, brt, cnt, sat):
            p_str = p.numpy().decode('utf-8')
            lbl_val = int(lbl.numpy())
            img = load_single_sample(
                p_str, input_type=input_type, augment=augment,
                flip=int(flp.numpy()), angle=float(ang.numpy()),
                brightness_f=float(brt.numpy()), contrast_f=float(cnt.numpy()), sat_f=float(sat.numpy())
            )
            return img, np.int32(lbl_val)

        outputs = tf.py_function(
            _py_fn,
            inp=[path, label, flip_val, angle_val, brightness_val, contrast_val, sat_val],
            Tout=[tf.float32, tf.int32]
        )

        # 모달리티에 따라 채널 수가 달라지므로 set_shape도 분기한다.
        if input_type == "crop_rgb":
            outputs[0].set_shape((224, 224, 3))
        else:
            outputs[0].set_shape((224, 224, 1))
        outputs[1].set_shape(())
        
        return outputs[0], outputs[1]

    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# 아래 두 함수는 현재 호출부가 없는 잔존 코드다.
# 실제 INT8 변환은 convert_keras_to_tflite.py의 _make_representative_dataset_gen을 사용한다.
def representative_dataset(items, max_samples=200):
    for rgb_path, ir_path, _ in items[:max_samples]:
        rgb, ir = load_sample(rgb_path, ir_path, augment=False)
        yield [
            np.expand_dims(rgb, axis=0).astype(np.float32),
            np.expand_dims(ir, axis=0).astype(np.float32),
        ]


def representative_single_dataset(items, input_type="crop_rgb", max_samples=200):
    """단일 입력용 대표 표본을 순회한다. 현재 호출부가 없는 잔존 코드다."""
    for rgb_path, ir_path, _ in items[:max_samples]:
        if input_type == "crop_rgb":
            img = load_single_sample(rgb_path, input_type="crop_rgb", augment=False)
        else:
            img = load_single_sample(ir_path, input_type="crop_ir", augment=False)
        yield [np.expand_dims(img, axis=0).astype(np.float32)]
