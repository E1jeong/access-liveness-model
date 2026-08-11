"""tf.data 입력 파이프라인.

이미지 디코딩·증강·정규화를 OpenCV로 수행하고 tf.py_function으로 감싸 tf.data에 태운다.
TF 네이티브 연산 대신 OpenCV를 쓰는 이유는 PyTorch 파이프라인(pytorch_pipeline/dataset.py)과
픽셀 단위로 같은 결과를 내야 두 파이프라인의 지표를 비교할 수 있기 때문이다.

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

    # 회전을 원본 해상도에서 먼저 하고 그다음 224로 축소한다(순서가 바뀌면 보간 품질이 나빠진다).
    # INTER_AREA는 축소 전용으로 에일리어싱이 가장 적은 보간법.
    rgb = cv2.resize(rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    ir = cv2.resize(ir, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    if augment:
        # ColorJitter (RGB only): match PyTorch after resize.
        # IR에는 색 증강을 걸지 않는다 — IR 밝기는 물리적 반사 특성이고 스푸핑 판별의
        # 핵심 신호라, 흔들면 오히려 단서를 지우게 된다.
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
        rgb = rgb_f.astype(np.uint8)  # PyTorch ColorJitter와 동일하게 uint8로 되돌린 뒤 정규화

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
            # ColorJitter (RGB only)
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
    else: # crop_ir
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
      - index가 다르면 → 서로 다른 증강
    이 두 가지가 동시에 성립한다. 일반 tf.random을 쓰면 tf.data가 map을 병렬 실행하는 순간
    호출 순서가 비결정적이 되어 재현이 깨진다.

    index는 파일 목록 내 위치가 아니라 repeat 뒤에 붙인 enumerate가 매기는 '스트림 전역
    카운터'다(make_dataset 참고). 따라서 e번째 에폭의 j번째 샘플은 index = e*N + j 를 받고,
    같은 이미지라도 에폭이 바뀌면 다른 증강이 걸린다. 그러면서도 그 수열 자체는 seed로
    고정되므로 실행 간 재현성은 유지된다.
    """
    if augment:
        # 2원소 int64 텐서가 stateless 계열이 요구하는 seed 형식이다.
        seed_tensor = tf.stack([index, tf.cast(seed, tf.int64)])
        # 파라미터마다 seed의 두 번째 성분을 +1씩 밀어 서로 독립된 난수열을 뽑는다.
        # (같은 seed를 재사용하면 flip과 angle이 완전히 상관돼 버린다.)
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
        # 대가로 GIL 때문에 병렬성이 제한되지만, 여기서는 GPU 학습 속도가 병목이라 문제되지 않는다.
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


# --- 아래 두 함수는 학습이 아니라 INT8 변환 경로(convert_keras_to_tflite.py)에서 쓰인다. ---
# 양자화 스케일을 정하려면 실제 입력 분포 표본이 필요해서, 증강 없이 원본 그대로를 흘려보낸다.
# 학습 데이터와 동일한 전처리를 써야 스케일이 맞으므로 load_sample을 공유한다.
def representative_dataset(items, max_samples=200):
    for rgb_path, ir_path, _ in items[:max_samples]:
        rgb, ir = load_sample(rgb_path, ir_path, augment=False)
        yield [
            np.expand_dims(rgb, axis=0).astype(np.float32),
            np.expand_dims(ir, axis=0).astype(np.float32),
        ]


def representative_single_dataset(items, input_type="crop_rgb", max_samples=200):
    for rgb_path, ir_path, _ in items[:max_samples]:
        if input_type == "crop_rgb":
            img = load_single_sample(rgb_path, input_type="crop_rgb", augment=False)
        else:
            img = load_single_sample(ir_path, input_type="crop_ir", augment=False)
        yield [np.expand_dims(img, axis=0).astype(np.float32)]



