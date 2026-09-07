"""tf.data 입력 파이프라인.

이미지 디코딩·증강·정규화를 OpenCV로 수행하고 tf.py_function으로 감싸 tf.data에 태운다.
공간 증강 방식과 정규화 계약은 PyTorch 파이프라인(pytorch_pipeline/dataset.py)에 맞췄지만,
resize 구현과 ColorJitter 연산 순서가 달라 픽셀 단위 결과까지 같지는 않다.

정규화 계약(앱/PyTorch와 동일해야 함):
  RGB: BGR→RGB → 0~1 → (x - ImageNet_mean) / ImageNet_std
  IR : 그레이스케일 → 0~1 → (x - 0.5) / 0.5   즉 [-1, 1]

Multi-Task Auxiliary 지도학습 지원:
  - `aux_depth=True` 시 14x14 크기의 3D 깊이 지도(`depth_output`)를 타겟 딕셔너리로 함께 반환한다.
  - `aux_binary_pad=True` 시 같은 12-class label을 `pad_output`에도 전달한다.
  - `aux_supcon=True` 시 같은 label을 학습 전용 `supcon_output`에도 전달한다.
    binary target 변환은 loss에서 Phase 2 policy로 수행한다.
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

from keras_pipeline.data.depth_generator import generate_pseudo_depth_map
from keras_pipeline.data.spec import IMAGE_SIZE, RGB_MEAN, RGB_STD, IR_MEAN, IR_STD
from common.classes import CLASS_MAPPING


DISPLAY_LABEL = CLASS_MAPPING["display"]


def _apply_rgb_moire(rgb, strength, frequency, angle, phase):
    """RGB display 재촬영 때의 서브픽셀-센서 간섭 줄무늬를 합성한다."""
    height, width = rgb.shape[:2]
    x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    stripe = np.sin(2.0 * np.pi * frequency * (x * np.cos(angle) + y * np.sin(angle)) + phase)
    channel_phase = np.asarray([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0], dtype=np.float32)
    interference = np.sin(
        2.0 * np.pi * frequency * (x[..., np.newaxis] * np.cos(angle) + y[..., np.newaxis] * np.sin(angle))
        + phase + channel_phase
    )
    # 공통 휘도 줄무늬와 RGB 서브픽셀 위상차를 함께 넣어 재촬영 모아레를 근사한다.
    overlay = 0.6 * stripe[..., np.newaxis] + 0.4 * interference
    return np.clip(rgb.astype(np.float32) + 255.0 * strength * overlay, 0, 255).astype(np.uint8)


def _apply_ir_glare(ir, strength, center_x, center_y, sigma):
    """IR LED가 화면 유리에 정반사되어 생기는 흰 글레어를 합성한다."""
    height, width = ir.shape[:2]
    x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    distance = ((x / width - center_x) ** 2 + (y / height - center_y) ** 2) / (2.0 * sigma ** 2)
    alpha = strength * np.exp(-distance)
    return np.clip(ir.astype(np.float32) + (255.0 - ir.astype(np.float32)) * alpha, 0, 255).astype(np.uint8)


def load_sample(rgb_path, ir_path, augment=False, flip=0, angle=0.0, brightness_f=1.0, contrast_f=1.0, sat_f=1.0,
                label=None, moire_strength=0.0, moire_frequency=0.0, moire_angle=0.0, moire_phase=0.0,
                glare_strength=0.0, glare_x=0.5, glare_y=0.5, glare_sigma=0.15):
    """이미지를 불러와 정규화한다. augment=True이면 학습용 데이터 증강을 적용한다."""
    rgb = cv2.imread(rgb_path)
    if rgb is None:
        raise ValueError(f"Failed to read RGB image: {rgb_path}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    if ir is None:
        raise ValueError(f"Failed to read IR image: {ir_path}")

    if augment:
        if flip == 1:
            rgb = cv2.flip(rgb, 1)
            ir = cv2.flip(ir, 1)
        h, w = rgb.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rgb = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR)
        ir = cv2.warpAffine(ir, M, (w, h), flags=cv2.INTER_LINEAR)

    rgb = cv2.resize(rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    ir = cv2.resize(ir, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    if augment:
        rgb_f = rgb.astype(np.float32)
        rgb_f = np.clip(rgb_f * brightness_f, 0, 255)
        mean_val = rgb_f.mean()
        rgb_f = np.clip((rgb_f - mean_val) * contrast_f + mean_val, 0, 255)
        gray = (0.299 * rgb_f[:, :, 0] + 0.587 * rgb_f[:, :, 1] + 0.114 * rgb_f[:, :, 2])[:, :, np.newaxis]
        rgb_f = np.clip(gray + sat_f * (rgb_f - gray), 0, 255)
        rgb = rgb_f.astype(np.uint8)

        if label == DISPLAY_LABEL:
            rgb = _apply_rgb_moire(rgb, moire_strength, moire_frequency, moire_angle, moire_phase)
            ir = _apply_ir_glare(ir, glare_strength, glare_x, glare_y, glare_sigma)

    rgb = rgb.astype(np.float32) / 255.0
    rgb = (rgb - RGB_MEAN) / RGB_STD

    ir = ir.astype(np.float32) / 255.0
    ir = np.expand_dims(ir, axis=-1)
    ir = (ir - IR_MEAN) / IR_STD

    return rgb.astype(np.float32), ir.astype(np.float32)


def load_single_sample(path, input_type="crop_rgb", augment=False, flip=0, angle=0.0, brightness_f=1.0, contrast_f=1.0, sat_f=1.0,
                       label=None, moire_strength=0.0, moire_frequency=0.0, moire_angle=0.0, moire_phase=0.0,
                       glare_strength=0.0, glare_x=0.5, glare_y=0.5, glare_sigma=0.15):
    """단일 이미지(RGB 혹은 IR)를 불러와 정규화하고 필요시 증강한다."""
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
            rgb_f = rgb.astype(np.float32)
            rgb_f = np.clip(rgb_f * brightness_f, 0, 255)
            mean_val = rgb_f.mean()
            rgb_f = np.clip((rgb_f - mean_val) * contrast_f + mean_val, 0, 255)
            gray = (0.299 * rgb_f[:, :, 0] + 0.587 * rgb_f[:, :, 1] + 0.114 * rgb_f[:, :, 2])[:, :, np.newaxis]
            rgb_f = np.clip(gray + sat_f * (rgb_f - gray), 0, 255)
            rgb = rgb_f.astype(np.uint8)
            if label == DISPLAY_LABEL:
                rgb = _apply_rgb_moire(rgb, moire_strength, moire_frequency, moire_angle, moire_phase)

        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - RGB_MEAN) / RGB_STD
        return rgb.astype(np.float32)
    else:
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

        if augment and label == DISPLAY_LABEL:
            ir = _apply_ir_glare(ir, glare_strength, glare_x, glare_y, glare_sigma)

        ir = ir.astype(np.float32) / 255.0
        ir = np.expand_dims(ir, axis=-1)
        ir = (ir - IR_MEAN) / IR_STD
        return ir.astype(np.float32)


def _sample_augmentation_params(index, seed, augment):
    """tf.data map 함수 내에서 index와 seed 기반의 결정론적 증강 파라미터를 추출한다."""
    if augment:
        seed_tensor = tf.stack([index, tf.cast(seed, tf.int64)])
        flip_val = tf.random.stateless_uniform([], seed=seed_tensor, minval=0, maxval=2, dtype=tf.int32)
        angle_val = tf.random.stateless_uniform([], seed=seed_tensor + [0, 1], minval=-10.0, maxval=10.0, dtype=tf.float32)
        brightness_val = tf.random.stateless_uniform([], seed=seed_tensor + [0, 2], minval=0.7, maxval=1.3, dtype=tf.float32)
        contrast_val = tf.random.stateless_uniform([], seed=seed_tensor + [0, 3], minval=0.7, maxval=1.3, dtype=tf.float32)
        sat_val = tf.random.stateless_uniform([], seed=seed_tensor + [0, 4], minval=0.8, maxval=1.2, dtype=tf.float32)
        moire_strength_val = tf.random.stateless_uniform([], seed=seed_tensor + [0, 5], minval=0.08, maxval=0.18, dtype=tf.float32)
        moire_frequency_val = tf.random.stateless_uniform([], seed=seed_tensor + [0, 6], minval=0.04, maxval=0.10, dtype=tf.float32)
        moire_angle_val = tf.random.stateless_uniform([], seed=seed_tensor + [0, 7], minval=0.0, maxval=np.pi, dtype=tf.float32)
        moire_phase_val = tf.random.stateless_uniform([], seed=seed_tensor + [0, 8], minval=0.0, maxval=2.0 * np.pi, dtype=tf.float32)
        glare_strength_val = tf.random.stateless_uniform([], seed=seed_tensor + [0, 9], minval=0.35, maxval=0.70, dtype=tf.float32)
        glare_x_val = tf.random.stateless_uniform([], seed=seed_tensor + [0, 10], minval=0.25, maxval=0.75, dtype=tf.float32)
        glare_y_val = tf.random.stateless_uniform([], seed=seed_tensor + [0, 11], minval=0.25, maxval=0.75, dtype=tf.float32)
        glare_sigma_val = tf.random.stateless_uniform([], seed=seed_tensor + [0, 12], minval=0.10, maxval=0.22, dtype=tf.float32)
    else:
        flip_val = tf.constant(0, dtype=tf.int32)
        angle_val = tf.constant(0.0, dtype=tf.float32)
        brightness_val = tf.constant(1.0, dtype=tf.float32)
        contrast_val = tf.constant(1.0, dtype=tf.float32)
        sat_val = tf.constant(1.0, dtype=tf.float32)
        moire_strength_val = tf.constant(0.0, dtype=tf.float32)
        moire_frequency_val = tf.constant(0.0, dtype=tf.float32)
        moire_angle_val = tf.constant(0.0, dtype=tf.float32)
        moire_phase_val = tf.constant(0.0, dtype=tf.float32)
        glare_strength_val = tf.constant(0.0, dtype=tf.float32)
        glare_x_val = tf.constant(0.5, dtype=tf.float32)
        glare_y_val = tf.constant(0.5, dtype=tf.float32)
        glare_sigma_val = tf.constant(0.15, dtype=tf.float32)
    return (flip_val, angle_val, brightness_val, contrast_val, sat_val,
            moire_strength_val, moire_frequency_val, moire_angle_val, moire_phase_val,
            glare_strength_val, glare_x_val, glare_y_val, glare_sigma_val)


def make_dataset(items, batch_size=8, shuffle=False, seed=42, augment=False, repeat=False,
                 aux_depth=False, aux_binary_pad=False, aux_supcon=False):
    """dual(RGB+IR) 모델용 tf.data 데이터셋을 만든다."""
    items = list(items)
    if not items:
        raise ValueError("Dataset items list cannot be empty.")
    if shuffle:
        random.Random(seed).shuffle(items)

    rgb_paths = [item[0] for item in items]
    ir_paths = [item[1] for item in items]
    labels = [item[2] for item in items]

    ds = tf.data.Dataset.from_tensor_slices((rgb_paths, ir_paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(items), seed=seed, reshuffle_each_iteration=True)

    if repeat:
        ds = ds.repeat()

    ds = ds.enumerate()

    def map_fn(index, element):
        rgb_path, ir_path, label = element
        (flip_val, angle_val, brightness_val, contrast_val, sat_val,
         moire_strength_val, moire_frequency_val, moire_angle_val, moire_phase_val,
         glare_strength_val, glare_x_val, glare_y_val, glare_sigma_val) = _sample_augmentation_params(index, seed, augment)

        def _py_fn(r_path, i_path, lbl, flp, ang, brt, cnt, sat, moire_str, moire_freq, moire_ang, moire_ph,
                   glare_str, glare_x, glare_y, glare_sig):
            r_path_str = r_path.numpy().decode('utf-8')
            i_path_str = i_path.numpy().decode('utf-8')
            lbl_val = int(lbl.numpy())
            flp_int = int(flp.numpy())
            ang_flt = float(ang.numpy())
            rgb, ir = load_sample(
                r_path_str, i_path_str, augment=augment,
                flip=flp_int, angle=ang_flt,
                brightness_f=float(brt.numpy()), contrast_f=float(cnt.numpy()), sat_f=float(sat.numpy()),
                label=lbl_val, moire_strength=float(moire_str.numpy()), moire_frequency=float(moire_freq.numpy()),
                moire_angle=float(moire_ang.numpy()), moire_phase=float(moire_ph.numpy()),
                glare_strength=float(glare_str.numpy()), glare_x=float(glare_x.numpy()),
                glare_y=float(glare_y.numpy()), glare_sigma=float(glare_sig.numpy())
            )
            if aux_depth:
                depth = generate_pseudo_depth_map(lbl_val, size=(14, 14), flip=flp_int, angle=ang_flt)
                return rgb, ir, np.int32(lbl_val), depth
            return rgb, ir, np.int32(lbl_val)

        if aux_depth:
            outputs = tf.py_function(
                _py_fn,
                inp=[rgb_path, ir_path, label, flip_val, angle_val, brightness_val, contrast_val, sat_val,
                     moire_strength_val, moire_frequency_val, moire_angle_val, moire_phase_val,
                     glare_strength_val, glare_x_val, glare_y_val, glare_sigma_val],
                Tout=[tf.float32, tf.float32, tf.int32, tf.float32]
            )
            outputs[0].set_shape((224, 224, 3))
            outputs[1].set_shape((224, 224, 1))
            outputs[2].set_shape(())
            outputs[3].set_shape((14, 14, 1))
            targets = {"logits": outputs[2], "depth_output": outputs[3]}
            if aux_binary_pad:
                targets["pad_output"] = outputs[2]
            if aux_supcon:
                targets["supcon_output"] = outputs[2]
            return (outputs[0], outputs[1]), targets
        else:
            outputs = tf.py_function(
                _py_fn,
                inp=[rgb_path, ir_path, label, flip_val, angle_val, brightness_val, contrast_val, sat_val,
                     moire_strength_val, moire_frequency_val, moire_angle_val, moire_phase_val,
                     glare_strength_val, glare_x_val, glare_y_val, glare_sigma_val],
                Tout=[tf.float32, tf.float32, tf.int32]
            )
            outputs[0].set_shape((224, 224, 3))
            outputs[1].set_shape((224, 224, 1))
            outputs[2].set_shape(())
            if aux_binary_pad or aux_supcon:
                targets = {"logits": outputs[2]}
                if aux_binary_pad:
                    targets["pad_output"] = outputs[2]
                if aux_supcon:
                    targets["supcon_output"] = outputs[2]
                return (outputs[0], outputs[1]), targets
            return (outputs[0], outputs[1]), outputs[2]

    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_single_dataset(items, input_type="crop_rgb", batch_size=8, shuffle=False, seed=42,
                        augment=False, repeat=False, aux_depth=False, aux_binary_pad=False,
                        aux_supcon=False):
    """crop_rgb / crop_ir 단일 입력 모델용 데이터셋."""
    items = list(items)
    if not items:
        raise ValueError("Dataset items list cannot be empty.")
    if shuffle:
        random.Random(seed).shuffle(items)

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

    ds = ds.enumerate()

    def map_fn(index, element):
        path, label = element
        (flip_val, angle_val, brightness_val, contrast_val, sat_val,
         moire_strength_val, moire_frequency_val, moire_angle_val, moire_phase_val,
         glare_strength_val, glare_x_val, glare_y_val, glare_sigma_val) = _sample_augmentation_params(index, seed, augment)

        def _py_fn(p, lbl, flp, ang, brt, cnt, sat, moire_str, moire_freq, moire_ang, moire_ph,
                   glare_str, glare_x, glare_y, glare_sig):
            p_str = p.numpy().decode('utf-8')
            lbl_val = int(lbl.numpy())
            flp_int = int(flp.numpy())
            ang_flt = float(ang.numpy())
            img = load_single_sample(
                p_str, input_type=input_type, augment=augment,
                flip=flp_int, angle=ang_flt,
                brightness_f=float(brt.numpy()), contrast_f=float(cnt.numpy()), sat_f=float(sat.numpy()),
                label=lbl_val, moire_strength=float(moire_str.numpy()), moire_frequency=float(moire_freq.numpy()),
                moire_angle=float(moire_ang.numpy()), moire_phase=float(moire_ph.numpy()),
                glare_strength=float(glare_str.numpy()), glare_x=float(glare_x.numpy()),
                glare_y=float(glare_y.numpy()), glare_sigma=float(glare_sig.numpy())
            )
            if aux_depth:
                depth = generate_pseudo_depth_map(lbl_val, size=(14, 14), flip=flp_int, angle=ang_flt)
                return img, np.int32(lbl_val), depth
            return img, np.int32(lbl_val)

        if aux_depth:
            outputs = tf.py_function(
                _py_fn,
                inp=[path, label, flip_val, angle_val, brightness_val, contrast_val, sat_val,
                     moire_strength_val, moire_frequency_val, moire_angle_val, moire_phase_val,
                     glare_strength_val, glare_x_val, glare_y_val, glare_sigma_val],
                Tout=[tf.float32, tf.int32, tf.float32]
            )
            if input_type == "crop_rgb":
                outputs[0].set_shape((224, 224, 3))
            else:
                outputs[0].set_shape((224, 224, 1))
            outputs[1].set_shape(())
            outputs[2].set_shape((14, 14, 1))
            targets = {"logits": outputs[1], "depth_output": outputs[2]}
            if aux_binary_pad:
                targets["pad_output"] = outputs[1]
            if aux_supcon:
                targets["supcon_output"] = outputs[1]
            return outputs[0], targets
        else:
            outputs = tf.py_function(
                _py_fn,
                inp=[path, label, flip_val, angle_val, brightness_val, contrast_val, sat_val,
                     moire_strength_val, moire_frequency_val, moire_angle_val, moire_phase_val,
                     glare_strength_val, glare_x_val, glare_y_val, glare_sigma_val],
                Tout=[tf.float32, tf.int32]
            )
            if input_type == "crop_rgb":
                outputs[0].set_shape((224, 224, 3))
            else:
                outputs[0].set_shape((224, 224, 1))
            outputs[1].set_shape(())
            if aux_binary_pad or aux_supcon:
                targets = {"logits": outputs[1]}
                if aux_binary_pad:
                    targets["pad_output"] = outputs[1]
                if aux_supcon:
                    targets["supcon_output"] = outputs[1]
                return outputs[0], targets
            return outputs[0], outputs[1]

    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
