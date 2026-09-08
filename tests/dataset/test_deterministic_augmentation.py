import pytest
import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path

from keras_pipeline.data.dataset import (
    make_dataset,
    make_single_dataset,
    load_single_sample,
)

@pytest.fixture
def fake_noise_images(tmp_path):
    # 증강 차이가 분명히 보이도록 무작위 노이즈 이미지를 만든다.
    np.random.seed(42)
    img_rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    img_gray = np.random.randint(0, 256, (224, 224), dtype=np.uint8)

    rgb_path = tmp_path / "cropRGB.bmp"
    ir_path = tmp_path / "cropIR.bmp"

    cv2.imwrite(str(rgb_path), img_rgb)
    cv2.imwrite(str(ir_path), img_gray)

    return str(rgb_path), str(ir_path)

def test_deterministic_augmentation_dual(fake_noise_images):
    rgb_path, ir_path = fake_noise_images
    items = [(rgb_path, ir_path, 0)] * 5
    
    # 1. 같은 시드로 데이터셋 두 개를 만든다.
    ds1 = make_dataset(items, batch_size=2, shuffle=True, seed=42, augment=True)
    ds2 = make_dataset(items, batch_size=2, shuffle=True, seed=42, augment=True)
    
    # 2. 두 데이터셋의 배치를 비교한다.
    batches1 = list(ds1)
    batches2 = list(ds2)
    
    assert len(batches1) == len(batches2)
    
    for (inputs1, labels1), (inputs2, labels2) in zip(batches1, batches2):
        rgb_b1, ir_b1 = inputs1
        rgb_b2, ir_b2 = inputs2
        
        np.testing.assert_allclose(rgb_b1.numpy(), rgb_b2.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(ir_b1.numpy(), ir_b2.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(labels1.numpy(), labels2.numpy())

    # 3. 다른 시드가 다른 증강 결과를 만드는지 확인한다.
    ds3 = make_dataset(items, batch_size=2, shuffle=True, seed=43, augment=True)
    batches3 = list(ds3)
    
    diff_rgb = False
    for (inputs1, _), (inputs3, _) in zip(batches1, batches3):
        rgb_b1, _ = inputs1
        rgb_b3, _ = inputs3
        if not np.allclose(rgb_b1.numpy(), rgb_b3.numpy(), atol=1e-3):
            diff_rgb = True
            break
            
    assert diff_rgb, "Different seeds must produce different augmented outputs"


def test_augmentation_varies_across_epochs(fake_noise_images):
    """같은 이미지가 에폭마다 '다른' 증강을 받아야 한다.

    증강 시드로 쓰는 index는 repeat 뒤 enumerate가 매기는 전역 카운터라 에폭 경계에서
    리셋되지 않는다. 셔플을 끄고 batch_size=1로 두면 n번째 원소가 곧 n번째 샘플이므로,
    앞 N개(1에폭)와 뒤 N개(2에폭)를 위치별로 비교하면 된다.
    """
    rgb_path, ir_path = fake_noise_images
    n = 4
    items = [(rgb_path, ir_path, 0)] * n

    ds = make_dataset(items, batch_size=1, shuffle=False, seed=42, augment=True, repeat=True)
    rgb_seq = [inputs[0].numpy() for inputs, _ in ds.take(2 * n)]

    # 1에폭과 2에폭의 같은 위치 샘플이 하나라도 다르면 통과.
    assert any(
        not np.allclose(rgb_seq[i], rgb_seq[n + i], atol=1e-3) for i in range(n)
    ), "Same sample must receive different augmentation on a later epoch"

    # 그러면서도 시드가 같으면 그 수열 자체는 재현돼야 한다.
    ds2 = make_dataset(items, batch_size=1, shuffle=False, seed=42, augment=True, repeat=True)
    rgb_seq2 = [inputs[0].numpy() for inputs, _ in ds2.take(2 * n)]
    for a, b in zip(rgb_seq, rgb_seq2):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)


def test_deterministic_augmentation_single(fake_noise_images):
    rgb_path, ir_path = fake_noise_images
    items = [(rgb_path, ir_path, 0)] * 5
    
    # crop_rgb 단일 입력 데이터셋
    ds1_rgb = make_single_dataset(items, input_type="crop_rgb", batch_size=2, shuffle=True, seed=42, augment=True)
    ds2_rgb = make_single_dataset(items, input_type="crop_rgb", batch_size=2, shuffle=True, seed=42, augment=True)
    
    for (img1, lbl1), (img2, lbl2) in zip(ds1_rgb, ds2_rgb):
        np.testing.assert_allclose(img1.numpy(), img2.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(lbl1.numpy(), lbl2.numpy())
        
    # crop_ir 단일 입력 데이터셋
    ds1_ir = make_single_dataset(items, input_type="crop_ir", batch_size=2, shuffle=True, seed=42, augment=True)
    ds2_ir = make_single_dataset(items, input_type="crop_ir", batch_size=2, shuffle=True, seed=42, augment=True)
    
    for (img1, lbl1), (img2, lbl2) in zip(ds1_ir, ds2_ir):
        np.testing.assert_allclose(img1.numpy(), img2.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(lbl1.numpy(), lbl2.numpy())


def test_ir_brightness_augmentation(fake_noise_images):
    """IR 이미지의 선형 밝기 스케일링 증강 동작을 검증한다."""
    _, ir_path = fake_noise_images
    # augment=False: ir_brightness_f=1.0 기본 동작
    base_ir = load_single_sample(ir_path, input_type="crop_ir", augment=False)
    # augment=True, flip=0, angle=0.0, ir_brightness_f=1.10
    scaled_ir = load_single_sample(
        ir_path, input_type="crop_ir", augment=True, flip=0, angle=0.0, ir_brightness_f=1.10
    )
    # 픽셀 값이 스케일링되어 원본과 달라야 함
    assert not np.allclose(base_ir, scaled_ir, atol=1e-3)
    # ir_brightness_f=1.0으로 augment=True 적용 시 (flip=0, angle=0) base_ir와 동일해야 함
    ident_ir = load_single_sample(
        ir_path, input_type="crop_ir", augment=True, flip=0, angle=0.0, ir_brightness_f=1.0
    )
    np.testing.assert_allclose(base_ir, ident_ir, rtol=1e-5, atol=1e-5)
