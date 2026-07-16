import pytest
import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path

from keras_pipeline.tf_dataset import (
    make_dataset,
    make_single_dataset,
)

@pytest.fixture
def fake_noise_images(tmp_path):
    # random noise image to see augmentation differences clearly
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
    
    # 1. Generate two datasets with same seed
    ds1 = make_dataset(items, batch_size=2, shuffle=True, seed=42, augment=True)
    ds2 = make_dataset(items, batch_size=2, shuffle=True, seed=42, augment=True)
    
    # 2. Compare batches
    batches1 = list(ds1)
    batches2 = list(ds2)
    
    assert len(batches1) == len(batches2)
    
    for (inputs1, labels1), (inputs2, labels2) in zip(batches1, batches2):
        rgb_b1, ir_b1 = inputs1
        rgb_b2, ir_b2 = inputs2
        
        np.testing.assert_allclose(rgb_b1.numpy(), rgb_b2.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(ir_b1.numpy(), ir_b2.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(labels1.numpy(), labels2.numpy())

    # 3. Check that different seed produces different augmented outputs
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


def test_deterministic_augmentation_single(fake_noise_images):
    rgb_path, ir_path = fake_noise_images
    items = [(rgb_path, ir_path, 0)] * 5
    
    # single_dataset crop_rgb
    ds1_rgb = make_single_dataset(items, input_type="crop_rgb", batch_size=2, shuffle=True, seed=42, augment=True)
    ds2_rgb = make_single_dataset(items, input_type="crop_rgb", batch_size=2, shuffle=True, seed=42, augment=True)
    
    for (img1, lbl1), (img2, lbl2) in zip(ds1_rgb, ds2_rgb):
        np.testing.assert_allclose(img1.numpy(), img2.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(lbl1.numpy(), lbl2.numpy())
        
    # single_dataset crop_ir
    ds1_ir = make_single_dataset(items, input_type="crop_ir", batch_size=2, shuffle=True, seed=42, augment=True)
    ds2_ir = make_single_dataset(items, input_type="crop_ir", batch_size=2, shuffle=True, seed=42, augment=True)
    
    for (img1, lbl1), (img2, lbl2) in zip(ds1_ir, ds2_ir):
        np.testing.assert_allclose(img1.numpy(), img2.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(lbl1.numpy(), lbl2.numpy())
