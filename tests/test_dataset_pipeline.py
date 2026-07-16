import pytest
import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path

from keras_pipeline.tf_dataset import (
    make_dataset,
    make_single_dataset,
    make_multimodal_dataset,
)


@pytest.fixture
def fake_images(tmp_path):
    # Create valid dummy images to satisfy cv2.imread and required multimodal paths
    img_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    img_gray = np.zeros((224, 224), dtype=np.uint8)

    rgb_path = tmp_path / "cropRGB.bmp"
    ir_path = tmp_path / "cropIR.bmp"
    raw_rgb_path = tmp_path / "RGB.bmp"
    raw_ir_path = tmp_path / "IR.bmp"
    heatmap_path = tmp_path / "face_heatmap.bmp"

    cv2.imwrite(str(rgb_path), img_rgb)
    cv2.imwrite(str(ir_path), img_gray)
    cv2.imwrite(str(raw_rgb_path), img_rgb)
    cv2.imwrite(str(raw_ir_path), img_gray)
    cv2.imwrite(str(heatmap_path), img_gray)

    return str(rgb_path), str(ir_path)


def test_dataset_builders_reject_empty_items():
    # Empty items list should immediately raise ValueError
    with pytest.raises(ValueError, match="cannot be empty"):
        make_dataset(items=[], batch_size=8)

    with pytest.raises(ValueError, match="cannot be empty"):
        make_single_dataset(items=[], input_type="crop_rgb", batch_size=8)

    with pytest.raises(ValueError, match="cannot be empty"):
        make_multimodal_dataset(items=[], batch_size=8)


def test_dataset_cardinality_and_steps(fake_images):
    rgb_path, ir_path = fake_images
    items = [(rgb_path, ir_path, 0)] * 10
    
    # Batch size 4, 10 samples -> 3 batches
    ds = make_dataset(items, batch_size=4, shuffle=False)
    
    # Check cardinality (number of batches)
    cardinality = ds.cardinality().numpy()
    assert cardinality == 3  # math.ceil(10 / 4) = 3
    
    # Check shape inside batches
    batch_index = 0
    for inputs, labels in ds:
        rgb_batch, ir_batch = inputs
        current_batch_size = labels.shape[0]
        if batch_index < 2:
            assert current_batch_size == 4
        else:
            assert current_batch_size == 2
        
        assert rgb_batch.shape == (current_batch_size, 224, 224, 3)
        assert ir_batch.shape == (current_batch_size, 224, 224, 1)
        batch_index += 1
    
    assert batch_index == 3


def test_corrupt_or_missing_image_path_raises_on_iteration():
    # Missing images in input list
    items = [("non_existent_rgb.bmp", "non_existent_ir.bmp", 0)]
    
    ds = make_dataset(items, batch_size=2, shuffle=False)
    
    # During dataset iteration, TF py_function runs load_sample which will fail with ValueError
    with pytest.raises(tf.errors.InvalidArgumentError):
        for _ in ds:
            pass


def test_prefetch_applied_to_pipelines(fake_images):
    rgb_path, ir_path = fake_images
    items = [(rgb_path, ir_path, 0)] * 4
    
    ds_dual = make_dataset(items, batch_size=2)
    ds_single = make_single_dataset(items, input_type="crop_rgb", batch_size=2)
    ds_multi = make_multimodal_dataset(items, batch_size=2)
    
    # Check if prefetch is correctly injected (usually wrapped in PrefetchDataset)
    assert "Prefetch" in type(ds_dual).__name__
    assert "Prefetch" in type(ds_single).__name__
    assert "Prefetch" in type(ds_multi).__name__
