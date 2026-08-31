"""Multi-Task 3D Depth 보조 지도학습 파이프라인 단위 테스트."""
import numpy as np
import pytest
import tensorflow as tf

from keras_pipeline.depth_generator import generate_pseudo_depth_map
from keras_pipeline.tf_model import (
    build_single_model, build_dual_model, extract_deploy_model
)
from keras_pipeline.tf_dataset import make_single_dataset, make_dataset


def test_depth_generator_shapes_and_values():
    """모든 12개 클래스에 대해 14x14 깊이 지도의 형상과 값 범위가 올바른지 검증."""
    # Live (0)
    d_live = generate_pseudo_depth_map(0, (14, 14), flip=1, angle=10.0)
    assert d_live.shape == (14, 14, 1)
    assert 0.0 <= d_live.min() and d_live.max() <= 1.0
    assert d_live.max() > 0.5  # 코/이마 영역 볼록

    # Flat Spoof (1~5)
    for c in range(1, 6):
        d_flat = generate_pseudo_depth_map(c, (14, 14))
        assert d_flat.shape == (14, 14, 1)
        assert np.all(d_flat == 0.0)

    # Curved Spoof (6~9)
    for c in range(6, 10):
        d_curved = generate_pseudo_depth_map(c, (14, 14))
        assert d_curved.shape == (14, 14, 1)
        assert 0.0 <= d_curved.min() and d_curved.max() <= 0.25

    # Dental Mask (10, 11)
    for c in (10, 11):
        d_dental = generate_pseudo_depth_map(c, (14, 14))
        assert d_dental.shape == (14, 14, 1)
        assert 0.0 <= d_dental.min() and d_dental.max() <= 1.0


def test_model_aux_depth_single_and_deploy_extract():
    """Single IR 모델에서 aux_depth 헤드 생성 및 배포용 모델 추출 검증."""
    model = build_single_model(input_type="crop_ir", aux_depth=True, rgb_weights=None)
    dummy_in = np.zeros((2, 224, 224, 1), dtype=np.float32)
    outputs = model(dummy_in)

    assert isinstance(outputs, (list, tuple))
    assert len(outputs) == 2
    logits, depth = outputs
    assert logits.shape == (2, 12)
    assert depth.shape == (2, 14, 14, 1)

    # 배포용 모델 추출
    deploy_model = extract_deploy_model(model)
    deploy_out = deploy_model(dummy_in)
    assert deploy_out.shape == (2, 12)
    assert len(deploy_model.outputs) == 1


def test_model_aux_depth_dual():
    """Dual 모델에서 aux_depth 헤드 생성 검증."""
    model = build_dual_model(aux_depth=True, rgb_weights=None)
    dummy_rgb = np.zeros((2, 224, 224, 3), dtype=np.float32)
    dummy_ir = np.zeros((2, 224, 224, 1), dtype=np.float32)
    outputs = model([dummy_rgb, dummy_ir])

    assert len(outputs) == 2
    logits, depth = outputs
    assert logits.shape == (2, 12)
    assert depth.shape == (2, 14, 14, 1)


def test_dataset_aux_depth_structure(tmp_path):
    """tf.data 파이프라인에서 aux_depth=True 시 타겟 딕셔너리 구조 검증."""
    import cv2
    img_path = str(tmp_path / "test.bmp")
    dummy_img = np.zeros((224, 224), dtype=np.uint8)
    cv2.imwrite(img_path, dummy_img)

    items = [(img_path, img_path, 0), (img_path, img_path, 1)]
    ds = make_single_dataset(items, input_type="crop_ir", batch_size=2, aux_depth=True)
    
    for batch_x, batch_y in ds.take(1):
        assert batch_x.shape == (2, 224, 224, 1)
        assert isinstance(batch_y, dict)
        assert "logits" in batch_y and "depth_output" in batch_y
        assert batch_y["logits"].shape == (2,)
        assert batch_y["depth_output"].shape == (2, 14, 14, 1)
