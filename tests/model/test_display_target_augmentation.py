import cv2
import numpy as np

from common.classes import CLASS_MAPPING
from keras_pipeline.data.dataset import load_sample, load_single_sample, make_single_dataset


DISPLAY_PARAMS = {
    "augment": True,
    "moire_strength": 0.15,
    "moire_frequency": 0.07,
    "moire_angle": 0.4,
    "moire_phase": 1.2,
    "glare_strength": 0.6,
    "glare_x": 0.5,
    "glare_y": 0.5,
    "glare_sigma": 0.15,
}


def _write_blank_pair(tmp_path):
    rgb_path = tmp_path / "cropRGB.bmp"
    ir_path = tmp_path / "cropIR.bmp"
    cv2.imwrite(str(rgb_path), np.zeros((224, 224, 3), dtype=np.uint8))
    cv2.imwrite(str(ir_path), np.zeros((224, 224), dtype=np.uint8))
    return str(rgb_path), str(ir_path)


def test_display_target_augmentation_does_not_change_live_or_other_attacks(tmp_path):
    """모아레와 IR 글레어는 display 클래스에만 합성해야 한다."""
    rgb_path, ir_path = _write_blank_pair(tmp_path)

    live_rgb, live_ir = load_sample(rgb_path, ir_path, label=CLASS_MAPPING["live"], **DISPLAY_PARAMS)
    print_rgb, print_ir = load_sample(rgb_path, ir_path, label=CLASS_MAPPING["print"], **DISPLAY_PARAMS)
    display_rgb, display_ir = load_sample(rgb_path, ir_path, label=CLASS_MAPPING["display"], **DISPLAY_PARAMS)

    np.testing.assert_array_equal(live_rgb, print_rgb)
    np.testing.assert_array_equal(live_ir, print_ir)
    assert not np.array_equal(display_rgb, live_rgb)
    assert not np.array_equal(display_ir, live_ir)


def test_ir_display_applies_moire_then_glare(tmp_path):
    """IR display는 휘도 모아레와 글레어를 함께 받되, 둘은 서로 다른 변형이다."""
    rgb_path, ir_path = _write_blank_pair(tmp_path)
    label = CLASS_MAPPING["display"]

    _, live_ir = load_sample(rgb_path, ir_path, label=CLASS_MAPPING["live"], **DISPLAY_PARAMS)
    _, moire_only = load_sample(
        rgb_path, ir_path, label=label, **{**DISPLAY_PARAMS, "glare_strength": 0.0}
    )
    _, glare_only = load_sample(
        rgb_path, ir_path, label=label, **{**DISPLAY_PARAMS, "moire_strength": 0.0}
    )
    _, both = load_sample(rgb_path, ir_path, label=label, **DISPLAY_PARAMS)
    crop_ir_both = load_single_sample(
        ir_path, input_type="crop_ir", label=label, **DISPLAY_PARAMS
    )

    assert not np.array_equal(moire_only, live_ir)
    assert not np.array_equal(glare_only, live_ir)
    assert not np.array_equal(moire_only, glare_only)
    assert not np.array_equal(both, moire_only)
    assert not np.array_equal(both, glare_only)
    np.testing.assert_array_equal(both, crop_ir_both)


def test_dataset_display_artifacts_option(tmp_path):
    """make_single_dataset에서 augment_display_artifacts 플래그가 동작해야 한다."""
    rgb_path, ir_path = _write_blank_pair(tmp_path)
    label = CLASS_MAPPING["display"]
    items = [(rgb_path, ir_path, label)]

    # 기본값 (augment_display_artifacts=False): 모아레/글레어가 적용되지 않아야 함
    ds_disabled = make_single_dataset(
        items, input_type="crop_ir", batch_size=1, shuffle=False, seed=42,
        augment=True, augment_display_artifacts=False,
    )
    # 활성화 (augment_display_artifacts=True): 모아레/글레어가 적용되어야 함
    ds_enabled = make_single_dataset(
        items, input_type="crop_ir", batch_size=1, shuffle=False, seed=42,
        augment=True, augment_display_artifacts=True,
    )

    batch_disabled = next(iter(ds_disabled))
    batch_enabled = next(iter(ds_enabled))

    ir_disabled = batch_disabled[0].numpy()
    ir_enabled = batch_enabled[0].numpy()

    assert not np.array_equal(ir_disabled, ir_enabled)
