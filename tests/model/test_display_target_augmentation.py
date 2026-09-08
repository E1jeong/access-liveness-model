import cv2
import numpy as np

from common.classes import CLASS_MAPPING
from keras_pipeline.data.dataset import load_sample, load_single_sample


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
