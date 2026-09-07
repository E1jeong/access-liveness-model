import cv2
import numpy as np

from common.classes import CLASS_MAPPING
from keras_pipeline.data.dataset import load_sample


def test_display_target_augmentation_does_not_change_live_or_other_attacks(tmp_path):
    """모아레와 IR 글레어는 display 클래스에만 합성해야 한다."""
    rgb_path = tmp_path / "cropRGB.bmp"
    ir_path = tmp_path / "cropIR.bmp"
    cv2.imwrite(str(rgb_path), np.zeros((224, 224, 3), dtype=np.uint8))
    cv2.imwrite(str(ir_path), np.zeros((224, 224), dtype=np.uint8))

    params = {
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

    live_rgb, live_ir = load_sample(rgb_path, ir_path, label=CLASS_MAPPING["live"], **params)
    print_rgb, print_ir = load_sample(rgb_path, ir_path, label=CLASS_MAPPING["print"], **params)
    display_rgb, display_ir = load_sample(rgb_path, ir_path, label=CLASS_MAPPING["display"], **params)

    np.testing.assert_array_equal(live_rgb, print_rgb)
    np.testing.assert_array_equal(live_ir, print_ir)
    assert not np.array_equal(display_rgb, live_rgb)
    assert not np.array_equal(display_ir, live_ir)
