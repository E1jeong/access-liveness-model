import numpy as np
import tensorflow as tf
from keras_pipeline.tf_model import build_single_mobilenetv2


def test_conv1_parity():
    # 1. crop_ir 모델을 mean과 sum 방식으로 각각 빌드 (ImageNet 가중치 전이)
    model_mean = build_single_mobilenetv2(
        input_type="crop_ir",
        rgb_weights="imagenet",
        conv1_reduction="mean"
    )
    model_sum = build_single_mobilenetv2(
        input_type="crop_ir",
        rgb_weights="imagenet",
        conv1_reduction="sum"
    )

    # 2. 비교용 오리지널 RGB crop_rgb 모델 빌드
    model_rgb = build_single_mobilenetv2(
        input_type="crop_rgb",
        rgb_weights="imagenet"
    )

    # 3. 임의의 1채널 Grayscale 입력 생성 [1, 224, 224, 1]
    np.random.seed(42)
    x_gray = np.random.uniform(-1.0, 1.0, size=(1, 224, 224, 1)).astype(np.float32)

    # 4. 3채널 복제 입력 생성 [1, 224, 224, 3] (R=G=B=x)
    x_rgb = np.concatenate([x_gray, x_gray, x_gray], axis=-1)

    # 5. 각 백본에서 Conv1 레이어 인스턴스 획득
    rgb_backbone = model_rgb.get_layer("crop_rgb_mobilenetv2")
    ir_mean_backbone = model_mean.get_layer("crop_ir_mobilenetv2")
    ir_sum_backbone = model_sum.get_layer("crop_ir_mobilenetv2")

    conv1_rgb = rgb_backbone.get_layer("Conv1")
    conv1_mean = ir_mean_backbone.get_layer("Conv1")
    conv1_sum = ir_sum_backbone.get_layer("Conv1")

    # 6. 각 Conv1 레이어 연산 수행
    out_rgb = conv1_rgb(x_rgb)
    out_mean = conv1_mean(x_gray)
    out_sum = conv1_sum(x_gray)

    # NumPy 변환
    out_rgb_np = out_rgb.numpy()
    out_mean_np = out_mean.numpy()
    out_sum_np = out_sum.numpy()

    # 7. 수학적 Parity 검증
    # - Sum 방식은 RGB 오리지널 Grayscale 복제본 출력과 수학적으로 완전히 일치해야 함.
    # - Mean 방식은 RGB 오리지널 Grayscale 복제본 출력의 1/3 스케일이어야 함.
    sum_diff = np.abs(out_rgb_np - out_sum_np)
    max_sum_diff = np.max(sum_diff)
    print(f"\n[Parity Test] Max absolute difference (Sum vs RGB): {max_sum_diff:.7e}")

    mean_diff = np.abs(out_rgb_np / 3.0 - out_mean_np)
    max_mean_diff = np.max(mean_diff)
    print(f"[Parity Test] Max absolute difference (Mean vs RGB/3): {max_mean_diff:.7e}")

    # 두 방식의 스케일 비율 검증 (Sum == 3 * Mean)
    sum_vs_mean_diff = np.abs(out_sum_np - 3.0 * out_mean_np)
    max_sum_vs_mean_diff = np.max(sum_vs_mean_diff)
    print(f"[Parity Test] Max absolute difference (Sum vs 3*Mean): {max_sum_vs_mean_diff:.7e}")

    # 허용 오차(floating point precision 고려, atol=1e-5) 내에서 검증
    assert max_sum_diff < 1e-5, f"Sum reduction parity failed: diff={max_sum_diff}"
    assert max_mean_diff < 1e-5, f"Mean reduction parity failed: diff={max_mean_diff}"
    assert max_sum_vs_mean_diff < 1e-5, f"Sum vs 3*Mean scaling parity failed: diff={max_sum_vs_mean_diff}"

    print("[Parity Test] All Conv1 weight transfer parity checks passed successfully!")
