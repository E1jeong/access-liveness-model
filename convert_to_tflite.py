import argparse
import torch
import sys
import os

# Windows 콘솔 한글/이모지 출력 인코딩 에러 방지
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def export_float_tflite(pth_path, float_tflite_path):
    """PyTorch 체크포인트를 NHWC 입력의 float tflite로 변환해 저장합니다."""
    # 1. PyTorch 모델 로드
    from model import get_anti_spoof_model
    model = get_anti_spoof_model()
    state_dict = torch.load(pth_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    # 2. 더미 NHWC 입력 데이터 정의 (Android 배포 사양 준수)
    # RGB: [1, 224, 224, 3], IR: [1, 224, 224, 1]
    sample_rgb = torch.randn(1, 224, 224, 3)
    sample_ir = torch.randn(1, 224, 224, 1)

    # 3. Google litert_torch를 이용한 변환 진행
    import litert_torch

    # 모델의 입출력을 채널 마지막(NHWC) 포맷으로 매핑
    nhwc_model = litert_torch.to_channel_last_io(model, args=[0, 1])
    nhwc_model.eval()

    edge_model = litert_torch.convert(nhwc_model, (sample_rgb, sample_ir))
    os.makedirs(os.path.dirname(float_tflite_path), exist_ok=True)
    edge_model.export(float_tflite_path)
    print(f"[float 변환 완료] {float_tflite_path}")


def _resolve_io_names(float_tflite_path):
    """float tflite의 시그니처 키와 RGB/IR 입력 텐서 이름을 채널 수로 식별합니다."""
    from ai_edge_litert.interpreter import Interpreter
    from ai_edge_quantizer.utils import tfl_interpreter_utils

    interp = Interpreter(model_path=float_tflite_path)
    interp.allocate_tensors()

    sig_list = interp.get_signature_list()
    if sig_list:
        sig_key = list(sig_list.keys())[0]
        in_details = interp.get_signature_runner(sig_key).get_input_details()
    else:
        sig_key = tfl_interpreter_utils.DEFAULT_SIGNATURE_KEY
        in_details = {d['name']: d for d in interp.get_input_details()}

    rgb_name, ir_name = None, None
    for name, d in in_details.items():
        if int(d['shape'][-1]) == 3:
            rgb_name = name
        elif int(d['shape'][-1]) == 1:
            ir_name = name
    if rgb_name is None or ir_name is None:
        raise RuntimeError(f"입력 텐서(RGB 3ch / IR 1ch)를 식별하지 못했습니다: {list(in_details)}")
    return sig_key, rgb_name, ir_name


def _build_calibration_data(sig_key, rgb_name, ir_name, data_dir, folds, seed, num_samples):
    """실제 학습 이미지를 NHWC float32로 변환해 calibration 데이터를 만듭니다."""
    import numpy as np
    from dataset import get_data_loaders

    # 학습과 동일한 전처리(resize+normalize)가 적용된 실제 샘플을 사용 (fold 0 train)
    train_loader, _ = get_data_loaders(
        data_dir, batch_size=8, k_folds=folds, fold_idx=0, seed=seed, num_workers=0
    )

    samples = []
    for rgb_b, ir_b, _ in train_loader:
        for i in range(rgb_b.shape[0]):
            # NCHW -> NHWC, 배치 차원 1 유지
            rgb = rgb_b[i].permute(1, 2, 0).unsqueeze(0).numpy().astype(np.float32)
            ir = ir_b[i].permute(1, 2, 0).unsqueeze(0).numpy().astype(np.float32)
            samples.append({rgb_name: rgb, ir_name: ir})
            if len(samples) >= num_samples:
                break
        if len(samples) >= num_samples:
            break

    print(f"[calibration] 실제 이미지 {len(samples)}장으로 보정합니다.")
    return {sig_key: samples}


def _report_io_dtype(int8_tflite_path):
    """양자화 모델의 입출력 dtype을 출력해 NPU(int8) 호환 여부를 확인합니다."""
    from ai_edge_litert.interpreter import Interpreter
    interp = Interpreter(model_path=int8_tflite_path)
    interp.allocate_tensors()
    print(" - 입력 텐서:")
    for d in interp.get_input_details():
        print(f"    {d['name']}: dtype={d['dtype'].__name__}, shape={list(d['shape'])}")
    print(" - 출력 텐서:")
    for d in interp.get_output_details():
        print(f"    {d['name']}: dtype={d['dtype'].__name__}, shape={list(d['shape'])}")


def quantize_int8(float_tflite_path, int8_tflite_path, data_dir, folds, seed, num_samples):
    """float tflite를 풀 INT8(가중치+활성) PTQ로 양자화합니다. i.MX 8M Plus NPU용."""
    from ai_edge_quantizer import quantizer, recipe

    sig_key, rgb_name, ir_name = _resolve_io_names(float_tflite_path)
    calibration_data = _build_calibration_data(
        sig_key, rgb_name, ir_name, data_dir, folds, seed, num_samples
    )

    print(f"\n[INT8 양자화 중...] recipe=static_wi8_ai8")
    qt = quantizer.Quantizer(float_tflite_path)
    qt.load_quantization_recipe(recipe.static_wi8_ai8())
    calib_result = qt.calibrate(calibration_data)
    result = qt.quantize(calib_result)
    result.export_model(int8_tflite_path, overwrite=True)

    float_mb = os.path.getsize(float_tflite_path) / (1024 * 1024)
    int8_mb = os.path.getsize(int8_tflite_path) / (1024 * 1024)
    print(f"[INT8 양자화 성공] {int8_tflite_path}")
    print(f" - 크기: {float_mb:.2f}MB(float) -> {int8_mb:.2f}MB(int8)")
    _report_io_dtype(int8_tflite_path)


def convert_pytorch_to_tflite(
    pth_path="model/best_model_fold0.pth",
    tflite_path="model/anti_spoofing.tflite",
    quantize=False,
    data_dir="dataset/raw",
    folds=5,
    seed=42,
    calib_samples=200,
):
    if not os.path.exists(pth_path):
        print(f"[-] {pth_path}가 존재하지 않습니다. 먼저 모델을 학습시켜주세요.")
        return

    try:
        if not quantize:
            # float tflite를 최종 산출물로 바로 생성 (기존 동작)
            print(f"\n[TFLite 변환 중...] {pth_path} -> {tflite_path}")
            export_float_tflite(pth_path, tflite_path)
            print(f"[TFLite 변환 성공] {tflite_path} 파일이 성공적으로 생성되었습니다!")
            return

        # 양자화: float를 중간 산출물로 만든 뒤 INT8로 변환
        base, ext = os.path.splitext(tflite_path)
        float_tflite_path = f"{base}_float{ext}"
        print(f"\n[TFLite 변환 중...] {pth_path} -> {float_tflite_path} (float 중간 산출물)")
        export_float_tflite(pth_path, float_tflite_path)
        quantize_int8(float_tflite_path, tflite_path, data_dir, folds, seed, calib_samples)
    except ImportError as e:
        print(f"[-] 필요한 라이브러리가 설치되지 않았습니다: {e}")
    except Exception as e:
        print(f"[-] 변환 오류 발생: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert trained PyTorch checkpoint to TFLite")
    parser.add_argument("--pth-path", default="model/best_model_fold0.pth")
    parser.add_argument("--tflite-path", default="model/anti_spoofing.tflite")
    parser.add_argument("--quantize", action="store_true", help="i.MX 8M Plus NPU용 풀 INT8 양자화 수행")
    parser.add_argument("--data-dir", default="dataset/raw", help="calibration에 사용할 데이터 경로")
    parser.add_argument("--folds", type=int, default=5, help="calibration 데이터 분할(fold 0 train 사용)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calib-samples", type=int, default=200, help="calibration 샘플 수")
    args = parser.parse_args()

    convert_pytorch_to_tflite(
        args.pth_path,
        args.tflite_path,
        quantize=args.quantize,
        data_dir=args.data_dir,
        folds=args.folds,
        seed=args.seed,
        calib_samples=args.calib_samples,
    )
