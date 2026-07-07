import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# torch.export.export 멍키 패치 (torchao/pytorch 버전 호환성 문제 해결용)
_original_export = torch.export.export
def _patched_export(mod, args, kwargs=None, *opt_args, **opt_kwargs):
    if not isinstance(mod, nn.Module) and callable(mod):
        class FunctionWrapper(nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn
            def forward(self, *args, **kwargs):
                return self.fn(*args, **kwargs)
        mod = FunctionWrapper(mod)
    return _original_export(mod, args, kwargs, *opt_args, **opt_kwargs)
torch.export.export = _patched_export


def convert_pytorch_to_tflite(pth_path="model/best_model_fold0.pth", tflite_path="model/anti_spoofing.tflite", quantize=False):
    """학습한 PyTorch 체크포인트를 NHWC 입력의 TFLite(Float 또는 INT8 양자화)로 변환한다."""
    if not os.path.exists(pth_path):
        print(f"[-] {pth_path}가 존재하지 않습니다. 먼저 모델을 학습시켜주세요.")
        return

    from pytorch_pipeline.model import get_anti_spoof_model
    model = get_anti_spoof_model()
    state_dict = torch.load(pth_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    # 더미 NHWC 입력 (Android 배포 사양: RGB [1,224,224,3], IR [1,224,224,1])
    sample_rgb = torch.randn(1, 224, 224, 3)
    sample_ir = torch.randn(1, 224, 224, 1)

    print(f"\n[TFLite 변환 중...] {pth_path} -> {tflite_path} (Quantize={quantize})")
    try:
        import litert_torch

        # channels-last wrapper
        nhwc_model = litert_torch.to_channel_last_io(model, args=[0, 1])
        nhwc_model.eval()

        # 1단계: 순수 Float32 TFLite 모델 변환 및 저장
        print("[1단계] Float32 TFLite 변환 진행 중...")
        edge_model = litert_torch.convert(nhwc_model, (sample_rgb, sample_ir))
        
        dirpath = os.path.dirname(tflite_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
            
        float_tflite_path = tflite_path
        if quantize:
            float_tflite_path = tflite_path.replace(".tflite", "_float.tflite")
            if float_tflite_path == tflite_path:
                float_tflite_path = tflite_path + "_float.tflite"
                
        edge_model.export(float_tflite_path)
        print(f" -> Float32 TFLite 생성 완료: {float_tflite_path}")

        # 2단계: 양자화(INT8) 처리 진행
        if quantize:
            print("[2단계] ai_edge_quantizer를 이용한 INT8 정적 양자화 적용 중...")
            import ai_edge_quantizer as aq
            import numpy as np
            from pytorch_pipeline.dataset import get_data_loaders

            # Quantizer 인스턴스 생성
            qt = aq.Quantizer(float_tflite_path)
            # 빌트인 static 레시피 로드 (가중치 INT8, 활성화 INT8)
            qt.load_quantization_recipe("static_wi8_ai8")

            # 셔플된 train_loader로 보정 데이터 생성
            print(" -> Calibration 데이터 수집 중...")
            train_loader, _ = get_data_loaders(
                "dataset/raw",
                batch_size=1,
                k_folds=5,
                fold_idx=0,
                num_workers=2
            )

            calibration_samples = []
            for idx, (rgb, ir, _) in enumerate(train_loader):
                # NHWC float32 NumPy 배열로 변환
                rgb_np = rgb.permute(0, 2, 3, 1).numpy().astype(np.float32)
                ir_np = ir.permute(0, 2, 3, 1).numpy().astype(np.float32)
                # 시그니처 텐서 인풋 키 매핑
                calibration_samples.append({
                    "args_0": rgb_np,
                    "args_1": ir_np
                })
                if idx >= 200:  # 200개 샘플 보정
                    break

            calibration_data = {
                "serving_default": calibration_samples
            }

            # 보정(Calibration) 실행
            print(" -> 모델 보정 추론 중...")
            calibration_result = qt.calibrate(calibration_data)

            # 양자화 빌드 및 최종 저장
            print(" -> INT8 TFLite 파일 생성 중...")
            quant_result = qt.quantize(calibration_result)

            with open(tflite_path, "wb") as f:
                f.write(quant_result.quantized_model)
            print(f"[TFLite 양자화 성공] {tflite_path} 파일이 성공적으로 생성되었습니다!")
        else:
            dirpath = os.path.dirname(tflite_path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            edge_model.export(tflite_path)
            print(f"[TFLite 변환 성공] {tflite_path} 파일이 성공적으로 생성되었습니다!")
    except ImportError as e:
        print(f"[-] 라이브러리 임포트 오류: {e}")
    except Exception as e:
        print(f"[-] 변환 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert trained PyTorch checkpoint to float or INT8 TFLite")
    parser.add_argument("--pth-path", default="model/best_model_fold0.pth")
    parser.add_argument("--tflite-path", default="model/anti_spoofing.tflite")
    parser.add_argument("--quantize", action="store_true", help="Perform static INT8 quantization")
    args = parser.parse_args()
    convert_pytorch_to_tflite(args.pth_path, args.tflite_path, args.quantize)
