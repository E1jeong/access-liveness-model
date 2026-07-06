import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import torch

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


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

        if quantize:
            print("[INT8 양자화 파이프라인 활성화]")
            from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
            from litert_torch.quantize.pt2e_quantizer import PT2EQuantizer
            from litert_torch.quantize.quant_config import QuantConfig
            from pytorch_pipeline.dataset import get_data_loaders

            # Step 1: Export structure using torch.export
            exported_model = torch.export.export(nhwc_model, (sample_rgb, sample_ir)).module()

            # Step 2: PT2E Quantizer 준비 (Symmetric Per-Channel)
            quantizer = PT2EQuantizer()
            quantizer.set_global(
                quantizer.get_supported_quantization_configs()[2]
            )
            prepared_model = prepare_pt2e(exported_model, quantizer)

            # Step 3: Calibration 데이터 로드 및 캡처
            print("[Calibration 데이터 로드 중...]")
            _, val_loader = get_data_loaders(
                "dataset/raw",
                batch_size=8,
                k_folds=5,
                fold_idx=0,
                num_workers=2
            )

            print("[보정(Calibration) 추론 실행 중...]")
            with torch.no_grad():
                for idx, (rgb, ir, _) in enumerate(val_loader):
                    # ToTensor()는 NCHW [B, 3, H, W]를 리턴하므로 NHWC [B, H, W, C]로 변환
                    rgb_nhwc = rgb.permute(0, 2, 3, 1)
                    ir_nhwc = ir.permute(0, 2, 3, 1)
                    prepared_model(rgb_nhwc, ir_nhwc)
                    if idx >= 30:  # 30개 배치만 보정에 사용
                        break

            # Step 4: PT2E 양자화 변환 완료
            quantized_model = convert_pt2e(prepared_model, fold_quantize=False)

            # Step 5: LiteRT 변환 진행 (QuantConfig 래핑)
            q_config = QuantConfig(pt2e_quantizer=quantizer)
            edge_model = litert_torch.convert(
                quantized_model,
                (sample_rgb, sample_ir),
                quant_config=q_config
            )
        else:
            edge_model = litert_torch.convert(nhwc_model, (sample_rgb, sample_ir))

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
