import argparse
import torch
import sys
import os

# Windows 콘솔 한글/이모지 출력 인코딩 에러 방지
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def convert_pytorch_to_tflite(pth_path="model/best_model_fold0.pth", tflite_path="model/anti_spoofing.tflite"):
    """학습한 PyTorch 체크포인트를 NHWC 입력의 float TFLite로 변환한다.

    INT8 양자화는 현재 도구 체계로는 동작하지 않아 보류됨(docs/project_status.md §3 참고).
    배포는 이 float 모델을 CPU로 사용한다.
    """
    if not os.path.exists(pth_path):
        print(f"[-] {pth_path}가 존재하지 않습니다. 먼저 모델을 학습시켜주세요.")
        return

    # PyTorch 모델 로드
    from model import get_anti_spoof_model
    model = get_anti_spoof_model()
    state_dict = torch.load(pth_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    # 더미 NHWC 입력 (Android 배포 사양: RGB [1,224,224,3], IR [1,224,224,1])
    sample_rgb = torch.randn(1, 224, 224, 3)
    sample_ir = torch.randn(1, 224, 224, 1)

    print(f"\n[TFLite 변환 중...] {pth_path} -> {tflite_path}")
    try:
        import litert_torch

        # 입출력을 채널 마지막(NHWC) 포맷으로 매핑
        nhwc_model = litert_torch.to_channel_last_io(model, args=[0, 1])
        nhwc_model.eval()

        edge_model = litert_torch.convert(nhwc_model, (sample_rgb, sample_ir))
        os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
        edge_model.export(tflite_path)
        print(f"[TFLite 변환 성공] {tflite_path} 파일이 성공적으로 생성되었습니다!")
    except ImportError:
        print("[-] litert_torch 라이브러리가 설치되지 않았습니다. 설치를 진행해 주세요.")
    except Exception as e:
        print(f"[-] 변환 오류 발생: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert trained PyTorch checkpoint to float TFLite")
    parser.add_argument("--pth-path", default="model/best_model_fold0.pth")
    parser.add_argument("--tflite-path", default="model/anti_spoofing.tflite")
    args = parser.parse_args()
    convert_pytorch_to_tflite(args.pth_path, args.tflite_path)
