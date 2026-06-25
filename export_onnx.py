"""학습한 PyTorch 모델을 ONNX로 export한다 (NXP eIQ Toolkit import용).

eIQ Portal은 ONNX를 import해 representative 데이터로 INT8 양자화 후
i.MX 8M Plus NPU용 TFLite를 생성할 수 있다. 여기서는 양자화 전 float ONNX만 만든다.

사용:
    python export_onnx.py --pth-path model/best_model_fold0.pth --onnx-path model/anti_spoofing.onnx
"""

import argparse
import os
import sys

import torch

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def export(pth_path, onnx_path, opset):
    from model import get_anti_spoof_model

    if not os.path.exists(pth_path):
        print(f"[-] {pth_path} 없음. 먼저 모델을 학습하세요.")
        return

    model = get_anti_spoof_model()
    model.load_state_dict(torch.load(pth_path, map_location="cpu"))
    model.eval()

    # 원본 모델 forward(rgb, ir)는 NCHW 입력 (ONNX 표준 레이아웃)
    dummy_rgb = torch.randn(1, 3, 224, 224)
    dummy_ir = torch.randn(1, 1, 224, 224)

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_rgb, dummy_ir),
        onnx_path,
        input_names=["rgb", "ir"],
        output_names=["logits"],
        opset_version=opset,
        dynamic_axes={
            "rgb": {0: "batch"},
            "ir": {0: "batch"},
            "logits": {0: "batch"},
        },
    )
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"[ONNX export 성공] {onnx_path} ({size_mb:.2f}MB, opset={opset})")
    print(" - 입력: rgb [N,3,224,224], ir [N,1,224,224] (NCHW)")
    print(" - 출력: logits [N,5]")
    print("eIQ Portal에서 이 ONNX를 import → representative 데이터로 INT8 양자화하세요.")


def main():
    parser = argparse.ArgumentParser(description="Export trained model to ONNX for eIQ")
    parser.add_argument("--pth-path", default="model/best_model_fold0.pth")
    parser.add_argument("--onnx-path", default="model/anti_spoofing.onnx")
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()
    export(args.pth_path, args.onnx_path, args.opset)


if __name__ == "__main__":
    main()
