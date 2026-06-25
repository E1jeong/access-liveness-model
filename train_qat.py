"""QAT-then-PTQ: 원본 모델을 활성(activation) fake-quant로 fine-tune하여
양자화에 강건한 가중치를 만든 뒤, 기존 PTQ 경로로 int8 tflite를 만든다.

배경:
  - w8only 실험에서 int8 '가중치'는 정상(ACER≈0)임이 확인됨.
  - 붕괴는 '활성' 양자화에서만 발생 → 활성을 양자화에 강하게 학습시키면 됨.
  - litert_torch의 PT2E 변환기는 MobileNetV3 SE 블록에서 깨지므로 사용하지 않는다.
    여기서는 torch만으로 fake-quant fine-tune을 하고, 원본 모델 구조 그대로
    .pth를 저장한다. 이후 검증된 PTQ 경로로 변환한다:

      python convert_to_tflite.py --quantize --pth-path model/best_model_qat.pth
      python evaluate_tflite.py --models model/anti_spoofing.tflite model/anti_spoofing_float.tflite
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


class ActFakeQuant(nn.Module):
    """per-tensor 대칭 int8 활성 fake-quant. STE로 그래디언트를 통과시키고,
    학습 중 EMA로 스케일을 갱신한다(=관찰된 활성 범위에 맞춰 양자화 시뮬레이션)."""

    def __init__(self, momentum=0.01):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("ready", torch.tensor(0))

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                amax = x.detach().abs().max().clamp(min=1e-8)
                s = amax / 127.0
                if int(self.ready) == 0:
                    self.scale.copy_(s)
                    self.ready.fill_(1)
                else:
                    self.scale.mul_(1 - self.momentum).add_(self.momentum * s)
        q = torch.clamp(torch.round(x / self.scale), -128, 127) * self.scale
        return x + (q - x).detach()  # straight-through estimator


def attach_act_fakequant(model, device):
    """활성 출력(conv-bn 융합 결과 ~ BN 출력, 활성함수 출력, Linear)에 fake-quant 부착."""
    handles, fqs = [], []
    # conv 단독 출력은 BN과 융합되므로 굳이 양자화하지 않고, 융합 결과에 해당하는
    # BatchNorm 출력과 활성함수/Linear 출력을 양자화한다(실제 int8 그래프와 유사).
    target = (nn.BatchNorm2d, nn.Hardswish, nn.Hardsigmoid, nn.ReLU, nn.Linear)
    for m in model.modules():
        if isinstance(m, target):
            fq = ActFakeQuant().to(device)
            fqs.append(fq)

            def hook(mod, inp, out, _fq=fq):
                if isinstance(out, torch.Tensor):
                    return _fq(out)
                return out

            handles.append(m.register_forward_hook(hook))
    return handles, fqs


def run_qat(args):
    from dataset import get_data_loaders
    from model import get_anti_spoof_model

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"QAT 디바이스: {device}")

    if not os.path.exists(args.pth_path):
        print(f"[-] {args.pth_path} 없음. 먼저 float 모델을 학습하세요.")
        return

    model = get_anti_spoof_model()
    model.load_state_dict(torch.load(args.pth_path, map_location="cpu"))
    model.to(device)

    handles, fqs = attach_act_fakequant(model, device)
    # 입력도 최종 int8 모델에서 양자화되므로 입력 fake-quant 추가
    in_fq_rgb = ActFakeQuant().to(device)
    in_fq_ir = ActFakeQuant().to(device)
    print(f"활성 fake-quant {len(fqs)}개 + 입력 2개 부착")

    train_loader, val_loader = get_data_loaders(
        args.data_dir, batch_size=args.batch_size, k_folds=args.folds,
        fold_idx=args.fold_idx, seed=args.seed, num_workers=args.num_workers,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        in_fq_rgb.train(); in_fq_ir.train()
        run, correct, total = 0.0, 0, 0
        for rgb, ir, labels in tqdm(train_loader, desc=f"QAT {epoch+1}/{args.epochs}"):
            rgb, ir, labels = rgb.to(device), ir.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(in_fq_rgb(rgb), in_fq_ir(ir))
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            run += loss.item() * rgb.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += rgb.size(0)
        print(f" -> train_loss {run/total:.4f} | train_acc {correct/total*100:.2f}%")

        model.eval()
        in_fq_rgb.eval(); in_fq_ir.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for rgb, ir, labels in val_loader:
                rgb, ir, labels = rgb.to(device), ir.to(device), labels.to(device)
                out = model(in_fq_rgb(rgb), in_fq_ir(ir))
                vc += (out.argmax(1) == labels).sum().item()
                vt += rgb.size(0)
        print(f" -> val_acc(fake-quant) {vc/vt*100:.2f}%")

    # 후크 제거 → 원본 모델 구조 그대로의 강건 가중치 저장
    for h in handles:
        h.remove()
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    torch.save(model.state_dict(), args.out_path)
    print(f"\n[QAT 완료] 양자화-강건 가중치 저장: {args.out_path}")
    print("다음 단계:")
    print(f"  python convert_to_tflite.py --quantize --pth-path {args.out_path}")
    print("  python evaluate_tflite.py --models model/anti_spoofing.tflite model/anti_spoofing_float.tflite")


def parse_args():
    p = argparse.ArgumentParser(description="Activation fake-quant fine-tune (QAT-then-PTQ)")
    p.add_argument("--pth-path", default="model/best_model_fold0.pth", help="시작 float 가중치")
    p.add_argument("--out-path", default="model/best_model_qat.pth", help="강건 가중치 저장 경로")
    p.add_argument("--data-dir", default="dataset/raw")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-5, help="QAT는 낮은 학습률 권장")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold-idx", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default=None, help="cuda/cpu (기본: 자동)")
    return p.parse_args()


if __name__ == "__main__":
    run_qat(parse_args())
