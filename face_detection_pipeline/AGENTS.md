# Face Detection Pipeline Module Guide

## Scope

- Own the RGB face detector architecture, training data adapter, losses, and evaluation. Export is not implemented yet.
- Keep detector outputs separate from PAD classes and decisions.

## Orient First

- Model: `models/yunet.py`
- Data adapter: `data/dataset.py`
- Loss: `training/loss.py`
- Training entry point: `training/train.py`
- Decoding and metrics: `evaluation/detection.py`

## Boundary & Architecture Constraints

- The v1 contract is one RGB image to zero or more axis-aligned face boxes with confidence.
- Use the box-only YuNet-N source pinned in `models/yunet.py`; do not add public pretrained weights or public face-detection datasets.
- Preserve the upstream BSD-3-Clause license in `third_party/yunet/LICENSE`.
- Keep landmarks out of v1. Add a landmark head only when the downstream SDK contract requires it.

## Change Gates

- Preserve the fixed `dataset/raw/{train,validation,test}` split and original capture files.
- Store generated checkpoints and exports only under gitignored `model/`.
- Keep 320×320 and 256×448 as candidates until a reviewed full comparison fixes the export shape.

## Verify

```bash
uv run pytest tests/face_detection
bash -n scripts/face_detection/*.sh
```
