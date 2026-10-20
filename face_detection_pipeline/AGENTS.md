# Face Detection Pipeline Module Guide

## Scope

- Own the RGB face detector architecture, training data adapter, losses, evaluation, and fixed-shape TFLite export.
- Keep detector outputs separate from PAD classes and decisions.

## Orient First

- Model: `models/yunet.py`
- Data adapter: `data/dataset.py`
- Loss: `training/loss.py`
- Training entry point: `training/train.py`
- Decoding and metrics: `evaluation/detection.py`
- Float/optional INT8 export and parity checks: `export/convert.py` via `scripts/face_detection/run_convert.sh`

## Boundary & Architecture Constraints

- The v1 contract is one RGB image to zero or more axis-aligned face boxes with confidence.
- Use the box-only YuNet-N source pinned in `models/yunet.py`; do not add public pretrained weights or public face-detection datasets.
- Preserve the upstream BSD-3-Clause license in `third_party/yunet/LICENSE`.
- Keep landmarks out of v1. Add a landmark head only when the downstream SDK contract requires it.

## Change Gates

- Preserve the fixed `dataset/raw/{train,validation,test}` split and original capture files.
- Store generated checkpoints and exports only under gitignored `model/`.
- Keep 320×320 and 256×448 as candidates until a reviewed full comparison fixes the export shape.
- Export NHWC RGB pixels in `[0,255]` with the training letterbox; preserve the documented stride/row/channel order in `export/convert.py`. Require float parity before INT8, report quantized detection metrics, and do not claim NPU support without device verification.

## Verify

```bash
uv run pytest tests/face_detection
for script in scripts/face_detection/*.sh; do bash -n "$script" || exit; done
```
