# access-liveness-model Instructions

`AGENTS.md` is the repository's project-rule source of truth; `CLAUDE.md` points here. Project state, history, metrics, commands, and detailed contracts belong in the Obsidian wiki, not here.

## Read Before Acting

- Wiki root: `Project/Company/access-liveness-model/` in `E1jeong/obsidian-vault`. Clone the vault locally if unavailable.
- Every session: read `이슈/확인 필요.md` → `log.md` → `핸드오프.md`; identify the machine with `nvidia-smi 2>/dev/null | grep -q "GTX 1660 Ti" && echo "sub-laptop" || echo "company PC"`; then run `git status --short`. Read `로드맵/` only for progress requests or deferred-P2 resume/design.
- Before proposing or running a command, explain the command and reason in Korean. Report the machine, latest completion, next work, and Android/NPU status in Korean.
- Before training or data work, read `운영.md` and `기술/학습 명령어 가이드.md`. Before model-contract or NPU work or reporting either, also read paired `Project/Company/android-anti-spoofing-lab/이슈/확인 필요.md`; never infer both repositories' state from one. Before INT8 work read `기술/INT8 양자화와 NPU.md`; for current metrics read `테스트/평가지표와 결과.md`; for the Android contract read `기술/Android 배포 계약.md`.

## Immutable Project Boundaries

- The company PC is code/docs/git only: never train there. Only the GTX 1660 Ti sub-laptop (`sub`) is authoritative for training, data, and quantization. Use project `uv` environments; transfer company-PC edits with `rsync` to `sub`.
- Future Keras training uses fixed `dataset/raw/{train,validation,test}`, never K-Fold: train/calibration use `train`, selection uses `validation`, and `test` is final-only after settings freeze. Before every training run on the sub-laptop, run `validate_fixed_splits.py`; `run_fixed_split.sh` evaluates validation only, and test requires `evaluate_tflite.py --split test`.
- Do not retrain or replace a verified model candidate without an explicit team decision. `multimodal` 5-input was removed; do not restore or deploy it. Keep `--conv1-reduction sum` for 1-channel IR ImageNet Conv1 transfer.
- Model variants use `keras_pipeline --model-type`. `classes.py:CLASS_NAMES` is the only class-index source. Follow the exact tensor, layout, normalization, and conversion requirements in the Android contract; specifically, `npu_int8` evaluation must use `[-1,1]` inputs.
- Write generated `.tflite` and `.pth` only under gitignored `model/`; do not sync artifacts with git. Android deployment manually copies each selected model together with its matching sidecar manifest into the app assets and registers the correct slot type.
- `Backend CPU` means no NNAPI acceleration. Current Android `master` rejects a model slot when NNAPI preparation or warmup fails; it must not fall back to CPU. Do not generalize one model's NNAPI success to another architecture.
- Do not retry stopped PyTorch/MobileNetV3 INT8 paths before reading their chronology. Diagnose an NPU failure by isolating unsupported operations, not by retraining.

## Safety

- Antigravity `write_to_file` temporary scripts must use an agent-artifact scratch `TargetFile`, never an Obsidian absolute path or other external local path; external paths cause `invalid_args`.
- If an unexpected performance blocker appears, do not independently switch pipelines or convert through another path. Stop, summarize the fact, and obtain the user's explicit decision on the next deployment step.
