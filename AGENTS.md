# access-liveness-model Instructions

`AGENTS.md` is the repository's project-rule source of truth; `CLAUDE.md` points here. Project state, history, metrics, commands, and detailed contracts belong in the Obsidian wiki, not here.

## Read Before Acting

- Wiki root: `Dev/Project/Company/access-liveness-model/` in `E1jeong/obsidian-vault`. Clone the vault locally if unavailable. Every page is in English.
- Every session: read `README.md` → `handoff.md` → `issues/needs-verification.md`; identify the machine with `nvidia-smi 2>/dev/null | grep -q "GTX 1660 Ti" && echo "sub-laptop" || echo "company PC"`; then run `git status --short`. `log.md` is a decision index read on demand, not on entry. Read `roadmap/` only for progress requests or deferred-P2 resume/design.
- Before proposing or running a command, explain the command and reason in Korean. Report the machine, latest completion, next work, and Android/NPU status in Korean.
- Before training or data work, read `operations/working-environment.md` and `technical/training-command-guide.md`. Before model-contract or NPU work or reporting either, also read paired `Dev/Project/Company/android-anti-spoofing-lab/issues/needs-verification.md`; never infer both repositories' state from one. Before INT8 work read `technical/int8-quantization-npu.md`; for current metrics read `tests/evaluation-metrics-results.md`; for the Android contract read `technical/android-deployment-agreement.md`.

## Immutable Project Boundaries

- The company PC is code/docs/git only: never train there. Only the GTX 1660 Ti sub-laptop (`sub`) is authoritative for training, data, and quantization. Use project `uv` environments; transfer company-PC edits with `rsync` to `sub`.
- Future Keras training uses fixed `dataset/raw/{train,validation,test}`, never K-Fold: train/calibration use `train`, selection uses `validation`, and `test` is final-only after settings freeze. Before every training run on the sub-laptop, run `validate_fixed_splits.py`; `scripts/keras/run_fixed_split.sh` evaluates validation only, and test requires `evaluate_tflite.py --split test`.
- Do not retrain or replace a verified model candidate without an explicit team decision. `multimodal` 5-input was removed; do not restore or deploy it. Keep `sum` conv1 reduction for 1-channel IR ImageNet Conv1 transfer. Since 2026-08-10 `sum` is the default in `scripts/keras/run_fixed_split.sh`, both parsers, and both `build_*_mobilenetv2` signatures, so a run needs no flag; `mean` is the rejected variant and must never be passed on a run whose metrics will be published.
- Shell wrappers live in `scripts/keras/` and `scripts/pytorch/`; run them from the repository root. Never invoke the Keras path as bare `python` — only `scripts/keras/*.sh` set the `LD_LIBRARY_PATH` that `.venv-tf` needs for `libcudnn`.
- Model variants use `keras_pipeline --model-type`. `classes.py:CLASS_NAMES` is the only class-index source. Follow the exact tensor, layout, normalization, and conversion requirements in the Android contract; specifically, `npu_int8` evaluation must use `[-1,1]` inputs.
- `docs/keras-concept-review.md` is the user's own comprehension record. It carries two status axes: `근거 상태` (is the claim true — settable from code) and `이해 상태` (can the user reproduce it unaided). Explain concepts in conversation; write into that file only when the user asks, and never set an `이해 상태` line on their behalf — raise it only after the user has answered a question that demonstrates it.
- Write generated `.tflite` and `.pth` only under gitignored `model/`; do not sync artifacts with git. Android deployment manually copies each selected model together with its matching sidecar manifest into the app assets and registers the correct slot type.
- `Backend CPU` means no NNAPI acceleration. Current Android `master` rejects a model slot when NNAPI preparation or warmup fails; it must not fall back to CPU. Do not generalize one model's NNAPI success to another architecture.
- Do not retry stopped PyTorch/MobileNetV3 INT8 paths before reading their chronology. Diagnose an NPU failure by isolating unsupported operations, not by retraining.

## Safety

- Antigravity `write_to_file` temporary scripts must use an agent-artifact scratch `TargetFile`, never an Obsidian absolute path or other external local path; external paths cause `invalid_args`.
- If an unexpected performance blocker appears, do not independently switch pipelines or convert through another path. Stop, summarize the fact, and obtain the user's explicit decision on the next deployment step.
