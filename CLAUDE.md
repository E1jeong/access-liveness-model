# Claude Code 세션 시작 절차

매 세션 시작 시 아래 순서를 반드시 따른다. 사용자가 별도로 요청하지 않아도 자동으로 수행한다.

## 1. 머신 확인

```bash
nvidia-smi 2>/dev/null | grep -q "GTX 1660 Ti" && echo "서브노트북" || echo "회사 PC"
```

- **서브노트북**: GPU 학습·변환 가능. `run_keras_*.sh` 스크립트 사용.
- **회사 PC**: CPU 전용. 코드·문서 편집 및 git push/pull만 수행. 학습 명령은 실행하지 않는다.

## 2. 문서 읽기

이 저장소에는 `docs/` 폴더가 없다. 고정 개발 기준과 현재 상태는 옵시디언 vault(GitHub `E1jeong/obsidian-vault`, 로컬에 없으면 clone)의 다음 경로에 있다.

- `Project/Company/access-liveness-model/개요.md` — 제품 목표, 현재 구현 범위, 정보 우선순위
- `Project/Company/access-liveness-model/운영.md` — 머신별 세팅 상태(uv venv 구성 등), 검증 명령
- `Project/Company/access-liveness-model/로드맵/개발 단계.md` — 고정 개발 게이트, 다음 작업 순서
- `Project/Company/access-liveness-model/테스트/평가지표와 결과.md` — 현재 성능 수치
- `Project/Company/access-liveness-model/이슈/확인 필요.md` — 미해결·확인 필요 항목 (가장 먼저 확인)
- `Project/Company/access-liveness-model/로드맵/전면 개편 작업 백로그.md` — 세션 간 전면 개편 작업의 우선순위·상태·완료 기준
- `Project/Company/access-liveness-model/log.md` — 변경 이력(append-only, 영어)
- `Project/Company/android-anti-spoofing-lab/` — 이 프로젝트와 세트로 움직이는 Android 평가 앱(`android-anti-spoofing-lab`) 위키. 모델 계약(입출력 텐서 구성)이나 NPU 상태를 바꿀 때는 반드시 같이 확인한다.

## 3. 저장소 상태 확인

```bash
git status --short
```

## 4. 현재 모델 선택 방향

- 현재 비교 대상은 `dual` 2-input 모델과 RGB/IR `paired_1_input` 모델이다.
- `multimodal` 5-input 모델은 과거 테스트에서 성능 미달로 인해 코드베이스 및 학습/평가 파이프라인에서 완전히 폐기 및 제거되었습니다.
- 6클래스 `dual` 2-input 학습은 완료되지 않았고 실행 도중 멈춘 상태에서 보류 중이다. 후보에서 제외된 것은 아니므로 폐기된 것으로 기록하지 말고, 사용자 지시 없이 자동으로 재개하지 않는다.
- 현재 실기기 검증 기준선은 RGB fold3 + IR fold4의 6클래스 `paired_1_input` NPU-friendly INT8 조합이다. Android manifest는 RGB를 fold4로 바꾸는 변경이 있으므로, fold4/fold4 조합은 실기기 검증 전까지 기준선으로 보고하지 않는다.

## 5. 고정 데이터 분할 및 정밀 누수 검사

- `[검증 완료]` 신규 Keras 학습은 K-Fold 대신 `dataset/raw/train`, `dataset/raw/validation`, `dataset/raw/test` 고정 분할을 사용한다.
- `train`은 가중치 학습과 INT8 calibration, `validation`은 best checkpoint 선택, `test`는 설정 확정 후 최종 평가에만 사용한다.
- `validate_fixed_splits.py`가 클래스/파일 완전성과 `subject`/`frame` split 누수를 검사한다. 2026-07-16부터 **파일 MD5 콘텐츠 해시 대조** 및 `meta.json` 기반 **`session`/`video` ID 오버랩 검출** 장치가 강화되었습니다.
- 실제 6클래스 고정 split은 GPU 서브노트북에서 누수 검사를 통과했다(train 12,000 / validation 1,200 / test 1,198). 신규 학습 전에는 항상 `validate_fixed_splits.py`를 다시 실행하여 데이터셋 오염을 차단한다.

## 6. 환경 락(Lock) 및 Git 동기화

- **의존성 환경 재현**: 각 머신/목적에 적합한 락(lock) 파일이 `requirements/` 디렉터리에 제공됩니다.
  - WSL CPU 개발 환경: `uv pip install -r requirements/wsl-cpu.lock`
  - 원격 PyTorch GPU 환경: `uv pip install -r requirements/sub-gpu-pytorch.lock`
  - 원격 TensorFlow GPU 환경: `uv pip install -r requirements/sub-gpu-keras.lock`
- **Git 꼬임 방지**: 여러 머신(Mac, WSL 등) 간의 깃 충돌을 피하기 위해 다음 글로벌 깃 설정이 권장됩니다.
  - `git config --global pull.rebase true && git config --global rebase.autoStash true && git config --global core.autocrlf input`
  - 서브노트북 및 개인 개발 장비에서 안전하게 원격과 소스코드를 동기화할 수 있도록 `./scripts/git_pull_clean.sh` (강제 초기화 시 `-f`) 헬퍼 스크립트가 제공됩니다.

## 7. 사용자에게 보고

위 확인 결과를 바탕으로 다음을 한국어로 간단히 보고한다.

- 현재 머신
- 마지막으로 완료된 작업 (옵시디언 `log.md` 기준)
- 다음으로 할 작업 (옵시디언 `이슈/확인 필요.md` 기준)
- 현재 Android/NPU 상태: `Backend CPU`이면 NNAPI 실패 후 CPU/XNNPACK fallback이며, NPU 가속 성공으로 보고하지 않는다. 최신 상태는 `Project/Company/android-anti-spoofing-lab/기능/모델 계약과 브랜치.md` 참조.
