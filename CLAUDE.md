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
- `Project/Company/access-liveness-model/log.md` — 변경 이력(append-only, 영어)
- `Project/Company/android-anti-spoofing-lab/` — 이 프로젝트와 세트로 움직이는 Android 평가 앱(`android-anti-spoofing-lab`) 위키. 모델 계약(입출력 텐서 구성)이나 NPU 상태를 바꿀 때는 반드시 같이 확인한다.

## 3. 저장소 상태 확인

```bash
git status --short
```

## 4. 현재 모델 선택 방향

- 현재 비교 대상은 `dual` 2-input 모델과 RGB/IR `paired_1_input` 모델이다.
- `multimodal` 5-input 모델은 과거 테스트에서 2-input보다 성능이 현저히 낮았다는 팀 피드백이 있으나 정량 수치는 남아 있지 않다. 폐기 확정은 아니지만 당분간 사용하지 않으며, 사용자의 명시적 요청 없이 5-input 학습·배포를 우선하지 않는다.
- 6클래스 `dual` 2-input 학습은 완료되지 않았고 실행 도중 멈춘 상태에서 보류 중이다. 후보에서 제외된 것은 아니므로 폐기된 것으로 기록하지 말고, 사용자 지시 없이 자동으로 재개하지 않는다.
- 현재 실기기 검증 기준선은 RGB fold3 + IR fold4의 6클래스 `paired_1_input` NPU-friendly INT8 조합이다. Android manifest는 RGB를 fold4로 바꾸는 변경이 있으므로, fold4/fold4 조합은 실기기 검증 전까지 기준선으로 보고하지 않는다.

## 5. 사용자에게 보고

위 확인 결과를 바탕으로 다음을 한국어로 간단히 보고한다.

- 현재 머신
- 마지막으로 완료된 작업 (옵시디언 `log.md` 기준)
- 다음으로 할 작업 (옵시디언 `이슈/확인 필요.md` 기준)
- 현재 Android/NPU 상태: `Backend CPU`이면 NNAPI 실패 후 CPU/XNNPACK fallback이며, NPU 가속 성공으로 보고하지 않는다. 최신 상태는 `Project/Company/android-anti-spoofing-lab/기능/모델 계약과 브랜치.md` 참조.
