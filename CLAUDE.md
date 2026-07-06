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

- `Project/Company/Access Liveness Model/핸드오프.md` — **가장 먼저 읽는다.** 지난 세션 요약, 현재 상태, 다음 시작점, 블로커(예: ALL-STOP 여부)
- `Project/Company/Access Liveness Model/개요.md` — 제품 목표, 현재 구현 범위, 정보 우선순위
- `Project/Company/Access Liveness Model/운영.md` — 머신별 세팅 상태(uv venv 구성 등), 검증 명령
- `Project/Company/Access Liveness Model/로드맵/개발 단계.md` — 고정 개발 게이트, 다음 작업 순서
- `Project/Company/Access Liveness Model/테스트/평가지표와 결과.md` — 현재 성능 수치
- `Project/Company/Access Liveness Model/이슈/확인 필요.md` — 미해결·확인 필요 항목 (가장 먼저 확인)
- `Project/Company/Access Liveness Model/log.md` — 변경 이력(append-only, 영어)
- `Project/Company/Anti-Spoofing Viewer/` — 이 프로젝트와 세트로 움직이는 Android 평가 앱(`android-anti-spoofing-lab`) 위키. 모델 계약(입출력 텐서 구성)이나 NPU 상태를 바꿀 때는 반드시 같이 확인한다.

## 3. 저장소 상태 확인

```bash
git status --short
```

## 4. 사용자에게 보고

위 확인 결과를 바탕으로 다음을 한국어로 간단히 보고한다.

- 현재 머신
- 마지막으로 완료된 작업 (옵시디언 `log.md` 기준)
- 다음으로 할 작업 (옵시디언 `이슈/확인 필요.md` 기준)
- 현재 Android/NPU 상태: `Backend CPU`이면 NNAPI 실패 후 CPU/XNNPACK fallback이며, NPU 가속 성공으로 보고하지 않는다. 최신 상태는 `Project/Company/Anti-Spoofing Viewer/기능/모델 계약과 브랜치.md` 참조.

## 5. 위키 업데이트 ("위키 업데이트해" 요청 시 또는 세션 종료 시)

- 이번 세션에서 바뀐 것(학습 결과, 양자화 시도, 발견한 버그, 결정 사항)을 정리한다.
- 확정된 사실은 관련 위키 문서에 반영하고, 불확실한 내용은 `이슈/확인 필요.md`에 남긴다. 모델 계약/NPU 상태가 바뀌었다면 `Project/Company/Anti-Spoofing Viewer/` 쪽도 같이 갱신한다.
- `log.md`에 append-only로 기록을 남긴다.
- `핸드오프.md`를 **덮어써서**(append 아님) 최신 상태로 갱신한다.
