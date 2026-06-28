# Claude Code 세션 시작 절차

매 세션 시작 시 아래 순서를 반드시 따른다. 사용자가 별도로 요청하지 않아도 자동으로 수행한다.

## 1. 머신 확인

```bash
nvidia-smi 2>/dev/null | grep -q "GTX 1660 Ti" && echo "서브노트북" || echo "회사 PC"
```

- **서브노트북**: GPU 학습·변환 가능. `run_keras_*.sh` 스크립트 사용.
- **회사 PC**: CPU 전용. 코드·문서 편집 및 git push/pull만 수행. 학습 명령은 실행하지 않는다.

## 2. 문서 읽기

다음 두 파일을 반드시 읽는다.

- `docs/project_guide.md` — 고정 개발 기준, 단계별 게이트, AI 행동 규칙
- `docs/project_status.md` — 현재 상태, 검증 결과, 다음 작업 순서

## 3. 저장소 상태 확인

```bash
git status --short
```

## 4. 사용자에게 보고

위 확인 결과를 바탕으로 다음을 한국어로 간단히 보고한다.

- 현재 머신
- 마지막으로 완료된 작업 (`project_status.md` §8 변경 로그 기준)
- 다음으로 할 작업 (`project_status.md` §0.1 핸드오프 기준)
