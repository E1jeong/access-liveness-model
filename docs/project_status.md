# 안티스푸핑 프로젝트 현재 상태

이 문서는 작업할 때마다 바뀌는 사실과 검증 결과를 기록한다. 고정 개발 절차와 판단 기준은 [project_guide.md](project_guide.md)를 따른다.

- **마지막 검증일**: 2026-06-22
- **검증 환경**: Windows, PowerShell, 프로젝트 `.venv`
- **현재 단계**: 단계 1 기준선 확인 완료, 단계 2~3 평가·ROI·데이터 수집 기준 정비 필요
- **다음 작업**: 독립 데이터 분할·평가 규격과 원본 보존 ROI 파이프라인을 구현 가능한 수준으로 확정

## 1. 상태 요약

### 완료 및 검증된 항목

- `[검증 완료]` 모든 현재 Python 파일이 문법 컴파일을 통과했다.
- `[검증 완료]` 가상환경의 주요 버전이 기존 기록과 일치한다.
- `[검증 완료]` MobileNetV3-Small 체크포인트가 정상 로드되며 분류 출력은 2개다.
- `[검증 완료]` 현재 클래스 매핑은 `real: 0`, `spoof: 1`이다.
- `[검증 완료]` ONNX checker를 통과했다.
- `[검증 완료]` 동일 무작위 입력에서 PyTorch와 ONNX 최대 절대 출력 차이는 약 `6.63e-7`이며 예측 클래스가 일치했다.
- `[검증 완료]` 현재 validation 45장의 confusion matrix는 `[[21, 0], [0, 24]]`, Accuracy는 100%다.

### 구현됐지만 제품 성능으로 검증되지 않은 항목

- `[구현 완료]` 웹캠 RGB 프레임 수집
- `[구현 완료]` 중앙 250×250 크롭과 224×224 모델 입력 변환
- `[구현 완료]` MobileNetV3-Small 학습
- `[구현 완료]` ONNX 변환 및 PC 웹캠 추론
- `[미검증]` 미등록 인물, 다른 장소, 다른 카메라, 다양한 조명에서의 성능
- `[미검증]` 마스크 착용 REAL 성능
- `[미검증]` Paper Mask Attack 성능
- `[미검증]` 독립 test set의 APCER/BPCER/ACER

### 미구현 항목

- `[미구현]` 사람·촬영 세션 단위 데이터 분할 검사
- `[미구현]` 독립 test split
- `[미구현]` 얼굴 검출 기반 ROI
- `[미구현]` 원본과 processed 데이터 분리 및 `metadata.csv`
- `[미구현]` TFLite INT8 변환
- `[미구현]` i.MX 8M Plus NPU 실기기 검증
- `[미구현]` Android 애플리케이션 통합
- `[미구현]` RGB+IR 모델

## 2. 데이터 현황

| Split | REAL | SPOOF | 합계 |
|---|---:|---:|---:|
| Train | 80 | 55 | 135 |
| Validation | 21 | 24 | 45 |
| 합계 | 101 | 79 | 180 |

- 모든 현재 이미지는 250×250 JPEG다.
- 정확히 동일한 파일 해시 중복은 발견되지 않았다.
- 표본 확인과 파일명 시퀀스상 동일 인물·동일 장소·연속 촬영 데이터로 판단된다.
- 현재 SPOOF는 스마트폰 화면 Replay Attack에 한정된다.
- 현재 데이터는 초기 파이프라인 동작 확인용이며 제품 성능 평가용이 아니다.
- 사람 수: 표본상 1명으로 판단되지만 전체 metadata가 없어 확정 기록은 없다.
- 세션 수: metadata가 없어 신뢰할 수 있게 산정할 수 없다.
- 공격 종류: `replay_phone` 1종만 확인됨

## 3. 현재 모델과 전처리

- 모델: `torchvision.models.mobilenet_v3_small`
- 초기 가중치: ImageNet pretrained weights
- 출력: REAL/SPOOF 2개 logits
- 입력: RGB, float32, `[batch, 3, 224, 224]`
- 정규화: ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
- 학습 설정: CPU, 10 epochs, batch size 8, learning rate `1e-4`, Adam, CrossEntropyLoss
- 체크포인트: `best_model.pth`
- ONNX: `model.onnx`과 외부 데이터 파일 `model.onnx.data`
- ONNX 입력/출력: `input [batch,3,224,224]` → `output [batch,2]`

현재 validation 100%는 동일한 사람과 환경에서 연속 촬영된 작은 데이터에 대한 결과다. 새로운 사람이나 공격 방식에 대한 성능 근거로 사용하지 않는다.

## 4. 환경 버전

| 구성요소 | 확인 버전 |
|---|---|
| Python | 3.14.3 |
| PyTorch | 2.12.1+cpu |
| Torchvision | 0.27.1+cpu |
| OpenCV | 4.13.0 |
| Matplotlib | 3.11.0 |
| tqdm | 4.68.3 |
| ONNX | 1.22.0 |
| ONNX Script | 0.7.0 |
| ONNX Runtime | 1.27.0 |

CUDA는 사용할 수 없으며 현재 학습 코드도 CPU를 명시적으로 사용한다.

## 5. 마지막 검증 명령과 결과

프로젝트 루트의 Windows PowerShell에서 실행했다.

```powershell
.\.venv\Scripts\python.exe -m py_compile model.py dataset.py train.py collect_data.py crop_dataset.py convert_to_onnx.py inference_onnx.py verify_setup.py
.\.venv\Scripts\python.exe verify_setup.py
```

결과:

- 모든 파일 문법 컴파일 통과
- 주요 라이브러리 import 및 버전 확인 통과
- CUDA 사용 불가 확인

추가 Python 검증으로 다음을 확인했다.

- `best_model.pth`의 state tensor 244개 로드 성공
- classifier weight shape `(2, 1024)`
- ONNX CPUExecutionProvider 로드 성공
- ONNX checker 통과
- PyTorch와 ONNX 동일 입력 출력 최대 절대 오차 약 `6.63e-7`
- validation Accuracy `45/45 = 100%`
- validation 예측 confidence 범위 약 62.25~79.76%, 평균 약 71.34%

웹캠 실시간 90~99% confidence는 과거 관찰 내용이지만 실행 로그가 없어 현재 검증 결과로 기록하지 않는다.

## 6. 알려진 문제와 재현 정보

### Print Attack OOD 오분류

- **현상**: 종이에 인쇄한 얼굴 사진을 REAL로 오인한 사례가 사용자 테스트에서 확인됐다.
- **의미**: 현재 모델은 일반적인 생체 여부보다 스마트폰 화면의 반사·픽셀·테두리 같은 특징에 의존했을 가능성이 높다.
- **재현 상태**: 당시 영상이나 자동 평가 로그가 없어 정량 재현은 아직 불가능하다.
- **처리 원칙**: Print 데이터를 바로 기존 train/val에 섞지 말고 ROI와 독립 split 규격을 확정한 후 수집한다.

### 데이터 누수 위험

- train과 validation이 동일 인물·장소의 연속 촬영으로 보인다.
- validation 100%는 실제 일반화 성능보다 높게 측정됐을 가능성이 크다.
- metadata가 없어 subject/session 중복을 자동 판정할 수 없다.

### 전처리 위험

- `collect_data.py`는 전체 프레임을 저장한다.
- `crop_dataset.py`는 중앙 크롭 결과로 기존 파일을 덮어쓴다.
- `inference_onnx.py`도 얼굴 검출 없이 화면 중앙을 사용한다.
- 현재 방식으로 새 데이터를 추가하면 향후 얼굴 검출 ROI와 학습 데이터 분포가 달라질 수 있다.

### 재현성과 배포 위험

- 의존성 lock 파일, random seed, 독립 test, 자동 평가 스크립트가 없다.
- 데이터, 체크포인트, ONNX 산출물이 `.gitignore` 대상이므로 저장소만으로는 현재 결과를 재현할 수 없다.
- TFLite, INT8, NPU, Android 관련 결과는 아직 없다.

## 7. 사용자에게 필요한 작업

현재 즉시 필요한 물리 작업은 없다. AI가 다음 항목을 먼저 준비해야 한다.

1. 원본을 보존하는 데이터 구조
2. 얼굴 검출과 ROI 규격
3. subject/session 분리 규칙
4. 공격 종류별 metadata 형식
5. 필요한 인원·세션·공격 유형·촬영 수량이 포함된 체크리스트

위 항목이 검증된 뒤 사용자에게 신규 촬영을 요청한다. 요청 시에는 Python 명령을 요구하지 말고 다음 내용을 쉬운 체크리스트로 제공한다.

- 누구를 촬영해야 하는지
- 어떤 얼굴 상태와 공격 도구가 필요한지
- 각 조건을 몇 번, 어떤 거리와 조명에서 촬영하는지
- 파일 저장이 정상인지 사용자가 어떻게 확인하는지
- 촬영 중 개인정보와 원본을 어떻게 보호하는지

## 8. 다음 단계 진입 조건

단계 4 데이터 수집으로 넘어가기 전에 다음 조건을 모두 충족해야 한다.

- [ ] 현재 기준선 산출물이 별도 보존되어 있다.
- [ ] 원본 파일을 덮어쓰지 않는 구조가 정해졌다.
- [ ] 얼굴 검출 및 ROI 정책이 코드와 테스트로 검증됐다.
- [ ] train/val/test의 subject/session 중복 검사 규칙이 정해졌다.
- [ ] `metadata.csv` 형식과 필수 값이 정해졌다.
- [ ] 필요한 데이터 조건을 사용자용 촬영 체크리스트로 설명할 수 있다.

하나라도 충족하지 못하면 신규 데이터 촬영이나 재학습을 시작하지 않는다.

## 9. 변경 이력

| 날짜 | 변경 내용 | 근거 |
|---|---|---|
| 2026-06-22 | 초기 코드·환경·데이터·체크포인트·ONNX 상태 검증 | 저장소 정적 검사 및 로컬 실행 |
| 2026-06-22 | 고정 개발 가이드와 변경 상태 문서 분리 | AI 자율 개발 문서 개편 |
