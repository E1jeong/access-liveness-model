# 안면인식 출입통제기 안티스푸핑 프로젝트 개발 가이드

이 문서는 이 프로젝트를 이어서 작업하는 AI 에이전트가 따라야 하는 **고정 개발 기준**이다. 자주 바뀌는 데이터 수량, 모델 결과, 현재 작업 단계는 [project_status.md](project_status.md)에 기록한다.

사용자는 Python이나 모델 학습 구현을 직접 판단하지 않는다. AI는 실제 코드와 실행 결과를 확인해 기술적인 결정을 내리고, 물리적인 데이터 촬영·장비 접근·제품 요구사항 변경처럼 사용자의 결정이나 행동이 필요한 경우에만 쉬운 말로 요청해야 한다.

## 1. 상태 표기와 정보 우선순위

문서에서는 다음 표기를 사용한다.

- `[목표]`: 최종적으로 달성해야 하지만 아직 검증되지 않은 내용
- `[구현 완료]`: 코드나 산출물이 존재하는 내용
- `[검증 완료]`: 명령 실행이나 측정으로 정상 동작을 확인한 내용
- `[미구현]`: 아직 코드나 산출물이 없는 내용
- `[사용자 확인 필요]`: 물리 작업, 장비 정보 또는 제품 정책 결정이 필요한 내용

정보가 충돌하면 다음 순서를 따른다.

1. 현재 저장소의 코드와 데이터
2. 같은 환경에서 다시 실행한 검증 결과
3. `project_status.md`의 최근 검증 기록
4. 이 가이드의 일반 원칙
5. 과거 대화나 기존 성능 설명

문서에 적힌 숫자를 그대로 믿지 말고 재현 가능한 명령으로 확인한다. 확인할 수 없는 내용은 추정하지 말고 `[미검증]`이라고 보고한다.

## 2. 프로젝트 목표와 현재 범위

### 2.1 최종 목표

- `[목표]` NXP i.MX 8M Plus 기반 Android 출입통제기에서 실제 얼굴(REAL)과 위조 얼굴(SPOOF)을 실시간 판정한다.
- `[목표]` Replay Attack, Print Attack, Paper Mask Attack을 탐지한다.
- `[목표]` 맨얼굴과 덴탈 마스크 착용자를 정상 사용자로 처리한다.
- `[목표]` 최종 후보 모델은 독립 test set에서 Accuracy 98% 이상을 만족하고 APCER, BPCER, ACER도 함께 평가한다.
- `[목표]` 최종 배포 후보 포맷은 TFLite INT8이며 실제 NPU에서 정확도, 지연시간, FPS, 메모리 사용량을 검증한다.
- `[목표]` RGB 모델을 먼저 제품 기준선으로 만든 뒤 RGB+IR 모델을 별도 단계로 개발한다.

98%는 목표이지 현재 성능 보장이 아니다. 독립 test set은 학습에 사용하지 않은 사람, 촬영 세션, 공격 매체로 구성해야 한다.

### 2.2 현재 구현 범위

> 참고: 과거 PC 웹캠·RGB 단일·REAL/SPOOF 2클래스·ONNX·중앙 250×250 크롭 단계는 폐기되었다. 아래는 현재(디바이스 수집 + RGB+IR 5클래스 + TFLite) 기준이다.

- `[구현 완료]` Android 디바이스에서 취득한 RGB+IR 크롭 이미지(`cropRGB.bmp`/`cropIR.bmp`)를 파일로 전달받아 학습
- `[구현 완료]` Dual-Input MobileNetV3-Small 기반 5클래스 분류(live/print/picture/mask/display) — PyTorch 파이프라인
- `[구현 완료]` Dual-Input MobileNetV2 기반 5클래스 분류 — Keras 파이프라인 (`keras_pipeline/`), INT8 양자화 목적
- `[구현 완료]` subject 단위 K-fold 분할 및 누수 검사 (`utils.py`에 통합, 양쪽 파이프라인 공유)
- `[구현 완료]` Accuracy/APCER/BPCER/ACER 평가 및 ACER 기준 best 저장 (양쪽 파이프라인)
- `[구현 완료]` 학습 데이터 증강: RGB+IR 공동 flip/rotation, RGB ColorJitter (양쪽 파이프라인)
- `[구현 완료]` `litert_torch` 기반 TFLite(NHWC) 변환 및 Android 통합(추론 경로 존재)
- `[구현 완료]` Keras/MobileNetV2 full INT8 및 NPU-friendly INT8 export 경로 (`--npu-int8`)
- `[구현 완료]` `run_keras_*.sh` 실행 스크립트 — TF GPU용 `LD_LIBRARY_PATH` 자동 설정 포함
- `[검증 완료]` float 모델 학습(서브노트북 GPU, subject 5명): liveness ACER≈0, 5클래스 val_acc≈0.89
- `[검증 완료]` TF GPU 동작 (`LD_LIBRARY_PATH` 설정 시 GTX 1660 Ti 인식)
- `[검증 완료]` Keras/MobileNetV2 fold 4 full INT8 validation: 표준 INT8 `ACER=0.0060`, NPU-friendly INT8 `ACER=0.0160`
- `[미구현]` 독립 test set(현재는 K-fold 교차검증만)
- `[시도 후 보류]` TFLite INT8 양자화 (MobileNetV3 기반) — PTQ는 활성 양자화에서 붕괴, QAT는 학습은 되나 직렬화(litert/eIQ) 실패. 전체 시도 기록은 `project_status.md` §3 참조.
- `[검증 실패]` NPU 실기기 실행 — Android NNAPI delegate를 시도하지만 현재 Keras NPU-friendly INT8 모델도 `ANEURALNETWORKS_BAD_DATA ... while adding operation`으로 실패하고 CPU/XNNPACK으로 fallback된다. `Backend CPU`는 NPU 가속이 아니다.
- `[미구현]` 의존성 lock 파일(재현성)

> 작업 머신은 2대다: 코드/문서/Android는 **회사 머신(WSL, torch CPU)**, 학습·양자화·데이터는 **서브노트북(GPU, SSH `mysub`)**. 상세는 `project_status.md` §0.

서브노트북에서도 프레임워크별 가상환경을 분리한다. PyTorch 학습은 기존 `.venv`를 유지하고, TensorFlow/Keras 실험은 별도 `.venv-tf`를 만들어 사용한다. PyTorch에서 GPU가 잡힌다고 TensorFlow에서도 자동으로 GPU가 잡히는 것은 아니므로, Keras 학습 전에는 `tf.config.list_physical_devices('GPU')` 결과를 확인한다.

현재 성능은 실제 디바이스 수집 데이터가 정비된 뒤에야 측정 가능하며, 그 전까지를 제품 성능이나 일반화 성능으로 표현하지 않는다.

## 3. 필수 평가 지표

- **Accuracy**: 전체 표본 중 올바르게 분류한 비율이다.
- **APCER**(Attack Presentation Classification Error Rate): 공격 표본을 REAL로 통과시킨 비율이다. 보안 실패를 나타낸다.
- **BPCER**(Bona Fide Presentation Classification Error Rate): 정상 사용자를 SPOOF로 거부한 비율이다. 사용성 실패를 나타낸다.
- **ACER**(Average Classification Error Rate): `(APCER + BPCER) / 2`이다.

다음 규칙을 적용한다.

- `[목표]` 최종 후보는 독립 test set에서 Accuracy 98% 이상을 만족해야 한다.
- `[목표]` 개발 기본 기준은 APCER 2% 이하, BPCER 2% 이하, ACER 2% 이하로 둔다.
- `[사용자 확인 필요]` 실제 출입통제 제품의 최종 허용 기준은 보안 정책과 현장 요구에 맞춰 확정해야 한다.
- 전체 평균만 보고하지 않고 공격 종류, 마스크 여부, 조명, 카메라, 사람별 결과를 함께 보고한다.
- test set 결과를 보고 모델을 다시 조정했다면 그 test set은 더 이상 독립 test set이 아니다. 새 test set을 구성해야 한다.

## 4. AI 에이전트 실행 규칙

### 4.1 작업 시작 절차

AI는 매 작업을 다음 순서로 시작한다.

1. `docs/project_guide.md`와 `docs/project_status.md`를 읽는다.
2. `git status`, 관련 코드, 데이터 폴더, 산출물 존재 여부를 확인한다.
3. 상태 문서의 최근 기록이 실제 저장소와 일치하는지 대조한다.
4. 이번 작업의 성공 기준과 검증 명령을 먼저 정한다.
5. 현재 단계의 통과 조건을 만족하지 못했다면 다음 단계 작업을 시작하지 않는다.

작업 시작 시 먼저 현재 머신을 확인한다.

```bash
nvidia-smi 2>/dev/null | grep -q "GTX 1660 Ti" && echo "서브노트북" || echo "회사 PC"
```

- **서브노트북**: GPU 학습·변환 가능. `run_keras_*.sh` 스크립트 사용.
- **회사 PC**: CPU 전용. 코드·문서 편집 및 git push/pull만 수행. 학습은 실행하지 않는다.

머신별 기본 확인 명령:

```bash
git status --short
.venv/bin/python -m py_compile model.py dataset.py train.py classes.py utils.py convert_to_tflite.py verify_setup.py
.venv/bin/python verify_setup.py
```

Keras 파이프라인 확인은 반드시 셸 스크립트를 사용한다(TF GPU `LD_LIBRARY_PATH` 설정 포함, 서브노트북 전용):

```bash
./run_keras_model.sh
```

Git 소유권 경고가 발생하면 저장소 설정을 영구 변경하지 말고 해당 명령에만 `-c safe.directory=<절대 경로>`를 적용한다.

### 4.2 자율 실행 범위

AI가 별도 승인 없이 수행할 수 있는 작업:

- 코드·문서·데이터 구조의 읽기와 분석
- 성공 기준이 명확한 코드 수정
- 정적 검사, 단위 테스트, 모델 평가
- 기존 원본과 모델을 보존하는 신규 학습·변환
- 실패 원인이 확인된 범위의 최소 수정

사용자에게 먼저 확인할 작업:

- 얼굴·공격 이미지의 신규 촬영
- 실제 NXP 장비 또는 Android BSP/eIQ 정보 제공
- 보안 기준, 허용 지연시간, 대상 환경 등 제품 요구 변경
- 유료 서비스, 외부 데이터 구매, 개인정보 외부 전송
- 기존 원본 데이터나 검증된 모델의 삭제·덮어쓰기

### 4.3 데이터와 모델 보호 규칙

- 원본 촬영 이미지는 수정하거나 덮어쓰지 않는다.
- 크롭·리사이즈 이미지는 별도 경로에 생성한다.
- 새 학습 결과는 실행 ID가 포함된 별도 폴더에 저장하고 평가 후에만 현재 후보로 지정한다.
- 기존 체크포인트(`model/best_model_fold*.pth`)와 배포 자산(`anti_spoofing.tflite`)을 바로 덮어쓰지 않는다.
- 학습 전 클래스 수, 사람 수, 세션 수, 공격 종류, split 중복을 검사한다.
- 같은 사람이나 같은 연속 촬영 세션이 train과 test에 동시에 들어가지 않게 한다.
- 정확히 같은 파일뿐 아니라 같은 영상에서 추출한 인접 프레임도 다른 split으로 나누지 않는다.

### 4.4 변경 원칙

- 가장 단순한 기준 모델부터 검증한다.
- 데이터 문제인지 모델 문제인지 구분하기 전에 구조를 변경하지 않는다.
- 모델 구조나 하이퍼파라미터를 변경할 때는 변경 이유와 비교 기준을 기록한다.
- 한 번의 실험에서는 원인 분석이 가능하도록 변경 변수를 최소화한다.
- 작업 후 실행 명령, 결과, 실패 내용, 생성 산출물을 `project_status.md`에 기록한다.

### 4.5 사용자 보고 형식

전문 용어만 나열하지 말고 다음 순서로 보고한다.

1. 무엇이 확인됐는지
2. 그 결과가 실제 제품 관점에서 무엇을 의미하는지
3. 아직 믿을 수 없는 부분과 위험
4. AI가 다음으로 할 작업
5. 사용자에게 필요한 행동이 있다면 촬영 대상·수량·방법을 포함한 체크리스트

`완벽`, `상용 등급`, `매우 안정적`, `완성` 같은 표현은 독립 test와 실기기 검증 근거 없이 사용하지 않는다.

## 5. 데이터 표준

### 5.1 데이터 구조

학습 데이터는 Android 수집기가 만든 구조를 그대로 사용한다(`dataset.py`가 이 구조를 읽는다).

```text
dataset/
└── raw/
    └── <class>/                         # live, print, picture, mask, display
        └── <class>_<subjectId>/         # 폴더 = 하나의 subject (예: live_0)
            └── <frameId>/               # 0, 1, 2, ...
                ├── cropRGB.bmp          # 학습 입력(RGB)
                ├── cropIR.bmp           # 학습 입력(IR)
                ├── RGB.bmp              # 원본 보존
                └── IR.bmp               # 원본 보존
```

- 클래스 정의의 단일 출처는 `classes.py`다.
- subject 폴더(`<class>_<id>`) 단위로 K-fold가 나뉘며, 한 subject의 frame들이 train/val로 쪼개지지 않는다.
- 동일 인물의 live와 그 인물을 본뜬 mask는 서로 다른 물리 객체로 수집한다(mask 폴더에 live 인물 자체가 들어가지 않음).
- 얼굴 개인정보를 외부로 전송하지 않는다.
- 디바이스 수집분은 사용자가 직접 전달·관리한다. 학습 코드는 원본(`RGB.bmp`/`IR.bmp`)을 수정하지 않는다.

### 5.2 전처리 일치 규칙

- 얼굴 검출·크롭은 **디바이스 수집 시점**에 수행되어 `cropRGB.bmp`/`cropIR.bmp`로 저장된다. 학습은 이 크롭을 224×224로 리사이즈해 사용하므로, Android 추론 크롭(동일 검출기·`cropMarginRatio`)과 일치해야 한다.
- 정규화 명세(단일 출처는 Android `model_spec.json`, 학습은 `dataset.py`가 동일하게 적용):
  - RGB: 224×224, float32, mean `[0.485,0.456,0.406]` / std `[0.229,0.224,0.225]`
  - IR: 224×224, 1채널, float32, mean `[0.5]` / std `[0.5]`
- NPU-friendly Keras INT8 export는 RGB Lambda 전처리를 TFLite 그래프 밖으로 빼기 때문에 RGB도 mean `[0.5]` / std `[0.5]`를 사용한다. Android `model_spec.json`과 TFLite input quantization을 항상 함께 확인한다.
- 학습·TFLite 평가가 같은 입력에 대해 같은 전처리 결과를 내는지 테스트한다.
- INT8 모델은 별도의 양자화 입력 규격을 기록한다.

## 6. 단계별 개발 게이트

모든 단계는 입력, AI 수행 작업, 사용자 작업, 산출물, 검증, 통과 조건, 실패 처리를 갖는다. 통과 조건을 충족하지 못하면 다음 단계로 넘어가지 않는다.

### 단계 1. 기준선 보존

- **입력**: 현재 코드, 디바이스 수집 데이터(`dataset/raw`), `model/best_model_fold*.pth`, 가상환경
- **AI 수행 작업**: 환경 버전, 파일 수, 클래스 매핑, 모델 입출력, TFLite 자산 입출력 형상을 기록하고 재현 가능한 의존성·seed·학습 설정의 누락을 식별한다.
- **사용자 작업**: 없음
- **산출물**: 기준선 기록, 이후 구현할 의존성 파일과 실험 기록 형식
- **검증**: Python 컴파일, `verify_setup.py`, 체크포인트 로드, TFLite 자산 입출력 형상 확인, 학습-추론 전처리 일치 비교
- **통과 조건**: 기존 산출물이 손상되지 않았고 현재 결과를 다시 측정할 수 있다.
- **실패 시 처리**: 새 학습을 중단하고 누락 파일 또는 환경 차이를 상태 문서에 기록한다.

### 단계 2. 평가 기준 확정

- **입력**: 제품 목표, 데이터 표준, 현재 데이터 한계
- **AI 수행 작업**: subject/session 단위 분할 검사와 Accuracy/APCER/BPCER/ACER 평가 절차를 구현하고 공격별 결과 형식을 정의한다.
- **사용자 작업**: 최종 제품의 허용 오탐·미탐 기준이 따로 있다면 제공한다.
- **산출물**: 독립 test 규칙, 분할 검사 결과, 재사용 가능한 평가 명령
- **검증**: 의도적으로 중복 subject/session을 넣었을 때 검사가 실패하는지 확인하고 알려진 예측으로 지표 계산을 테스트한다.
- **통과 조건**: 데이터 누수를 자동 검출하며 동일 체크포인트 평가가 같은 결과를 낸다.
- **실패 시 처리**: 모델 학습을 진행하지 않고 분할·지표 오류부터 수정한다.

### 단계 3. 얼굴 검출과 ROI 확정

- **입력**: 원본 프레임, 얼굴 검출 후보, 현재 224×224 모델 입력
- **AI 수행 작업**: 얼굴 검출, 얼굴 주변 여백, 경계 처리, 검출 실패, 복수 얼굴 정책을 구현하고 공통 전처리로 통합한다.
- **사용자 작업**: 실제 출입통제 카메라의 예상 거리와 한 화면에 허용할 인원 정책을 확인한다.
- **산출물**: 원본을 보존하는 processed 데이터와 공통 전처리 코드
- **검증**: 중앙·가장자리·작은 얼굴·얼굴 없음·복수 얼굴 사례를 테스트하고 학습/추론 크롭이 일치하는지 비교한다.
- **통과 조건**: 정상 사용 범위에서 ROI가 안정적이며 원본 파일이 변경되지 않는다.
- **실패 시 처리**: 데이터 촬영을 요청하지 않고 검출기 또는 ROI 정책을 먼저 수정한다.

### 단계 4. 데이터 수집

- **입력**: 확정된 ROI, metadata 형식, split 규칙
- **AI 수행 작업**: 필요한 인원·공격 종류·촬영 조건·수량을 계산해 쉬운 촬영 체크리스트를 만들고 수집 결과를 검사한다.
- **사용자 작업**: 체크리스트에 따라 여러 사람, 세션, 조명, 마스크 REAL과 Replay/Print/Paper Mask를 촬영한다.
- **산출물**: 보존된 raw 데이터, processed 데이터, `metadata.csv`
- **검증**: 손상 파일, 라벨 누락, 클래스/공격 불균형, subject/session split 중복을 검사한다.
- **통과 조건**: 모든 목표 시나리오가 독립 train/val/test에 존재하고 누수가 없다.
- **실패 시 처리**: 부족한 항목만 사용자에게 구체적으로 재촬영 요청한다.

### 단계 5. RGB 기준 모델 학습

- **입력**: 검증된 RGB 데이터셋과 MobileNetV3-Small 기준 모델
- **AI 수행 작업**: seed와 설정을 저장해 학습하고 validation으로 체크포인트와 임계값을 선택한 뒤 test를 한 번 평가한다.
- **사용자 작업**: 없음
- **산출물**: 실행별 체크포인트, 설정, 학습 곡선, 공격별 평가 보고서
- **검증**: 반복 실행 재현성, confusion matrix, Accuracy/APCER/BPCER/ACER, 오분류 표본 분석
- **통과 조건**: 독립 test에서 목표 지표를 만족하고 특정 배경·사람·공격 도구에만 의존한다는 증거가 없다.
- **실패 시 처리**: 데이터·ROI·임계값 문제를 먼저 분석한다. 증거가 있을 때만 멀티프레임 또는 모델 구조 변경을 제안한다.

### 단계 6. NPU 배포 가능성 확인

- **입력**: RGB 후보 모델, Android/BSP/eIQ 정보
- **AI 수행 작업**: 지원 연산자와 변환 경로를 확인하고 소규모 TFLite 변환 및 NPU 실행 실험을 설계한다.
- **사용자 작업**: 실제 장비 접근과 BSP/eIQ 버전을 제공한다.
- **산출물**: 호환성 결과, 미지원 연산자 목록, 초기 latency/FPS 결과
- **검증**: 실제 i.MX 8M Plus에서 delegate 사용 여부와 추론 결과를 확인한다.
- **통과 조건**: NPU에서 전체 모델이 실행되고 제품 목표에 접근 가능한 성능을 보인다.
- **실패 시 처리**: CPU fallback을 성공으로 처리하지 않는다. 현재 알려진 실패는 `ANEURALNETWORKS_BAD_DATA ... while adding operation`이며, 다음 조사는 남은 `AVERAGE_POOL_2D`, `RESHAPE`, `CONCATENATION`, `FULLY_CONNECTED`, 또는 quantized conv/depthwise 제약을 하나씩 분리하는 방향으로 진행한다.

### 단계 7. TFLite INT8 최종 변환

- **입력**: 제품 후보 모델, 대표 calibration dataset
- **AI 수행 작업**: full INT8 양자화와 변환 검증을 수행한다.
- **사용자 작업**: 없음. calibration 데이터에 개인정보 외부 전송이 필요하면 먼저 승인한다.
- **산출물**: `.tflite`, 양자화 설정, 변환 로그, 정확도 비교 보고서
- **검증**: PyTorch/TFLite 예측 비교, 양자화 전후 지표, NPU latency/FPS/메모리 비교
- **통과 조건**: 정확도 저하가 허용 범위이고 NPU 실행 및 성능 목표를 만족한다.
- **실패 시 처리**: calibration 대표성, 미지원 연산, 전처리 불일치를 순서대로 조사한다.

### 단계 8. Android 통합

- **입력**: 검증된 TFLite INT8 모델과 ROI 규격
- **AI 수행 작업**: 카메라 입력, 얼굴 검출, NPU 추론, 시간축 결과 안정화, 오류 처리를 애플리케이션에 통합한다.
- **사용자 작업**: 출입 허용 정책, UI 요구, 실제 장비 테스트 환경을 제공한다.
- **산출물**: Android 통합 코드와 실기기 시험 보고서
- **검증**: 정상·공격·얼굴 없음·복수 얼굴·카메라 오류·장시간 실행을 시험한다.
- **통과 조건**: 정확도뿐 아니라 지연시간, 안정성, 메모리, 발열 기준을 만족한다.
- **실패 시 처리**: 모델 문제와 애플리케이션 파이프라인 문제를 분리 측정한다.

### 단계 9. RGB+IR 확장

- **입력**: 검증된 RGB 제품 기준선, 동기화된 RGB/IR 센서 정보
- **AI 수행 작업**: 카메라 동기화와 융합 방식을 설계하고 RGB 기준선과 동일 test protocol로 비교한다.
- **사용자 작업**: IR 센서 사양, 프레임 동기화 방식, 실기기 접근을 제공한다.
- **산출물**: RGB+IR 데이터셋·모델·비교 보고서
- **검증**: RGB 단독 대비 공격별 개선, NPU 호환성, 지연시간을 측정한다.
- **통과 조건**: 복잡도 증가보다 명확한 보안 성능 개선이 크다.
- **실패 시 처리**: RGB 기준선을 유지하고 IR 확장을 제품 필수 기능으로 간주하지 않는다.

## 7. 현재 에이전트가 따라야 할 다음 순서

현재 단계와 검증 수치는 반드시 `project_status.md`에서 확인한다. 현 상태에서 새 AI는 다음 순서로 작업한다.

1. `project_status.md`의 최신 fold 4 INT8 / NPU-friendly INT8 평가 수치와 Android NNAPI 실패 로그를 확인한다.
2. 코드와 모델 artifact가 같은 머신에 있는지 확인한다. 모델 파일은 gitignored이므로 `rsync`/`scp`로 별도 이동한다.
3. 표준 INT8와 NPU-friendly INT8의 전처리 계약 차이를 확인한다. NPU-friendly export는 RGB/IR 모두 mean `[0.5]`, std `[0.5]`다.
4. Android에서 `Backend CPU`가 뜨면 NPU 가속 실패로 기록한다. `Backend NNAPI`가 뜨기 전까지 inference timing을 NPU 성능으로 보고하지 않는다.
5. 다음 NPU 디버깅은 남은 TFLite op를 줄이는 작은 실험으로 진행한다. 후보는 `AVERAGE_POOL_2D`, `RESHAPE`, `CONCATENATION`, `FULLY_CONNECTED`, quantized conv/depthwise 제약이다.

사용자에게 바로 “데이터를 더 촬영해 달라”고 요청하거나 새 학습을 시작하지 않는다. 현재 문제는 우선 데이터/학습 문제가 아니라 NNAPI/NPU 호환성 문제다.
