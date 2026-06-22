# 📋 안면 인식 출입통제기용 안티스푸핑(위변조 방지) 프로젝트 기술 가이드

본 문서는 NPU 탑재 안드로이드 출입통제 디바이스(**NXP i.MX 8M Plus**) 배포를 목적으로 하는 안티스푸핑(Liveness Detection) 모델 학습 프로젝트의 구조와 개발 가이드를 다룹니다.  
인간 개발자뿐 아니라 **다른 AI 에이전트(LLM)**가 본 프로젝트를 이어서 작업하거나 유지보수할 수 있도록 구체적인 기술 스펙과 데이터 포맷을 명시합니다.

---

## 1. 프로젝트 개요 (Project Objective)
* **목표**: 카메라에 비친 얼굴이 실제 사람(REAL)인지 위조된 얼굴(SPOOF)인지를 실시간으로 판정하는 이진 분류(Binary Classification) 모델을 학습하고 이식하는 것입니다.
* **타겟 정확도**: **98% 이상**
* **지원해야 하는 위조 시나리오 (SPOOF)**:
  - 스마트폰 / 태블릿 디스플레이 화면 속의 얼굴 (Replay Attack)
  - 종이 인쇄 사진 (Print Attack)
  - 눈 부위에 구멍을 뚫어 실제 사람이 뒤에서 깜빡이는 위조 종이 가면 (Paper Mask Attack)
* **지원해야 하는 진짜 시나리오 (REAL)**:
  - 맨얼굴 상태의 실제 사람
  - **덴탈 마스크 등 다양한 마스크를 착용한 상태의 실제 사람** (위조로 오판하지 않고 통과시켜야 함)

---

## 2. 하드웨어 스펙 및 최적화 대상 (Target Hardware Specification)
* **메인 프로세서(MCU/SoC)**: **NXP i.MX 8M Plus**
* **NPU 가속기**: Vivante NPU 내장 (2.3 TOPS 연산 성능)
* **지원 개발 환경**: NXP eIQ ML Software Development Environment
* **최종 탑재 포맷**: **TensorFlow Lite (TFLite) - 8비트 정수 양자화(INT8 Quantized)**
  - *이유*: i.MX 8M Plus의 Vivante NPU 하드웨어 가속은 TFLite INT8 포맷에서 최고의 성능을 냅니다.
* **센서 사양**: 디바이스에 RGB(컬러) 카메라와 IR(적외선) 카메라가 동시에 탑재되어 있으므로, 추후 RGB+IR 듀얼 카메라 연동 모델로 고도화 예정입니다. (현재는 로컬 PC 웹캠 기준 RGB 단일 모델 개발 진행 중)

---

## 3. 개발 환경 및 기술 스택 (Tech Stack)
* **개발 언어**: Python 3.14.3
* **딥러닝 프레임워크**: PyTorch (CPU-only 버전, `2.12.1+cpu` 설치됨)
* **카메라 & 이미지 처리**: OpenCV (`opencv-python` 4.13.0)
  - Windows 환경의 카메라 로딩 지연 방지를 위해 DirectShow(`cv2.CAP_DSHOW`) 백엔드를 명시적으로 사용합니다.
* **포맷 변환 도구**: ONNX (`onnx` 1.22.0), `onnxscript` 0.7.0
* **실시간 PC 검증 엔진**: ONNX Runtime (`onnxruntime` 1.27.0)

---

## 4. 폴더 구조 및 파일 가이드 (Directory Structure)

```
c:/Users/Unionbiometrics/Desktop/company/10.ai_model/
├── .venv/                      # 파이썬 가상환경 폴더
├── dataset/                    # 학습 및 검증용 이미지 데이터셋
│   ├── train/                  # 학습용 데이터 (Train)
│   │   ├── real/               # 진짜 사람 얼굴 이미지
│   │   └── spoof/              # 위조 얼굴 이미지 (화면, 종이 등)
│   └── val/                    # 검증용 데이터 (Validation - 모의고사)
│       ├── real/
│       └── spoof/
├── docs/
│   └── project_guide.md        # [본 문서] 전체 기술 명세서
├── collect_data.py             # 웹캠 기반 얼굴 수집 스크립트 (가이드 박스 표시)
├── crop_dataset.py             # 수집된 사진에서 얼굴만 (250x250) 잘라내는 전처리 스크립트
├── dataset.py                  # PyTorch DataLoader 정의 (224x224 리사이징 및 정규화)
├── model.py                    # MobileNetV3-Small 신경망 설계 및 출력 레이어 개조 (2진 분류)
├── train.py                    # 학습 진행, 최고성능 모델 저장 (best_model.pth), 학습곡선 그리기
├── convert_to_onnx.py          # PyTorch (.pth) 가중치를 범용 ONNX (.onnx)로 변환
├── inference_onnx.py           # ONNX 모델 기반의 실시간 웹캠 안티스푸핑 판정 프로그램
└── verify_setup.py             # 파이썬 라이브러리 정상 작동 여부 검증용 유틸
```

---

## 5. 데이터 처리 및 전처리 규격 (Data Preprocessing Specification)

AI 에이전트들이 모델 수정이나 추가 학습을 진행할 때 반드시 준수해야 하는 전처리 규격입니다.

1. **얼굴 영역 크롭 (Crop)**:
   - 입력 해상도(예: `640x480`) 이미지의 **정중앙을 기준**으로 가로/세로 **`250x250`** 크기로 영역을 잘라내어 얼굴 정보만 가져옵니다.
   - 배경의 픽셀 간섭을 줄이고 얼굴 패턴에만 집중시키기 위함입니다.
2. **이미지 리사이징**:
   - `250x250` 크롭 이미지를 최종적으로 **`224x224`** 크기로 리사이징합니다.
3. **입력 텐서 변환**:
   - [H, W, C] 구조의 이미지를 PyTorch가 기대하는 [C, H, W] 텐서로 변환하며, 픽셀 값 범위는 `0~1.0` 사이로 정규화합니다.
4. **ImageNet 정규화 공식**:
   - 변환된 텐서에 아래 평균과 표준편차를 적용하여 정규화합니다:
     - `Mean = [0.485, 0.456, 0.406]`
     - `Std  = [0.229, 0.224, 0.225]`

---

## 6. 인공지능 모델 명세 (Model Architecture)
* **기반 네트워크**: `torchvision.models.mobilenet_v3_small` (ImageNet 사전 학습 가중치 사용)
* **분류 레이어 수정 코드 (`model.py`)**:
  ```python
  import torchvision.models as models
  import torch.nn as nn
  
  # 베이스 모델 로드
  model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
  
  # 기존 1000개 출력을 2개 출력(진짜/가짜)으로 변경
  in_features = model.classifier[3].in_features
  model.classifier[3] = nn.Linear(in_features, 2)
  ```
* **출력 포맷**:
  - Class 0: **`real`** (진짜 얼굴)
  - Class 1: **`spoof`** (위조 얼굴)

---

## 7. 현재 진행 상태 및 에이전트 핵심 행동 가이드 (Next Steps for Agents)

새로운 AI 에이전트가 투입될 경우, 아래 우선순위에 따라 작업을 진행해 주시기 바랍니다.

### 작업 1순위: 데이터 다양성 보강 및 재학습
- 현재 수집된 데이터는 스마트폰 액정 화면에 한정되어 있어, **인쇄된 종이 사진 및 종이 마스크**를 위조로 탐지하지 못하거나, **마스크 착용자**를 진짜 사람으로 통과시키지 못하는 한계가 있습니다.
- `collect_data.py`를 활용해 신규 데이터를 보강한 뒤 `crop_dataset.py`로 얼굴 부위만 자르고, `train.py`로 재학습하여 `best_model.pth`를 갱신해야 합니다.

### 작업 2순위: TFLite 변환 스크립트 구축
- `model.onnx` 파일을 안드로이드 및 i.MX 8M Plus NPU 전용인 `.tflite`로 변환하는 파이썬 변환 모듈(`convert_to_tflite.py`)을 만들어야 합니다.
- NPU 최적화를 위해 **8비트 양자화(INT8 Quantization)** 옵션을 설계해야 합니다.

### 작업 3순위: 얼굴 검출 엔진(Face Detector) 연동
- 사용자가 카메라 영역 내 어디에 서 있더라도 자동으로 얼굴 부위만 실시간 크롭해 주는 **MediaPipe 얼굴 탐지기**를 `inference_onnx.py`에 적용해 상용 제품급 유연성을 확보해야 합니다.
