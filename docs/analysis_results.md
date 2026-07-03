# 안티스푸핑 전용 사전 학습 가중치(Pretrained Weights) 분석 보고서

얼굴 안티스푸핑(Face Anti-Spoofing, FAS) 작업은 일반 객체 분류(ImageNet)와 비교했을 때 미세한 텍스처(종이 질감, 모아레 패턴, 반사광, 마스크 경계 등)와 멀티모달(RGB, IR, Depth) 정보를 다루는 특성이 다릅니다. 따라서 안티스푸핑 전용 데이터셋으로 사전 학습된 가중치를 사용하면 모델의 수렴 속도와 예측 신뢰도를 향상시키는 데 큰 도움이 될 수 있습니다. 

본 보고서에서는 대표적인 안티스푸핑 데이터셋과 공개된 사전 학습 가중치의 종류, 그리고 Keras 파이프라인에 이를 이식하여 적용할 수 있는 방안을 분석합니다.

---

## 1. 대표적인 안티스푸핑 데이터셋 및 사전 학습 가중치

학계와 오픈소스 커뮤니티에서 안티스푸핑 챌린지 및 연구용으로 널리 사용되는 데이터셋과 사전 학습 모델 정보는 다음과 같습니다.

### ① CASIA-SURF (Multi-modal)
* **특징**: RGB, Depth, IR 3가지 모달리티를 동시에 제공하는 최초의 대규모 멀티모달 얼굴 안티스푸핑 데이터셋입니다.
* **데이터 규모**: 약 1,000명의 피험자로부터 얻은 21,000개의 비디오 클립.
* **사전 학습 모델 특징**: 
  - 주로 ResNet, MobileNetV2 등을 백본으로 하여 멀티스트림(Multi-stream) 혹은 융합(Fusion) 네트워크 형태로 학습된 체크포인트가 논문 저자들의 GitHub를 통해 제공됩니다.
  - 우리 프로젝트와 같이 **RGB와 IR 채널을 동시에 사용하는 모델**에 매우 적합한 사전 학습 소스입니다.

### ② CASIA-SURF CeFA (Large-scale Multi-modal)
* **특징**: CVPR 2020 챌린지용으로 공개된 대규모 멀티모달 데이터셋으로, 크로스 에스니시티(Cross-ethnicity, 다양한 인종) 및 크로스 모달리티(Cross-modality) 안티스푸핑 성능 검증에 특화되어 있습니다.
* **데이터 규모**: 3가지 인종(황인, 흑인, 백인)의 1,600명 피험자, RGB/Depth/IR 이미지 990,000장 이상.
* **사전 학습 모델 특징**:
  - 인종이나 조명 변화에 강인한 특징을 추출하도록 백본이 훈련되어 있어 일반화 성능이 매우 뛰어납니다.
  - 다양한 챌린지 참가팀(예: AlexanderParkin 등)이 PyTorch 기반의 MobileNetV2, ResNet 기반 베이스라인 모델 가중치를 공개하고 있습니다.

### ③ CelebA-Spoof (Large-scale Single-modal RGB)
* **특징**: 2020년에 발표된 세계 최대 규모 of RGB 싱글모달 안티스푸핑 데이터셋입니다.
* **데이터 규모**: 10,078명의 피험자, 625,537장의 이미지. 바운딩 박스, 랜드마크뿐만 아니라 스푸핑 타입(인쇄물, 사진, 디스플레이 등) 및 촬영 환경(조명, 실내외) 등 풍부한 속성 어노테이션을 제공합니다.
* **사전 학습 모델 특징**:
  - RGB 싱글모달 기반이지만 데이터양이 압도적으로 많아, RGB 백본의 특징 추출기(Feature Extractor) 가중치 초기화용으로 매우 훌륭한 성능을 냅니다.
  - `kprokofi/light-weight-face-anti-spoofing` 등에서 MobileNetV2/V3 기반 백본 가중치를 주로 이 데이터셋으로 프리트레이닝하여 제공하고 있습니다.

### ④ Minivision Silent-Face-Anti-Spoofing (MiniFASNet)
* **특징**: 실제 상용 서비스(에지 장치)에 최적화된 모바일 안티스푸핑 오픈소스 프로젝트입니다.
* **사전 학습 모델 특징**:
  - 자체 정의한 경량 신경망인 `MiniFASNetV1`/`MiniFASNetV2`(Squeeze-and-Excitation 블록 적용 버전 포함) 모델의 사전 학습 가중치를 제공합니다.
  - 이 가중치들은 안티스푸핑을 타겟으로 고도로 최적화되어 있으나, MobileNetV2와 같은 완전 표준 규격 백본이 아니어서 레이어 커스텀이 필요할 수 있습니다.

---

## 2. Keras 파이프라인에서 이미지넷 외 가중치 활용 방안

Keras 공식 라이브러리(`tf.keras.applications`)는 기본적으로 `weights='imagenet'`과 `None`만 제공합니다. 따라서 안티스푸핑 가중치를 사용하려면 외부(대부분 PyTorch 형식인 `.pth` 파일)에서 다운로드한 가중치 파일을 **Keras 모델 규격에 맞게 변환하여 수동 로딩**해야 합니다.

### 방안 A: PyTorch 가중치를 Keras에 수동 매핑 (추천)
현재 우리 코드의 `tf_model.py` 에는 이미 3채널 ImageNet 가중치를 1채널 IR 백본 가중치로 이식하는 함수(`_transfer_imagenet_weights_to_ir_backbone`)가 구현되어 있습니다. 이 메커니즘을 동일하게 확장할 수 있습니다.

1. **가중치 파일 파싱**: PyTorch 가중치(`.pth`)를 Python에서 `torch.load()`로 로드하여 레이어별 Weight Tensor 이름과 차원 정보를 확인합니다.
2. **커널 차원 전치(Transpose)**: PyTorch의 합성곱(Convolution) 가중치는 `[OutChannels, InChannels, Height, Width]` 레이아웃을 가지는 반면, Keras/TensorFlow는 `[Height, Width, InChannels, OutChannels]` 레이아웃을 사용합니다. 따라서 커널 가중치 축을 정렬해주어야 합니다.
   ```python
   # PyTorch -> Keras 텐서 변환 예시 (Concept)
   keras_weight = torch_weight.permute(2, 3, 1, 0).numpy()
   ```
3. **Keras 레이어 매핑**: `tf.keras.applications.MobileNetV2`를 생성한 뒤, 레이어 이름을 기준으로 대조하여 `layer.set_weights()`로 변환된 가중치 배열을 주입합니다.

### 방안 B: ONNX 변환 및 포팅 도구 활용
PyTorch 모델을 중간 포맷인 ONNX로 내보낸 후 Keras 모델로 변환하는 자동화 도구를 사용할 수도 있습니다.

1. **PyTorch -> ONNX**: `torch.onnx.export`를 사용하여 모델 그래프와 가중치를 통합한 `.onnx` 파일 생성.
2. **ONNX -> Keras**: `pt2keras` 또는 `nobuco` 같은 변환 도구를 이용하여 Keras 모델 인스턴스를 직접 복원하고 이를 `.h5` 또는 `.keras` 가중치 파일로 저장.
3. **파이프라인 로딩**: 저장된 `.h5` 파일을 Keras 파이프라인에서 `model.load_weights()`를 통해 불러와 학습을 시작합니다.

### 방안 C: 전사적 자체 사전학습(Pre-training)
외부 가중치를 로드하고 변환하는 과정이 에러에 취약하다면, Keras 파이프라인 내에서 공개 안티스푸핑 데이터셋(예: CelebA-Spoof 등 오픈 데이터셋)을 다운로드하여 **Keras 백본을 먼저 1차적으로 학습시켜 자체적인 안티스푸핑 사전학습 가중치를 만들고**, 이를 Android 배포용 타겟 데이터셋 파이프라인에 전이학습(Transfer Learning) 형태로 태우는 방법입니다.

---

## 3. 안티스푸핑 가중치 도입 시 핵심 챌린지 요인

* **채널 불일치 (Modality Mismatch)**:
  현재 우리 프로젝트는 RGB(3ch)와 IR(1ch)을 병렬로 취급하는 듀얼 스트림 구조입니다. 만약 가져오려는 사전학습 모델이 RGB-D-IR(5ch)을 하나의 네트워크로 한 번에 처리하도록 설계되어 있다면 가중치를 분해하여 개별 백본에 맞추어 주입하는 전처리(Weight Splitting)가 필요합니다.
* **패딩과 스트라이드 불일치 (Padding & Alignment)**:
  PyTorch 모델의 경우 Conv 레이어의 패딩 처리가 TensorFlow의 `SAME`/`VALID` 방식과 다르게 explicit 정수 패딩으로 처리된 경우가 많습니다. 가중치만 복사해서 주입했을 때 레이어 내부 연산 방식의 차이로 인해 출력 텐서의 수치가 미세하게 어긋나 정확도가 저하될 위험이 있으므로 정확도 비교(Validation Check)가 선행되어야 합니다.
* **라이선스 및 접근 권한**:
  CASIA-SURF 및 CeFA 등의 데이터셋과 그 가중치는 상업적 이용 가능 여부(비상업적 연구 목적으로만 제한되는 경우가 많음)를 꼼꼼히 확인하고 도입해야 합니다.

---

## 4. PyTorch MobileNetV3 INT8 양자화 및 Mac 환경 활용 방안

기존 리눅스/윈도우(WSL) 환경에서 PyTorch 기반 MobileNetV3 양자화(INT8) 시도는 하드스위시(hard-swish) 및 SE(Squeeze-and-Excitation) 블록의 활성화 함수가 붕괴(PTQ 시 상수 클래스 고정)하거나, 변환/직렬화 툴체인(litert_torch 0.9.1)의 호환성 문제로 실패했습니다.

새롭게 확보된 **Mac 디바이스(macOS, Unix 계열 컴파일러 환경 및 Apple Silicon MPS 가속)** 환경을 활용한다면, 이러한 에러를 극복하고 양자화 모델을 확보할 수 있는 3가지 현실적인 루트가 존재합니다.

### 🚀 루트 1: Google의 최신 `ai-edge-torch` + `ai-edge-quantizer` 파이프라인
구글의 최신 PyTorch-to-TFLite 공식 툴체인인 `ai-edge-torch`와 후속 양자화 도구인 `ai-edge-quantizer`를 활용하는 방법입니다.

* **동작 원리**:
  1. PyTorch 기반 MobileNetV3 Float 모델을 `ai-edge-torch.convert()`를 사용해 표준 TFLite 플랫버퍼(`.tflite`) 파일로 내보냅니다.
  2. 내보낸 TFLite 모델을 구글의 최신 `ai_edge_quantizer.Quantizer` 라이브러리에 로드하고, Calibration 데이터셋을 제공하여 TFLite 레벨에서 안정적인 `INT8` 포스트 트레이닝 양자화(PTQ)를 수행합니다.
* **Mac 환경의 강점**:
  - `ai-edge-torch` 라이브러리는 C++ 컴파일러 및 의존 패키지 빌드 환경에 매우 예민합니다. WSL이나 윈도우 환경에 비해 macOS(특히 clang 컴파일러 기반)는 이 공식 툴체인의 설치와 모델 빌드가 훨씬 부드럽고 호환성 에러(특히 직렬화 과정의 flatbuffer 버전 충돌 에러)를 피하기 쉽습니다.
  - 빌드 과정이 단순화되므로 에러 추적이 쉽습니다.

### 🚀 루트 2: `torchao` + ExecuTorch를 통한 QAT 및 배포
PyTorch 생태계의 최신 에지 배포 표준인 **ExecuTorch**와 최적화 도구인 **`torchao`**를 이용하는 QAT(Quantization-Aware Training) 경로입니다.

* **동작 원리**:
  1. PyTorch 2.x 이상에서 `torchao`의 PT2E(PyTorch 2 Export) QAT 흐름을 적용하여 MobileNetV3 모델을 미세조정(Fine-tuning) 학습합니다. 
  2. 학습된 가중치를 ExecuTorch 컴파일러를 통해 `.pte` 형식으로 변환하고 Android용 백엔드(XNNPACK 또는 NPU 연동 컴파일러)에 타겟팅합니다.
* **Mac 환경의 강점**:
  - macOS 환경은 Apple Silicon의 MPS(Metal Performance Shaders) 백엔드를 통해 PyTorch 모델 학습 및 양자화 시뮬레이션을 가속할 수 있습니다. 
  - `torchao`는 최신 PyTorch 릴리즈와 궁합이 잘 맞으며, 모바일 가속 시 발생하는 수치 붕괴를 보정해주는 양자화 커널이 빌트인되어 있어 정확도 저하를 최소화합니다.

### 🚀 루트 3: ONNX Runtime Mobile로의 패러다임 전환 (우회 해결책)
만약 TFLite(`.tflite`)로 모델을 변환하고 직렬화하는 과정(Serialization)에서 SE 블록이나 특정 연산자(ReduceMean 등) 컴파일 에러가 지속된다면, 안드로이드 배포 포맷을 **TFLite에서 ONNX Runtime Mobile로 우회**하는 것이 가장 강력하고 안정적인 대안입니다.

* **동작 원리**:
  1. PyTorch에서 가공된 MobileNetV3 모델을 표준 ONNX 포맷(opset=18 이상)으로 내보냅니다.
  2. ONNX Runtime의 공식 양자화 툴(`onnxruntime.quantization`)을 사용하여 Calibration 데이터셋 기반의 정교한 Per-Channel INT8 양자화를 수행합니다.
  3. Android 앱 측에 `org.onnxruntime:onnxruntime-android` AAR 의존성을 추가하고 NNAPI(NPU) 디바이스 백엔드를 활성화하여 실행합니다.
* **Mac 환경의 강점**:
  - macOS 최신 PyTorch 버전에서는 ONNX 변환기가 가장 에러 없이 매끄럽게 작동합니다. PyTorch의 ONNX Exporter는 윈도우/리눅스 환경의 CUDA 버전 종속성으로 인해 특정 opset이나 QDQ 노드 변환 에러를 내뿜을 때가 많은데, 맥 환경에서는 이러한 드라이버/패키지 불일치 문제가 거의 없습니다.
  - **장점**: ONNX Runtime은 Android NPU(NNAPI) 가속을 매우 강력하게 지원하며, TFLite의 까다로운 연산자 호환성 및 컴파일러 에러 문제를 완벽히 우회할 수 있습니다.

