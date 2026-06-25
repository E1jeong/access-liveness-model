# 안티스푸핑 프로젝트 현재 상태

이 문서는 작업할 때마다 바뀌는 사실과 검증 결과를 기록한다. 고정 개발 절차와 판단 기준은 [project_guide.md](project_guide.md)를 따른다.

- **마지막 갱신일**: 2026-06-25
- **검증 환경**: WSL Ubuntu 24.04, 프로젝트 `.venv`
- **현재 단계**: 코드 구조를 RGB+IR 5클래스 / 디바이스 수집 / TFLite 배포 체계로 전환 완료. 실제 학습용 데이터 정비 및 성능 측정 대기.

## 0. 프로젝트 전환 이력 (중요)

이 프로젝트는 두 시기를 거쳤다. 과거 기록을 읽을 때 혼동하지 않도록 명시한다.

- **이전(웹캠 시기, 폐기됨)**: PC 웹캠으로 RGB 단일 프레임을 취득해 학습하고 웹캠으로 추론을 확인하던 단계. REAL/SPOOF 2클래스, 중앙 250×250 크롭, ONNX 변환, Haar Cascade 등이 이 시기의 산물이다. **현재 코드에는 남아있지 않다.**
- **현재(디바이스 시기)**: Android 출입통제기에서 RGB+IR 이미지를 직접 취득해 파일로 전달받고, 그 파일로 학습하며, 배포는 TFLite로 고정한다. 웹캠 수집·ONNX 경로는 제거했다.

## 1. 상태 요약

### 검증된 항목 (코드/실행 근거 있음)

- `[검증 완료]` `model.py` 실행 시 듀얼 인풋(RGB+IR) 출력 텐서가 `[1, 5]`다.
- `[검증 완료]` `classes.py`가 클래스 단일 출처이며 `0=live, 1=print, 2=picture, 3=mask, 4=display`다.
- `[검증 완료]` `dataset.py`가 subject(`<class>_<id>` 폴더) 단위 K-fold로 분할하며 train/val 비겹침 assert를 통과한다. 더미 데이터에서 RGB `[B,3,224,224]`, IR `[B,1,224,224]`를 출력한다.
- `[검증 완료]` `train.py`가 5×5 confusion matrix, 클래스별 recall, APCER/BPCER/ACER를 산출하고 best 체크포인트를 **ACER 최저** 기준으로 저장한다. APCER는 "spoof를 live(0)로 오분류한 비율"로 정의되어 있다(self-check 포함).
- `[검증 완료]` 배포 자산 `app/src/main/assets/anti_spoofing.tflite`의 입력은 `[1,224,224,3]`+`[1,224,224,1]`(float32), 출력은 `[1,5]`(float32)다. Android `model_spec.json`의 정규화(RGB ImageNet, IR 0.5/0.5)와 학습 전처리가 일치한다.

### 구현됐으나 성능 미측정

- `[미검증]` 실제 디바이스 수집 데이터에 대한 학습 성능(현재 `dataset/raw`에 subject 데이터가 전개되어 있지 않음).
- `[미검증]` 미등록 인물·다른 조명·다른 거리에서의 일반화 성능.
- `[미검증]` 클래스별(print/picture/mask/display) APCER 세부 성능.
- `[미검증]` i.MX 8M Plus NPU 실기기 정확도·지연시간·FPS.

### 미구현 항목

- `[미구현]` TFLite INT8 양자화.
- `[미구현]` 독립 test split(현재는 K-fold 교차검증만).
- `[미구현]` 의존성 lock 파일(재현성).

## 2. 데이터 현황

- 학습 데이터는 Android 수집기가 만든 구조 `dataset/raw/<class>/<class>_<subjectId>/<frame>/`를 그대로 사용한다. 각 frame 폴더에는 `cropRGB.bmp`, `cropIR.bmp`, `RGB.bmp`, `IR.bmp`가 있어야 한다(`dataset.py`가 4개 모두 존재를 요구).
- 현재 `dataset/raw/`에는 클래스 폴더만 있고 subject 데이터는 비어 있다 → 현 상태로는 실제 학습 불가, 더미 데이터로만 파이프라인 동작을 확인했다.
- `dataset/data.zip`은 이전 데이터셋이며 사용자가 직접 관리·삭제 예정이다. 자동으로 전개하지 않는다.
- 클래스 정의: 동일 인물의 live(맨얼굴)와 그 인물을 본뜬 mask는 서로 다른 물리 객체로 수집한다. mask 폴더에 live 인물 자체가 들어가지 않으므로 클래스 간 동일 인물 누수 위험은 낮다고 사용자가 확인했다.

## 3. 현재 모델과 전처리

- 모델: Dual-Input MobileNetV3-Small (RGB 백본 + IR 백본, IR은 1채널 입력으로 첫 conv 교체). 두 백본 feature(576+576=1152)를 concat 후 `Linear(1152,1024)→Hardswish→Dropout→Linear(1024,5)`.
- 초기 가중치: 두 백본 모두 ImageNet pretrained.
- 입력: RGB `[1,3,224,224]`, IR `[1,1,224,224]` (PyTorch NCHW). 배포 TFLite는 NHWC로 래핑.
- 정규화: RGB mean `[0.485,0.456,0.406]`/std `[0.229,0.224,0.225]`, IR mean `[0.5]`/std `[0.5]`.
- 학습 설정: CPU, K-fold 교차검증, Adam, CrossEntropyLoss (기본값은 `train.py` argparse 참고).
- 체크포인트: `model/best_model_fold{N}.pth` (ACER 최저 기준).
- 배포: `convert_to_tflite.py`로 `model/anti_spoofing.tflite` 생성 후 Android `app/src/main/assets/`로 수동 복사.

## 4. 환경 버전

현재 환경은 `verify_setup.py`로 확인한다(과거 고정 표 대신 실행으로 확인).

```bash
.venv/bin/python verify_setup.py
```

확인 시점 주요 버전: PyTorch `2.12.1+cpu`. CUDA 불가(CPU 학습). 그 외 버전은 위 명령 출력으로 확인한다.

## 5. 검증 명령

```bash
.venv/bin/python model.py        # 출력 [1,5] 확인
.venv/bin/python dataset.py      # dataset/raw에 데이터가 있을 때 RGB/IR 형상 확인
.venv/bin/python train.py        # K-fold 학습 및 APCER/BPCER/ACER 산출
.venv/bin/python convert_to_tflite.py   # TFLite 변환 (litert_torch 필요)
```

## 6. 알려진 위험

- **재현성**: 의존성 lock 파일·고정 seed 외 자동 평가 스크립트가 부족하다. 데이터·체크포인트·tflite는 `.gitignore` 대상이라 저장소만으로 결과를 재현할 수 없다.
- **데이터 정비 대기**: 실제 디바이스 수집 데이터가 `dataset/raw`에 전개되기 전까지 모든 성능 수치는 미측정이다.
- **NPU 미검증**: TFLite INT8·i.MX 8M Plus 실행 결과가 아직 없다.

## 7. 변경 이력

| 일시 | 변경 내용 |
|---|---|
| 2026-06-25 | 웹캠/ONNX 시기 잔재 정리. 문서를 현재 RGB+IR 5클래스·디바이스 수집·TFLite 배포 체계로 재작성. `convert_to_onnx.py`·`create_dummy_tflite.py` 삭제, `verify_setup.py`/`.gitignore`에서 ONNX 제거. |
