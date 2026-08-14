# Keras 학습 파이프라인 핵심 개념 정리

팀 코드리뷰 및 개발을 위해 `keras_pipeline/` 코드가 "왜 이렇게 설계되었는가"를 딥러닝 핵심 작동 원리부터 학습 파이프라인, 평가지표까지 체계적으로 재편성한 문서.

- **SSOT(단일 진실 공급원)**: 프로젝트 상태·지표·명령어의 출처는 Obsidian 위키(`Dev/Project/Company/access-liveness-model/`)입니다. 이 문서는 개념 이해 및 코드리뷰용입니다.
- **상태 표기 기준**:
  - **근거 상태**: `확인됨`(코드/테스트로 검증) · `기록됨`(위키에 근거 있음) · `미확인`(추정)

---

## 목차

1. **[Part 1] CNN 핵심 동작 원리 (입력에서 Logits까지)**
   - 1-A. 커널(필터)과 합성곱(Conv)
   - 1-B. 특징 맵(Feature Map)과 채널(Channel)
   - 1-C. ReLU 활성화 함수 (비선형성과 신호 필터링)
   - 1-D. 파라미터와 가중치 공유(Weight Sharing)
   - 1-E. 해상도 사다리(224→7)와 GAP(Global Average Pooling)
   - 1-F. 백본(1280차원 요약)과 헤드(10개 판단)의 분리
   - 1-G. Logits (출력 점수와 Softmax가 없는 이유)
2. **[Part 2] 학습 파이프라인과 하이퍼파라미터**
   - 2-A. 정규화(MEAN / STD)의 수학적 본질
   - 2-B. 가중치 이식(Transfer) vs 스크래치(Scratch) & Conv1 `sum`
   - 2-C. `tf.data` 파이프라인 (2중 셔플, 증강, `repeat=True` 무한 스트림)
   - 2-D. CosineDecay와 미세조정(Fine-tuning) 학습률
   - 2-E. 손실 함수(Loss)와 One-Hot 인코딩, Label Smoothing
   - 2-F. Dropout과 과적합 방지
   - 2-G. `model.compile`(도구 장착) vs `model.fit`(실제 학습 실행)
   - 2-H. 실험 재현성 보장 (`--seed`)
3. **[Part 3] 안티스푸핑 평가 지표와 과적합 판별**
   - 3-A. 손실(Loss)과 과적합(Overfitting)의 판별 원리
   - 3-B. APCER · BPCER · ACER (불균형 데이터와 보안/사용성 분리)
4. **[부록] 프로젝트 특수 계약 및 배포 장치** (TFLite 알파벳 정렬, RGB Lambda 보정)

---

# Part 1. CNN 핵심 동작 원리 (입력에서 Logits까지)

```text
[입력 이미지]             [합성곱 + ReLU]              [백본 압축 사다리]           [GAP]            [분류 헤드]
224×224 (RGB/IR)  ──▶  커널 훑기 + 음수 제거  ──▶  224→112→56→28→14→7  ──▶  1280차원 벡터  ──▶  Dense ──▶ 10개 점수(Logits)
```

## 1-A. 커널(필터)과 합성곱(Conv)
- **커널(필터)**: 입력 데이터에서 특정 시각 패턴(가로선, 세로선, 질감 등)을 찾는 작은 창(예: 3×3).
- **합성곱(Convolution)**: 커널을 이미지 위에 올려놓고 미끄러뜨리면서(Stride), 겹치는 영역의 숫자를 곱하고 모두 더해 새로운 지도를 만드는 연산.
- **근거 상태**: `확인됨` (`keras.applications.MobileNetV2` 커널 값 실측)

## 1-B. 특징 맵(Feature Map)과 채널(Channel)
- **특징 맵(Feature Map)**: 커널들이 이미지를 훑고 지나간 뒤 만들어낸 결과물 전체 3D 덩어리 (예: `112×112×32`).
- **채널(Channel)**: 특징 맵 안에 쌓여 있는 2D 지도의 장수(깊이). 커널 1장이 지도 1장(1채널)을 만들며, 커널 32장을 적용하면 결과물은 32채널이 된다.
- **근거 상태**: `확인됨` (`model.summary()` 층별 shape)

## 1-C. ReLU 활성화 함수
- **수식**: $f(x) = \max(0, x)$
- **동작**: 이전 레이어(Conv, Dense)의 계산 결과가 양수($>0$)면 그대로 통과시키고, 음수($\le 0$)면 $0$으로 버림.
- **존재 이유**: 곱셈/덧셈(선형 연산)만 반복하면 복잡한 경계를 구분할 수 없음. 음수를 잘라내는 비선형성을 부여하여 선, 모서리, 질감, 얼굴 형태 등의 복잡한 패턴을 학습할 수 있게 만듦.
- **근거 상태**: `확인됨` (`tf_model.py:107` `activation="relu"`)

## 1-D. 파라미터와 가중치 공유(Weight Sharing)
- **파라미터(가중치)**: 커널 안에 적혀 있는 숫자 그 자체. 이미지를 바꿔도 안 변하며, 학습을 돌려야 수정되고 `.keras` 파일에 저장됨.
- **가중치 공유**: 이미지의 왼쪽 위를 훑을 때 쓴 커널 864개를 오른쪽 아래를 볼 때도 똑같이 재사용함. 이를 통해 Dense 방식 대비 파라미터 수를 7천만 배 절감함.
- **근거 상태**: `확인됨`

## 1-E. 해상도 사다리(224→7)와 GAP(Global Average Pooling)
- **224 해상도 사다리**: MobileNetV2는 Stride 2 합성곱으로 공간 크기를 5번 반토막 냄 ($224 \rightarrow 112 \rightarrow 56 \rightarrow 28 \rightarrow 14 \rightarrow 7$).
- **GAP (전역 평균 풀링)**: 마지막 $7\times 7\times 1280$ 특징 맵에서 각 채널($7\times 7=49$칸)의 평균을 내어 **1280차원의 1D 벡터**로 압축함. Flatten 방식 대비 파라미터를 수십 배 절약함.
- **근거 상태**: `확인됨` (`spec.py:IMAGE_SIZE`, `tf_model.py:AveragePooling2D`)

## 1-F. 백본(Backbone)과 헤드(Head)의 분리
- **백본(등뼈)**: 이미지를 1280개의 특징 숫자로 요약하는 범용 시각 추출기 (`include_top=False`).
- **헤드(머리)**: 1280개 요약을 받아 "live인가 스푸핑인가" 최종 점수(10개)를 계산하는 안티스푸핑 전용 분류기 (`Dense(1024) + ReLU` $\rightarrow$ `Dense(10)`).
- **근거 상태**: `확인됨` (`tf_model.py:56-70`, `107-120`)

## 1-G. Logits (출력 점수)
- **정의**: 모델 맨 끝 층이 뱉는 softmax 이전의 10개 원시 점수.
- **Softmax가 없는 이유**: 가장 높은 점수의 클래스를 찾는 `argmax` 판정에는 확률 변환이 불필요하며, Softmax를 제거해야 수치 안정성과 NPU 연산 효율이 높아짐.
- **근거 상태**: `확인됨` (`tf_train.py:140` `np.argmax(logits, axis=1)`)

---

# Part 2. 학습 파이프라인과 하이퍼파라미터

```text
[배치 8장] ──▶ 순전파 ──▶ Logits ──▶ Loss 계산 (One-Hot / Label Smoothing) ──▶ 역전파(Backprop) ──▶ Adam (CosineDecay LR) ──▶ 가중치 갱신
```

## 2-A. 정규화(MEAN / STD)의 본질
- **수식**: $x_{\text{norm}} = (x - \text{MEAN}) / \text{STD}$
- **MEAN(0점 조절)**: 중심을 0으로 이동.
- **STD(눈금 조절)**: 데이터 분포 폭을 1로 스케일링.
- **목적**: 0~255 원본 픽셀을 그대로 넣으면 경사하강법이 불안정해지므로, 평균 0 / 분산 1 근처인 $[-1, 1]$ 범위로 정렬하여 안정적으로 학습시킴.
- **근거 상태**: `확인됨` (`spec.py:9-12`, `tf_dataset.py:52-57`)

## 2-B. 가중치 이식(Transfer) vs 스크래치 & Conv1 `sum`
- **스크래치(`--rgb-weights none`)**: 난수로 시작하여 밑바닥부터 학습.
- **이식 모델(`--rgb-weights imagenet`)**: ImageNet 120만 장으로 학습된 우수한 시각 필터를 가져와 시작 (수렴 속도 및 성능 우수).
- **1채널 IR 이식 메커니즘**:
  - `crop_ir` 생성 시 임시 RGB 백본을 띄워 가중치를 추출함.
  - 첫 번째 레이어 `Conv1`의 커널 `(3,3,3,32)`을 **`sum` (더하기)** 방식으로 `(3,3,1,32)`로 축소 이식함.
  - `mean`은 출력이 1/3로 줄어 BatchNorm 통계가 깨지지만, `sum`은 RGB 동일 입력 시 원본 응답 크기를 100% 보존하여 BatchNorm 통계와 완벽히 일치함.
- **근거 상태**: `확인됨` (`tf_model.py:34-81`, `tests/model/test_conv1_reduction.py`)

## 2-C. `tf.data` 파이프라인 (2중 셔플, 증강, `repeat=True` 무한 스트림)
- **1차 파이썬 셔플 + 2차 TF 버퍼 셔플 (2중 셔플)**:
  - 1차: `random.Random(seed).shuffle(items)`로 클래스별로 정렬된 원본 리스트를 미리 섞어 증강 시드 카운터 편향 방지.
  - 2차: `ds.shuffle(buffer_size=len(items), reshuffle_each_iteration=True)`로 에폭마다 배치의 조합과 순서를 무작위화.
- **데이터 증강(Augmentation)**:
  - 공간 증강(공통): 좌우 반전(50%), 미세 회전($\pm 10^\circ$).
  - 광도 증강(RGB 전용): ColorJitter(밝기 $\pm 30\%$, 대비 $\pm 30\%$, 채도 $\pm 20\%$). IR은 광도 왜곡 방지를 위해 제외.
- **`repeat=True` 무한 스트림과 증강 시드 카운터 누적**:
  - `ds.repeat().enumerate()` 순서로 구성하여 데이터 스트림이 끝없이 이어짐.
  - 에폭이 바뀌어도 인덱스 카운터가 0으로 리셋되지 않고 누적되므로, 같은 이미지라도 매 에폭마다 서로 다른 증강 변형을 적용받음.
- **`steps_per_epoch` 계산**:
  $$\text{steps\_per\_epoch} = \text{math.ceil}\left(\frac{\text{len(train\_items)}}{\text{batch\_size}}\right)$$
  무한 데이터셋에 1에폭의 경계(가중치 갱신 횟수)를 수동 지정하여 `AcerCheckpoint` 검증 시점을 제어함.
- **근거 상태**: `확인됨` (`tf_dataset.py:182-317`, `tf_train.py:281`)

## 2-D. CosineDecay와 미세조정(Fine-tuning) 학습률
- **CosineDecay**: 학습률을 고정하지 않고 코사인 곡선을 따라 시작값($10^{-4}$)에서 최종값($10^{-6}$)으로 점진적 감쇠.
- **시작 학습률 $10^{-4}$**: 백본을 동결하지 않고 전체 학습(`trainable=True`)하므로, 이미 완성된 ImageNet 가중치가 초반에 망가지지 않도록 낮은 학습률로 조심스럽게 미세조정함.
- **근거 상태**: `확인됨` (`tf_train.py:287-291`)

## 2-E. 손실 함수(Loss)와 One-Hot 인코딩, Label Smoothing
- **손실 함수(Loss Function)**: 모델 예측과 실제 정답의 차이를 수치화하는 벌점 계산기. 점수(Loss)가 0에 가까울수록 정답을 잘 맞춘 것임.
- **정수 라벨(Sparse) vs 원-핫(One-Hot) 인코딩**:
  - 정수 라벨: 정답 번호 1개만 표현 (예: `live`=0, `mask`=2).
  - 원-핫 인코딩: 정답 자리에만 1.0(불 켜짐), 나머지 9개 오답 자리는 0.0(불 꺼짐)인 10칸짜리 배열.
- **라벨 스무딩과 `tf.one_hot` 변환의 필요성**:
  - 목표 정답을 `[1.0, 0, 0, ...]` 대신 `[0.91, 0.01, 0.01, ...]`로 완화하여 모델의 과도한 확신(Overconfidence)에 벌점을 줌.
  - 10개 클래스에 완화된 확률을 분배하려면 10차원 배열 형태가 필수적이므로, `loss_fn`에서 `tf.one_hot`으로 정수 라벨을 펼쳐 `CategoricalCrossentropy`에 전달함.
- **근거 상태**: `확인됨` (`tf_train.py:319-334`)

## 2-F. Dropout과 과적합 방지
- **Dropout (`--dropout 0.2`)**:
  - 학습 시 매 스텝마다 분류 헤드의 1024개 뉴런 중 20%를 무작위로 꺼서 특정 뉴런 의존성을 방지함.
  - 추론/검증/단말 배포 시에는 자동으로 비활성화(100% 정상 작동)됨.
- **근거 상태**: `확인됨` (`tf_model.py:109`)

## 2-G. `model.compile`(도구 장착) vs `model.fit`(실제 학습 실행)
- **`model.compile` (학습 준비/규칙 설정)**:
  - 옵티마이저(Adam), 손실 함수(Cross-Entropy), 평가지표(Accuracy) 등 학습에 필요한 도구를 모델에 장착하는 단계.
  - 데이터를 읽지 않으며 0.01초 내에 완료되고 모델 가중치는 전혀 변경되지 않음.
- **`model.fit` (실제 반복 학습 루프)**:
  - 배치 데이터 공급 $\rightarrow$ 순전파(예측) $\rightarrow$ 손실 계산 $\rightarrow$ 역전파(미분) $\rightarrow$ 가중치 갱신을 수천 번 반복하며 실제로 모델을 학습시키는 핵심 실행 함수.
- **근거 상태**: `확인됨` (`tf_train.py:335-358`)

## 2-H. 실험 재현성 보장 (`--seed 42`)
- **역할**: Python 난수, NumPy 난수, TensorFlow 가중치 초기화·셔플·증강 난수원을 시드 하나로 일괄 고정 (`set_random_seed`).
- **목적**: 실행 시점과 무관하게 100% 동일한 학습 결과를 재현하여 공정한 A/B 비교 보장.
- **근거 상태**: `확인됨` (`tf_train.py:242`)

---

# Part 3. 안티스푸핑 평가 지표와 과적합 판별

## 3-A. 손실(Loss)과 과적합(Overfitting) 판별
- **원리**: Train Loss는 계속 감소하는데 Validation 지표가 돌아서서 악화되는 시점이 바로 **"데이터를 응용하지 않고 통째로 외우기 시작했다(과적합)"**는 신호.
- **`AcerCheckpoint`**: 파이프라인에서 매 에폭마다 Validation셋에 대해 `predict`를 돌려 **ACER가 역대 최저일 때만 `.keras` 모델을 저장**함.
- **근거 상태**: `확인됨` (`tf_train.py:115-173`)

## 3-B. APCER · BPCER · ACER 지표 분리
- **왜 단순 정확도(Accuracy)를 쓰지 않는가**:
  스푸핑 90%, 정상 10%처럼 데이터가 불균형할 때, 정상인을 전부 스푸핑으로 거부해도 정확도는 90%가 나오는 왜곡(사기)이 발생함.
- **지표의 정의**:
  - **APCER (보안 실패)**: 가짜(Spoof)를 진짜(Live)로 통과시킨 비율 = $\frac{\text{뚫린 공격 수}}{\text{전체 공격 수}}$
  - **BPCER (사용성 실패)**: 진짜(Live)를 가짜(Spoof)로 오인 거부한 비율 = $\frac{\text{거부된 정상인 수}}{\text{전체 정상인 수}}$
  - **ACER**: 위 둘의 단순 평균 = $\frac{\text{APCER} + \text{BPCER}}{2}$
- **근거 상태**: `확인됨` (`utils.py:386-395`)

---

# [부록] 프로젝트 특수 규칙 및 계약 (참고용)

| 항목 | 목적 및 규칙 | 관련 코드 |
| :--- | :--- | :--- |
| **TFLite 텐서 정렬 방지** | TFLite 변환기가 입력 이름을 알파벳순 정렬하므로, `a_rgb`, `b_ir` 접두사로 순서를 강제 고정함 | `spec.py:MODEL_INPUT_SIGNATURES` |
| **RGB Lambda 보정 레이어** | 안드로이드 앱이 ImageNet 표준화로 입력하므로, 앱 수정 없이 모델 내부에서 `[-1, 1]`로 역산 변환함 | `tf_model.py:121-128` |
| **1x1 Conv 치환** | NPU(NNAPI)에서 Fully Connected(Dense) 미지원 문제를 방지하기 위해 1×1 Conv2D로 치환 가능하게 설계 | `tf_model.py:84-103` |
