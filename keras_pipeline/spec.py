"""파이프라인 전체가 공유하는 입력 규격 상수.

데이터 로딩(tf_dataset), 모델 정의(tf_model), 변환/평가가 모두 이 파일을 참조한다.
여기 값을 바꾸면 학습·변환·안드로이드 앱 전처리가 동시에 어긋나므로,
안드로이드 계약 문서와 함께 검토하지 않고는 수정하지 말 것.
"""
import numpy as np
from classes import CLASS_NAMES

# 이미지 규격 — cv2.resize에 넘기는 순서라 (width, height)지만 정사각이라 구분이 무의미하다.
IMAGE_SIZE = (224, 224)
IMAGE_HEIGHT, IMAGE_WIDTH = IMAGE_SIZE

# 정규화 파라미터
# RGB는 ImageNet 채널별 평균/표준편차 (PyTorch 파이프라인·안드로이드 앱과 동일한 값).
RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# IR은 통계를 따로 추정하지 않고 0.5/0.5를 써서 결과 범위를 정확히 [-1, 1]로 만든다.
IR_MEAN = np.array([0.5], dtype=np.float32)
IR_STD = np.array([0.5], dtype=np.float32)

# 출력 클래스 수 — classes.py의 CLASS_NAMES가 유일한 클래스 인덱스 출처다.
NUM_CLASSES = len(CLASS_NAMES)

# 모델별 입력 텐서 규격 (이름, shape)
# 접두사 a_/b_는 장식이 아니다: TFLite가 입력을 이름 사전순으로 정렬하므로
# 이 접두사가 "0번=RGB, 1번=IR"이라는 앱과의 인덱스 계약을 강제한다.
MODEL_INPUT_SIGNATURES = {
    "dual": (("a_rgb", (224, 224, 3)), ("b_ir", (224, 224, 1))),
    "crop_rgb": (("a_crop_rgb", (224, 224, 3)),),
    "crop_ir": (("b_crop_ir", (224, 224, 1)),),
}
