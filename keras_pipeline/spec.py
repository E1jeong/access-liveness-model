import numpy as np
from classes import CLASS_NAMES

# 이미지 규격
IMAGE_SIZE = (224, 224)
IMAGE_HEIGHT, IMAGE_WIDTH = IMAGE_SIZE

# 정규화 파라미터
RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IR_MEAN = np.array([0.5], dtype=np.float32)
IR_STD = np.array([0.5], dtype=np.float32)

# 출력 클래스 수
NUM_CLASSES = len(CLASS_NAMES)

# 모델별 입력 텐서 규격 (이름, shape)
MODEL_INPUT_SIGNATURES = {
    "dual": (("a_rgb", (224, 224, 3)), ("b_ir", (224, 224, 1))),
    "crop_rgb": (("a_crop_rgb", (224, 224, 3)),),
    "crop_ir": (("b_crop_ir", (224, 224, 1)),),
}
