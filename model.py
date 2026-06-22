import torch
import torch.nn as nn
import torchvision.models as models

def get_anti_spoof_model():
    """
    안드로이드 기기/NPU에 적합한 초경량 모델인 MobileNetV3-Small 모델을 불러옵니다.
    사전 학습된 가중치(Pretrained Weights)를 사용하여 얼굴 특징을 빠르게 파악할 수 있게 합니다.
    """
    
    # 1. 이미 학습된 MobileNetV3-Small 모델 불러오기 (전이 학습 - Transfer Learning)
    # 기본 이미지넷(ImageNet) 데이터셋으로 학습된 상태이므로 사물, 얼굴의 기하학적 특징을 이미 알고 있습니다.
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    
    # 2. 모델의 마지막 분류 레이어(Classifier) 수정하기
    # MobileNetV3-Small의 기본 출력 클래스는 1000개(사물 종류)입니다.
    # 우리는 진짜 얼굴(0)과 위조 얼굴(1) 2개의 클래스만 분류하면 되므로 출력을 2로 변경합니다.
    # MobileNetV3-Small의 분류 레이어 구조:
    # model.classifier는 Sequential 구조이며, 마지막 레이어는 model.classifier[3]에 위치한 Linear 레이어입니다.
    
    # 마지막 Linear 레이어의 입력 특징 수(in_features) 파악 (보통 1024개)
    in_features = model.classifier[3].in_features
    
    # 마지막 레이어를 입력 특징 수 -> 출력 2로 변경
    model.classifier[3] = nn.Linear(in_features, 2)
    
    print("[모델 생성 완료]")
    print(f" - 베이스 모델: MobileNetV3-Small")
    print(f" - 분류 클래스 수: 2 (진짜 / 위조)")
    
    return model

if __name__ == "__main__":
    # 모델 생성 테스트
    model = get_anti_spoof_model()
    # 테스트용 가짜 이미지 텐서 생성 [배치크기=1, 채널=3, 높이=224, 너비=224]
    dummy_input = torch.randn(1, 3, 224, 224)
    # 모델에 집어넣어 결과 확인
    output = model(dummy_input)
    print(f"모델 출력 텐서 크기: {output.shape}")  # [1, 2] -> 2개 클래스에 대한 예측 점수값
    print(f"예측 출력값: {output}")
