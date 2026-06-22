import torch
import onnx
import sys
from model import get_anti_spoof_model

# Windows 콘솔에서 ✅ 이모지 같은 특수문자를 출력할 때 발생하는 인코딩 에러 방지
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

def export_to_onnx():
    # 1. 저장된 가중치 파일명 및 출력할 ONNX 파일명 정의
    pth_path = "best_model.pth"
    onnx_path = "model.onnx"

    # 2. 모델 구조 생성 및 가중치(Weights) 로드
    model = get_anti_spoof_model()
    
    # 학습된 가중치 로드 (CPU 환경에 맞춰 로드)
    state_dict = torch.load(pth_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    
    # 추론 모드로 전환 (Dropout, Batch Normalization 동작 고정)
    model.eval()

    # 3. 더미 입력(Dummy Input) 데이터 생성
    # ONNX로 변환할 때는 입력데이터가 흘러가는 통로를 만들어주기 위해 실제 입력과 동일한 크기의 모조 데이터가 필요합니다.
    # [배치 크기=1, RGB 채널=3, 높이=224, 너비=224]
    dummy_input = torch.randn(1, 3, 224, 224)

    # 4. PyTorch 모델을 ONNX 포맷으로 내보내기
    print(f"\n[ONNX 변환 중...] {pth_path} -> {onnx_path}")
    torch.onnx.export(
        model,                      # 변환할 PyTorch 모델
        dummy_input,                # 모델 입력을 위한 더미 데이터
        onnx_path,                  # 저장될 ONNX 파일 경로
        export_params=True,         # 모델 내의 학습된 파라미터(가중치)를 같이 저장할지 여부
        opset_version=15,           # ONNX 연산자 셋 버전 (최신 버전 15 사용)
        do_constant_folding=True,   # 연산 최적화 적용 (상수 폴딩)
        input_names=['input'],      # 입력 노드 이름 정의
        output_names=['output'],    # 출력 노드 이름 정의
        dynamic_axes={              # 배치 크기를 유연하게 조절할 수 있도록 설정 (선택 사항)
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print("[ONNX 변환 완료]")

    # 5. ONNX 파일 검증
    # 변환된 파일이 깨지지 않고 ONNX 표준 스펙에 맞게 잘 생성되었는지 검사합니다.
    print("\n[ONNX 모델 검증 중...]")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("[검증 성공] ONNX 모델 구조가 올바릅니다. 기기에 탑재할 수 있습니다!")

if __name__ == "__main__":
    export_to_onnx()
