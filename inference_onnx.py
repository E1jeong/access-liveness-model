import cv2
import numpy as np
import onnxruntime as ort
import sys

# Windows 콘솔 출력 인코딩 설정
sys.stdout.reconfigure(encoding='utf-8')

def preprocess_image(frame):
    """
    OpenCV 카메라 프레임(BGR)을 ONNX 모델이 기대하는 포맷(RGB, 정규화)으로 전처리합니다.
    """
    # 1. BGR -> RGB 변환
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 2. 224x224 크기로 조정
    image = cv2.resize(image, (224, 224))
    
    # 3. 0~255 값을 0~1 값으로 변환 (Float32)
    image = image.astype(np.float32) / 255.0
    
    # 4. ImageNet 데이터셋 정규화 공식 적용 (mean, std를 float32로 명시하여 double 타입 변환 방지)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    
    # 5. [높이, 너비, 채널] -> [채널, 높이, 너비] 순서 변경 (HWC -> CHW)
    image = np.transpose(image, (2, 0, 1))
    
    # 6. 배치 차원 추가 [1, 3, 224, 224]
    image = np.expand_dims(image, axis=0)
    
    return image

def softmax(x):
    """
    모델의 출력 점수(Logits)를 확률 값(0~1 사이)으로 변환합니다.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

def main():
    # 1. ONNX 모델 로드
    onnx_path = "model.onnx"
    try:
        # ONNX Runtime으로 변환된 모델을 메모리에 로드합니다.
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        print(f"[+] ONNX 모델 로드 완료: {onnx_path}")
    except Exception as e:
        print(f"[-] 에러: ONNX 모델을 불러오지 못했습니다. {e}")
        return

    # 2. 카메라(웹캠) 열기
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[-] 에러: 카메라를 열 수 없습니다.")
        return

    print("\n==================================================")
    print("      ONNX 실시간 안티스푸핑(위변조 방지) 테스트      ")
    print("==================================================")
    print("  - 초록색 네모: 진짜 얼굴 (REAL)")
    print("  - 빨간색 네모: 위조 얼굴 (SPOOF) - 스마트폰 화면 등")
    print("  - 종료하려면 카메라 창에서 [ Q ] 키를 누르세요.")
    print("==================================================")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        
        # 3. 모델 입력을 위해 화면 중앙 부분 가이드박스 설정
        box_size = 250
        x1 = int(w/2 - box_size/2)
        y1 = int(h/2 - box_size/2)
        x2 = int(w/2 + box_size/2)
        y2 = int(h/2 + box_size/2)

        # 가이드 박스 내 얼굴 부분만 크롭(잘라내기)하여 입력으로 사용
        face_crop = frame[y1:y2, x1:x2]
        
        # 크롭된 영역이 유효한지 확인
        if face_crop.size > 0:
            # 4. 이미지 전처리 수행 (얼굴 영역만 사용)
            input_tensor = preprocess_image(face_crop)

            # 5. ONNX 추론 실행
            # PyTorch가 아닌 ONNX Runtime을 사용하여 초고속 추론 수행
            outputs = session.run(None, {input_name: input_tensor})
            logits = outputs[0]  # 모델 예측 원본 점수
            
            # 확률값 계산
            probabilities = softmax(logits)[0]
            pred_class = np.argmax(probabilities) # 0: real, 1: spoof
            confidence = probabilities[pred_class] * 100

            # 6. 예측 결과에 따른 화면 표시
            # 진짜 얼굴(0) -> 초록색 가이드박스
            if pred_class == 0:
                color = (0, 255, 0)
                label = f"REAL: {confidence:.2f}%"
            # 위조 얼굴(1) -> 빨간색 가이드박스
            else:
                color = (0, 0, 255)
                label = f"SPOOF: {confidence:.2f}%"

            # 화면에 박스 및 라벨 표시
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(frame, (x1, y1 - 35), (x1 + 180, y1), color, -1) # 텍스트 배경 박스
            cv2.putText(frame, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(frame, "Press Q to Quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Real-time Anti-Spoofing Test (ONNX)", frame)

        # Q 키 입력 시 루프 탈출
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
