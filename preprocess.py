import cv2
import os

# OpenCV 내장 Haar Cascade 파일 경로 로드
CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def detect_and_crop_face(image, crop_size=250, margin_ratio=0.2):
    """
    이미지에서 얼굴을 검출하고 여백을 두어 크롭한 후 지정된 크기로 리사이즈합니다.
    얼굴이 없으면 중앙 크롭으로 대체(Fallback)합니다.
    
    Args:
        image: BGR 이미지 (numpy array)
        crop_size: 출력 이미지 가로세로 크기 (기본값: 250)
        margin_ratio: 얼굴 박스 대비 추가할 상하좌우 여백 비율 (기본값: 0.2, 즉 20%)
        
    Returns:
        cropped_image: 크롭 및 리사이즈된 이미지
        face_detected: 얼굴 검출 성공 여부 (bool)
        bbox: 검출된 얼굴의 원본 바운딩 박스 (x, y, w, h) 또는 None
    """
    h_img, w_img, _ = image.shape
    
    # 1. Haar Cascade 검출을 위해 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. 얼굴 검출 실행
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,      # 오탐 방지를 위해 이웃 사각형 개수 임계값 상향 (기본 5 -> 8)
        minSize=(80, 80)     # 멀리 있는 작은 서랍 손잡이 등의 오탐을 막기 위해 최소 얼굴 크기 상향 (기본 30x30 -> 80x80)
    )
    
    if len(faces) > 0:
        # 3. 여러 개의 얼굴이 검출되는 경우 가장 크기가 큰 얼굴을 선택
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        
        # 4. 여백(Margin) 계산
        margin_x = int(w * margin_ratio)
        margin_y = int(h * margin_ratio)
        
        # 5. 여백을 포함한 크롭 영역 계산 (이미지 경계를 벗어나지 않도록 클리핑)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w_img, x + w + margin_x)
        y2 = min(h_img, y + h + margin_y)
        
        # 6. 크롭 및 리사이즈
        cropped = image[y1:y2, x1:x2]
        if cropped.size > 0:
            cropped = cv2.resize(cropped, (crop_size, crop_size))
            return cropped, True, (x, y, w, h)
            
    # --- 7. 얼굴 검출 실패 시 정중앙 크롭 Fallback 적용 ---
    x1 = int(w_img / 2 - crop_size / 2)
    y1 = int(h_img / 2 - crop_size / 2)
    
    # 이미지 경계 예외 처리
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_img, x1 + crop_size)
    y2 = min(h_img, y1 + crop_size)
    
    cropped = image[y1:y2, x1:x2]
    
    # 최종 크기가 crop_size와 다르면 리사이즈 (예: 원본이 250보다 작은 특수 케이스)
    if cropped.shape[0] != crop_size or cropped.shape[1] != crop_size:
        cropped = cv2.resize(cropped, (crop_size, crop_size))
        
    return cropped, False, None

if __name__ == "__main__":
    # 단독 테스트 코드 (테스트용 이미지 로드 및 검출 결과 출력)
    print(f"Haar Cascade 파일 로드 상태: {not face_cascade.empty()}")
    print(f"Haar Cascade 파일 경로: {CASCADE_PATH}")
