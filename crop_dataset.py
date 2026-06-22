import cv2
import os
import sys
from preprocess import detect_and_crop_face

# Windows 콘솔 인코딩 설정
sys.stdout.reconfigure(encoding='utf-8')

def crop_center_faces():
    base_dir = "dataset"
    dirs = [
        os.path.join(base_dir, "train", "real"),
        os.path.join(base_dir, "train", "spoof"),
        os.path.join(base_dir, "val", "real"),
        os.path.join(base_dir, "val", "spoof")
    ]

    print("========================================")
    print("      얼굴 영역 크롭(Crop) 이미지 전처리      ")
    print("========================================")

    # 1. 4개 폴더를 돌며 이미지 크롭 수행
    crop_size = 250
    total_cropped = 0
    faces_detected_count = 0

    for folder_path in dirs:
        if not os.path.exists(folder_path):
            continue
            
        print(f"[{folder_path}] 처리 중...")
        file_list = os.listdir(folder_path)
        
        for filename in file_list:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            filepath = os.path.join(folder_path, filename)
            
            # 이미지 불러오기
            img = cv2.imread(filepath)
            if img is None:
                continue
                
            h, w, _ = img.shape
            
            # 이미 250x250 크기로 잘려진 이미지는 건너뜁니다 (중복 전처리 방지)
            if h == crop_size and w == crop_size:
                continue
            
            # 2. 얼굴 검출 기반 크롭 수행 (검출 실패 시 중앙 크롭 Fallback)
            cropped_img, face_detected, _ = detect_and_crop_face(img, crop_size=crop_size)
            if face_detected:
                faces_detected_count += 1
            
            # 3. 크롭된 이미지를 기존 경로에 덮어쓰기
            cv2.imwrite(filepath, cropped_img)
            total_cropped += 1

    print("========================================")
    print(f"전처리 완료: 총 {total_cropped}장 중 {faces_detected_count}장에서 얼굴을 감지해 크롭했습니다.")

if __name__ == "__main__":
    crop_center_faces()
