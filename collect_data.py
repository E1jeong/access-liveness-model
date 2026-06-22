import cv2
import os
import time
from preprocess import face_cascade

def main():
    # 1. 데이터를 저장할 폴더 경로 정의
    base_dir = "dataset"
    dirs = {
        "train_real": os.path.join(base_dir, "train", "real"),
        "train_spoof": os.path.join(base_dir, "train", "spoof"),
        "val_real": os.path.join(base_dir, "val", "real"),
        "val_spoof": os.path.join(base_dir, "val", "spoof")
    }

    # 2. 폴더가 없으면 자동으로 생성
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    # 3. PC의 웹캠(카메라) 열기 (Windows에서 MSMF 에러 방지를 위해 cv2.CAP_DSHOW 사용)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("[-] 에러: 웹캠을 열 수 없습니다. 카메라가 올바르게 연결되어 있는지 확인해 주세요.")
        return

    print("==================================================")
    print("      웹캠 기반 안티스푸핑 데이터 수집 프로그램      ")
    print("==================================================")
    print("카메라 창을 클릭한 상태에서 아래 키보드 단추를 누르세요:")
    print("  [ R ] 키 : 진짜 얼굴 (Train Real) 사진 저장")
    print("  [ S ] 키 : 위조 얼굴 (Train Spoof) 사진 저장 (화면/종이 등)")
    print("  [ E ] 키 : 검증용 진짜 얼굴 (Val Real) 사진 저장")
    print("  [ D ] 키 : 검증용 위조 얼굴 (Val Spoof) 사진 저장")
    print("  [ Q ] 키 : 프로그램 종료 (Quit)")
    print("==================================================")

    # 각 폴더별로 저장된 이미지 개수를 세기 위한 카운터
    counts = {k: len(os.listdir(v)) for k, v in dirs.items()}
    print(f"현재 수집된 데이터 현황:")
    print(f" - 학습용 진짜(Train Real): {counts['train_real']}장")
    print(f" - 학습용 위조(Train Spoof): {counts['train_spoof']}장")
    print(f" - 검증용 진짜(Val Real): {counts['val_real']}장")
    print(f" - 검증용 위조(Val Spoof): {counts['val_spoof']}장")
    print("--------------------------------------------------")

    last_saved_face = None

    while True:
        # 카메라로부터 한 프레임 읽어오기
        ret, frame = cap.read()
        if not ret:
            print("[-] 카메라 프레임을 읽어오지 못했습니다.")
            break

        # 좌우 반전 (거울 모드)
        frame = cv2.flip(frame, 1)

        # 원본 이미지 복사 (텍스트 없는 깔끔한 이미지를 저장하기 위함)
        save_frame = frame.copy()

        # 화면에 조작 가이드 안내 텍스트 그리기
        # cv2.putText(이미지, 텍스트, 좌표, 폰트, 크기, 색상(B,G,R), 두께)
        h, w, _ = frame.shape
        cv2.putText(frame, "Press R: Train Real | S: Train Spoof", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press E: Val Real   | D: Val Spoof", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press Q: Quit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 실시간 얼굴 검출
        gray = cv2.cvtColor(save_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # 검출된 얼굴이 있는 경우 그 위치에 박스 그리기
        if len(faces) > 0:
            # 가장 큰 얼굴 선택
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w_face, h_face = faces[0]
            
            # 크롭될 마진 영역 계산
            margin_x = int(w_face * 0.2)
            margin_y = int(h_face * 0.2)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(w, x + w_face + margin_x)
            y2 = min(h, y + h_face + margin_y)
            
            # 시각화: 감지된 얼굴(초록) 및 학습용 크롭 영역(노랑)
            cv2.rectangle(frame, (x, y), (x + w_face, y + h_face), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "Face Detected (Crop Margin)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            # 얼굴 미감지 시 정중앙 유도 박스 (빨간색)
            box_size = 250
            x1 = int(w/2 - box_size/2)
            y1 = int(h/2 - box_size/2)
            x2 = int(w/2 + box_size/2)
            y2 = int(h/2 + box_size/2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(frame, "Align Face (No Face Detected)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 수집된 카운트 표시
        cv2.putText(frame, f"T_Real: {counts['train_real']} | T_Spoof: {counts['train_spoof']}", (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"V_Real: {counts['val_real']} | V_Spoof: {counts['val_spoof']}", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 우측 하단에 최근 저장된 크롭 이미지 오버레이 (있는 경우)
        if last_saved_face is not None:
            preview_size = 120
            preview = cv2.resize(last_saved_face, (preview_size, preview_size))
            
            # 오버레이 위치 계산 (우측 하단 여백 15px)
            px = w - preview_size - 15
            py = h - preview_size - 15
            
            # frame에 복사
            frame[py:py+preview_size, px:px+preview_size] = preview
            
            # 테두리 및 라벨 그리기 (노란색)
            cv2.rectangle(frame, (px, py), (px+preview_size, py+preview_size), (0, 255, 255), 2)
            cv2.putText(frame, "SAVED FACE", (px, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

        # 화면 보여주기
        cv2.imshow("Anti-Spoofing Data Collector", frame)

        # 키 입력 대기 (1밀리초 동안 대기하며 입력이 있으면 key에 아스키 코드 저장)
        key = cv2.waitKey(1) & 0xFF

        target_key = None
        if key == ord('r') or key == ord('R'):
            target_key = "train_real"
        elif key == ord('s') or key == ord('S'):
            target_key = "train_spoof"
        elif key == ord('e') or key == ord('E'):
            target_key = "val_real"
        elif key == ord('d') or key == ord('D'):
            target_key = "val_spoof"
        elif key == ord('q') or key == ord('Q'):
            print("[+] 데이터 수집을 종료합니다.")
            break

        # 키가 눌렸을 경우 이미지 저장 처리
        if target_key:
            # 고유한 파일명을 위해 타임스탬프 사용 (예: train_real_1719273928.jpg)
            timestamp = int(time.time() * 1000)
            filename = f"{target_key}_{timestamp}.jpg"
            filepath = os.path.join(dirs[target_key], filename)
            
            # 이미지 파일로 저장
            cv2.imwrite(filepath, save_frame)
            counts[target_key] += 1
            print(f"[저장 완료] -> {filepath} (현재 총 {counts[target_key]}장)")
            
            # 우측 하단 프리뷰용 이미지 추출 (얼굴 영역 크롭본)
            from preprocess import detect_and_crop_face
            cropped_face, face_detected, _ = detect_and_crop_face(save_frame, crop_size=250)
            last_saved_face = cropped_face

    # 4. 카메라 해제 및 모든 창 닫기
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
