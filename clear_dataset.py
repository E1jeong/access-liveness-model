import os
import shutil
import sys

# Windows 콘솔 인코딩 설정
sys.stdout.reconfigure(encoding='utf-8')

def clear_dataset():
    dirs = [
        "dataset/train/real",
        "dataset/train/spoof",
        "dataset/val/real",
        "dataset/val/spoof"
    ]
    
    print("========================================")
    print("       데이터셋 이미지 초기화 프로그램       ")
    print("========================================")
    
    deleted_count = 0
    
    for folder in dirs:
        if not os.path.exists(folder):
            continue
            
        print(f"[{folder}] 비우는 중...")
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            try:
                if os.path.isfile(filepath):
                    os.unlink(filepath)
                    deleted_count += 1
                elif os.path.isdir(filepath):
                    shutil.rmtree(filepath)
                    deleted_count += 1
            except Exception as e:
                print(f"[-] 파일 삭제 실패: {filepath} ({e})")
                
    print("========================================")
    print(f"초기화 완료: 총 {deleted_count}개의 이미지를 삭제했습니다.")

if __name__ == "__main__":
    clear_dataset()
