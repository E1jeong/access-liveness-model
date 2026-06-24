import os
import shutil
import sys

# Windows 콘솔 인코딩 설정
sys.stdout.reconfigure(encoding='utf-8')

def clear_dataset():
    target = "dataset"
    print("========================================")
    print("       데이터셋 이미지 초기화 프로그램       ")
    print("========================================")
    
    if os.path.exists(target):
        print(f"[{target}] 폴더 및 하위 파일 전체 삭제 중...")
        try:
            shutil.rmtree(target)
            print("[+] 기존 dataset 폴더가 성공적으로 제거되었습니다.")
        except Exception as e:
            print(f"[-] dataset 폴더 삭제 실패: {e}")
    else:
        print("[*] 기존 dataset 폴더가 존재하지 않습니다.")

    # 새 데이터셋 폴더 준비
    os.makedirs(os.path.join(target, "raw"), exist_ok=True)
    print("[+] dataset/raw 폴더가 생성되었습니다.")
    print("========================================")

if __name__ == "__main__":
    clear_dataset()
