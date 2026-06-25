import os
import sys

print("========================================")
print("     AI Model Setup Verification        ")
print("========================================")

# 1. Check Python version
print(f"Python Version: {sys.version}")

# 2. Check Torch & Torchvision
try:
    import torch
    import torchvision
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Torchvision Version: {torchvision.__version__}")
    print(f"CUDA Available (GPU Support): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
         print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"[-] PyTorch or Torchvision is not installed: {e}")

# 3. Check OpenCV
try:
    import cv2
    print(f"OpenCV Version: {cv2.__version__}")
except ImportError as e:
    print(f"[-] OpenCV is not installed: {e}")

# 4. Check Matplotlib & tqdm
for lib_name in ["matplotlib", "tqdm"]:
    try:
        lib = __import__(lib_name)
        print(f"{lib_name.capitalize()} Version: {getattr(lib, '__version__', 'Installed')}")
    except ImportError as e:
        print(f"[-] {lib_name.capitalize()} is not installed: {e}")

# 5. Check litert_torch (TFLite deployment path)
try:
    import litert_torch
    print(f"litert_torch: {getattr(litert_torch, '__version__', 'Installed')}")
except ImportError as e:
    print(f"[-] litert_torch is not installed (needed for convert_to_tflite.py): {e}")

print("========================================")
print("Verification complete.")
