import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np
import torch
from pytorch_pipeline.dataset import get_data_loaders
from classes import CLASS_NAMES
from utils import calculate_validation_metrics

def verify_tflite_quantization(tflite_path):
    print(f"\n==========================================")
    print(f"[양자화 모델 검증 시작] 모델 파일: {tflite_path}")
    print(f"==========================================")
    
    if not os.path.exists(tflite_path):
        print(f"[-] 에러: {tflite_path} 파일이 존재하지 않습니다.")
        return

    import ai_edge_litert.interpreter as litert

    # 1. Interpreter 초기화
    interpreter = litert.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"[*] 모델 입력 텐서 정보:")
    for idx, details in enumerate(input_details):
        print(f"  - Input {idx}: Name={details['name']}, Shape={details['shape']}, Type={details['dtype']}")
        
    print(f"[*] 모델 출력 텐서 정보:")
    for idx, details in enumerate(output_details):
        print(f"  - Output {idx}: Name={details['name']}, Shape={details['shape']}, Type={details['dtype']}")

    # 입력 텐서 매핑 매칭
    rgb_detail = None
    ir_detail = None
    for details in input_details:
        shape = details['shape']
        if len(shape) == 4:
            if shape[3] == 3:
                rgb_detail = details
            elif shape[3] == 1:
                ir_detail = details

    if rgb_detail is None or ir_detail is None:
        print("[-] 에러: RGB([*, 224, 224, 3]) 및 IR([*, 224, 224, 1]) 입력 구조를 판별할 수 없습니다.")
        return

    # 2. Calibration / Validation 데이터 로드
    print("[*] 검증용 데이터셋 로드 중...")
    _, val_loader = get_data_loaders(
        "dataset/raw",
        batch_size=1,  # TFLite는 일반적으로 배치 크기 1로 추론하므로 1로 설정
        k_folds=5,
        fold_idx=0,
        num_workers=1
    )

    all_labels = []
    all_preds = []
    
    # 3. 양자화 도메인 변환 헬퍼 (Int8일 경우)
    def to_quantized(data, details):
        if details['dtype'] == np.int8 or details['dtype'] == np.uint8:
            scale, zero_point = details['quantization']
            if scale == 0.0:
                scale = 1.0
            q_data = np.round(data / scale) + zero_point
            # dtype 범위 내로 클리핑
            min_val = -128 if details['dtype'] == np.int8 else 0
            max_val = 127 if details['dtype'] == np.int8 else 255
            return np.clip(q_data, min_val, max_val).astype(details['dtype'])
        return data.astype(np.float32)

    def from_quantized(data, details):
        if details['dtype'] == np.int8 or details['dtype'] == np.uint8:
            scale, zero_point = details['quantization']
            return (data.astype(np.float32) - zero_point) * scale
        return data

    print("[*] 추론 테스트 진행 중...")
    count = 0
    distribution = {}

    for rgb, ir, label in val_loader:
        # NCHW -> NHWC 변환
        rgb_np = rgb.permute(0, 2, 3, 1).numpy()
        ir_np = ir.permute(0, 2, 3, 1).numpy()

        # 양자화 스케일 적용
        rgb_input = to_quantized(rgb_np, rgb_detail)
        ir_input = to_quantized(ir_np, ir_detail)

        # 텐서 설정
        interpreter.set_tensor(rgb_detail['index'], rgb_input)
        interpreter.set_tensor(ir_detail['index'], ir_input)
        
        # 추론
        interpreter.invoke()

        # 출력값 획득
        output_data = interpreter.get_tensor(output_details[0]['index'])
        # 양자화 출력 역변환
        output_float = from_quantized(output_data, output_details[0])

        pred = int(np.argmax(output_float, axis=1)[0])
        label_val = int(label[0])

        all_labels.append(label_val)
        all_preds.append(pred)

        count += 1
        # 실시간 진행 출력 (최대 100개만 진행)
        if count >= 100:
            break

    # 4. 결과 분석 및 요약
    all_preds_arr = np.array(all_preds)
    unique, counts = np.unique(all_preds_arr, return_counts=True)
    distribution = dict(zip(unique, counts))

    print(f"\n==========================================")
    print(f"[검증 요약] 총 {count}개 샘플 평가 완료")
    print(f"==========================================")
    print(f"[*] 예측 클래스별 분포:")
    for class_idx in range(len(CLASS_NAMES)):
        class_name = CLASS_NAMES[class_idx]
        freq = distribution.get(class_idx, 0)
        percentage = (freq / count) * 100
        print(f"  - {class_name} (클래스 {class_idx}): {freq}회 ({percentage:.2f}%)")

    # APCER, BPCER, ACER 계산
    confusion_matrix, recalls, apcer, bpcer, acer = calculate_validation_metrics(all_labels, all_preds)
    
    print(f"\n[*] 성능 평가지표 (샘플 수={count}):")
    print(" -> Confusion Matrix (행=실제, 열=예측):")
    print(confusion_matrix)
    print(" -> 클래스별 Recall:")
    for class_name, recall in zip(CLASS_NAMES, recalls):
        print(f"    {class_name}: {recall:.4f}")
    print(f" -> APCER (스푸핑 오검증율): {apcer:.4f}")
    print(f" -> BPCER (라이브 오검증율): {bpcer:.4f}")
    print(f" -> ACER (평균 오검증율): {acer:.4f}")
    print(f"==========================================")

    # 모델 붕괴 유무 판단
    for class_idx, freq in distribution.items():
        if freq >= count * 0.90:
            print(f"\n[⚠️ 경고] 모델 붕괴 감지! {CLASS_NAMES[class_idx]} 클래스 쏠림 비율: {(freq/count)*100:.1f}%")
            print(" -> 가중치 양자화 단계에서 활성화 오차가 과도하게 반영되어 모든 입력을 특정 클래스로 고정 출력하고 있습니다.")
            return

    print("\n[정상] 예측이 골고루 분산되어 있으며 출력 붕괴 현상이 발견되지 않았습니다!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify if TFLite model output collapses into one class.")
    parser.add_argument("--tflite-path", default="model/anti_spoofing.tflite", help="Path to compiled TFLite model")
    args = parser.parse_args()
    verify_tflite_quantization(args.tflite_path)
