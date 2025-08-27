#!/usr/bin/env python3
"""
YOLO11n → ONNX → TensorFlow 변환 전체 스크립트
- Dynamic axis 문제 해결
- ONNX simplifier 적용
"""

import os
import numpy as np
from ultralytics import YOLO
import onnx
from onnxsim import simplify
import onnx2tf
import onnxruntime as ort

# -----------------------------
# 1. YOLO11n → ONNX 변환
# -----------------------------
def convert_yolo_to_onnx(model_path="yolo11n.pt"):
    """train 모델을 ONNX로 변환"""
    if not os.path.exists(model_path):
        print(f"모델 파일을 찾을 수 없습니다: {model_path}")
        return None

    print("모델 로딩 중...")
    model = YOLO(model_path)

    print("ONNX로 변환 중 (dynamic=False, simplify=True)...")
    onnx_path = model.export(
        format='onnx',
        dynamic=False,   # dynamic axis 제거
        simplify=True,
        opset=12
    )

    print(f"변환 완료! ONNX 파일: {onnx_path}")
    return onnx_path

# -----------------------------
# 2. ONNX simplifier 적용
# -----------------------------
def simplify_onnx(onnx_path):
    """ONNX simplifier 적용"""
    simplified_path = "yolo11n_simplified.onnx"
    model = onnx.load(onnx_path)
    model_simp, check = simplify(model)
    onnx.save(model_simp, simplified_path)
    print(f"Simplified ONNX 저장 완료: {simplified_path}")
    return simplified_path

# -----------------------------
# 3. ONNX → TensorFlow 변환
# -----------------------------
def convert_onnx_to_tf(onnx_path):
    """ONNX 모델을 TensorFlow로 변환"""
    if not os.path.exists(onnx_path):
        print(f"ONNX 파일을 찾을 수 없습니다: {onnx_path}")
        return None

    output_folder = "train_tf"
    print("ONNX → TensorFlow 변환 시작...")
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=output_folder,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=False,
        output_signaturedefs=True
    )
    print(f"변환 완료! TensorFlow 모델 폴더: {output_folder}")
    return output_folder

# -----------------------------
# 4. TensorFlow 모델 구조 확인
# -----------------------------
def check_tf_model(output_folder):
    """변환된 TensorFlow 모델 확인"""
    if not os.path.exists(output_folder):
        print(f"출력 폴더를 찾을 수 없습니다: {output_folder}")
        return

    print(f"\n=== TensorFlow 모델 구조 확인 ===")
    for root, dirs, files in os.walk(output_folder):
        level = root.replace(output_folder, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")

# -----------------------------
# 5. ONNX 모델 추론 테스트
# -----------------------------
def test_onnx_inference(onnx_path):
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_dtype = np.float32  # fallback

    print(f"ONNX Input: {input_name}, shape: {input_shape}, dtype: {input_dtype}")

    dummy_input = np.random.randn(1, 3, 640, 640).astype(input_dtype)
    outputs = session.run(None, {input_name: dummy_input})
    print(f"ONNX Output shape: {[o.shape for o in outputs]}")

# -----------------------------
# Main 실행
# -----------------------------
if __name__ == "__main__":
    print("=== YOLO11n → ONNX → TensorFlow 변환 시작 ===")
    
    # 1. YOLO → ONNX
    onnx_file = convert_yolo_to_onnx("train.pt")
    if not onnx_file:
        exit(1)

    # 2. ONNX simplifier
    onnx_simp_file = simplify_onnx(onnx_file)

    # 3. ONNX 추론 테스트
    test_onnx_inference(onnx_simp_file)

    # 4. ONNX → TensorFlow
    tf_folder = convert_onnx_to_tf(onnx_simp_file)
    if tf_folder:
        print("\n TensorFlow 변환 성공!")
        check_tf_model(tf_folder)
    else:
        print("\n TensorFlow 변환 실패!")
