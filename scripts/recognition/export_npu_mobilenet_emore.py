import os
import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from calibration import (
    CALIBRATION_SAMPLES,
    collect_live_face_paths,
    load_normalized_face_image,
    select_stratified_face_paths,
)

# Dynamic path resolution relative to repository root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
WEIGHTS_PATH = os.path.join(REPO_ROOT, "model", "recognition", "mobilenet_emb256.h5")
OUT_DIR = os.path.join(REPO_ROOT, "model", "recognition", "tflite")
OUT_PATH = os.path.join(OUT_DIR, "mobilenet_emore_npu_int8.tflite")
DATASET_ROOT = os.path.join(REPO_ROOT, "dataset", "raw")
EXPECTED_INT8_OPERATOR_COUNT = 31
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(f"Recognition weights file not found: {WEIGHTS_PATH}")

f = h5py.File(WEIGHTS_PATH, "r")

# Rule 1: Fixed batch size 1
inputs = keras.Input(batch_shape=(1, 112, 112, 3), name="input_1")

# Backbone
base = keras.applications.MobileNet(input_shape=(112, 112, 3), include_top=False, weights=None)
x = base(inputs)  # Output (1, 3, 3, 1024)

# Rule 2: Explicit 3x3 Depthwise Conv (kernel 3x3 on 3x3 feature map -> 1x1x1024)
dw = layers.DepthwiseConv2D(kernel_size=(3, 3), padding="valid", use_bias=False, name="depthwise_conv2d")(x)
bn1 = layers.BatchNormalization(name="batch_normalization")(dw)

# Rule 3: 1x1 Conv (1x1x1024 -> 1x1x256)
conv_head = layers.Conv2D(256, kernel_size=(1, 1), use_bias=True, name="conv2d")(bn1)
emb_bn = layers.BatchNormalization(scale=False, name="embedding")(conv_head)

# Rule 4: Reshape to [1, 256] instead of Flatten
out = layers.Reshape((256,), name="out_embedding")(emb_bn)

m = keras.Model(inputs=inputs, outputs=out, name="mobilenet_emore_npu")

def get_layer_weights(layer_name):
    if layer_name not in f["model_weights"]:
        return None
    grp = f["model_weights"][layer_name]
    sub = list(grp.keys())[0] if len(grp.keys()) > 0 else None
    if not sub:
        return None
    sub_grp = grp[sub]
    w_keys = list(sub_grp.keys())

    if "kernel:0" in w_keys and "bias:0" in w_keys:
        return [np.array(sub_grp["kernel:0"]), np.array(sub_grp["bias:0"])]
    elif "kernel:0" in w_keys:
        return [np.array(sub_grp["kernel:0"])]
    elif "depthwise_kernel:0" in w_keys:
        return [np.array(sub_grp["depthwise_kernel:0"])]
    elif "gamma:0" in w_keys:
        return [
            np.array(sub_grp["gamma:0"]),
            np.array(sub_grp["beta:0"]),
            np.array(sub_grp["moving_mean:0"]),
            np.array(sub_grp["moving_variance:0"])
        ]
    elif "beta:0" in w_keys:
        return [
            np.array(sub_grp["beta:0"]),
            np.array(sub_grp["moving_mean:0"]),
            np.array(sub_grp["moving_variance:0"])
        ]
    return None

for layer in m.layers:
    w = get_layer_weights(layer.name)
    if w is not None:
        layer.set_weights(w)

for layer in base.layers:
    w = get_layer_weights(layer.name)
    if w is not None:
        layer.set_weights(w)

print("Weights loaded into NPU-compliant model.")

# Calibration dataset: real live faces from both fixed training splits.
samples_by_subject = collect_live_face_paths(DATASET_ROOT)
calibration_paths, selected_by_subject = select_stratified_face_paths(
    samples_by_subject,
    CALIBRATION_SAMPLES,
)
print(
    f"Selected {len(calibration_paths)} calibration images from "
    f"{len(selected_by_subject)} live subjects."
)

def rep_gen():
    for image_path in calibration_paths:
        yield [load_normalized_face_image(image_path)]


def verify_int8_operator_graph(tflite_model):
    """Reject an export that deviates from the 31-node all-INT8 NPU contract."""
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    op_details = interpreter._get_ops_details()
    tensor_dtypes = {
        tensor["index"]: tensor["dtype"]
        for tensor in interpreter.get_tensor_details()
    }
    floating_point_ops = []
    for op_index, op in enumerate(op_details):
        tensor_indices = [
            index
            for index in (*op["inputs"], *op["outputs"])
            if index >= 0
        ]
        # INT32 bias and shape tensors are required by otherwise quantized TFLite
        # operators. A floating-point tensor is the actual fallback indicator.
        if any(np.issubdtype(tensor_dtypes[index], np.floating) for index in tensor_indices):
            floating_point_ops.append((op_index, op["op_name"]))

    io_dtypes = [
        detail["dtype"]
        for detail in (*interpreter.get_input_details(), *interpreter.get_output_details())
    ]
    if (
        len(op_details) != EXPECTED_INT8_OPERATOR_COUNT
        or floating_point_ops
        or any(dtype != np.int8 for dtype in io_dtypes)
    ):
        raise RuntimeError(
            "Recognition INT8 export contract failed: "
            f"expected {EXPECTED_INT8_OPERATOR_COUNT}/{EXPECTED_INT8_OPERATOR_COUNT} "
            f"INT8 operators, found {len(op_details) - len(floating_point_ops)}/{len(op_details)}; "
            f"floating-point={floating_point_ops}, io_dtypes={io_dtypes}"
        )
    print(
        f"Verified {len(op_details)}/{len(op_details)} TFLite operators use INT8 tensors."
    )

converter = tf.lite.TFLiteConverter.from_keras_model(m)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_int8 = converter.convert()
verify_int8_operator_graph(tflite_int8)
with open(OUT_PATH, "wb") as f_out:
    f_out.write(tflite_int8)

print(f"Exported NPU-compliant INT8 model ({len(tflite_int8) / (1024 * 1024):.2f} MB) to {OUT_PATH}")
