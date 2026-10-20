"""Verification of NPU INT8 operator graph contract for face recognition."""

from typing import List, Tuple
import numpy as np
import tensorflow as tf

EXPECTED_INT8_OPERATOR_COUNT = 31
EXPECTED_INPUT_SHAPE = (1, 112, 112, 3)
EXPECTED_OUTPUT_SHAPE = (1, 256)


def verify_int8_operator_graph(
    tflite_model: bytes,
    expected_operator_count: int = EXPECTED_INT8_OPERATOR_COUNT,
    expected_input_shape: Tuple[int, ...] = EXPECTED_INPUT_SHAPE,
    expected_output_shape: Tuple[int, ...] = EXPECTED_OUTPUT_SHAPE,
) -> None:
    """Reject an export that deviates from the all-INT8 NPU contract.

    Verifies:
      1. Exactly `expected_operator_count` operators.
      2. Zero floating-point tensors in inputs/outputs of any operator.
      3. Input and output tensor dtypes are INT8.
      4. Input and output tensor shapes match contract (default: [1, 112, 112, 3] -> [1, 256]).
    """
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    op_details = interpreter._get_ops_details()
    tensor_dtypes = {
        tensor["index"]: tensor["dtype"]
        for tensor in interpreter.get_tensor_details()
    }

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Verify I/O count and shapes
    if len(input_details) != 1 or tuple(input_details[0]["shape"]) != tuple(expected_input_shape):
        actual_input_shape = [tuple(d["shape"]) for d in input_details]
        raise RuntimeError(
            f"Recognition INT8 export shape contract failed: expected 1 input with shape "
            f"{expected_input_shape}, found {actual_input_shape}"
        )

    if len(output_details) != 1 or tuple(output_details[0]["shape"]) != tuple(expected_output_shape):
        actual_output_shape = [tuple(d["shape"]) for d in output_details]
        raise RuntimeError(
            f"Recognition INT8 export shape contract failed: expected 1 output with shape "
            f"{expected_output_shape}, found {actual_output_shape}"
        )

    floating_point_ops: List[Tuple[int, str]] = []
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
        len(op_details) != expected_operator_count
        or floating_point_ops
        or any(dtype != np.int8 for dtype in io_dtypes)
    ):
        raise RuntimeError(
            "Recognition INT8 export contract failed: "
            f"expected {expected_operator_count}/{expected_operator_count} "
            f"INT8 operators, found {len(op_details) - len(floating_point_ops)}/{len(op_details)}; "
            f"floating-point={floating_point_ops}, io_dtypes={io_dtypes}"
        )

    print(
        f"Verified {len(op_details)}/{len(op_details)} TFLite operators conform to NPU INT8 contract."
    )
