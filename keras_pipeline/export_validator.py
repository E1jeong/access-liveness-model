import json
import os
import numpy as np
import tensorflow as tf

from classes import CLASS_NAMES
from keras_pipeline.model_signature import validate_tflite_model_signature
from keras_pipeline.tf_dataset import RGB_MEAN, RGB_STD


def _rgb_imagenet_norm_to_mobilenet_range(rgb):
    raw_0_1 = rgb * RGB_STD + RGB_MEAN
    return raw_0_1 * 2.0 - 1.0


def _copy_nested_weights(source_model, target_model, layer_name):
    source_layer = source_model.get_layer(layer_name)
    target_layer = target_model.get_layer(layer_name)

    source_sub_layers = {layer.name: layer for layer in source_layer.layers}
    target_weighted_layers = [layer for layer in target_layer.layers if layer.get_weights()]
    source_weighted_names = {
        layer.name for layer in source_layer.layers if layer.get_weights()
    }
    target_weighted_names = {layer.name for layer in target_weighted_layers}
    missing_in_target = source_weighted_names - target_weighted_names
    if missing_in_target:
        raise ValueError(
            f"{layer_name}: export backbone에 없는 source weighted layer: "
            f"{sorted(missing_in_target)}"
        )

    copied = []
    for target_sub_layer in target_layer.layers:
        target_weights = target_sub_layer.get_weights()
        if not target_weights:
            continue

        source_sub_layer = source_sub_layers.get(target_sub_layer.name)
        if source_sub_layer is None:
            raise ValueError(
                f"{layer_name}: source backbone에 없는 export weighted layer: "
                f"{target_sub_layer.name}"
            )
        source_weights = source_sub_layer.get_weights()
        source_shapes = [tuple(weight.shape) for weight in source_weights]
        target_shapes = [tuple(weight.shape) for weight in target_weights]
        if source_shapes != target_shapes:
            raise ValueError(
                f"{layer_name}/{target_sub_layer.name}: weight tensor shape mismatch "
                f"(source={source_shapes}, target={target_shapes})"
            )
        target_sub_layer.set_weights(source_weights)
        copied.append((target_sub_layer.name, source_shapes))

    print(f"[weight copy] {layer_name}: {copied}")


def _npu_parity_inputs(sample, model_type):
    def batch(value):
        return np.expand_dims(value, axis=0).astype(np.float32)

    if model_type == "dual":
        return [batch(sample[0]), batch(sample[1])], [
            batch(_rgb_imagenet_norm_to_mobilenet_range(sample[0])),
            batch(sample[1]),
        ]
    if model_type == "crop_rgb":
        return [batch(sample[0])], [batch(_rgb_imagenet_norm_to_mobilenet_range(sample[0]))]
    if model_type == "crop_ir":
        return [batch(sample[1])], [batch(sample[1])]

    source_inputs = [batch(value) for value in sample]
    export_inputs = [
        batch(_rgb_imagenet_norm_to_mobilenet_range(sample[0])),
        batch(sample[1]),
        batch(_rgb_imagenet_norm_to_mobilenet_range(sample[2])),
        batch(sample[3]),
        batch(sample[4]),
    ]
    return source_inputs, export_inputs


def validate_npu_export_parity(trained_model, export_model, sample, model_type):
    """Fail when the NPU export graph no longer preserves Keras logits."""
    source_inputs, export_inputs = _npu_parity_inputs(sample, model_type)
    source_call_inputs = source_inputs[0] if len(source_inputs) == 1 else source_inputs
    export_call_inputs = export_inputs[0] if len(export_inputs) == 1 else export_inputs
    trained_logits = trained_model(source_call_inputs, training=False).numpy()
    export_logits = export_model(export_call_inputs, training=False).numpy()
    error = np.abs(trained_logits - export_logits)
    result = {
        "max_error": float(error.max()),
        "mean_error": float(error.mean()),
        "argmax_agreement": bool(np.array_equal(
            np.argmax(trained_logits, axis=-1), np.argmax(export_logits, axis=-1)
        )),
    }
    if not result["argmax_agreement"] or not np.allclose(
        trained_logits, export_logits, rtol=1e-5, atol=1e-5
    ):
        raise ValueError(f"NPU export Keras logits parity failed: {result}")
    print(f"[NPU export Keras logits parity] {result}")
    return result


def build_npu_export_model(trained_model, model_type):
    # 학습 체크포인트에서 classifier 구성을 읽는다 (--classifier-units 0 지원).
    try:
        trained_dense = trained_model.get_layer("classifier_dense")
        classifier_units = trained_dense.units
    except ValueError:
        trained_dense = None
        classifier_units = 0

    if model_type == "dual":
        from keras_pipeline.tf_model import build_dual_mobilenetv2
        export_model = build_dual_mobilenetv2(
            rgb_weights=None,
            dropout=0.0,
            classifier_units=classifier_units,
            gray_imagenet_init=False,
            rgb_input_mobilenet_range=True,
            average_pool_op=True,
            fixed_batch_size=1,
            classifier_as_conv=True,
        )
        backbones = ["rgb_mobilenetv2", "ir_mobilenetv2"]
    else:
        from keras_pipeline.tf_model import build_single_mobilenetv2
        export_model = build_single_mobilenetv2(
            input_type=model_type,
            rgb_weights=None,
            dropout=0.0,
            classifier_units=classifier_units,
            gray_imagenet_init=False,
            rgb_input_mobilenet_range=True,
            average_pool_op=True,
            fixed_batch_size=1,
            classifier_as_conv=True,
        )
        backbones = [f"{model_type}_mobilenetv2"]

    for layer_name in backbones:
        _copy_nested_weights(trained_model, export_model, layer_name)

    # classifier_dense (Dense) -> classifier_dense_conv (Conv2D 1x1)
    if trained_dense is not None:
        export_dense_conv = export_model.get_layer("classifier_dense_conv")
        dense_w, dense_b = trained_dense.get_weights()
        conv_w = np.reshape(dense_w, (1, 1, dense_w.shape[0], dense_w.shape[1]))
        export_dense_conv.set_weights([conv_w, dense_b])

    # logits (Dense) -> logits_conv (Conv2D 1x1)
    trained_logits = trained_model.get_layer("logits")
    export_logits_conv = export_model.get_layer("logits_conv")
    logits_w, logits_b = trained_logits.get_weights()
    logits_conv_w = np.reshape(logits_w, (1, 1, logits_w.shape[0], logits_w.shape[1]))
    export_logits_conv.set_weights([logits_conv_w, logits_b])

    return export_model


def inspect_tflite(path, model_type):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    validate_tflite_model_signature(input_details, output_details, model_type)
    print("[tflite tensors]")
    for idx, detail in enumerate(input_details):
        print(
            f" input {idx}: name={detail['name']} shape={detail['shape'].tolist()} "
            f"dtype={detail['dtype']} quant={detail['quantization']}"
        )
    for idx, detail in enumerate(output_details):
        print(
            f" output {idx}: name={detail['name']} shape={detail['shape'].tolist()} "
            f"dtype={detail['dtype']} quant={detail['quantization']}"
        )


def write_tflite_sidecar_manifest(tflite_path, model_type):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    is_npu_int8 = "npu_int8" in os.path.basename(tflite_path)

    inputs_info = []
    for idx, detail in enumerate(input_details):
        name = detail['name']
        shape = detail['shape'].tolist()
        dtype = detail['dtype'].__name__
        channels = shape[-1]

        input_kind = "unknown"
        if model_type == "crop_rgb":
            input_kind = "rgb"
        elif model_type == "crop_ir":
            input_kind = "ir"
        elif model_type == "dual":
            input_kind = "rgb" if channels == 3 else "ir"

        scale, zero_point = detail['quantization']
        quant = None
        if scale != 0.0 or zero_point != 0:
            quant = {
                "scale": float(scale),
                "zero_point": int(zero_point)
            }

        if channels == 3:
            if is_npu_int8:
                norm = {
                    "mean": [0.5, 0.5, 0.5],
                    "std": [0.5, 0.5, 0.5],
                    "range": "[-1, 1]"
                }
            else:
                norm = {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "range": "imagenet"
                }
        else:
            norm = {
                "mean": [0.5],
                "std": [0.5],
                "range": "[-1, 1]"
            }

        inputs_info.append({
            "name": name,
            "index": idx,
            "shape": shape,
            "dtype": dtype,
            "input_kind": input_kind,
            "quantization": quant,
            "normalization": norm
        })

    outputs_info = []
    for idx, detail in enumerate(output_details):
        name = detail['name']
        shape = detail['shape'].tolist()
        dtype = detail['dtype'].__name__

        scale, zero_point = detail['quantization']
        quant = None
        if scale != 0.0 or zero_point != 0:
            quant = {
                "scale": float(scale),
                "zero_point": int(zero_point)
            }

        outputs_info.append({
            "name": name,
            "index": idx,
            "shape": shape,
            "dtype": dtype,
            "quantization": quant,
            "output_is_logits": True
        })

    manifest = {
        "model_type": model_type,
        "file_name": os.path.basename(tflite_path),
        "delegate": "nnapi" if is_npu_int8 else "cpu",
        "inputs": inputs_info,
        "outputs": outputs_info,
        "class_order": CLASS_NAMES,
        "crop_margin_ratio": 0.10
    }

    manifest_path = tflite_path.replace(".tflite", "_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[tflite sidecar manifest saved] {manifest_path}")
