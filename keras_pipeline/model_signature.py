from classes import CLASS_NAMES


MODEL_INPUT_SIGNATURES = {
    "dual": (("a_rgb", (224, 224, 3)), ("b_ir", (224, 224, 1))),
    "crop_rgb": (("a_crop_rgb", (224, 224, 3)),),
    "crop_ir": (("b_crop_ir", (224, 224, 1)),),
}


def _expected_inputs(model_type):
    try:
        return MODEL_INPUT_SIGNATURES[model_type]
    except KeyError as error:
        raise ValueError(f"알 수 없는 model type: {model_type}") from error


def _shape_tuple(shape):
    if hasattr(shape, "as_list"):
        shape = shape.as_list()
    return tuple(shape)


def _logical_input_name(name):
    name = name.split(":", 1)[0]
    if name.startswith("serving_default_"):
        name = name.removeprefix("serving_default_")
    return name


def _validate_inputs(inputs, model_type, tflite=False):
    expected_inputs = _expected_inputs(model_type)
    if len(inputs) != len(expected_inputs):
        raise ValueError(
            f"{model_type} 모델은 입력 {len(expected_inputs)}개여야 하지만 {len(inputs)}개입니다."
        )

    expected_by_name = dict(expected_inputs)
    actual_names = []
    for item in inputs:
        name = _logical_input_name(item["name"] if tflite else item.name)
        actual_names.append(name)
        if name not in expected_by_name:
            raise ValueError(f"{model_type} 모델에 허용되지 않은 입력 이름이 있습니다: {name}")

        shape = _shape_tuple(item["shape"] if tflite else item.shape)
        expected_shape = expected_by_name[name]
        if len(shape) != len(expected_shape) + 1 or shape[1:] != expected_shape:
            raise ValueError(
                f"{model_type} 입력 {name}의 shape는 "
                f"(batch, {', '.join(map(str, expected_shape))})여야 하지만 {shape}입니다."
            )

    expected_names = set(expected_by_name)
    if set(actual_names) != expected_names:
        raise ValueError(
            f"{model_type} 입력 이름은 {sorted(expected_names)}여야 하지만 {sorted(actual_names)}입니다."
        )


def _validate_outputs(outputs, model_type, tflite=False):
    if len(outputs) != 1:
        raise ValueError(f"{model_type} 모델은 출력 1개여야 하지만 {len(outputs)}개입니다.")

    shape = _shape_tuple(outputs[0]["shape"] if tflite else outputs[0].shape)
    if len(shape) != 2 or shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            f"{model_type} 모델 출력 shape는 (batch, {len(CLASS_NAMES)})여야 하지만 {shape}입니다."
        )


def validate_keras_model_signature(model, model_type):
    """Validate a loaded Keras model before TFLite conversion."""
    _validate_inputs(model.inputs, model_type)
    _validate_outputs(model.outputs, model_type)


def validate_tflite_model_signature(input_details, output_details, model_type):
    """Validate TFLite interpreter tensor details before an artifact is accepted."""
    _validate_inputs(input_details, model_type, tflite=True)
    _validate_outputs(output_details, model_type, tflite=True)
