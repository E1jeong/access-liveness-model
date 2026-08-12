"""모델 입출력 텐서 규격 검문소.

spec.py의 MODEL_INPUT_SIGNATURES에 적힌 계약(입력 개수·이름·shape, 출력 shape)을
실제 모델이 지키는지 확인한다. 검사 항목은 셋이다.

  dual 모델이라면
    입력 개수  : 정확히 2개
    입력 이름  : {a_rgb, b_ir}
    입력 shape : (batch, 224, 224, 3), (batch, 224, 224, 1)
    출력       : 1개, (batch, 10)

왜 필요한가: 입력 개수·이름·shape가 계약과 다른 모델을 배포 전에 거부하기 위해서다.
이 검사는 입력을 이름으로 대조하므로 리스트 순서 자체는 검사하지 않는다. 입력 순서는
spec.py의 a_/b_ 접두사와 TFLite 변환 결과로 관리하며, inspect_tflite 로그에서 별도로 확인한다.

같은 검사를 Keras 모델과 TFLite 파일 양쪽에서 하는 이유도 같다 — 변환 전에 한 번,
변환 후에 한 번 확인해야 변환 단계에서 생긴 어긋남을 잡을 수 있다.
"""
from classes import CLASS_NAMES
from keras_pipeline.spec import MODEL_INPUT_SIGNATURES


# 모델 종류에 해당하는 계약 튜플을 꺼낸다. 오타 난 model_type이 조용히 통과하지 않도록
# KeyError를 의미가 분명한 ValueError로 바꿔 던진다.
def _expected_inputs(model_type):
    try:
        return MODEL_INPUT_SIGNATURES[model_type]
    except KeyError as error:
        raise ValueError(f"알 수 없는 model type: {model_type}") from error


# shape 표현을 파이썬 튜플 하나로 통일한다.
# Keras는 TensorShape 객체(as_list()를 가짐)를, TFLite는 numpy 배열을 주기 때문에
# 그대로 비교하면 같은 shape인데도 다르다고 판정될 수 있다.
def _shape_tuple(shape):
    if hasattr(shape, "as_list"):
        shape = shape.as_list()
    return tuple(shape)


# TFLite 변환기가 붙인 장식을 떼고 원래 이름만 남긴다.
# 예: "serving_default_a_rgb:0"  →  "a_rgb"
# ":0"은 텐서 출력 인덱스, "serving_default_"는 서명 이름에서 온 접두사다.
# Keras 쪽 이름에는 둘 다 없으므로 이 함수를 통과해도 그대로 나온다.
def _logical_input_name(name):
    name = name.split(":", 1)[0]
    if name.startswith("serving_default_"):
        name = name.removeprefix("serving_default_")
    return name


# 입력 텐서 검사 본체. Keras와 TFLite 양쪽에서 재사용하며, tflite 플래그로
# 속성 접근 방식만 갈아 끼운다(Keras는 item.name / TFLite는 item["name"]).
def _validate_inputs(inputs, model_type, tflite=False):
    expected_inputs = _expected_inputs(model_type)
    # ① 개수 검사 — dual인데 입력이 1개면 잘못된 모델 타입으로 변환한 것이다.
    if len(inputs) != len(expected_inputs):
        raise ValueError(
            f"{model_type} 모델은 입력 {len(expected_inputs)}개여야 하지만 {len(inputs)}개입니다."
        )

    expected_by_name = dict(expected_inputs)  # 예: {"a_rgb": (224,224,3), "b_ir": (224,224,1)}
    actual_names = []
    for item in inputs:
        name = _logical_input_name(item["name"] if tflite else item.name)
        actual_names.append(name)
        # ② 이름 검사 — 계약에 없는 이름이면 즉시 실패.
        if name not in expected_by_name:
            raise ValueError(f"{model_type} 모델에 허용되지 않은 입력 이름이 있습니다: {name}")

        # ③ shape 검사. 계약에는 배치 축이 빠져 있으므로(=(224,224,3)),
        #    실제 shape은 그보다 한 축 많아야 하고 뒷부분이 정확히 일치해야 한다.
        #    예: (1,224,224,3) → len 4 == 3+1 이고 [1:] == (224,224,3) → 통과.
        #    배치 크기 자체는 검사하지 않는다(학습은 동적 None, NPU export는 1로 고정).
        shape = _shape_tuple(item["shape"] if tflite else item.shape)
        expected_shape = expected_by_name[name]
        if len(shape) != len(expected_shape) + 1 or shape[1:] != expected_shape:
            raise ValueError(
                f"{model_type} 입력 {name}의 shape는 "
                f"(batch, {', '.join(map(str, expected_shape))})여야 하지만 {shape}입니다."
            )

    # ④ 집합 비교 — 위 루프는 "허용되지 않은 이름"만 잡는다. 같은 이름이 두 번 나오고
    #    다른 하나가 통째로 빠진 경우는 개수·이름 검사를 모두 통과해 버리므로,
    #    집합이 정확히 같은지 마지막에 한 번 더 확인한다.
    expected_names = set(expected_by_name)
    if set(actual_names) != expected_names:
        raise ValueError(
            f"{model_type} 입력 이름은 {sorted(expected_names)}여야 하지만 {sorted(actual_names)}입니다."
        )


# 출력은 모델 종류와 무관하게 항상 "logits 1개, (batch, 클래스 수)"다.
# 출력 이름은 검사하지 않는다 — 앱은 출력이 하나뿐이라 이름 없이 인덱스로 읽는다.
def _validate_outputs(outputs, model_type, tflite=False):
    if len(outputs) != 1:
        raise ValueError(f"{model_type} 모델은 출력 1개여야 하지만 {len(outputs)}개입니다.")

    # 마지막 축이 CLASS_NAMES 길이와 다르면 클래스 수가 바뀐 채 변환된 것이다.
    shape = _shape_tuple(outputs[0]["shape"] if tflite else outputs[0].shape)
    if len(shape) != 2 or shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            f"{model_type} 모델 출력 shape는 (batch, {len(CLASS_NAMES)})여야 하지만 {shape}입니다."
        )


# 진입점 1: 변환 '전'. convert_keras_to_tflite.main()이 .keras를 로드한 직후 호출한다.
# 여기서 걸리면 애초에 잘못 학습된 모델이므로 변환을 시작하지 않는다.
def validate_keras_model_signature(model, model_type):
    """불러온 Keras 모델의 서명을 TFLite 변환 전에 검사한다."""
    _validate_inputs(model.inputs, model_type)
    _validate_outputs(model.outputs, model_type)


# 진입점 2: 변환 '후'. 아직 파일로 쓰기 전의 tflite 바이트를 인터프리터에 올려 검사한다
# (_validate_tflite_bytes 참고). 여기서 걸리면 변환 과정이 서명을 망가뜨린 것이다.
def validate_tflite_model_signature(input_details, output_details, model_type):
    """산출물을 채택하기 전에 TFLite 인터프리터의 텐서 정보를 검사한다."""
    _validate_inputs(input_details, model_type, tflite=True)
    _validate_outputs(output_details, model_type, tflite=True)
