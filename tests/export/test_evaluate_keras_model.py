import tensorflow as tf

import common.utils as utils
import antispoofing_pipeline.keras.contracts.model_signature as model_signature
import antispoofing_pipeline.keras.models.model as keras_models
from common.evaluate_tflite import evaluate_keras_model


def test_evaluate_keras_model_strips_auxiliary_heads(monkeypatch):
    raw_model = object()
    deploy_model = object()
    extract_calls = []
    validated_models = []

    monkeypatch.setattr(tf.keras.models, "load_model", lambda *a, **k: raw_model)

    def fake_extract_deploy_model(model):
        extract_calls.append(model)
        return deploy_model

    monkeypatch.setattr(keras_models, "extract_deploy_model", fake_extract_deploy_model)
    monkeypatch.setattr(
        model_signature,
        "validate_keras_model_signature",
        lambda model, model_type: validated_models.append(model),
    )
    monkeypatch.setattr(utils, "validate_fixed_split_coverage", lambda data_dir: {})
    monkeypatch.setattr(utils, "collect_split_items", lambda data_dir, split: [])

    evaluate_keras_model("fake.keras", "dataset/raw", "validation", "crop_ir")

    # 로드한 원본 체크포인트가 extract_deploy_model에 전달되고,
    # 서명 검증은 보조 헤드가 제거된 배포 모델에 대해 수행되어야 한다.
    assert extract_calls == [raw_model]
    assert validated_models == [deploy_model]
