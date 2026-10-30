import pytest
import numpy as np
import tensorflow as tf

from common.classes import CLASS_NAMES
from antispoofing_pipeline.keras.training.train import (
    _build_optimizer,
    _print_validation_results,
    _set_backbone_trainable,
    _merge_histories,
    CombinedHistory,
    AcerCheckpoint,
    TrainingProgbarLogger,
    parse_args,
)
from antispoofing_pipeline.keras.models.model import build_single_model, build_dual_model


pytestmark = pytest.mark.tensorflow


@pytest.mark.parametrize("entrypoint", ["train", "model"])
@pytest.mark.parametrize("weights", ["imagenet", "none"])
def test_keras_weights_option(monkeypatch, entrypoint, weights):
    from antispoofing_pipeline.keras.models.model import parse_args as parse_model_args

    parse = parse_args if entrypoint == "train" else parse_model_args
    monkeypatch.setattr("sys.argv", [entrypoint, "--weights", weights])
    assert parse().weights == weights

    monkeypatch.setattr("sys.argv", [entrypoint])
    assert parse().weights == "imagenet"


@pytest.mark.parametrize("entrypoint", ["train", "model"])
@pytest.mark.parametrize("arguments", [
    ["--conv1-reduction", "mean"],
    ["--conv1-reduction", "sum"],
    ["--no-gray-imagenet-init"],
    ["--rgb-weights", "none"],
    ["--weights", "invalid"],
])
def test_keras_rejects_removed_or_invalid_weights_options(monkeypatch, entrypoint, arguments):
    from antispoofing_pipeline.keras.models.model import parse_args as parse_model_args

    parse = parse_args if entrypoint == "train" else parse_model_args
    monkeypatch.setattr("sys.argv", [entrypoint, *arguments])
    with pytest.raises(SystemExit):
        parse()


def test_build_optimizer():
    # 1. AdamW with weight decay
    opt_adamw = _build_optimizer("adamw", learning_rate=1e-3, weight_decay=0.02)
    assert float(opt_adamw.learning_rate) == pytest.approx(1e-3)
    if hasattr(opt_adamw, "weight_decay"):
        wd = opt_adamw.weight_decay
        wd_val = float(wd.numpy()) if hasattr(wd, "numpy") else float(wd)
        assert wd_val == pytest.approx(0.02)

    # 2. AdamW with EMA
    opt_ema = _build_optimizer("adamw", learning_rate=1e-3, weight_decay=0.01, use_ema=True, ema_momentum=0.95)
    assert getattr(opt_ema, "use_ema", False) is True
    assert getattr(opt_ema, "ema_momentum", None) == pytest.approx(0.95)

    # 3. Standard Adam
    opt_adam = _build_optimizer("adam", learning_rate=2e-4)
    assert opt_adam.learning_rate.numpy() == pytest.approx(2e-4)

    # 4. Unknown optimizer raises ValueError
    with pytest.raises(ValueError, match="지원하지 않는 옵티마이저"):
        _build_optimizer("invalid_opt", learning_rate=1e-3)


def test_set_backbone_trainable_single():
    # Single crop_ir model
    model = build_single_model(input_type="crop_ir", weights=None, backbone="mobilenetv2")
    initial_trainable = len(model.trainable_variables)
    assert initial_trainable > 4

    # Freeze backbone
    _set_backbone_trainable(model, trainable=False)
    frozen_trainable = len(model.trainable_variables)
    assert frozen_trainable == 4  # Only classifier head Dense layers (weights + bias = 4)

    # Unfreeze backbone
    _set_backbone_trainable(model, trainable=True)
    unfrozen_trainable = len(model.trainable_variables)
    assert unfrozen_trainable == initial_trainable


def test_set_backbone_trainable_dual():
    # Dual model
    model = build_dual_model(weights=None, backbone="mobilenetv2")
    initial_trainable = len(model.trainable_variables)
    assert initial_trainable > 4

    # Freeze backbone
    _set_backbone_trainable(model, trainable=False)
    frozen_trainable = len(model.trainable_variables)
    assert frozen_trainable == 4  # Only classifier head Dense layers

    # Unfreeze backbone
    _set_backbone_trainable(model, trainable=True)
    unfrozen_trainable = len(model.trainable_variables)
    assert unfrozen_trainable == initial_trainable


def test_combined_history():
    class DummyHistory:
        def __init__(self, d):
            self.history = d

    h1 = DummyHistory({"loss": [0.5, 0.4], "acc": [0.8, 0.85]})
    h2 = DummyHistory({"loss": [0.3, 0.2, 0.1], "acc": [0.9, 0.95, 0.98]})

    merged = _merge_histories(h1, h2)
    assert merged.history["loss"] == [0.5, 0.4, 0.3, 0.2, 0.1]
    assert merged.history["acc"] == [0.8, 0.85, 0.9, 0.95, 0.98]

    # Single history wrapper
    single_merged = _merge_histories(h1, None)
    assert single_merged.history["loss"] == [0.5, 0.4]


@pytest.mark.parametrize("model_type", ["crop_ir", "crop_rgb", "dual"])
@pytest.mark.parametrize("freeze_epochs", [0, 1, 2])
def test_training_keeps_augmentation_counter_across_stages(monkeypatch, tmp_path, model_type, freeze_epochs):
    from antispoofing_pipeline.keras.data import dataset
    from antispoofing_pipeline.keras.training import train

    # Encode the real augmentation index in synthetic pixels; record only batches
    # consumed by train_step, not speculative map/prefetch calls.
    sample_params = dataset._sample_augmentation_params

    def indexed_params(index, *args, **kwargs):
        values = list(sample_params(index, *args, **kwargs))
        values[2] = tf.cast(index, tf.float32)
        return tuple(values)

    def synthetic_single(path, input_type, brightness_f, **kwargs):
        channels = 1 if input_type == "crop_ir" else 3
        return np.full((224, 224, channels), brightness_f, dtype=np.float32)

    def synthetic_dual(rgb_path, ir_path, brightness_f, **kwargs):
        return tuple(np.full((224, 224, c), brightness_f, dtype=np.float32) for c in (3, 1))

    monkeypatch.setattr(dataset, "_sample_augmentation_params", indexed_params)
    monkeypatch.setattr(dataset, "load_single_sample", synthetic_single)
    monkeypatch.setattr(dataset, "load_sample", synthetic_dual)
    # Five items with batch size two exercises ceil(N / batch_size): each epoch
    # consumes six samples from the repeated stream, including across stages.
    items = [("synthetic-rgb", "synthetic-ir", 0)] * 5
    monkeypatch.setattr(train, "validate_fixed_split_coverage", lambda _: {"test": 0})
    monkeypatch.setattr(train, "collect_split_items", lambda *args: items)
    monkeypatch.setattr(train, "_save_learning_curves", lambda *args: None)

    class Checkpoint(tf.keras.callbacks.Callback):
        best_metrics = None
        acer_history = []

    monkeypatch.setattr(train, "AcerCheckpoint", lambda **kwargs: Checkpoint())
    monkeypatch.setattr(train, "TrainingProgbarLogger", tf.keras.callbacks.Callback)
    seen = []

    class Recorder(tf.keras.Model):
        def compile(self, **kwargs):
            # The observation hook uses py_function, which XLA cannot compile.
            super().compile(jit_compile=False, **kwargs)

        def train_step(self, data):
            inputs, _ = data
            images = inputs[0] if model_type == "dual" else inputs
            tf.py_function(
                lambda values: seen.extend(values.numpy().astype(int).tolist()),
                [images[:, 0, 0, 0]], [],
            )
            return super().train_step(data)

    def build_model(**kwargs):
        if model_type == "dual":
            inputs = [tf.keras.Input((224, 224, c)) for c in (3, 1)]
            pooled = tf.keras.layers.Concatenate()([
                tf.keras.layers.GlobalAveragePooling2D()(x) for x in inputs
            ])
        else:
            channels = 1 if model_type == "crop_ir" else 3
            inputs = tf.keras.Input((224, 224, channels))
            pooled = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        return Recorder(inputs, tf.keras.layers.Dense(len(CLASS_NAMES))(pooled))

    monkeypatch.setattr(train, "build_single_model", build_model)
    monkeypatch.setattr(train, "build_dual_model", build_model)
    monkeypatch.setattr("sys.argv", [
        "train", "--model-type", model_type, "--weights", "none",
        "--epochs", "3", "--freeze-backbone-epochs", str(freeze_epochs),
        "--batch-size", "2", "--output-dir", str(tmp_path),
    ])
    train.main()
    assert seen == list(range(18))


def test_training_progbar_omits_duplicate_epoch_end_output(capsys):
    progress = TrainingProgbarLogger()
    progress.set_params({"verbose": 1, "epochs": 1, "steps": 1})

    progress.on_epoch_begin(0)
    progress.on_train_batch_end(0, {"loss": 0.5})
    output_after_batch = capsys.readouterr().out

    progress.on_epoch_end(0, {"loss": 0.5})
    assert capsys.readouterr().out == ""
    assert "1/1" in output_after_batch


def test_print_validation_results_is_grouped_and_aligned(capsys):
    cm = np.eye(len(CLASS_NAMES), dtype=np.int32)
    recalls = np.linspace(0.5, 1.0, len(CLASS_NAMES))

    _print_validation_results(cm, recalls, 0.9, 0.1, 0.2, 0.15)
    output = capsys.readouterr().out

    assert "[Validation Results]" in output
    assert "Confusion matrix (rows=true, cols=pred)" in output
    assert "Class recall" in output
    assert "live             0.5000" in output
    assert "Accuracy 0.9000 | APCER 0.1000 | BPCER 0.2000 | ACER 0.1500" in output


def test_acer_checkpoint_ema_swap(tmp_path):
    # Dummy simple model with EMA optimizer
    inputs = tf.keras.Input(shape=(4,))
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES))(inputs)
    model = tf.keras.Model(inputs, outputs)

    opt = _build_optimizer("adamw", learning_rate=0.1, weight_decay=0.01, use_ema=True, ema_momentum=0.9)
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    x = np.random.randn(20, 4).astype(np.float32)
    y = np.random.randint(0, len(CLASS_NAMES), size=(20,)).astype(np.int32)
    val_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(10)

    out_ckpt = str(tmp_path / "test_model.keras")
    ckpt = AcerCheckpoint(val_ds=val_ds, output_path=out_ckpt)

    # Fit 1 step to populate variables & EMA
    model.fit(val_ds, epochs=2, callbacks=[ckpt], verbose=0)

    assert len(ckpt.acer_history) == 2
    assert ckpt.best_metrics is not None
