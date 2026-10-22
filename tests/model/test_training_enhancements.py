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


def test_train_rejects_policy_rejected_conv1_mean_reduction(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train.py", "--conv1-reduction", "mean"])
    with pytest.raises(SystemExit):
        parse_args()

    monkeypatch.setattr("sys.argv", ["train.py"])
    assert parse_args().conv1_reduction == "sum"


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
    model = build_single_model(input_type="crop_ir", rgb_weights=None, backbone="mobilenetv2")
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
    model = build_dual_model(rgb_weights=None, backbone="mobilenetv2")
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
