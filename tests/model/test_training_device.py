import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.tensorflow
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_training_process_device_policy():
    result = subprocess.run(
        [sys.executable, "-c", """
import json
import sys
import numpy as np
import tensorflow as tf

before = tf.config.get_visible_devices("GPU")
from antispoofing_pipeline.keras.training import train
after = tf.config.get_visible_devices("GPU")
if sys.platform == "darwin":
    assert not after, after
else:
    assert after == before, (before, after)

x = tf.ones((1, 4, 4, 1))
w = tf.Variable(tf.ones((1, 1, 1, 1)))
with tf.GradientTape() as tape:
    y = tf.nn.conv2d(x, w, strides=1, padding="SAME")
    loss = tf.reduce_sum(y)
gradient = tape.gradient(loss, w)
if sys.platform == "darwin":
    assert "CPU:0" in y.device, y.device
    assert "CPU:0" in gradient.device, gradient.device

model = tf.keras.Sequential([
    tf.keras.Input(shape=(4, 4, 1)),
    tf.keras.layers.Conv2D(1, 1),
])
model.compile(optimizer="sgd", loss="mse", jit_compile=False)
batch_loss = model.train_on_batch(np.ones((1, 4, 4, 1)), np.zeros((1, 4, 4, 1)))
assert np.isfinite(batch_loss), batch_loss
print(json.dumps({"visible_gpus": [d.name for d in after],
                  "convolution_device": y.device, "gradient_device": gradient.device,
                  "train_on_batch_loss": float(batch_loss)}))
"""],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS CPU-only training policy")
def test_training_import_rejects_initialized_gpu():
    result = subprocess.run(
        [sys.executable, "-c", """
import tensorflow as tf
if not tf.config.list_physical_devices("GPU"):
    raise SystemExit(77)
tf.config.list_logical_devices("GPU")
try:
    from antispoofing_pipeline.keras.training import train
except RuntimeError:
    pass
else:
    raise AssertionError("Training must reject an already initialized GPU runtime")
"""],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode == 77:
        pytest.skip("No physical TensorFlow GPU available")
    assert result.returncode == 0, result.stdout + result.stderr
