"""MobileNetV1 ArcFace architecture adapted for NPU INT8 deployment."""

import os
from typing import Optional

import h5py
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


MIN_EXPECTED_LOADED_LAYERS = 58


def _extract_layer_weights(weights_group, layer_name: str):
    """Extract weight arrays for a layer from Keras h5 model_weights group."""
    if layer_name not in weights_group:
        return None
    grp = weights_group[layer_name]
    sub_keys = list(grp.keys())
    if not sub_keys:
        return None
    sub_grp = grp[sub_keys[0]]
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
            np.array(sub_grp["moving_variance:0"]),
        ]
    elif "beta:0" in w_keys:
        return [
            np.array(sub_grp["beta:0"]),
            np.array(sub_grp["moving_mean:0"]),
            np.array(sub_grp["moving_variance:0"]),
        ]
    return None


def load_mobilenet_h5_weights(
    model: keras.Model,
    weights_path: str,
    min_loaded_layers: int = MIN_EXPECTED_LOADED_LAYERS,
) -> int:
    """Load weights from a pre-trained Keras h5 checkpoint into the NPU model.

    Raises:
        RuntimeError: If fewer than ``min_loaded_layers`` layers are mapped, preventing
            silent fallback to randomly-initialized layers due to layer name mismatches.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Recognition weights file not found: {weights_path}")

    loaded_layers_count = 0
    with h5py.File(weights_path, "r") as f:
        if "model_weights" not in f:
            raise KeyError(f"'model_weights' not found in weights file {weights_path}")
        mw = f["model_weights"]

        # Load weights into top-level model layers
        for layer in model.layers:
            weights = _extract_layer_weights(mw, layer.name)
            if weights is not None:
                layer.set_weights(weights)
                loaded_layers_count += 1

            # Check nested sub-models (e.g. base MobileNet backbone)
            if isinstance(layer, keras.Model):
                for sub_layer in layer.layers:
                    sub_weights = _extract_layer_weights(mw, sub_layer.name)
                    if sub_weights is not None:
                        sub_layer.set_weights(sub_weights)
                        loaded_layers_count += 1

    if loaded_layers_count < min_loaded_layers:
        raise RuntimeError(
            f"Insufficient weights loaded into MobileNet model: loaded {loaded_layers_count} "
            f"layers, expected at least {min_loaded_layers}. Check layer name alignment."
        )

    return loaded_layers_count


def build_npu_mobilenet_model(
    weights_path: Optional[str] = None,
    embedding_dim: int = 256,
) -> keras.Model:
    """Build NPU-compliant MobileNetV1 face recognition model.

    Rules for NPU compiler compatibility:
      1. Fixed batch size 1: batch_shape=(1, 112, 112, 3)
      2. Explicit 3x3 Depthwise Conv (kernel 3x3 on 3x3 feature map -> 1x1x1024)
      3. 1x1 Conv head (1x1x1024 -> 1x1xembedding_dim)
      4. Reshape to [1, embedding_dim] instead of Flatten
    """
    inputs = keras.Input(batch_shape=(1, 112, 112, 3), name="input_1")

    base = keras.applications.MobileNet(
        input_shape=(112, 112, 3),
        include_top=False,
        weights=None,
    )
    x = base(inputs)  # (1, 3, 3, 1024)

    dw = layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        padding="valid",
        use_bias=False,
        name="depthwise_conv2d",
    )(x)
    bn1 = layers.BatchNormalization(name="batch_normalization")(dw)

    conv_head = layers.Conv2D(
        embedding_dim,
        kernel_size=(1, 1),
        use_bias=True,
        name="conv2d",
    )(bn1)
    emb_bn = layers.BatchNormalization(scale=False, name="embedding")(conv_head)
    out = layers.Reshape((embedding_dim,), name="out_embedding")(emb_bn)

    model = keras.Model(inputs=inputs, outputs=out, name="mobilenet_emore_npu")

    if weights_path is not None:
        load_mobilenet_h5_weights(model, weights_path)

    return model
