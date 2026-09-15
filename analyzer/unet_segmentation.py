"""Compact TensorFlow/Keras U-Net, imported lazily so classical workflows stay light."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("TensorFlow is required for U-Net training or inference. Install requirements.txt in Python 3.10-3.12.") from exc
    return tf


def dice_metric(y_true, y_pred):
    tf = _tensorflow()
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    return tf.reduce_mean((2.0 * intersection + 1e-6) / (denominator + 1e-6))


def dice_bce_loss(y_true, y_pred):
    tf = _tensorflow()
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    return bce + (1.0 - dice_metric(y_true, y_pred))


def build_unet(input_shape: tuple[int, int, int] = (256, 256, 3)):
    tf = _tensorflow()
    layers = tf.keras.layers

    def block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        return x

    inputs = layers.Input(shape=input_shape)
    c1 = block(inputs, 32)
    p1 = layers.MaxPooling2D()(c1)
    c2 = block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)
    c3 = block(p2, 128)
    p3 = layers.MaxPooling2D()(c3)
    bottleneck = block(p3, 256)
    u3 = layers.UpSampling2D(interpolation="bilinear")(bottleneck)
    u3 = layers.Concatenate()([u3, c3])
    c4 = block(u3, 128)
    u2 = layers.UpSampling2D(interpolation="bilinear")(c4)
    u2 = layers.Concatenate()([u2, c2])
    c5 = block(u2, 64)
    u1 = layers.UpSampling2D(interpolation="bilinear")(c5)
    u1 = layers.Concatenate()([u1, c1])
    c6 = block(u1, 32)
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c6)
    return tf.keras.Model(inputs, outputs, name="compact_unet_skin_lesion")


def compile_unet(model, learning_rate: float = 1e-4):
    tf = _tensorflow()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=dice_bce_loss,
        metrics=[dice_metric, tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5, name="iou")],
    )
    return model


def load_unet_model(model_path: str | Path):
    tf = _tensorflow()
    return tf.keras.models.load_model(model_path, custom_objects={"dice_bce_loss": dice_bce_loss, "dice_metric": dice_metric})


def predict_unet(model, image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    rgb = np.asarray(image).astype(np.float32) / 255.0
    prediction = model.predict(rgb[None, ...], verbose=0)[0, ..., 0]
    return (prediction >= threshold).astype(np.uint8)

