"""Convert .keras models to .tflite format for lightweight deployment."""

import os
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

KERAS_MODELS = [
    "custom_cnn_best.keras",
    "mobilenetv2_stage1_best.keras",
]


def convert(keras_filename: str) -> None:
    keras_path = os.path.join(MODELS_DIR, keras_filename)
    tflite_path = keras_path.replace(".keras", ".tflite")

    print(f"Loading {keras_path} ...")
    model = tf.keras.models.load_model(keras_path, compile=False, safe_mode=False)

    print("Converting to TFLite (float32, no quantization) ...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_bytes = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_bytes)

    size_kb = len(tflite_bytes) / 1024
    print(f"Saved {tflite_path} ({size_kb:.0f} KB)\n")


if __name__ == "__main__":
    for name in KERAS_MODELS:
        convert(name)
    print("Done — all models converted.")
