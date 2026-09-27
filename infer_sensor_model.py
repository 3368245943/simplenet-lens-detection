#!/usr/bin/env python
"""Run the single sensor-aware ONNX model with an explicit sensor source."""
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

SENSORS = ("rgb", "stereo", "tof")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def preprocess(path):
    with Image.open(path) as image:
        gray = image.convert("RGB").convert("L").resize((240, 180), Image.BILINEAR)
        pixels = np.asarray(gray, dtype=np.float32) / 255.0
    image = np.broadcast_to(pixels, (3, 180, 240)).copy()
    return ((image - MEAN) / STD)[None, ...]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="results/LensDirt_Results/sensor_heads/run/deploy/sensor_model_opset11.onnx")
    parser.add_argument("--image", required=True)
    parser.add_argument("--sensor", required=True, choices=SENSORS,
                        help="Explicit source: rgb, stereo (left/right), or tof")
    args = parser.parse_args()
    if not Path(args.model).is_file() or not Path(args.image).is_file():
        parser.error("Model and image must both exist")
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    metadata = session.get_modelmeta().custom_metadata_map
    if metadata.get("sensor_ids") != "0=rgb,1=stereo,2=tof":
        raise ValueError("Unknown sensor ID mapping in ONNX model")
    score, is_anomaly = session.run(None, {
        "image": preprocess(args.image),
        "sensor_id": np.array([SENSORS.index(args.sensor)], dtype=np.int64),
    })
    print(f"sensor={args.sensor} anomaly_score={float(score[0]):.6f} "
          f"label={'dirty' if bool(is_anomaly[0]) else 'normal'}")


if __name__ == "__main__":
    main()
