#!/usr/bin/env python
"""Interactive viewer for RGB, stereo and ToF inference samples.

Configure LENS_MODEL_PATH and sensor image-directory environment variables,
or edit the defaults below for a local deployment.
"""
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from infer_sensor_model import preprocess

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get(
    "LENS_MODEL_PATH",
    ROOT / "results/LensDirt_Results/sensor_heads_cleaned_q95/run/"
    "sensor_model_normal_q95_count_opset11.onnx",
)).expanduser()
IMAGE_DIRS = {
    "rgb": os.environ.get("LENS_RGB_DIR"),
    "stereo": os.environ.get("LENS_STEREO_DIR"),
    "tof": os.environ.get("LENS_TOF_DIR"),
}
THRESHOLDS = {
    "rgb": float(os.environ.get("LENS_THRESHOLD_RGB", "2.1480555534362793")),
    "stereo": float(os.environ.get("LENS_THRESHOLD_STEREO", "127")),
    "tof": float(os.environ.get("LENS_THRESHOLD_TOF", "1.3620946407318115")),
}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SENSORS = ("rgb", "stereo", "tof")


def is_sensor_image(path, sensor):
    parts = path.as_posix().split("/")
    if sensor == "rgb":
        return "cam0" in parts and "depth_data" not in parts
    if sensor == "stereo":
        return "depth_data" in parts and "img" in parts and ("left" in parts or "right" in parts)
    if sensor == "tof":
        return "depth_data" in parts and "tof" in parts and "img" in parts
    return False


def collect_images(directory, sensor):
    if not directory:
        return []
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"{sensor} image directory does not exist: {directory}")
    return [
        (path, sensor)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in EXTENSIONS and is_sensor_image(path, sensor)
    ]


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"ONNX model not found: {MODEL_PATH}. Set LENS_MODEL_PATH to the model file."
        )
    configured = [(sensor, IMAGE_DIRS[sensor]) for sensor in SENSORS]
    directories = [Path(directory).expanduser().resolve() for _, directory in configured if directory]
    if len(directories) != len(set(directories)):
        raise ValueError("RGB, stereo and ToF must use separate image directories")
    images = [item for sensor, directory in configured for item in collect_images(directory, sensor)]
    counts = {sensor: sum(item[1] == sensor for item in images) for sensor in SENSORS}
    print("Image counts:", counts, flush=True)
    if not images:
        raise RuntimeError("Set at least one LENS_RGB_DIR, LENS_STEREO_DIR or LENS_TOF_DIR")

    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    index = 0
    window = "Single image inference | N next, P previous, Q quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    while True:
        path, sensor = images[index]
        score, _ = session.run(None, {
            "image": preprocess(str(path)),
            "sensor_id": np.array([SENSORS.index(sensor)], dtype=np.int64),
        })
        score = float(score[0])
        threshold = THRESHOLDS[sensor]
        prediction = "ANOMALY" if score >= threshold else "NORMAL"
        color = (0, 0, 220) if prediction == "ANOMALY" else (0, 150, 0)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {path}")
        header = np.full((100, image.shape[1], 3), 245, dtype=np.uint8)
        cv2.putText(header, f"{index + 1}/{len(images)}  {sensor.upper()}  {prediction}",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, color, 2, cv2.LINE_AA)
        cv2.putText(header, f"score={score:.4f}  threshold={threshold:.4f}",
                    (12, 63), cv2.FONT_HERSHEY_SIMPLEX, .58, (35, 45, 55), 1, cv2.LINE_AA)
        cv2.putText(header, str(path), (12, 88), cv2.FONT_HERSHEY_SIMPLEX,
                    .42, (35, 45, 55), 1, cv2.LINE_AA)
        cv2.imshow(window, np.vstack((header, image)))
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("n"), ord("N"), ord(" "), 13):
            index = (index + 1) % len(images)
        elif key in (ord("p"), ord("P")):
            index = (index - 1) % len(images)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
