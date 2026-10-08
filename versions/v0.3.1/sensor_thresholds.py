"""Sensor-aware threshold calibration for the mixed RGB/stereo/ToF dataset."""

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_curve

SENSORS = ("rgb", "stereo", "tof")


def sensor_from_path(path):
    parts = path.replace("\\", "/").split("/")
    for i in range(len(parts) - 2):
        if parts[i:i + 3] == ["depth_data", "tof", "img"]:
            return "tof"
        if parts[i:i + 2] == ["depth_data", "img"] and parts[i + 2] in ("left", "right"):
            return "stereo"
    if "cam0" in parts:
        return "rgb"
    raise ValueError("Cannot identify camera from image path: " + path)


def scene_from_path(path):
    parts = path.replace("\\", "/").split("/")
    for key in ("正样本", "负样本"):
        if key in parts:
            i = parts.index(key)
            if len(parts) > i + 2:
                return "/".join(parts[i + 1:i + 3])
    raise ValueError("Cannot identify scene from image path: " + path)


def youden_threshold(labels, scores):
    if len(np.unique(labels)) != 2:
        raise ValueError("Threshold calibration requires both normal and anomaly samples")
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return float(thresholds[np.argmax(tpr - fpr)])


def evaluate_by_scene(paths, labels, scores):
    """Fit thresholds on other scenes and predict each held-out scene."""
    paths = list(paths)
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    sensors = np.array([sensor_from_path(p) for p in paths])
    scenes = np.array([scene_from_path(p) for p in paths])
    if not (len(paths) == len(labels) == len(scores)):
        raise ValueError("Paths, labels, and scores must have equal lengths")
    global_pred = np.zeros(len(paths), dtype=int)
    sensor_pred = np.zeros(len(paths), dtype=int)
    folds = []
    for scene in sorted(set(scenes)):
        fit, holdout = scenes != scene, scenes == scene
        global_threshold = youden_threshold(labels[fit], scores[fit])
        global_pred[holdout] = scores[holdout] >= global_threshold
        thresholds = {}
        for sensor in SENSORS:
            sensor_fit = fit & (sensors == sensor)
            sensor_holdout = holdout & (sensors == sensor)
            if not sensor_holdout.any():
                continue
            thresholds[sensor] = youden_threshold(labels[sensor_fit], scores[sensor_fit])
            sensor_pred[sensor_holdout] = scores[sensor_holdout] >= thresholds[sensor]
        folds.append({"held_out_scene": scene, "samples": int(holdout.sum()),
                      "global_threshold": global_threshold, "sensor_thresholds": thresholds,
                      "global_errors": int(np.sum(global_pred[holdout] != labels[holdout])),
                      "sensor_errors": int(np.sum(sensor_pred[holdout] != labels[holdout]))})
    def metrics(pred):
        return {"confusion_matrix": confusion_matrix(labels, pred, labels=[0, 1]).tolist(),
                "f1": float(f1_score(labels, pred)),
                "accuracy": float(np.mean(labels == pred))}
    return {"global": metrics(global_pred), "sensor_aware": metrics(sensor_pred),
            "folds": folds}, global_pred, sensor_pred
