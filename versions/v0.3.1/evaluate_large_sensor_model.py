#!/usr/bin/env python
"""Evaluate a larger test subset and archive timestamped FP/FN images."""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import shutil

import numpy as np
import onnxruntime as ort

from datasets.lens_dirt import DatasetSplit, LensDirtDataset, _uniform_sample
from evaluate_sensor_model import plot_report, summarize
from infer_sensor_model import preprocess
from sensor_thresholds import SENSORS, sensor_from_path


def choose_records(dataset, anomaly_per_sensor):
    normal = [(p, 0) for p in dataset.normal_paths]
    anomalous = []
    for sensor in SENSORS:
        candidates = sorted(p for p in dataset.anomaly_paths if sensor_from_path(p) == sensor)
        selected = _uniform_sample(candidates, anomaly_per_sensor)
        anomalous.extend((p, 1) for p in selected)
    return normal + anomalous


def evaluate(records, session, out, error_dir):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if out.exists() or error_dir.exists():
        raise FileExistsError("Output path already exists; select another timestamp or path")
    out.parent.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True)
    groups = {sensor: ([], [], []) for sensor in SENSORS}
    mistakes = []
    for index, (path, label) in enumerate(records, 1):
        sensor = sensor_from_path(path)
        score, predicted = session.run(None, {
            "image": preprocess(path),
            "sensor_id": np.array([SENSORS.index(sensor)], dtype=np.int64),
        })
        score, predicted = float(score[0]), int(predicted[0])
        groups[sensor][0].append(label)
        groups[sensor][1].append(score)
        groups[sensor][2].append(predicted)
        if label != predicted:
            mistake = "FP" if label == 0 else "FN"
            name = f"{stamp}_{index:05d}_{sensor}_{mistake}_score{score:.4f}{Path(path).suffix.lower()}"
            mistakes.append((path, mistake, name, sensor, label, predicted, score))
        if index % 100 == 0:
            print(f"Processed {index}/{len(records)}", flush=True)
    totals = {key: summarize(*values) for key, values in groups.items()}
    combined = tuple(sum((groups[key][i] for key in SENSORS), []) for i in range(3))
    totals["overall"] = summarize(*combined)
    metadata = session.get_modelmeta().custom_metadata_map
    thresholds = dict(zip(SENSORS, map(float, metadata["thresholds"].split(","))))
    plot_report(groups, totals, thresholds, out)
    with (error_dir / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(("filename", "error", "sensor", "true_label", "predicted_label", "score", "source"))
        for source, mistake, name, sensor, label, predicted, score in mistakes:
            shutil.copy2(source, error_dir / name)
            writer.writerow((name, mistake, sensor, label, predicted, f"{score:.6f}", source))
    print(f"Counts: {[(k, v['normal'], v['anomaly']) for k,v in totals.items() if k != 'overall']}")
    print(f"Confusion matrix: {totals['overall']['confusion_matrix']}, F1={totals['overall']['f1']:.4f}")
    print(f"False positives: {sum(r[1] == 'FP' for r in mistakes)}, false negatives: {sum(r[1] == 'FN' for r in mistakes)}")
    print(f"Histogram: {out}\nErrors: {error_dir}")
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="results/LensDirt_Results/sensor_heads/run/deploy/sensor_model_opset11.onnx")
    parser.add_argument("--datapath", default="./data")
    parser.add_argument("--anomaly-per-sensor", type=int, default=200)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--out", default=None, help="PNG report, default timestamped report/ folder")
    parser.add_argument("--errors", default=None, help="Error directory, default timestamped error/ folder")
    args = parser.parse_args()
    if args.anomaly_per_sensor <= 0 or args.frame_stride <= 0:
        parser.error("Sampling parameters must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = Path(args.out or f"report/sensor_heads/large_test_{stamp}.png")
    errors = Path(args.errors or f"error/sensor_heads_large_test_{stamp}")
    if out.suffix.lower() != ".png" or out.exists() or errors.exists():
        parser.error("Output PNG and error directory must be new paths")
    dataset = LensDirtDataset(args.datapath, split=DatasetSplit.TEST, frame_stride=args.frame_stride,
                              balance_scenes=False)
    records = choose_records(dataset, args.anomaly_per_sensor)
    train = LensDirtDataset(args.datapath, split=DatasetSplit.TRAIN, frame_stride=args.frame_stride)
    overlap = set(train.normal_paths) & set(dataset.normal_paths)
    if overlap:
        raise ValueError(f"Training/test overlap: {len(overlap)} identical normal files")
    print("No identical normal training files; RGB clips may still share a recording session.", flush=True)
    if not records:
        parser.error("No test samples found")
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    if session.get_modelmeta().custom_metadata_map.get("sensor_ids") != "0=rgb,1=stereo,2=tof":
        raise ValueError("Unknown model sensor ID mapping")
    print(f"Testing {len(records)} images, without modifying model weights", flush=True)
    evaluate(records, session, out, errors)


if __name__ == "__main__":
    main()
