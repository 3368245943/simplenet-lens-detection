#!/usr/bin/env python
"""Compare a global score threshold with sensor thresholds on held-out scenes.

The checkpoint was selected using this dataset; scene CV only validates threshold
calibration and is not an independent estimate of model generalization.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

from datasets.lens_dirt import DatasetSplit, LensDirtDataset, IMAGENET_MEAN, IMAGENET_STD
from sensor_thresholds import SENSORS, evaluate_by_scene, sensor_from_path, youden_threshold
from test_report import IN_H, IN_W, build_model, infer_score, load_ckpt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default="./results/LensDirt_Results/simplenet_lens_tof_normal/run/models/0/lens_dirt_lens/ckpt.pth")
    parser.add_argument("--datapath", default="./data")
    parser.add_argument("--out", default="report/sensor_threshold_cv")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    load_ckpt(model, args.ckpt)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IN_H, IN_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    dataset = LensDirtDataset(args.datapath, split=DatasetSplit.TEST, frame_stride=5, seed=0)
    paths, labels, scores = [], [], []
    for _, label, path, _ in dataset.data_to_iterate:
        score = infer_score(model, path, transform, device)
        if score is not None:
            paths.append(path)
            labels.append(int(label != "good"))
            scores.append(score)

    labels = np.array(labels)
    scores = np.array(scores)
    summary, _, _ = evaluate_by_scene(paths, labels, scores)
    summary["checkpoint"] = args.ckpt
    summary["samples"] = len(paths)
    summary["warning"] = "Checkpoint selection used the same dataset; scene-CV results are not an independent model test."
    sensor_names = np.array([sensor_from_path(p) for p in paths])
    thresholds = {sensor: youden_threshold(labels[sensor_names == sensor], scores[sensor_names == sensor])
                  for sensor in SENSORS}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cross_validation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "experimental_thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n", encoding="utf-8")
    print("Leave-one-scene-out global:", summary["global"])
    print("Leave-one-scene-out sensor-aware:", summary["sensor_aware"])
    print("Experimental full-set thresholds (NOT independently tested):", thresholds)
    print("Results:", out)


if __name__ == "__main__":
    main()
