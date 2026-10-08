#!/usr/bin/env python
"""Evaluate the one-file ONNX model on the fixed 224-image test split."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from datasets.lens_dirt import DatasetSplit, LensDirtDataset
from infer_sensor_model import preprocess
from sensor_thresholds import SENSORS, evaluate_by_scene, sensor_from_path


def summarize(labels, scores, decisions):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    decisions = np.asarray(decisions)
    return {"samples": int(len(labels)),
            "normal": int((labels == 0).sum()), "anomaly": int((labels == 1).sum()),
            "auroc": float(roc_auc_score(labels, scores)),
            "confusion_matrix": confusion_matrix(labels, decisions, labels=[0, 1]).tolist(),
            "f1": float(f1_score(labels, decisions))}


def plot_report(groups, result, thresholds, out):
    for family in ("Noto Sans CJK SC", "WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "DejaVu Sans"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
            plt.rcParams["font.family"] = family
            break
        except ValueError:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    names = {"rgb": "RGB", "stereo": "双目", "tof": "ToF"}
    for ax, sensor in zip(axes.flat, SENSORS):
        labels, scores, _ = groups[sensor]
        labels = np.asarray(labels)
        scores = np.asarray(scores)
        left, right = float(scores.min()), float(scores.max())
        if left == right:
            left, right = left - 0.01, right + 0.01
        bins = np.linspace(left, right, 24)
        ax.hist(scores[labels == 0], bins=bins, color="#247b69", alpha=0.70,
                label=f"正常 {sum(labels == 0)}")
        ax.hist(scores[labels == 1], bins=bins, color="#c25746", alpha=0.65,
                label=f"异常 {sum(labels == 1)}")
        ax.axvline(thresholds[sensor], color="#202e39", linestyle="--", linewidth=1.8,
                   label=f"阈值 {thresholds[sensor]:.3f}")
        ax.set(title=f"{names[sensor]}  ·  AUROC {result[sensor]['auroc']:.3f}",
               xlabel="异常分数（越高越异常）", ylabel="图片数量")
        ax.legend(frameon=False, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    ax = axes.flat[3]
    cm = np.asarray(result["overall"]["confusion_matrix"])
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    ax.set(title=f"总体混淆矩阵  ·  F1 {result['overall']['f1']:.3f}",
           xlabel="预测", ylabel="真实", xticks=[0, 1], yticks=[0, 1])
    ax.set_xticklabels(["正常", "异常"])
    ax.set_yticklabels(["正常", "异常"])
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center",
                    fontsize=20, color="white" if cm[row, col] > cm.max() / 2 else "#202e39")
    fig.suptitle("三路传感器异常分数分布与判定结果", fontsize=17)
    fig.text(0.5, 0.012, "开发集用于模型与阈值选优；本图不能代替独立录制批次的验收结果。",
             ha="center", fontsize=10, color="#596775")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="results/LensDirt_Results/sensor_heads/run/deploy/sensor_model_opset11.onnx")
    parser.add_argument("--datapath", default="./data")
    parser.add_argument("--out", default="report/sensor_heads/report.png", help="直方图报告 PNG")
    parser.add_argument("--details", default=None, help="可选：保存详细 JSON")
    args = parser.parse_args()
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    metadata = session.get_modelmeta().custom_metadata_map
    if metadata.get("sensor_ids") != "0=rgb,1=stereo,2=tof":
        raise ValueError("Unknown sensor mapping in model")
    thresholds = np.array([float(value) for value in metadata["thresholds"].split(",")])
    if len(thresholds) != len(SENSORS) or not np.isfinite(thresholds).all():
        raise ValueError("Invalid model thresholds")
    threshold_by_sensor = dict(zip(SENSORS, thresholds))
    dataset = LensDirtDataset(args.datapath, split=DatasetSplit.TEST, frame_stride=5, seed=0)
    groups = {key: ([], [], []) for key in SENSORS}
    paths = {key: [] for key in SENSORS}
    for _, label, path, _ in dataset.data_to_iterate:
        sensor = sensor_from_path(path)
        score, decision = session.run(None, {
            "image": preprocess(path),
            "sensor_id": np.array([SENSORS.index(sensor)], dtype=np.int64),
        })
        paths[sensor].append(path)
        groups[sensor][0].append(int(label != "good"))
        groups[sensor][1].append(float(score[0]))
        groups[sensor][2].append(int(decision[0]))
    result = {key: summarize(*value) for key, value in groups.items()}
    combined = tuple(sum((groups[key][i] for key in SENSORS), []) for i in range(3))
    result["overall"] = summarize(*combined)
    combined_paths = sum((paths[key] for key in SENSORS), [])
    result["scene_cv"], _, _ = evaluate_by_scene(combined_paths, combined[0], combined[1])
    result["warning"] = "Checkpoint and thresholds were selected with this test split; this is not an independent generalization estimate."
    out = Path(args.out)
    if out.suffix.lower() != ".png":
        parser.error("--out 必须是 PNG 文件路径")
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_report(groups, result, threshold_by_sensor, out)
    if args.details:
        details = Path(args.details)
        details.parent.mkdir(parents=True, exist_ok=True)
        details.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"正常 {result['overall']['normal']} 张，异常 {result['overall']['anomaly']} 张；"
          f"混淆矩阵 {result['overall']['confusion_matrix']}，F1 {result['overall']['f1']:.4f}")
    print("直方图报告:", out)


if __name__ == "__main__":
    main()
