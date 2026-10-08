#!/usr/bin/env python
"""Evaluate two stereo ONNX models and always write JSON plus histogram PNG."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import onnxruntime as ort
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score

from infer_sensor_model import preprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", default="results/LensDirt_Results/sensor_heads/run/deploy/sensor_model_opset11.onnx")
    parser.add_argument("--new", default="results/LensDirt_Results/stereo_validated/run_v2/deploy/sensor_model_opset11.onnx")
    parser.add_argument("--split", default="results/LensDirt_Results/stereo_validated/run_v2/split.json")
    parser.add_argument("--out", default="report/stereo_validated")
    args = parser.parse_args()
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    records = split["test"]
    paths = [row[0] for row in records]
    labels = np.array([row[1] for row in records])
    sessions = [ort.InferenceSession(path, providers=["CPUExecutionProvider"])
                for path in (args.old, args.new)]
    results, score_lists = [], []
    for name, session in zip(("before", "after"), sessions):
        scores, decisions = [], []
        metadata = session.get_modelmeta().custom_metadata_map
        threshold = float(metadata["thresholds"].split(",")[1])
        for path in paths:
            score, decision = session.run(
                None, {"image": preprocess(path), "sensor_id": np.array([1], dtype=np.int64)})
            scores.append(float(score[0]))
            decisions.append(int(decision[0]))
        scores = np.asarray(scores)
        decisions = np.asarray(decisions)
        matrix = confusion_matrix(labels, decisions, labels=[0, 1]).tolist()
        score_lists.append(scores)
        results.append({
            "model": name,
            "n": len(labels),
            "normal": int((labels == 0).sum()),
            "anomaly": int((labels == 1).sum()),
            "auroc": float(roc_auc_score(labels, scores)),
            "pr_auc": float(average_precision_score(labels, scores)),
            "threshold": threshold,
            "f1": float(f1_score(labels, decisions)),
            "confusion_matrix": matrix,
            "score_normal_mean": float(scores[labels == 0].mean()),
            "score_anomaly_mean": float(scores[labels == 1].mean()),
        })
    for family in ("Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
            plt.rcParams["font.family"] = family
            break
        except ValueError:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    all_scores = np.concatenate(score_lists)
    low, high = float(all_scores.min()), float(all_scores.max())
    if low == high:
        low, high = low - 0.01, high + 0.01
    bins = np.linspace(low, high, 28)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for axis, result, scores in zip(axes, results, score_lists):
        axis.hist(scores[labels == 0], bins=bins, color="#247b69", alpha=0.72,
                  label=f"正常 {int((labels == 0).sum())}")
        axis.hist(scores[labels == 1], bins=bins, color="#c25746", alpha=0.65,
                  label=f"异常 {int((labels == 1).sum())}")
        axis.axvline(result["threshold"], color="#202e39", linestyle="--", linewidth=1.8,
                     label=f"阈值 {result['threshold']:.3f}")
        cm = result["confusion_matrix"]
        axis.set_title(f"{result['model']} | AUROC {result['auroc']:.3f} | PR-AUC {result['pr_auc']:.3f}\n"
                       f"TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
        axis.set_xlabel("双目异常分数（越高越异常）")
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("图片数量")
    figure.suptitle("双目模型前后对比：最终测试集分数分布", fontsize=16)
    figure.tight_layout()
    figure.savefig(output / "report.png", dpi=150)
    plt.close(figure)
    payload = {
        "results": results,
        "report_image": str(output / "report.png"),
        "note": "Final test split was fixed before training and was not used for epoch or threshold selection.",
    }
    (output / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("Histogram:", output / "report.png")


if __name__ == "__main__":
    main()
