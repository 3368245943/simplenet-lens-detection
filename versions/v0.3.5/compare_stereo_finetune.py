#!/usr/bin/env python
"""Compare old and stereo-finetuned ONNX on identical stereo holdout images."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import onnxruntime as ort
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from datasets.lens_dirt import DatasetSplit, LensDirtDataset, _uniform_sample
from infer_sensor_model import preprocess
from sensor_thresholds import sensor_from_path


def metrics(y, score, predicted):
    return {"auroc": float(roc_auc_score(y, score)),
            "f1": float(f1_score(y, predicted)),
            "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", default="results/LensDirt_Results/sensor_heads/run/deploy/sensor_model_opset11.onnx")
    parser.add_argument("--new", default="results/LensDirt_Results/stereo_finetune/run/deploy/sensor_model_opset11.onnx")
    parser.add_argument("--datapath", default="./data")
    parser.add_argument("--anomalies", type=int, default=300)
    parser.add_argument("--out", default="report/stereo_finetune")
    args = parser.parse_args()
    if args.anomalies <= 0:
        parser.error("Anomaly sample count must be positive")
    ds = LensDirtDataset(args.datapath, split=DatasetSplit.TEST,
                         frame_stride=2, balance_scenes=False, strict_test_split=True)
    train = LensDirtDataset(args.datapath, split=DatasetSplit.TRAIN,
                            frame_stride=2, per_scene_train=100000)
    clean = sorted(p for p in ds.normal_paths if sensor_from_path(p) == "stereo")
    dirty = _uniform_sample(sorted(p for p in ds.anomaly_paths if sensor_from_path(p) == "stereo"), args.anomalies)
    overlap = set(clean) & set(train.normal_paths)
    if not clean or not dirty or overlap:
        raise ValueError(f"Invalid holdout split: clean={len(clean)} dirty={len(dirty)} overlap={len(overlap)}")
    paths = clean + dirty
    labels = np.array([0]*len(clean) + [1]*len(dirty))
    sessions = [ort.InferenceSession(p, providers=["CPUExecutionProvider"]) for p in (args.old, args.new)]
    results = []
    score_lists = [[], []]
    predictions = [[], []]
    for index, path in enumerate(paths, 1):
        image = preprocess(path)
        for k, session in enumerate(sessions):
            score, decision = session.run(None, {"image": image, "sensor_id": np.array([1], dtype=np.int64)})
            score_lists[k].append(float(score[0]))
            predictions[k].append(int(decision[0]))
        if index % 100 == 0:
            print(f"Scored {index}/{len(paths)} images", flush=True)
    for name, scores, decisions in zip(("before", "after"), score_lists, predictions):
        result = metrics(labels, scores, decisions)
        result["name"] = name
        result["normal"] = len(clean)
        result["anomaly"] = len(dirty)
        results.append(result)
    for family in ("WenQuanYi Micro Hei", "DejaVu Sans"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
            plt.rcParams["font.family"] = family
            break
        except ValueError:
            continue
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, result, scores, session in zip(axes, results, score_lists, sessions):
        values = np.asarray(scores)
        threshold = float(session.get_modelmeta().custom_metadata_map["thresholds"].split(",")[1])
        bins = np.linspace(values.min(), values.max(), 26)
        ax.hist(values[labels == 0], bins=bins, alpha=.7, color="#247b69", label=f"正常 {len(clean)}")
        ax.hist(values[labels == 1], bins=bins, alpha=.6, color="#c25746", label=f"异常 {len(dirty)}")
        ax.axvline(threshold, color="black", linestyle="--", label=f"固定阈值 {threshold:.3f}")
        cm = result["confusion_matrix"]
        ax.set(title=f"{result['name']} | AUC {result['auroc']:.3f} F1 {result['f1']:.3f}\nTN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}",
               xlabel="双目异常分数", ylabel="图片数量")
        ax.legend(frameon=False)
    fig.suptitle("双目分支续训前后 · 相同测试图片与原阈值", fontsize=16)
    fig.tight_layout()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "report.png", dpi=150)
    plt.close(fig)
    payload = {"results": results, "test_normal": len(clean), "test_anomaly": len(dirty),
               "normal_training_file_overlap": len(overlap),
               "note": "Same recordings may still overlap; old model checkpoint selected with an earlier development set. No threshold tuned on these expanded test images."}
    (out / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    print("Histogram:", out / "report.png", flush=True)


if __name__ == "__main__":
    main()
