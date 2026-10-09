#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成分类测试统计图（类似 YOLO 的 results 图）：
对测试集跑推理，输出一张包含「混淆矩阵 + 各类指标 + 分数分布」的统计图。

用法：
    ./venv/bin/python test_report.py [--n 每类抽样数] [--out 输出目录]

默认对测试集（正常=正样本、脏污=负样本）二分类，输出 report.png。
"""
import os
import sys
import argparse

import numpy as np
import cv2
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
for _f in ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "AR PL UMing CN", "DejaVu Sans"]:
    try:
        font_manager.findfont(_f, fallback_to_default=False)
        matplotlib.rcParams["font.family"] = _f
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import backbones
from datasets.lens_dirt import LensDirtDataset, DatasetSplit, IMAGENET_MEAN, IMAGENET_STD
import simplenet


# 灰度实验版：ToF 宽 240、高 180，灰度复制为 3 通道。
IN_H, IN_W = 180, 240


def build_model(device):
    backbone = backbones.load("wideresnet50")
    backbone.name = "wideresnet50"
    backbone.seed = None
    net = simplenet.SimpleNet(device)
    net.load(
        backbone=backbone,
        layers_to_extract_from=["layer2", "layer3"],
        device=device,
        input_shape=(3, IN_H, IN_W),
        pretrain_embed_dimension=1536,
        target_embed_dimension=1536,
        patchsize=3,
        embedding_size=256,
        meta_epochs=10,
        gan_epochs=4,
        noise_std=0.015,
        dsc_hidden=1024,
        dsc_layers=2,
        dsc_margin=0.5,
        pre_proj=1,
    )
    return net


def load_ckpt(net, ckpt_path):
    state_dict = torch.load(ckpt_path, map_location=net.device)
    if "discriminator" in state_dict:
        net.discriminator.load_state_dict(state_dict["discriminator"])
        if "pre_projection" in state_dict and net.pre_proj > 0:
            net.pre_projection.load_state_dict(state_dict["pre_projection"])
    else:
        net.load_state_dict(state_dict, strict=False)


def infer_score(net, path, transform, device):
    img = cv2.imread(path)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    im = transform(im).unsqueeze(0).to(torch.float).to(device)
    with torch.no_grad():
        return float(net._predict(im)[0][0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="./results/LensDirt_Results/simplenet_lens/run/models/0/lens_dirt_lens/ckpt.pth")
    parser.add_argument("--datapath", type=str,
                        default="./data")
    parser.add_argument("--n", type=int, default=200, help="每类抽样数")
    parser.add_argument("--out", type=str, default="report")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"[错误] checkpoint 不存在: {args.ckpt}，请先训练")
        sys.exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = build_model(device)
    load_ckpt(net, args.ckpt)

    from torchvision import transforms
    # 与训练一致：先转灰度，再复制为 3 通道，整图 Resize 到 240x180。
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IN_H, IN_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    ds = LensDirtDataset(args.datapath, split=DatasetSplit.TEST,
                         frame_stride=5, seed=0)
    normal_paths = [d[2] for d in ds.data_to_iterate if d[1] == "good"]
    anomaly_paths = [d[2] for d in ds.data_to_iterate if d[1] != "good"]

    def sample(paths, k):
        if len(paths) <= k:
            return paths
        step = (len(paths) - 1) / (k - 1)
        return [paths[int(round(i * step))] for i in range(k)]

    normal_sample = sample(normal_paths, args.n)
    anomaly_sample = sample(anomaly_paths, args.n)

    # 推理
    y_true = []
    scores = []
    for p in normal_sample:
        s = infer_score(net, p, transform, device)
        if s is not None:
            y_true.append(0)  # 0=正常
            scores.append(s)
    for p in anomaly_sample:
        s = infer_score(net, p, transform, device)
        if s is not None:
            y_true.append(1)  # 1=脏污
            scores.append(s)

    y_true = np.array(y_true)
    scores = np.array(scores)

    # 用 ROC 最优阈值（约登指数）作为判定阈值
    from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    auroc = roc_auc_score(y_true, scores)
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j = tpr - fpr
    best_th = thresholds[np.argmax(j)]

    y_pred = (scores >= best_th).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # ============ 画图 ============
    fig = plt.figure(figsize=(15, 5.5))
    fig.suptitle("SimpleNet 镜头脏污检测 —— 分类测试统计", fontsize=16, fontweight="bold")

    # 1. 混淆矩阵
    ax1 = fig.add_subplot(1, 3, 1)
    im = ax1.imshow(cm, cmap="Reds", vmin=0)
    ax1.set_title("混淆矩阵", fontsize=13, fontweight="bold")
    labels = ["正常", "脏污"]
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(labels)
    ax1.set_yticks([0, 1]); ax1.set_yticklabels(labels)
    ax1.set_xlabel("预测"); ax1.set_ylabel("真实")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                     fontsize=18, color="white" if cm[i, j] > cm.max()/2 else "black",
                     fontweight="bold")
    fig.colorbar(im, ax=ax1, fraction=0.046)

    # 2. 指标柱状图
    ax2 = fig.add_subplot(1, 3, 2)
    metrics_names = ["准确率\nAccuracy", "精确率\nPrecision", "召回率\nRecall", "F1"]
    metrics_vals = [acc, prec, rec, f1]
    bars = ax2.bar(metrics_names, metrics_vals, color=["#4caf50", "#2196f3", "#ff9800", "#e060a8"])
    ax2.set_ylim(0, 1.05)
    ax2.set_title(f"各类指标 (阈值={best_th:.3f})", fontsize=13, fontweight="bold")
    ax2.set_ylabel("分数")
    for b, v in zip(bars, metrics_vals):
        ax2.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=12, fontweight="bold")

    # 3. 分数分布 + ROC
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.hist(scores[y_true == 0], bins=30, alpha=0.6, color="#4caf50", label="正常", density=True)
    ax3.hist(scores[y_true == 1], bins=30, alpha=0.6, color="#e53935", label="脏污", density=True)
    ax3.axvline(best_th, color="black", linestyle="--", linewidth=2, label=f"阈值 {best_th:.2f}")
    ax3.set_title(f"异常分数分布 (AUROC={auroc:.3f})", fontsize=13, fontweight="bold")
    ax3.set_xlabel("异常分数"); ax3.set_ylabel("密度")
    ax3.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(args.out, exist_ok=True)
    out_png = os.path.join(args.out, "report.png")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)

    # 打印汇总
    print("===== 分类测试统计 =====")
    print(f"抽样: 正常 {int((y_true==0).sum())} 张, 脏污 {int((y_true==1).sum())} 张")
    print(f"AUROC: {auroc:.4f}")
    print(f"判定阈值(约登): {best_th:.3f}")
    print(f"混淆矩阵 [[TN, FP], [FN, TP]] = {cm.tolist()}")
    print(f"准确率: {acc:.4f}")
    print(f"精确率: {prec:.4f}")
    print(f"召回率: {rec:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"\n统计图已保存: {out_png}")


if __name__ == "__main__":
    main()
