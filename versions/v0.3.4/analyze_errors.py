#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
客观定位误检样本：不臆测原因，只做两件客观的事——
1) 按路径里的客观字段（场景/光照/类型/程度）分组统计误检率；
2) 把误检图片复制到文件夹，供人工查看。

误检定义（相对判定阈值，默认用 ROC 约登阈值）：
  假阳性 FP = 真实"正常" 被判成"脏污"
  假阴性 FN = 真实"脏污" 被判成"正常"

用法：
    ./venv/bin/python analyze_errors.py [--n 每类抽样数] [--out 输出目录]
"""
import os
import sys
import shutil
import argparse
from collections import defaultdict

import numpy as np
import cv2
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import backbones
from datasets.lens_dirt import LensDirtDataset, DatasetSplit, IMAGENET_MEAN, IMAGENET_STD
import simplenet


def build_model(device, imagesize=224):
    backbone = backbones.load("wideresnet50")
    backbone.name = "wideresnet50"
    backbone.seed = None
    net = simplenet.SimpleNet(device)
    net.load(
        backbone=backbone,
        layers_to_extract_from=["layer2", "layer3"],
        device=device,
        input_shape=(3, imagesize, imagesize),
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


def parse_fields(path, label):
    """从路径里客观提取 {场景, 光照, 类型, 程度} 字段，不做任何语义猜测。"""
    parts = path.split(os.sep)
    fields = {"场景": None, "光照": None, "类型": None, "程度": None}
    # 找到 正样本/负样本 之后的位置
    if label == "normal":
        key = "正样本"
    else:
        key = "负样本"
    try:
        i = parts.index(key)
    except ValueError:
        return fields
    rest = parts[i + 1:]
    if len(rest) >= 1:
        fields["场景"] = rest[0]
    if len(rest) >= 2:
        fields["光照"] = rest[1]
    # 负样本才可能有 类型/程度
    if label == "anomaly":
        if len(rest) >= 3:
            fields["类型"] = rest[2]
        if len(rest) >= 4:
            fields["程度"] = rest[3]
    return fields


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="./results/LensDirt_Results/simplenet_lens/run/models/0/lens_dirt_lens/ckpt.pth")
    parser.add_argument("--datapath", type=str,
                        default="./data")
    parser.add_argument("--n", type=int, default=300, help="每类抽样数")
    parser.add_argument("--out", type=str, default="误检分析")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"[错误] checkpoint 不存在: {args.ckpt}")
        sys.exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = build_model(device)
    load_ckpt(net, args.ckpt)

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    ds = LensDirtDataset(args.datapath, split=DatasetSplit.TEST,
                         frames_per_segment=20, imagesize=224, resize=256, seed=0)
    normal_paths = [d[2] for d in ds.data_to_iterate if d[1] == "good"]
    anomaly_paths = [d[2] for d in ds.data_to_iterate if d[1] != "good"]

    def sample(paths, k):
        if len(paths) <= k:
            return paths
        step = (len(paths) - 1) / (k - 1)
        return [paths[int(round(i * step))] for i in range(k)]

    normal_sample = sample(normal_paths, args.n)
    anomaly_sample = sample(anomaly_paths, args.n)

    # 推理收集
    records = []  # (path, label, score, fields)
    for p in normal_sample:
        s = infer_score(net, p, transform, device)
        if s is not None:
            records.append((p, "normal", s, parse_fields(p, "normal")))
    for p in anomaly_sample:
        s = infer_score(net, p, transform, device)
        if s is not None:
            records.append((p, "anomaly", s, parse_fields(p, "anomaly")))

    scores = np.array([r[2] for r in records])
    y = np.array([0 if r[1] == "normal" else 1 for r in records])

    # 阈值
    from sklearn.metrics import roc_curve
    fpr, tpr, ths = roc_curve(y, scores)
    thr = ths[np.argmax(tpr - fpr)]

    # 分类
    fp = []  # 正常误判脏污
    fn = []  # 脏污误判正常
    for r in records:
        path, label, s, fields = r
        pred = 1 if s >= thr else 0
        gt = 0 if label == "normal" else 1
        if gt == 0 and pred == 1:
            fp.append(r)
        elif gt == 1 and pred == 0:
            fn.append(r)

    print(f"===== 客观统计 (阈值 {thr:.3f}) =====")
    print(f"正常样本 {len(normal_sample)} 张, 脏污样本 {len(anomaly_sample)} 张")
    print(f"假阳性 FP(正常误判脏污): {len(fp)} 张")
    print(f"假阴性 FN(脏污误判正常): {len(fn)} 张")

    # 分组统计误检率（纯客观字段）
    def group_error_rate(recs, field, total_by_key):
        err = defaultdict(int)
        tot = defaultdict(int)
        for r in recs:
            k = r[3].get(field) or "未知"
            tot[k] += 1
        # 误检数
        for r in (fp + fn):
            k = r[3].get(field) or "未知"
            err[k] += 1
        return err, tot

    print("\n--- 按 场景 分组的误检率 (FP+FN 合计) ---")
    err, tot = group_error_rate(records, "场景", None)
    for k in sorted(tot):
        print(f"  {k}: 总 {tot[k]} 张, 误检 {err.get(k,0)} 张 ({err.get(k,0)/tot[k]*100:.1f}%)")

    print("\n--- 按 光照 分组的误检率 ---")
    err, tot = group_error_rate(records, "光照", None)
    for k in sorted(tot):
        print(f"  {k}: 总 {tot[k]} 张, 误检 {err.get(k,0)} 张 ({err.get(k,0)/tot[k]*100:.1f}%)")

    print("\n--- 按 脏污类型 分组的误检率 (仅脏污样本) ---")
    err, tot = group_error_rate([r for r in records if r[1] == "anomaly"], "类型", None)
    for k in sorted(tot):
        print(f"  {k}: 总 {tot[k]} 张, 误检 {err.get(k,0)} 张 ({err.get(k,0)/tot[k]*100:.1f}%)")

    print("\n--- 按 脏污程度 分组的误检率 (仅脏污样本) ---")
    err, tot = group_error_rate([r for r in records if r[1] == "anomaly"], "程度", None)
    for k in sorted(tot):
        print(f"  {k}: 总 {tot[k]} 张, 误检 {err.get(k,0)} 张 ({err.get(k,0)/tot[k]*100:.1f}%)")

    # 复制误检图片到文件夹
    out_fp = os.path.join(args.out, "假阳性_正常误判为脏污")
    out_fn = os.path.join(args.out, "假阴性_脏污误判为正常")
    os.makedirs(out_fp, exist_ok=True)
    os.makedirs(out_fn, exist_ok=True)
    # 清空旧内容
    for d in (out_fp, out_fn):
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))

    for r in fp:
        name = f"{r[2]:.3f}_" + os.path.basename(r[0])
        shutil.copy2(r[0], os.path.join(out_fp, name))
    for r in fn:
        name = f"{r[2]:.3f}_" + os.path.basename(r[0])
        shutil.copy2(r[0], os.path.join(out_fn, name))

    print(f"\n误检图片已复制:")
    print(f"  假阳性(正常误判脏污): {out_fp}  ({len(fp)} 张)")
    print(f"  假阴性(脏污误判正常): {out_fn}  ({len(fn)} 张)")


if __name__ == "__main__":
    main()
