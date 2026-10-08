#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实时演示：把文件夹里的帧图当作实时流，逐帧判定"正常/脏污"，
在 OpenCV 窗口里连续显示（带判定标签），按 q 退出。

用法：
    ./venv/bin/python realtime_demo.py --input /path/to/folder

    # 指定帧率（每秒显示几帧，默认 15）
    ./venv/bin/python realtime_demo.py --input /path/to/folder --fps 10

    # 循环播放
    ./venv/bin/python realtime_demo.py --input /path/to/folder --loop
"""
import os
import sys
import time
import argparse

# 抑制 OpenCV Qt 后端的字体目录警告（无害刷屏）。
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

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
    print(f"[OK] 已加载 checkpoint: {ckpt_path}")


def infer_score(net, image_bgr, transform, device):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    img = transform(img).unsqueeze(0).to(torch.float).to(device)
    with torch.no_grad():
        score = net._predict(img)[0][0]
    return float(score)


def annotate(image_bgr, score, threshold):
    h, w = image_bgr.shape[:2]
    is_dirty = score >= threshold
    label = "脏污" if is_dirty else "正常"
    color = (0, 0, 255) if is_dirty else (0, 255, 0)  # BGR 红/绿
    text = f"{label} {score:.2f}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(h, w) / 350.0
    thickness = max(2, int(min(h, w) * 0.008))
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(image_bgr, (0, 0), (tw + 20, th + 20), color, -1)
    cv2.putText(image_bgr, text, (10, th + 10), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return image_bgr


def collect_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    paths = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                paths.append(os.path.join(root, f))
    return sorted(paths)


def estimate_threshold(net, datapath, transform, device):
    ds = LensDirtDataset(datapath, split=DatasetSplit.TEST,
                         frames_per_segment=20, imagesize=224, resize=256, seed=0)
    normal = [d[2] for d in ds.data_to_iterate if d[1] == "good"][:30]
    anomaly = [d[2] for d in ds.data_to_iterate if d[1] != "good"][:30]
    ns, as_ = [], []
    for p in normal:
        img = cv2.imread(p)
        if img is not None:
            ns.append(infer_score(net, img, transform, device))
    for p in anomaly:
        img = cv2.imread(p)
        if img is not None:
            as_.append(infer_score(net, img, transform, device))
    thr = (np.median(ns) + np.median(as_)) / 2
    print(f"[阈值] {thr:.3f}")
    return thr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="帧图文件夹")
    parser.add_argument("--ckpt", type=str,
                        default="./results/LensDirt_Results/simplenet_lens/run/models/0/lens_dirt_lens/ckpt.pth")
    parser.add_argument("--datapath", type=str,
                        default="./data")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--imagesize", type=int, default=224)
    parser.add_argument("--fps", type=int, default=15, help="每秒显示帧数")
    parser.add_argument("--loop", action="store_true", help="循环播放")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"[错误] 文件夹不存在: {args.input}")
        sys.exit(1)
    if not os.path.exists(args.ckpt):
        print(f"[错误] checkpoint 不存在: {args.ckpt}")
        sys.exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[信息] 设备: {device}")

    net = build_model(device, args.imagesize)
    load_ckpt(net, args.ckpt)

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.imagesize),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    threshold = args.threshold if args.threshold is not None else \
        estimate_threshold(net, args.datapath, transform, device)

    paths = collect_images(args.input)
    if not paths:
        print(f"[错误] 文件夹里没有图片: {args.input}")
        sys.exit(1)
    print(f"[信息] 共 {len(paths)} 帧，帧率 {args.fps} fps，按 q 退出\n")

    cv2.namedWindow("SimpleNet 实时脏污检测", cv2.WINDOW_NORMAL)
    interval = 1.0 / args.fps

    idx = 0
    while True:
        p = paths[idx]
        img = cv2.imread(p)
        if img is None:
            idx = (idx + 1) % len(paths)
            continue

        score = infer_score(net, img, transform, device)
        drawn = annotate(img.copy(), score, threshold)
        cv2.imshow("SimpleNet 实时脏污检测", drawn)

        tag = "脏污" if score >= threshold else "正常"
        print(f"[{idx+1}/{len(paths)}] {tag} score={score:.3f}  {os.path.basename(p)}")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q 或 ESC 退出
            break

        time.sleep(interval)
        idx += 1
        if idx >= len(paths):
            if args.loop:
                idx = 0
            else:
                break

    cv2.destroyAllWindows()
    print("\n[结束] 演示已停止")


if __name__ == "__main__":
    main()
