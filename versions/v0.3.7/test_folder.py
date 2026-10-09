#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量测试文件夹帧图效果：对指定文件夹里的所有图片逐帧判定"正常/脏污"，
输出带标注的结果图 + 汇总统计。

用法：
    ./venv/bin/python test_folder.py --input /path/to/folder [--out 输出目录]

    # 常用：直接测某个录制片段的 cam0 目录（相对项目根的 data 目录）
    ./venv/bin/python test_folder.py --input "./data/负样本/室内/光照不良/毛/大量/data_9/20260903_0604/cam0"

参数：
    --input      必填，要测试的文件夹路径（递归找所有 jpg/png 图片）
    --ckpt       checkpoint 路径（默认用训练好的）
    --out        输出目录（默认 test_result）
    --threshold  单一判定阈值，缺省自动估算
    --sensor-thresholds  JSON 文件提供 rgb/stereo/tof 独立阈值
"""
import os
import sys
import argparse
import glob
import json

import numpy as np
import cv2
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import backbones
from datasets.lens_dirt import LensDirtDataset, DatasetSplit, IMAGENET_MEAN, IMAGENET_STD
import simplenet
from sensor_thresholds import SENSORS, sensor_from_path

# ToF 图像大小（与训练一致）：宽 240 x 高 180。
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
    print(f"[OK] 已加载 checkpoint: {ckpt_path}")


def infer_score(net, image_bgr, transform, device):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    img = transform(img).unsqueeze(0).to(torch.float).to(device)
    with torch.no_grad():
        score = net._predict(img)[0][0]
    return float(score)


def annotate(image_bgr, score, threshold, show_score=True):
    """在图片顶部标注判定结果（无检测框）。"""
    h, w = image_bgr.shape[:2]
    is_dirty = score >= threshold
    label = "脏污" if is_dirty else "正常"
    color = (0, 0, 255) if is_dirty else (0, 255, 0)  # BGR 红/绿
    conf = score if is_dirty else (1.0 - score)
    text = f"{label} {conf:.2f}" if show_score else label

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(h, w) / 400.0
    thickness = max(2, int(min(h, w) * 0.008))
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(image_bgr, (0, 0), (tw + 16, th + 16), color, -1)
    cv2.putText(image_bgr, text, (8, th + 8), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return image_bgr


def collect_images(folder):
    """递归收集文件夹里所有图片。"""
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    paths = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                paths.append(os.path.join(root, f))
    return sorted(paths)


def estimate_threshold(net, datapath, transform, device):
    """用测试集正常/脏污各若干张的中值中点估算阈值。"""
    ds = LensDirtDataset(datapath, split=DatasetSplit.TEST,
                         frame_stride=5, seed=0)
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
    print(f"[阈值] 自动估算: {thr:.3f} (正常中值 {np.median(ns):.3f} / 脏污中值 {np.median(as_):.3f})")
    return thr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="要测试的图片文件夹")
    parser.add_argument("--ckpt", type=str,
                        default="./results/LensDirt_Results/simplenet_lens/run/models/0/lens_dirt_lens/ckpt.pth")
    parser.add_argument("--datapath", type=str,
                        default="./data")
    parser.add_argument("--out", type=str, default="test_result")
    thresholds = parser.add_mutually_exclusive_group()
    thresholds.add_argument("--threshold", type=float, default=None)
    thresholds.add_argument("--sensor-thresholds", type=str,
                            help="JSON 文件，包含 rgb/stereo/tof 三路独立阈值")
    parser.add_argument("--no_annotate", action="store_true",
                        help="只输出判定结果不保存标注图")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"[错误] 文件夹不存在: {args.input}")
        sys.exit(1)
    if not os.path.exists(args.ckpt):
        print(f"[错误] checkpoint 不存在: {args.ckpt}，请先训练")
        sys.exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[信息] 设备: {device}")

    net = build_model(device)
    load_ckpt(net, args.ckpt)

    from torchvision import transforms
    # 与训练一致：先转灰度并复制为 3 通道，再整图缩放到 240x180。
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IN_H, IN_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    sensor_thresholds = None
    if args.sensor_thresholds:
        with open(args.sensor_thresholds, encoding="utf-8") as f:
            sensor_thresholds = json.load(f)
        if not isinstance(sensor_thresholds, dict) or set(sensor_thresholds) != set(SENSORS) or any(
            isinstance(sensor_thresholds[k], bool) or not isinstance(sensor_thresholds[k], (int, float))
            or not np.isfinite(sensor_thresholds[k]) for k in SENSORS
        ):
            parser.error("--sensor-thresholds 必须包含 rgb/stereo/tof 三个有限数值阈值")
        print(f"[阈值] 按传感器: {sensor_thresholds}")
    threshold = None if sensor_thresholds else (
        args.threshold if args.threshold is not None else
        estimate_threshold(net, args.datapath, transform, device)
    )

    paths = collect_images(args.input)
    print(f"[信息] 文件夹共 {len(paths)} 张图片\n")

    os.makedirs(args.out, exist_ok=True)
    n_dirty = 0
    n_normal = 0
    scores = []

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"  [跳过] 无法读取: {p}")
            continue
        score = infer_score(net, img, transform, device)
        image_threshold = sensor_thresholds[sensor_from_path(p)] if sensor_thresholds else threshold
        scores.append(score)
        is_dirty = score >= image_threshold
        if is_dirty:
            n_dirty += 1
        else:
            n_normal += 1

        name = os.path.splitext(os.path.basename(p))[0]
        tag = "脏污" if is_dirty else "正常"
        print(f"  {tag:2s} score={score:.3f}  {name}")

        if not args.no_annotate:
            drawn = annotate(img.copy(), score, image_threshold)
            out_p = os.path.join(args.out, f"{name}_{tag}_{score:.3f}.jpg")
            cv2.imwrite(out_p, drawn)

    # 汇总
    print("\n===== 汇总 =====")
    print(f"总图片数: {len(scores)}")
    if scores:
        print(f"判为脏污: {n_dirty} 张 ({n_dirty/len(scores)*100:.1f}%)")
        print(f"判为正常: {n_normal} 张 ({n_normal/len(scores)*100:.1f}%)")
        print(f"分数范围: [{min(scores):.3f}, {max(scores):.3f}]  均值 {np.mean(scores):.3f}")
    if not args.no_annotate:
        print(f"标注图已保存到: {args.out}/")


if __name__ == "__main__":
    main()
