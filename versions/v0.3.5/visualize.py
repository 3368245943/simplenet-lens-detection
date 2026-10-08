#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SimpleNet 镜头脏污检测 —— YOLO 风格可视化。

加载训练好的 checkpoint，对图片推理，在画面顶部标注整体判定
（"脏污"/"正常"）+ 置信度分数，不定位脏污位置（只做场景级判定）。

用法：
    # 对测试集抽样图片可视化
    ./venv/bin/python visualize.py --n 8

    # 对指定图片/文件夹可视化
    ./venv/bin/python visualize.py --input /path/to/img.jpg
    ./venv/bin/python visualize.py --input /path/to/folder
"""
import os
import sys
import argparse

import numpy as np
import cv2
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import backbones
from datasets.lens_dirt import LensDirtDataset, DatasetSplit, IMAGENET_MEAN, IMAGENET_STD
import simplenet


def build_model(device, imagesize=224):
    """按 run_lens.sh 的参数重建 SimpleNet 模型结构。"""
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


def infer(net, image_bgr, transform, device):
    """对 BGR 图像推理，返回异常分数（越高越可能脏污）。"""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    img = transform(img).unsqueeze(0).to(torch.float).to(device)
    with torch.no_grad():
        score = net._predict(img)[0][0]
    return float(score)


def draw_detection(image_bgr, score, threshold):
    """在图像上标注判定结果（无检测框，仅文字标签）。"""
    h, w = image_bgr.shape[:2]
    is_dirty = score >= threshold
    label = "脏污" if is_dirty else "正常"
    color = (0, 0, 255) if is_dirty else (0, 255, 0)  # BGR: 红/绿
    conf = score if is_dirty else (1.0 - score)
    text = f"{label} {conf:.2f}"

    # 顶部标签条（无矩形框）
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(h, w) / 400.0
    thickness = max(2, int(min(h, w) * 0.008))
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    # 标签条背景
    cv2.rectangle(image_bgr, (0, 0), (tw + 16, th + 16), color, -1)
    cv2.putText(image_bgr, text, (8, th + 8), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return image_bgr


def process_paths(net, paths, transform, device, threshold, out_dir):
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        score = infer(net, img, transform, device)
        drawn = draw_detection(img.copy(), score, threshold)
        name = os.path.splitext(os.path.basename(p))[0]
        tag = "dirty" if score >= threshold else "normal"
        out_p = os.path.join(out_dir, f"{tag}_{score:.3f}_{name}.jpg")
        cv2.imwrite(out_p, drawn)
        print(f"  {tag:6s} score={score:.3f}  ->  {out_p}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="./results/LensDirt_Results/simplenet_lens/run/models/0/lens_dirt_lens/ckpt.pth")
    parser.add_argument("--datapath", type=str,
                        default="./data")
    parser.add_argument("--input", type=str, default=None,
                        help="指定单张图片或文件夹；缺省则从测试集抽样")
    parser.add_argument("--n", type=int, default=8, help="每类抽样图片数")
    parser.add_argument("--out", type=str, default="visualization")
    parser.add_argument("--imagesize", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=None,
                        help="判定阈值；缺省自动取正常/脏污分数均值中点")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"[错误] checkpoint 不存在: {args.ckpt}")
        print("请先运行 bash run_lens.sh 完成训练。")
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

    os.makedirs(args.out, exist_ok=True)

    # 判定阈值：默认用正常/脏污各抽一批的中值中点；用户也可手动指定
    if args.input is None:
        ds = LensDirtDataset(args.datapath, split=DatasetSplit.TEST,
                             frames_per_segment=20, imagesize=args.imagesize,
                             resize=256, seed=0)
        normal_paths = [d[2] for d in ds.data_to_iterate if d[1] == "good"]
        anomaly_paths = [d[2] for d in ds.data_to_iterate if d[1] != "good"]

        def sample(paths, k):
            if len(paths) <= k:
                return paths
            step = (len(paths) - 1) / (k - 1)
            return [paths[int(round(i * step))] for i in range(k)]

        normal_sample = sample(normal_paths, args.n)
        anomaly_sample = sample(anomaly_paths, args.n)

        # 用抽样图估阈值
        ns, as_ = [], []
        for p in normal_sample:
            img = cv2.imread(p)
            if img is not None:
                ns.append(infer(net, img, transform, device))
        for p in anomaly_sample:
            img = cv2.imread(p)
            if img is not None:
                as_.append(infer(net, img, transform, device))
        if args.threshold is None:
            threshold = (np.median(ns) + np.median(as_)) / 2
        else:
            threshold = args.threshold
        print(f"[阈值] 自动计算: {threshold:.3f} (正常中值 {np.median(ns):.3f} / 脏污中值 {np.median(as_):.3f})")

        print("\n[生成] 正常样本标注图:")
        process_paths(net, normal_sample, transform, device, threshold, args.out)
        print("\n[生成] 脏污样本标注图:")
        process_paths(net, anomaly_sample, transform, device, threshold, args.out)
    else:
        if args.threshold is None:
            # 用户指定输入但没给阈值时，仍按测试集估算
            ds = LensDirtDataset(args.datapath, split=DatasetSplit.TEST,
                                 frames_per_segment=20, imagesize=args.imagesize,
                                 resize=256, seed=0)
            normal_paths = [d[2] for d in ds.data_to_iterate if d[1] == "good"]
            anomaly_paths = [d[2] for d in ds.data_to_iterate if d[1] != "good"]
            ns, as_ = [], []
            for p in normal_paths[:20]:
                img = cv2.imread(p)
                if img is not None:
                    ns.append(infer(net, img, transform, device))
            for p in anomaly_paths[:20]:
                img = cv2.imread(p)
                if img is not None:
                    as_.append(infer(net, img, transform, device))
            threshold = (np.median(ns) + np.median(as_)) / 2
        else:
            threshold = args.threshold
        print(f"[阈值] {threshold:.3f}")

        if os.path.isdir(args.input):
            paths = sorted(
                os.path.join(args.input, f) for f in os.listdir(args.input)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            )
        else:
            paths = [args.input]
        print(f"\n[生成] 标注图 ({len(paths)} 张):")
        process_paths(net, paths, transform, device, threshold, args.out)

    print(f"\n[完成] 所有标注图已保存到: {args.out}/")


if __name__ == "__main__":
    main()
