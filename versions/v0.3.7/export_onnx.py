#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把训练好的 SimpleNet（灰度输入兼容版）导出为 ONNX。

部署输入仍为固定的 3 通道张量，但三个通道来自同一张灰度图。

输入尺寸按 ToF 图像大小写死（宽 240 x 高 180）：
    layer2: (1, 512, 23, 30) -> 690 patches
    layer3: (1, 1024, 12, 15) -> 180 patches
两层的 patch 数不同（690 vs 180），按 layer2 的空间 grid H=23, W=30 对齐，
把 layer3 插值到 23x30。

batch size 固定为 1（ONNX trace 需要静态 shape）。

输出: 单输入 image(1,3,180,240) float32 -> 单输出 anomaly_score(1,) float32
注意：部署端预处理必须是「RGB → 灰度 → resize 到 240x180 → 复制为 3 通道 + ImageNet 归一化」。
     不做灰度转换（原版 SimpleNet 直接吃 3 通道）。
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import backbones
import simplenet

# ---- 输入尺寸与由此推出的静态 shape 常量 ----
IN_H, IN_W = 180, 240
L2_H, L2_W = 23, 30      # layer2 特征图空间尺寸
L3_H, L3_W = 12, 15      # layer3 特征图空间尺寸
N_PATCH = L2_H * L2_W    # 690
CH2 = 512                # layer2 通道数
CH3 = 1024               # layer3 通道数


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


class ONNXWrapper(torch.nn.Module):
    """固定 batch=1、输入 3x180x240，所有 shape 用常量，保证 ONNX 静态推演。

    原版 SimpleNet 模型仍接收 3 通道张量；部署端应把灰度图复制到三个通道。
    """

    def __init__(self, net):
        super().__init__()
        self.forward_modules = net.forward_modules
        self.layers_to_extract_from = net.layers_to_extract_from
        self.patch_maker = net.patch_maker
        self.pre_proj = net.pre_proj
        self.pre_projection = net.pre_projection if net.pre_proj > 0 else None
        self.discriminator = net.discriminator

    def _embed(self, images):
        fm = self.forward_modules
        _ = fm["feature_aggregator"].eval()
        with torch.no_grad():
            features = fm["feature_aggregator"](images)
        features = [features[layer] for layer in self.layers_to_extract_from]

        f0 = self.patch_maker.patchify(features[0])  # (1,690,512,3,3)
        f1 = self.patch_maker.patchify(features[1])  # (1,180,1024,3,3)

        # layer2 -> (690, 512*9)
        f0 = f0.reshape(N_PATCH, CH2, 3, 3).reshape(N_PATCH, CH2 * 3 * 3)

        # layer3 (180=12x15) -> 插值到 23x30
        f1 = f1.reshape(1, L3_H, L3_W, CH3, 3, 3)
        f1 = f1.permute(0, 3, 4, 5, 1, 2)                 # (1,1024,3,3,12,15)
        f1 = f1.reshape(CH3 * 3 * 3, L3_H, L3_W)          # (9216,12,15)
        f1 = F.interpolate(
            f1.unsqueeze(1), size=(L2_H, L2_W),
            mode="bilinear", align_corners=False,
        ).squeeze(1)                                       # (9216,23,30)
        f1 = f1.reshape(CH3, 3, 3, L2_H, L2_W).permute(3, 4, 0, 1, 2)
        f1 = f1.reshape(N_PATCH, CH3 * 3 * 3)             # (690, 9216)

        pretrain_dim = fm["preprocessing"].preprocessing_modules[0].preprocessing_dim
        target_dim = fm["preadapt_aggregator"].target_dim

        p0 = F.adaptive_avg_pool1d(f0.reshape(N_PATCH, 1, -1), pretrain_dim).squeeze(1)
        p1 = F.adaptive_avg_pool1d(f1.reshape(N_PATCH, 1, -1), pretrain_dim).squeeze(1)
        stacked = torch.stack([p0, p1], dim=1)             # (690,2,1536)

        features = stacked.reshape(N_PATCH, 1, -1)
        features = F.adaptive_avg_pool1d(features, target_dim).reshape(N_PATCH, -1)
        return features

    def forward(self, images):
        images = images.to(torch.float)
        features = self._embed(images)
        if self.pre_proj > 0:
            features = self.pre_projection(features)
        scores = -self.discriminator(features)             # (690, 1)
        scores = scores.reshape(images.shape[0], -1)       # (1, 690)
        scores = scores.max(dim=-1).values                 # (1,)
        return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="results/LensDirt_Results/simplenet_lens/run/"
                                "models/0/lens_dirt_lens/ckpt.pth")
    parser.add_argument("--out", type=str, default="simplenet_lens_240x180.onnx")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.ckpt):
        print(f"[错误] checkpoint 不存在: {args.ckpt}")
        sys.exit(1)

    net = build_model(device)
    load_ckpt(net, args.ckpt)
    wrapper = ONNXWrapper(net).to(device)
    for m in wrapper.modules():
        m.eval()

    # 写死 batch=1、3 通道、180x240
    dummy = torch.randn(1, 3, IN_H, IN_W).to(device)
    with torch.no_grad():
        ref = wrapper(dummy)
    print(f"[信息] 参考分数(PyTorch): {float(ref)}")

    torch.onnx.export(
        wrapper, dummy, args.out,
        input_names=["image"], output_names=["anomaly_score"],
        opset_version=12,
        dynamic_axes=None,
        do_constant_folding=True,
    )
    print(f"[OK] ONNX 已导出: {args.out}")

    # 校验 ONNX 数值一致性
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        out = sess.run(None, {"image": dummy.cpu().numpy()})[0]
        err = float(abs(out.ravel()[0] - ref.cpu().numpy().ravel()[0]))
        print(f"[校验] ONNX 输出 {out.ravel()[0]:.6f} vs PyTorch {ref.item():.6f}, 误差 {err:.6f}")
    except Exception as e:
        print(f"[提示] onnxruntime 校验跳过: {e}")


if __name__ == "__main__":
    main()
