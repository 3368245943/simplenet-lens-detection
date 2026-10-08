#!/usr/bin/env python
"""Continue training only the stereo projection/discriminator; preserve the other heads."""
import argparse
import copy
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import models

import simplenet
import utils
from datasets.lens_dirt import DatasetSplit, LensDirtDataset
from sensor_thresholds import sensor_from_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="results/LensDirt_Results/sensor_heads/run/sensor_model.pth")
    parser.add_argument("--out", default="results/LensDirt_Results/stereo_finetune/run")
    parser.add_argument("--datapath", default="./data")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--gan-epochs", type=int, default=4)
    args = parser.parse_args()
    if min(args.stride, args.epochs, args.gan_epochs) < 1:
        parser.error("Stride and epoch counts must be positive")
    out = Path(args.out)
    if out.exists():
        parser.error(f"Refusing to overwrite: {out}")
    utils.fix_seeds(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bundle = torch.load(args.base, map_location="cpu")
    if bundle.get("format") != "lens_sensor_heads_v1":
        raise ValueError("Unsupported base checkpoint")
    ds = LensDirtDataset(args.datapath, split=DatasetSplit.TRAIN,
                         frame_stride=args.stride, per_scene_train=100000, seed=0)
    indices = [i for i, row in enumerate(ds.data_to_iterate) if sensor_from_path(row[2]) == "stereo"]
    if not indices:
        raise ValueError("No stereo normal training images")
    loader = DataLoader(Subset(ds, indices), batch_size=4, shuffle=True, num_workers=0)
    backbone = models.wide_resnet50_2(weights=None)
    backbone.name, backbone.seed = "wideresnet50", None
    net = simplenet.SimpleNet(device)
    net.load(backbone=backbone, layers_to_extract_from=["layer2", "layer3"],
             device=device, input_shape=(3, 180, 240),
             pretrain_embed_dimension=1536, target_embed_dimension=1536,
             patchsize=3, embedding_size=256, meta_epochs=args.epochs,
             gan_epochs=args.gan_epochs, noise_std=0.015, dsc_hidden=1024,
             dsc_layers=2, dsc_margin=0.5, pre_proj=1)
    net.backbone.load_state_dict(bundle["backbone"], strict=True)
    for parameter in net.backbone.parameters():
        parameter.requires_grad_(False)
    stereo = bundle["heads"]["stereo"]
    net.pre_projection.load_state_dict(stereo["pre_projection"], strict=True)
    net.discriminator.load_state_dict(stereo["discriminator"], strict=True)
    net.proj_opt = torch.optim.AdamW(net.pre_projection.parameters(), lr=1e-5)
    net.dsc_opt = torch.optim.Adam(net.discriminator.parameters(), lr=2e-5, weight_decay=1e-5)
    net.set_model_dir(str(out / "logs"), "stereo")
    print(f"Training only stereo head: {len(indices)} normal frames, {args.epochs} epochs", flush=True)
    for epoch in range(args.epochs):
        net._train_discriminator(loader)
        print(f"Finished stereo epoch {epoch+1}/{args.epochs}", flush=True)
    net.logger.logger.flush()
    net.logger.logger.close()
    result = copy.deepcopy(bundle)
    result["heads"]["stereo"] = {
        "pre_projection": {k: v.detach().cpu() for k, v in net.pre_projection.state_dict().items()},
        "discriminator": {k: v.detach().cpu() for k, v in net.discriminator.state_dict().items()},
    }
    # Keep the original thresholds: the holdout is used only for evaluation.
    torch.save(result, out / "sensor_model.pth")
    (out / "training.json").write_text(json.dumps({
        "base": args.base, "stereo_normal_train": len(indices), "frame_stride": args.stride,
        "epochs": args.epochs, "gan_epochs_per_epoch": args.gan_epochs,
        "thresholds_unchanged": True, "backbone_rgb_tof_unchanged": True,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved new bundle: {out / 'sensor_model.pth'}", flush=True)


if __name__ == "__main__":
    main()
