#!/usr/bin/env python
"""Train three SimpleNet heads with one frozen backbone and save one bundle."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import models

import simplenet
import utils
from datasets.lens_dirt import DatasetSplit, LensDirtDataset, IMAGENET_MEAN, IMAGENET_STD
from sensor_thresholds import SENSORS, sensor_from_path, youden_threshold


def make_model(device, gan_epochs):
    backbone = models.wide_resnet50_2(pretrained=True)
    backbone.name = "wideresnet50"
    backbone.seed = None
    net = simplenet.SimpleNet(device)
    net.load(backbone=backbone, layers_to_extract_from=["layer2", "layer3"],
             device=device, input_shape=(3, 180, 240),
             pretrain_embed_dimension=1536, target_embed_dimension=1536,
             patchsize=3, embedding_size=256, meta_epochs=10,
             gan_epochs=gan_epochs, noise_std=0.015, dsc_hidden=1024,
             dsc_layers=2, dsc_margin=0.5, pre_proj=1)
    return net


def train_head(net, train_loader, test_loader, epochs):
    best_auc = -1.0
    best_states = None
    for epoch in range(epochs):
        net._train_discriminator(train_loader)
        scores, _, _, labels, _ = net.predict(test_loader)
        raw = np.asarray(scores).reshape(-1)
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(labels, raw))
        print(f"  epoch {epoch + 1}/{epochs}: AUROC={auc:.4f} best={max(auc, best_auc):.4f}", flush=True)
        if auc > best_auc:
            best_auc = auc
            best_states = (copy.deepcopy(net.pre_projection.state_dict()),
                           copy.deepcopy(net.discriminator.state_dict()))
    net.pre_projection.load_state_dict(best_states[0])
    net.discriminator.load_state_dict(best_states[1])
    scores, _, _, labels, _ = net.predict(test_loader)
    threshold = youden_threshold(np.asarray(labels), np.asarray(scores).reshape(-1))
    return best_auc, threshold, best_states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datapath", default="./data")
    parser.add_argument("--out", default="results/LensDirt_Results/sensor_heads/run")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gan-epochs", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.gan_epochs < 1:
        parser.error("Epoch counts must be positive")
    out = Path(args.out)
    if out.exists():
        parser.error(f"Output already exists; will not overwrite: {out}")
    utils.fix_seeds(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train = LensDirtDataset(args.datapath, split=DatasetSplit.TRAIN, frame_stride=5, seed=0)
    test = LensDirtDataset(args.datapath, split=DatasetSplit.TEST, frame_stride=5, seed=0)
    net = make_model(device, args.gan_epochs)
    heads, metrics = {}, {}
    for index, sensor in enumerate(SENSORS):
        train_indices = [i for i, row in enumerate(train.data_to_iterate) if sensor_from_path(row[2]) == sensor]
        test_indices = [i for i, row in enumerate(test.data_to_iterate) if sensor_from_path(row[2]) == sensor]
        if not train_indices or not test_indices:
            raise ValueError(f"Missing training or test samples for {sensor}")
        train_loader = DataLoader(Subset(train, train_indices), batch_size=4, shuffle=True, num_workers=0)
        test_loader = DataLoader(Subset(test, test_indices), batch_size=4, shuffle=False, num_workers=0)
        net.pre_projection = simplenet.Projection(1536, 1536, 1).to(device)
        net.discriminator = simplenet.Discriminator(1536, n_layers=2, hidden=1024).to(device)
        net.proj_opt = torch.optim.AdamW(net.pre_projection.parameters(), lr=1e-4)
        net.dsc_opt = torch.optim.Adam(net.discriminator.parameters(), lr=2e-4, weight_decay=1e-5)
        net.set_model_dir(str(out / "logs" / sensor), "lens")
        print(f"[{sensor}] train={len(train_indices)} test={len(test_indices)}", flush=True)
        auc, threshold, states = train_head(net, train_loader, test_loader, args.epochs)
        heads[sensor] = {"pre_projection": {k: v.cpu() for k, v in states[0].items()},
                         "discriminator": {k: v.cpu() for k, v in states[1].items()}}
        metrics[sensor] = {"train_normal": len(train_indices), "test_samples": len(test_indices),
                           "selected_auc": auc, "experimental_threshold": threshold}
        net.logger.logger.flush()
        net.logger.logger.close()
        print(f"[{sensor}] best AUROC={auc:.4f}, threshold={threshold:.4f}", flush=True)
    bundle = {"format": "lens_sensor_heads_v1", "sensor_names": SENSORS,
              "backbone": {k: v.cpu() for k, v in net.backbone.state_dict().items()},
              "heads": heads, "thresholds": [metrics[s]["experimental_threshold"] for s in SENSORS],
              "preprocessing": {"height": 180, "width": 240,
                                "mean": IMAGENET_MEAN, "std": IMAGENET_STD}}
    torch.save(bundle, out / "sensor_model.pth")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Single complete checkpoint: {out / 'sensor_model.pth'}", flush=True)


if __name__ == "__main__":
    main()
