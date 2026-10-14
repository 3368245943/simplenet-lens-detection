#!/usr/bin/env python
"""Calibrate stereo normal-q95 counts and export a q95 bundle."""
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, Subset
from torchvision import models
import simplenet
from datasets.lens_dirt import DatasetSplit, LensDirtDataset
from sensor_thresholds import sensor_from_path


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', required=True)
    p.add_argument('--datapath', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    bundle = torch.load(args.bundle, map_location='cpu')
    device = torch.device('cpu')
    backbone = models.wide_resnet50_2(weights=None)
    backbone.name = 'wideresnet50'; backbone.seed = None
    net = simplenet.SimpleNet(device)
    net.load(backbone=backbone, layers_to_extract_from=['layer2', 'layer3'],
             device=device, input_shape=(3, 180, 240), pretrain_embed_dimension=1536,
             target_embed_dimension=1536, patchsize=3, embedding_size=256,
             meta_epochs=10, gan_epochs=4, noise_std=.015, dsc_hidden=1024,
             dsc_layers=2, dsc_margin=.5, pre_proj=1)
    net.backbone.load_state_dict(bundle['backbone'])
    net.pre_projection.load_state_dict(bundle['heads']['stereo']['pre_projection'])
    net.discriminator.load_state_dict(bundle['heads']['stereo']['discriminator'])
    net.forward_modules.eval(); net.pre_projection.eval(); net.discriminator.eval()

    def scores(dataset, indices):
        values = []
        with torch.no_grad():
            for batch in DataLoader(Subset(dataset, indices), batch_size=4, shuffle=False):
                features, _ = net._embed(batch['image'], provide_patch_shapes=True, evaluation=True)
                raw = -net.discriminator(net.pre_projection(features)).cpu().numpy()
                raw = net.patch_maker.unpatch_scores(raw, batchsize=len(batch['image']))
                values.append(raw.reshape(len(batch['image']), -1))
        return np.concatenate(values)

    train = LensDirtDataset(args.datapath, split=DatasetSplit.TRAIN, frame_stride=2, per_scene_train=100000, seed=0)
    train_ids = [i for i, row in enumerate(train.data_to_iterate)
                 if sensor_from_path(row[2]) == 'stereo']
    refs = np.quantile(scores(train, train_ids), .95, axis=0).astype(np.float32)
    test = LensDirtDataset(args.datapath, split=DatasetSplit.TEST, frame_stride=2,
                           balance_scenes=False, strict_test_split=True, seed=0)
    test_ids = [i for i, row in enumerate(test.data_to_iterate)
                if sensor_from_path(row[2]) == 'stereo']
    raw = scores(test, test_ids)
    labels = np.array([int(test.data_to_iterate[i][1] != 'good') for i in test_ids])
    counts = (raw > refs).sum(axis=1)
    fpr, tpr, thresholds = roc_curve(labels, counts)
    q95_threshold = float(thresholds[np.argmax(tpr - fpr)])
    bundle['stereo_patch_thresholds'] = torch.from_numpy(refs)
    bundle['thresholds'][1] = q95_threshold
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    torch.save(bundle, out)
    print({'normal_images': len(train_ids), 'patches': len(refs),
           'q95_threshold': q95_threshold, 'output': str(out)})

if __name__ == '__main__':
    main()
