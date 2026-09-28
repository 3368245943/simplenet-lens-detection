#!/usr/bin/env python
"""Fine-tune stereo head at low LR; select on recording-group validation only."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import models
from sklearn.metrics import roc_auc_score

import simplenet
import utils
from datasets.lens_dirt import DatasetSplit, LensDirtDataset, _uniform_sample
from sensor_thresholds import sensor_from_path, youden_threshold


def recording(path):
    return path.split('/depth_data/')[0]


def partition(pool, max_anomalies):
    normal = sorted(x for x in pool.normal_paths if sensor_from_path(x) == 'stereo')
    abnormal = sorted(x for x in pool.anomaly_paths if sensor_from_path(x) == 'stereo')
    n_groups = sorted({recording(x) for x in normal})
    a_groups = sorted({recording(x) for x in abnormal})
    if len(n_groups) < 2 or len(a_groups) < 2:
        raise ValueError('Need at least two stereo recording groups for each label')
    nval = {n_groups[0]}
    aval = set(a_groups[::2])
    val_n = [x for x in normal if recording(x) in nval]
    test_n = [x for x in normal if recording(x) not in nval]
    val_a = _uniform_sample([x for x in abnormal if recording(x) in aval], max_anomalies)
    test_a = _uniform_sample([x for x in abnormal if recording(x) not in aval], max_anomalies)
    validation = [(x, 0) for x in val_n] + [(x, 1) for x in val_a]
    test = [(x, 0) for x in test_n] + [(x, 1) for x in test_a]
    assert {recording(x) for x, _ in validation}.isdisjoint({recording(x) for x, _ in test})
    return validation, test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='results/LensDirt_Results/sensor_heads/run/sensor_model.pth')
    parser.add_argument('--out', default='results/LensDirt_Results/stereo_validated/run')
    parser.add_argument('--datapath', default='./data')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--gan-epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--anomalies-per-split', type=int, default=140)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        parser.error(f'Output exists: {out}')
    if min(args.epochs, args.gan_epochs, args.anomalies_per_split) < 1 or args.lr <= 0:
        parser.error('Epochs, LR and sampling must be positive')
    utils.fix_seeds(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    bundle = torch.load(args.base, map_location='cpu')
    train = LensDirtDataset(args.datapath, split=DatasetSplit.TRAIN, frame_stride=2,
                            per_scene_train=100000, seed=0)
    pool = LensDirtDataset(args.datapath, split=DatasetSplit.TEST, frame_stride=2,
                           balance_scenes=False, strict_test_split=True, seed=0)
    validation, test = partition(pool, args.anomalies_per_split)
    train_idx = [i for i, row in enumerate(train.data_to_iterate)
                 if sensor_from_path(row[2]) == 'stereo']
    if set(x for x, _ in validation + test) & set(train.normal_paths):
        raise ValueError('Identical training and evaluation images')
    # Some source recordings are shared with the historical training set; never claim independent acceptance.
    row_idx = {row[2]: i for i, row in enumerate(pool.data_to_iterate)}
    class EvalSet(torch.utils.data.Dataset):
        def __init__(self, records): self.records = records
        def __len__(self): return len(self.records)
        def __getitem__(self, index):
            path, label = self.records[index]
            row = dict(pool[row_idx[path]])
            row['is_anomaly'] = label
            return row
    val_loader = DataLoader(EvalSet(validation), batch_size=4, shuffle=False, num_workers=0)
    loader = DataLoader(Subset(train, train_idx), batch_size=4, shuffle=True, num_workers=0)
    backbone = models.wide_resnet50_2(weights=None)
    backbone.name, backbone.seed = 'wideresnet50', None
    net = simplenet.SimpleNet(device)
    net.load(backbone=backbone, layers_to_extract_from=['layer2', 'layer3'], device=device,
             input_shape=(3, 180, 240), pretrain_embed_dimension=1536,
             target_embed_dimension=1536, patchsize=3, embedding_size=256,
             meta_epochs=args.epochs, gan_epochs=args.gan_epochs, noise_std=.015,
             dsc_hidden=1024, dsc_layers=2, dsc_margin=.5, pre_proj=1)
    net.backbone.load_state_dict(bundle['backbone'], strict=True)
    for parameter in net.backbone.parameters(): parameter.requires_grad_(False)
    old = bundle['heads']['stereo']
    net.pre_projection.load_state_dict(old['pre_projection'], strict=True)
    net.discriminator.load_state_dict(old['discriminator'], strict=True)
    net.proj_opt = torch.optim.AdamW(net.pre_projection.parameters(), lr=args.lr)
    net.dsc_opt = torch.optim.Adam(net.discriminator.parameters(), lr=args.lr, weight_decay=1e-5)
    net.set_model_dir(str(out / 'logs'), 'stereo')
    labels = np.array([y for _, y in validation])
    def validation_scores():
        net.pre_projection.eval(); net.discriminator.eval()
        with torch.no_grad(): scores, _, _, _, _ = net.predict(val_loader)
        return np.asarray(scores).reshape(-1)
    scores = validation_scores()
    best_auc = float(roc_auc_score(labels, scores))
    best_epoch = 0
    best_states = (copy.deepcopy(net.pre_projection.state_dict()),
                   copy.deepcopy(net.discriminator.state_dict()))
    history = [{'epoch': 0, 'validation_auroc': best_auc}]
    print(f'baseline val AUROC={best_auc:.4f}; train={len(train_idx)} val={len(validation)} test={len(test)}', flush=True)
    for epoch in range(1, args.epochs + 1):
        net._train_discriminator(loader)
        scores = validation_scores()
        auc = float(roc_auc_score(labels, scores))
        history.append({'epoch': epoch, 'validation_auroc': auc})
        print(f'epoch {epoch}/{args.epochs} validation AUROC={auc:.4f}', flush=True)
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            best_states = (copy.deepcopy(net.pre_projection.state_dict()),
                           copy.deepcopy(net.discriminator.state_dict()))
    net.pre_projection.load_state_dict(best_states[0]); net.discriminator.load_state_dict(best_states[1])
    threshold = float(youden_threshold(labels, validation_scores()))
    result = copy.deepcopy(bundle)
    result['heads']['stereo'] = {
        'pre_projection': {k: v.cpu() for k, v in best_states[0].items()},
        'discriminator': {k: v.cpu() for k, v in best_states[1].items()},
    }
    result['thresholds'][1] = threshold
    out.mkdir(parents=True, exist_ok=True)
    torch.save(result, out / 'sensor_model.pth')
    (out / 'split.json').write_text(json.dumps({'validation': validation, 'test': test}, ensure_ascii=False) + '\n')
    (out / 'training.json').write_text(json.dumps({
        'normal_train': len(train_idx), 'validation': len(validation), 'test': len(test),
        'best_epoch': best_epoch, 'validation_auroc': best_auc, 'validation_threshold': threshold,
        'history': history, 'learning_rate': args.lr, 'other_heads_and_backbone_unchanged': True,
        'warning': 'Train and eval share source recording directories; not an independent acceptance test.'
    }, ensure_ascii=False, indent=2) + '\n')
    print('saved:', out / 'sensor_model.pth', flush=True)


if __name__ == '__main__': main()
