#!/usr/bin/env python
"""Evaluate all routes with embedded decisions; never tune on evaluation labels."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import onnxruntime as ort
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
from datasets.lens_dirt import DatasetSplit, LensDirtDataset
from evaluate_large_sensor_model import choose_records
from infer_sensor_model import preprocess
from sensor_thresholds import SENSORS, sensor_from_path


def summarize(rows):
    y = np.array([r['label'] for r in rows])
    p = np.array([r['decision'] for r in rows])
    s = np.array([r['score'] for r in rows])
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return dict(n=len(rows), normal=int(tn+fp), anomaly=int(fn+tp),
                auroc=float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None,
                pr_auc=float(average_precision_score(y, s)) if len(np.unique(y)) == 2 else None,
                confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]],
                precision=float(tp / (tp+fp)) if tp+fp else 0.,
                recall=float(tp / (tp+fn)) if tp+fn else 0.,
                f1=float(f1_score(y, p, zero_division=0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='results/LensDirt_Results/stereo_aggregation/run/sensor_model_normal_q95_count_calibrated_opset11.onnx')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = Path(args.out or f'report/all_sensors_q95_{stamp}')
    out.mkdir(parents=True, exist_ok=False)
    dataset = LensDirtDataset('./data', split=DatasetSplit.TEST, frame_stride=5, balance_scenes=False)
    records = choose_records(dataset, 200)
    session = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    meta = session.get_modelmeta().custom_metadata_map
    assert meta['sensor_ids'] == '0=rgb,1=stereo,2=tof'
    thresholds = dict(zip(SENSORS, map(float, meta['thresholds'].split(','))))
    rows = []
    for i, (path, label) in enumerate(records):
        sensor = sensor_from_path(path)
        score, decision = session.run(None, {'image': preprocess(path), 'sensor_id': np.array([SENSORS.index(sensor)], np.int64)})
        score, decision = float(score[0]), bool(decision[0])
        assert np.isfinite(score) and decision == (score >= thresholds[sensor])
        rows.append(dict(path=path, sensor=sensor, label=label, score=score, decision=int(decision)))
        if (i+1) % 100 == 0:
            print(f'{i+1}/{len(records)}', flush=True)
    groups = {sensor: [r for r in rows if r['sensor'] == sensor] for sensor in SENSORS}
    results = {sensor: summarize(group) for sensor, group in groups.items()}
    # Different score scales are not pooled for ranking metrics.
    cm = np.sum([v['confusion_matrix'] for v in results.values()], axis=0)
    overall = dict(n=len(rows), confusion_matrix=cm.tolist(), f1=float(f1_score([r['label'] for r in rows], [r['decision'] for r in rows])))
    payload = dict(model=args.model, sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest(), metadata=meta, thresholds=thresholds, results=results, overall=overall,
                   protocol='Large existing evaluation pool: frame_stride=5, all selected normals and up to 200 anomalies per sensor; embedded thresholds only. No fitting or threshold optimization.',
                   caveat='Historical development data, not an independent acceptance test. No new validation split was created. Historical training/calibration overlap is not ruled out.',
                   label_policy='ToF poor-light positive-folder images remain normal; RGB/stereo poor light is anomalous.')
    (out/'metrics.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
    (out/'predictions.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n')
    for family in ('Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans'):
        try:
            font_manager.findfont(family, fallback_to_default=False)
            plt.rcParams['font.family'] = family
            break
        except ValueError:
            continue
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for col, sensor in enumerate(SENSORS):
        group = groups[sensor]; m = results[sensor]
        y = np.array([r['label'] for r in group]); s = np.array([r['score'] for r in group])
        lo = min(s.min(), thresholds[sensor]); hi = max(s.max(), thresholds[sensor])
        bins = np.linspace(lo, max(hi, lo+.01), 26)
        ax = axes[0,col]
        for label, color, name in ((0, '#247b69', '正常'), (1, '#c25746', '异常')):
            ax.hist(s[y==label], bins=bins, alpha=.65, color=color, label=f'{name} {sum(y==label)}')
        ax.axvline(thresholds[sensor], color='#263746', linestyle='--', label=f'内置阈值 {thresholds[sensor]:.3f}')
        ax.set_title(f"{sensor} | AUROC {m['auroc']:.4f} | PR-AUC {m['pr_auc']:.4f}")
        ax.set_xlabel('分数（各路量纲不同）'); ax.set_ylabel('图片数量'); ax.legend(fontsize=9)
        ax = axes[1,col]; matrix = np.array(m['confusion_matrix'])
        ax.imshow(matrix, cmap='Blues', vmin=0, vmax=200)
        ax.set_xticks([0,1], labels=['正常','异常']); ax.set_yticks([0,1], labels=['正常','异常'])
        ax.set_xlabel('预测'); ax.set_ylabel('真实')
        ax.set_title(f"F1 {m['f1']:.4f} | Recall {m['recall']:.2%}")
        for row in range(2):
            for column in range(2):
                ax.text(column, row, str(matrix[row,column]), ha='center', va='center', fontsize=18, color='white' if matrix[row,column]>100 else '#263746')
    fig.suptitle('正常 q95 候选 ONNX：RGB / 双目 / ToF 三路评估', fontsize=17)
    fig.text(.5,.015,'使用模型内置判定；历史开发数据，不是独立盲测。各路 AUROC 不混合计算。',ha='center')
    fig.tight_layout(rect=(0,.04,1,.96)); fig.savefig(out/'report.png', dpi=150); plt.close(fig)
    print(json.dumps(payload, ensure_ascii=False, indent=2)); print('OUTPUT', out)


if __name__ == '__main__':
    main()
