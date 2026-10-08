#!/usr/bin/env python
"""Compare max, top-k and normal-only patch calibration on fixed stereo splits."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import onnxruntime as ort
from PIL import ImageFile
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
from infer_sensor_model import preprocess
from sensor_thresholds import youden_threshold

ImageFile.LOAD_TRUNCATED_IMAGES = True

def score_model(path, records):
    session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    scores = []
    for image_path, _ in records:
        score, _ = session.run(None, {
            'image': preprocess(image_path),
            'sensor_id': np.array([1], dtype=np.int64),
        })
        scores.append(float(score[0]))
    return np.asarray(scores)

def summarize(labels, scores, threshold):
    prediction = scores >= threshold
    return {
        'n': int(len(labels)),
        'normal': int((labels == 0).sum()),
        'anomaly': int((labels == 1).sum()),
        'auroc': float(roc_auc_score(labels, scores)),
        'pr_auc': float(average_precision_score(labels, scores)),
        'threshold_from_validation': float(threshold),
        'f1': float(f1_score(labels, prediction)),
        'confusion_matrix': confusion_matrix(labels, prediction, labels=[0, 1]).tolist(),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old', default='results/LensDirt_Results/sensor_heads/run/deploy/sensor_model_opset11.onnx')
    parser.add_argument('--topk', default='results/LensDirt_Results/stereo_aggregation/run/sensor_model_top10_calibrated_opset11.onnx')
    parser.add_argument('--normal-calibrated', default='results/LensDirt_Results/stereo_aggregation/run/sensor_model_normal_q95_count_calibrated_opset11.onnx')
    parser.add_argument('--split', default='results/LensDirt_Results/stereo_validated/run_v2/split.json')
    parser.add_argument('--out', default='report/stereo_aggregation')
    args = parser.parse_args()
    splits = json.loads(Path(args.split).read_text())
    y_val = np.asarray([label for _, label in splits['validation']])
    y_test = np.asarray([label for _, label in splits['test']])
    models = {
        '原模型 max': args.old,
        'top-10 平均': args.topk,
        '正常 q95 超阈计数': args.normal_calibrated,
    }
    predictions, results = {}, {}
    for name, path in models.items():
        validation_scores = score_model(path, splits['validation'])
        test_scores = score_model(path, splits['test'])
        threshold = youden_threshold(y_val, validation_scores)
        predictions[name] = {'validation': validation_scores, 'test': test_scores}
        results[name] = {
            'validation': summarize(y_val, validation_scores, threshold),
            'test': summarize(y_test, test_scores, threshold),
        }
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        'results': results,
        'selection': 'normal_q95_count selected by validation AUROC; image thresholds fit on validation only using Youden J; test labels are not used for selection.',
        'training': 'No anomalous image was used for training. Per-location q95 reference uses only normal stereo training images.',
        'caveat': 'The fixed test split was inspected in prior experiments and is not a pristine independent acceptance set.',
    }
    (output / 'comparison.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    for family in ('Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans'):
        try:
            font_manager.findfont(family, fallback_to_default=False)
            plt.rcParams['font.family'] = family
            break
        except ValueError:
            continue
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    colors = {0: '#247b69', 1: '#c25746'}
    for column, name in enumerate(models):
        scores = predictions[name]['test']
        threshold = results[name]['test']['threshold_from_validation']
        minimum, maximum = float(scores.min()), float(scores.max())
        bins = np.linspace(minimum, maximum if maximum > minimum else minimum + 1e-3, 25)
        ax = axes[0, column]
        ax.hist(scores[y_test == 0], bins=bins, color=colors[0], alpha=.72,
                label=f'正常 {int((y_test == 0).sum())}')
        ax.hist(scores[y_test == 1], bins=bins, color=colors[1], alpha=.63,
                label=f'异常 {int((y_test == 1).sum())}')
        ax.axvline(threshold, color='#263746', linestyle='--', linewidth=1.8,
                   label=f'验证集阈值 {threshold:.2f}')
        metrics = results[name]['test']
        ax.set_title(f"{name} · AUROC {metrics['auroc']:.3f} · PR-AUC {metrics['pr_auc']:.3f}")
        ax.set_xlabel('异常分数'); ax.set_ylabel('图片数'); ax.legend(frameon=False, fontsize=9)
        cm = np.asarray(metrics['confusion_matrix'])
        ax = axes[1, column]
        ax.imshow(cm, cmap='Blues', vmin=0, vmax=max(1, int(cm.max())))
        ax.set_title(f"测试集混淆矩阵 · F1 {metrics['f1']:.3f}")
        ax.set_xticks([0, 1], labels=['正常', '异常']); ax.set_yticks([0, 1], labels=['正常', '异常'])
        ax.set_xlabel('预测'); ax.set_ylabel('真实')
        for row in range(2):
            for col in range(2):
                ax.text(col, row, str(int(cm[row, col])), ha='center', va='center', fontsize=16,
                        color='white' if cm[row, col] > cm.max() / 2 else '#263746')
    fig.suptitle('双目纯无监督分数聚合对比', fontsize=16)
    fig.text(.5, .015, '阈值仅由验证集校准；固定测试集曾用于早期诊断，需新录制会话做最终验收。',
             ha='center', fontsize=9, color='#596775')
    fig.tight_layout(rect=(0, .04, 1, .96))
    fig.savefig(output / 'report.png', dpi=150)
    plt.close(fig)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print('PNG:', output / 'report.png')

if __name__ == '__main__':
    main()
