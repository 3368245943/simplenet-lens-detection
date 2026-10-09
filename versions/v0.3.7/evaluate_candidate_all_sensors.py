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

ROOT = Path(__file__).resolve().parent

# ===================== 用户配置区 =====================
MODEL_PATH = ROOT / 'results/LensDirt_Results/stereo_aggregation/run/sensor_model_normal_q95_count_calibrated_opset11.onnx'
DATASET_PATH = ROOT / 'data'
THRESHOLD_RGB = 1.6450339555740356
THRESHOLD_STEREO = 70.0
THRESHOLD_TOF = 2.686849355697632
FRAME_STRIDE = 5
ANOMALY_PER_SENSOR = 100000
OUTPUT_DIR = None  # 例如 ROOT / 'report/my_test'; None 表示自动按时间生成
# ======================================================


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
    parser.add_argument('--model', default=None, help='可选：覆盖文件顶部 MODEL_PATH')
    parser.add_argument('--datapath', default=None, help='可选：覆盖文件顶部 DATASET_PATH')
    parser.add_argument('--threshold-rgb', type=float, default=None, help='可选：覆盖文件顶部 THRESHOLD_RGB')
    parser.add_argument('--threshold-stereo', type=float, default=None, help='可选：覆盖文件顶部 THRESHOLD_STEREO')
    parser.add_argument('--threshold-tof', type=float, default=None, help='可选：覆盖文件顶部 THRESHOLD_TOF')
    parser.add_argument('--frame-stride', type=int, default=None)
    parser.add_argument('--anomaly-per-sensor', type=int, default=None)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    model_value = args.model or MODEL_PATH
    data_value = args.datapath or DATASET_PATH
    model_path = Path(model_value).expanduser().resolve()
    data_path = Path(data_value).expanduser().resolve()
    frame_stride = args.frame_stride if args.frame_stride is not None else FRAME_STRIDE
    anomaly_per_sensor = args.anomaly_per_sensor if args.anomaly_per_sensor is not None else ANOMALY_PER_SENSOR
    thresholds = {
        'rgb': args.threshold_rgb if args.threshold_rgb is not None else THRESHOLD_RGB,
        'stereo': args.threshold_stereo if args.threshold_stereo is not None else THRESHOLD_STEREO,
        'tof': args.threshold_tof if args.threshold_tof is not None else THRESHOLD_TOF,
    }
    if frame_stride < 1 or anomaly_per_sensor < 1:
        parser.error('FRAME_STRIDE 和 ANOMALY_PER_SENSOR 必须为正数')
    if not model_path.is_file():
        parser.error(f'ONNX 文件不存在: {model_path}')
    if not data_path.is_dir():
        parser.error(f'测试数据目录不存在: {data_path}')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = Path(args.out).expanduser().resolve() if args.out else (OUTPUT_DIR or ROOT / f'report/all_sensors_custom_{stamp}')

    out.mkdir(parents=True, exist_ok=False)
    dataset = LensDirtDataset(str(data_path), split=DatasetSplit.TEST, frame_stride=frame_stride, balance_scenes=False)
    records = choose_records(dataset, anomaly_per_sensor)
    session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    meta = session.get_modelmeta().custom_metadata_map
    assert meta['sensor_ids'] == '0=rgb,1=stereo,2=tof'
    metadata_thresholds = dict(zip(SENSORS, map(float, meta['thresholds'].split(','))))
    print('使用手工阈值:', thresholds, '；模型内置阈值仅供对照:', metadata_thresholds, flush=True)
    rows = []
    for i, (path, label) in enumerate(records):
        sensor = sensor_from_path(path)
        score, _ = session.run(None, {
            'image': preprocess(path),
            'sensor_id': np.array([SENSORS.index(sensor)], dtype=np.int64),
        })
        score = float(score[0])
        decision = int(score >= thresholds[sensor])
        assert np.isfinite(score)
        rows.append(dict(path=path, sensor=sensor, label=label, score=score, decision=int(decision)))
        if (i+1) % 100 == 0:
            print(f'{i+1}/{len(records)}', flush=True)
    groups = {sensor: [r for r in rows if r['sensor'] == sensor] for sensor in SENSORS}
    results = {sensor: summarize(group) for sensor, group in groups.items()}
    # Different score scales are not pooled for ranking metrics.
    cm = np.sum([v['confusion_matrix'] for v in results.values()], axis=0)
    overall = dict(n=len(rows), confusion_matrix=cm.tolist(), f1=float(f1_score([r['label'] for r in rows], [r['decision'] for r in rows])))
    payload = dict(model=str(model_path), sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(), metadata=meta, thresholds=thresholds, metadata_thresholds=metadata_thresholds, results=results, overall=overall,
                   protocol=f'Custom evaluation: datapath={data_path}, frame_stride={frame_stride}, up to {anomaly_per_sensor} anomalies per sensor; thresholds configured in the script.',
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
        ax.axvline(thresholds[sensor], color='#263746', linestyle='--', label=f'手工阈值 {thresholds[sensor]:.3f}')
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
    fig.suptitle('RGB / 双目 / ToF 三路自定义阈值评估', fontsize=17)
    fig.text(.5,.015,'使用命令行传入的模型路径、数据路径和三路阈值；各路 AUROC 不混合计算。',ha='center')
    fig.tight_layout(rect=(0,.04,1,.96)); fig.savefig(out/'report.png', dpi=150); plt.close(fig)
    print(json.dumps(payload, ensure_ascii=False, indent=2)); print('OUTPUT', out)


if __name__ == '__main__':
    main()
