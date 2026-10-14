# 与 SimpleNet 对比修改项

## 1. 基础模型

本版本仍以 SimpleNet 的 WideResNet-50 特征提取和 patch-level anomaly score 为基础。没有改变 SimpleNet 的核心异常检测思想：训练主要使用正常样本，异常样本用于评估和阈值选择。

## 2. 数据集读取修改

原始 SimpleNet 面向单一数据集/单一类别；本版本增加 `datasets/lens_dirt.py`，适配以下结构：

```text
数据根目录/
├── 正样本/
└── 负样本/
```

每个录制片段按目录组织，训练/测试按片段切分，而不是随机把同一录制片段的相邻帧拆到两侧，降低帧泄漏风险。

本版本额外根据路径识别传感器：

```text
cam0                         -> rgb
depth_data/img/left/right     -> stereo
depth_data/tof/img            -> tof
```

不能把一个混合父目录下所有图片无条件标记成某一个传感器。

## 3. 预处理修改

本版本使用与当前 ONNX metadata 一致的预处理：

1. 读取图片。
2. 转灰度。
3. Resize 到 `240x180`。
4. 复制成 3 通道以兼容 WideResNet-50。
5. 使用 ImageNet mean/std 归一化。

```text
grating: grayscale_3ch_resize_180x240_imagenet_normalize
```

这与直接保留 RGB 三通道颜色输入不同。交接、评估和实时查看必须使用同一预处理。

## 4. 多传感器 head 修改

新增 `train_sensor_heads.py`：

- 共用一个 backbone。
- RGB、双目、ToF 各自拥有独立 projection/discriminator head。
- 推理输入增加 `sensor_id`：
  - `0 = rgb`
  - `1 = stereo`
  - `2 = tof`
- 每个传感器独立计算训练指标和原始分数阈值。

这样可以避免不同传感器的分数分布直接混在一起。

## 5. 双目 q95 聚合修改

原始 max 聚合方式只取双目 patch 分数中的最大值。本版本新增 `normal_q95_count`：

1. 使用正常双目训练图像计算每个 patch 位置的 q95 参考分数。
2. 每张图片统计超过对应参考值的 patch 数量。
3. 使用该数量进行场景级判断。

ONNX 中的 metadata 必须显示：

```text
stereo_aggregation = normal_q95_count
```

本次清洗后重新训练的 q95 校准结果：

```text
patch 数：690
双目计数阈值：127
```

`max` 模型的普通分数阈值不能和 q95 计数阈值混用。之前出现的约 `2.x` 双目阈值属于 max/普通 patch score 量纲，不是 q95 计数阈值。

## 6. ONNX 导出修改

新增/扩展 `export_sensor_model.py`：

- 导出 RGB、双目、ToF 三路统一 ONNX。
- 输入：`image`、`sensor_id`。
- 输出：score、label。
- 支持 `max`、top-k 和 `normal_q95_count` 聚合。
- 导出后立即用 ONNX Runtime 对三个 sensor id 做接口和数值校验。
- 写入模型 metadata，包括传感器顺序、预处理、聚合方式和阈值。

最终部署模型必须从 `sensor_model_q95.pth` 导出，不得从普通 `sensor_model.pth` 直接默认导出后误当 q95。

## 7. 评估修改

新增/扩展 `evaluate_candidate_all_sensors.py`：

- 一次评估 RGB、双目、ToF。
- 使用明确的手工阈值，不在评估报告中重新拟合阈值。
- 输出 `metrics.json`、`predictions.json` 和 `report.png`。
- 记录模型 SHA256 和 metadata。
- 报告明确区分模型内置阈值与本次命令传入的阈值。
- 记录数据路径、frame stride、采样上限和标签策略。

## 8. 训练与评估脚本关系

```text
train_sensor_heads.py
    -> sensor_model.pth

calibrate_q95.py
    -> sensor_model_q95.pth

export_sensor_model.py --stereo-aggregation normal_q95_count
    -> sensor_model_normal_q95_count_opset11.onnx

evaluate_candidate_all_sensors.py
    -> metrics.json / predictions.json / report.png
```

## 9. 本版本没有做的事情

- 没有修改 SimpleNet 的基础网络结构。
- 没有把 RGB、双目、ToF 图片混为一个类别训练。
- 没有把 ToF 图片自动按双目处理。
- 没有把 max 阈值当作 q95 阈值。
- 没有自动排除 `光照不良`，除非数据集或命令明确指定。
- 没有修改原始数据集。
