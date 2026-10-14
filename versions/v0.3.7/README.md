# SimpleNet 镜头脏污检测 — v0.3.7 cleaned-q95 reproduction

这是清理后 q95 复现版本的独立源码快照。相较旧快照，本版本加入双目 normal-q95 patch 校准脚本，并整理训练、导出和评估流程。数据、checkpoint、ONNX 和机器专属配置不随源码发布。

## 本版本内容

- `train_sensor_heads.py`：训练 RGB / stereo / ToF 三路 head。
- `calibrate_q95.py`：用正常双目训练图计算 patch 位置 q95，并生成校准 bundle。
- `export_sensor_model.py`：导出支持 `normal_q95_count` 的单文件 ONNX。
- `evaluate_candidate_all_sensors.py`：使用明确阈值评估三路传感器。
- `view_error_images_interactive.py`：交互式逐张查看推理结果，使用环境变量配置路径。
- `训练测试交接.md`：训练、校准、导出和评估步骤。
- `与SimpleNet对比修改项.md`：数据适配及模型修改说明。

本目录没有该复现版本的报告图、指标 JSON、checkpoint 或 ONNX 文件，不用旧版本评估结果冒充本版本结果。

## 安装

建议 Python 3.8 和 PyTorch 2.3.1 / torchvision 0.18.1 CUDA 12.1：

```bash
python3.8 -m venv venv
./venv/bin/pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 \
  --extra-index-url https://download.pytorch.org/whl/cu121
./venv/bin/pip install -r requirements.txt
```

## 数据集

准备 `datasets/lens_dirt.py` 支持的目录结构，包含 `正样本/` 和 `负样本/`。下面用 `/path/to/dataset` 作为占位路径；请替换为本机数据集目录，勿将数据复制到仓库。

## 训练、q95 校准与导出

```bash
./venv/bin/python -u train_sensor_heads.py \
  --datapath /path/to/dataset \
  --out results/sensor_heads_cleaned_q95/run

./venv/bin/python calibrate_q95.py \
  --bundle results/sensor_heads_cleaned_q95/run/sensor_model.pth \
  --datapath /path/to/dataset \
  --out results/sensor_heads_cleaned_q95/run/sensor_model_q95.pth

./venv/bin/python export_sensor_model.py \
  --bundle results/sensor_heads_cleaned_q95/run/sensor_model_q95.pth \
  --out results/sensor_heads_cleaned_q95/run/sensor_model_normal_q95_count_opset11.onnx \
  --stereo-aggregation normal_q95_count
```

再运行 `evaluate_candidate_all_sensors.py`，显式传入本机模型和数据集路径。交接文档中的阈值仅为该复现记录的示例；运行前需核验模型 metadata、阈值、采样设置及标签策略。

## 验证限制

`calibrate_q95.py` 使用测试切分标签，通过 Youden J 选择 q95 计数阈值。因此这份测试切分参与了阈值校准，不能作为独立盲测结果。正式验收需要使用完全未参与训练、阈值校准或模型选择的新录制数据。

本快照未包含复现运行的评估报告产物。完成评估后，应将报告与对应模型、阈值和数据切分一并归档。

## 交互式逐图查看

```bash
export LENS_MODEL_PATH=/path/to/model.onnx
export LENS_STEREO_DIR=/path/to/stereo/images
export LENS_TOF_DIR=/path/to/tof/images
# 如需 RGB：export LENS_RGB_DIR=/path/to/rgb/images
./venv/bin/python view_error_images_interactive.py
```

阈值可通过 `LENS_THRESHOLD_RGB`、`LENS_THRESHOLD_STEREO` 和 `LENS_THRESHOLD_TOF` 覆盖。默认值仅对应交接记录，不应不经校准就用于其他模型或数据集。
