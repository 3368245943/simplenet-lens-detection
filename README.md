# SimpleNet 镜头脏污检测

基于 [SimpleNet](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.pdf)（CVPR 2023）的镜头脏污/遮挡检测项目。

项目只做场景级二分类：判断镜头画面是否脏污或被遮挡，不定位具体位置与类型。训练阶段只使用正常样本。

## 当前发布版本

**v0.3.0（多传感器稳定候选，2026-09-24）**

- 默认部署路线：单文件 ONNX 多传感器模型，输入 `image + sensor_id`，支持 `rgb`、`stereo`、`tof`。
- 开发集结果：224 张样本，整体 F1 `0.9298`；场景交叉验证 F1 `0.9107`。
- 重要限制：上述指标不是全新录制批次的独立验收结果。正式上线前必须使用完全未参与训练、checkpoint 选择和阈值校准的新批次复测。
- 双目 patch 聚合实验最高测试 AUROC `0.9205`，仍属于候选实验，不替换默认部署模型。

版本选择、指标来源和发布门槛见 [`docs/RELEASES.md`](docs/RELEASES.md)。周期操作见 [`docs/UPDATE_CYCLE.md`](docs/UPDATE_CYCLE.md)。

## 功能入口

| 脚本 | 用途 |
|---|---|
| `run_lens.sh` | RGB 基线训练 |
| `train_sensor_heads.py` | 训练 RGB / stereo / ToF 传感器头 |
| `export_sensor_model.py` | 导出带内置阈值的单文件 ONNX |
| `infer_sensor_model.py` | 对单张图片执行传感器感知推理 |
| `evaluate_sensor_model.py` | 评估开发集并生成报告 |
| `evaluate_large_sensor_model.py` | 在更大测试池上评估并保存误检/漏检清单 |
| `evaluate_candidate_all_sensors.py` | 对候选模型进行全传感器比较 |
| `test_sensor_model.py` | 传感器模型回归测试 |
| `visualize.py` / `test_folder.py` | 可视化和文件夹批量推理 |
| `realtime_demo.py` | OpenCV 实时演示 |

## 环境

- Python 3.8
- PyTorch 2.3.1 + CUDA 12.1（GPU 推荐）
- RTX 40 系列建议使用 PyTorch 2.0 或更高版本

```bash
python3.8 -m venv venv
./venv/bin/pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 \
  --extra-index-url https://download.pytorch.org/whl/cu121
./venv/bin/pip install -r requirements.txt
```

## 数据集

数据集默认放在项目根目录 `data/`，不进入 Git：

```text
data/
  正样本/
    <场景>/<光照>/<data_x>/<时间戳>/cam0/*.jpg
  负样本/
    <场景>/<光照>/<脏污类型>/<程度>/<data_x>/<时间戳>/cam0/*.jpg
```

RGB 基线使用 `cam0`。多传感器训练需要按脚本参数提供对应的 RGB、stereo 和 ToF 数据目录。训练和测试应按录制片段划分，避免同一片段泄漏到两侧。

## 多传感器稳定主线

先训练并导出模型：

```bash
./venv/bin/python train_sensor_heads.py --datapath /path/to/data
./venv/bin/python export_sensor_model.py
```

单图推理时必须显式传入采集来源，不能根据图像外观猜测传感器：

```bash
./venv/bin/python infer_sensor_model.py --image /path/to/image.png --sensor rgb
./venv/bin/python infer_sensor_model.py --image /path/to/image.png --sensor stereo
./venv/bin/python infer_sensor_model.py --image /path/to/image.png --sensor tof
```

评估与回归：

```bash
./venv/bin/python evaluate_sensor_model.py --datapath /path/to/data
./venv/bin/python test_sensor_model.py
```

部署产物是单个 opset 11 ONNX 文件，包含骨干网络、传感器头和阈值；模型文件通常较大，默认不提交到 Git，建议通过 GitHub Release 或对象存储分发。

## RGB 基线

```bash
DATAPATH=/path/to/data bash run_lens.sh
./venv/bin/python test_report.py --datapath /path/to/data
```

RGB 基线在早期固定切分上的图像级 AUROC 约 `0.741`，主要用于对照，不是当前推荐部署路线。

## 版本与更新周期

更新按“实验记录 → 独立验证 → 发布候选 → GitHub Release”推进：

1. 在独立分支完成训练或推理改动，记录数据切分、环境、checkpoint、阈值和指标。
2. 运行编译检查和回归测试，确认旧接口没有变化。
3. 使用全新录制批次完成验收；不能只依据训练集、阈值校准集或已经查看过的固定测试集发布。
4. 更新 `CHANGELOG.md`、`docs/RELEASES.md` 和版本号，创建 `vX.Y.Z` 标签。
5. 推送到 `main` 后创建 GitHub Release；大模型文件作为 Release Asset，不放进源码提交。

GitHub Actions 会在 push 和 pull request 时执行 Python 编译检查；发布前还应在有数据和 GPU 的环境运行完整评估。

## 项目结构

```text
.
├── main.py / simplenet.py / backbones.py / resnet.py
├── train_sensor_heads.py / export_sensor_model.py / infer_sensor_model.py
├── evaluate_sensor_model.py / evaluate_large_sensor_model.py
├── datasets/
├── docs/RELEASES.md
├── docs/UPDATE_CYCLE.md
├── .github/workflows/quality.yml
└── requirements.txt
```

## 致谢

- [SimpleNet (CVPR 2023)](https://github.com/DonaldRR/SimpleNet)
- 本项目在其基础上适配自定义镜头数据集并做内存优化。
