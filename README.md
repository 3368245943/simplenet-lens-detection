# SimpleNet 镜头脏污检测版本仓库

项目代码按版本快照分目录保存。每个目录都是可独立查看的完整源码状态，不把不同阶段的新增实验脚本混在同一目录。

| 版本 | 内容 | 代码快照 | 报告预览 |
|---|---|---|---|
| `v0.3.0` | RGB 基线 | [`versions/v0.3.0/`](versions/v0.3.0/) | [查看报告](versions/v0.3.0/report/rgb-baseline.png) |
| `v0.3.1` | 多传感器训练、导出、推理、评估和阈值验证 | [`versions/v0.3.1/`](versions/v0.3.1/) | [开发集](versions/v0.3.1/report/sensor-heads-development.png) · [大样本](versions/v0.3.1/report/sensor-heads-large-test.png) |
| `v0.3.2` | 双目预处理诊断 | [`versions/v0.3.2/`](versions/v0.3.2/) | [分析](versions/v0.3.2/report/stereo-preprocessing-analysis.png) · [验证](versions/v0.3.2/report/stereo-preprocessing-validation.png) |
| `v0.3.3` | 双目续训对照 | [`versions/v0.3.3/`](versions/v0.3.3/) | [查看报告](versions/v0.3.3/report/stereo-finetune-comparison.png) |
| `v0.3.4` | 双目验证 | [`versions/v0.3.4/`](versions/v0.3.4/) | [查看报告](versions/v0.3.4/report/stereo-validation.png) |
| `v0.3.5` | 双目聚合实验 | [`versions/v0.3.5/`](versions/v0.3.5/) | [查看报告](versions/v0.3.5/report/stereo-aggregation.png) |
| `v0.3.6` | 全传感器候选评估 | [`versions/v0.3.6/`](versions/v0.3.6/) | [查看报告](versions/v0.3.6/report/all-sensors-large-test.png) |

每个快照目录含该阶段完整的训练/推理代码、数据适配代码、依赖说明和版本 README。数据集、模型权重和运行结果不属于源码快照。

Git 标签 `v0.3.0` 至 `v0.3.6` 仍对应相同版本状态；仓库根目录只提供版本索引，方便按目录浏览和下载。
