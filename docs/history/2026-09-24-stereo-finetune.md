# v0.3.3：双目续训对照

日期：2026-09-24

本阶段只作为反例记录：使用 318 张双目正常帧续训双目 projection/discriminator，冻结共享骨干及 RGB / ToF 分支。

| 模型 | AUROC | TN | FP | FN | TP | F1 |
|---|---:|---:|---:|---:|---:|---:|
| 原模型 | 0.8551 | 103 | 37 | 68 | 232 | 0.8155 |
| 双目续训 | 0.8200 | 1 | 139 | 0 | 300 | 0.8119 |

结论：不采用续训模型。AUROC 下降，且原阈值下正常双目几乎全部误报；没有替换默认部署模型。

对应入口：`finetune_stereo_head.py`、`compare_stereo_finetune.py`、`report/stereo_finetune/README.md`（源实验目录）。
