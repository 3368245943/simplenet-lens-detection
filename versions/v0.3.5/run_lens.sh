#!/usr/bin/env bash
# SimpleNet 训练脚本 —— 镜头脏污/遮挡检测 (二分类异常检测)
# 使用自定义数据集 datasets/lens_dirt.py，正样本=正常，负样本=脏污/遮挡。
#
# 收敛验证：每录制片段均匀抽 20 帧去冗余（121片段 → ~2400张），10 个 meta-epoch。
# 实时进度条会显示在终端（tqdm），无需后台运行。

set -e

cd "$(dirname "$0")"

PYTHON=${PYTHON:-./venv/bin/python}
# 数据集路径：默认取项目内 data 目录，也可用环境变量覆盖：
#   DATAPATH=/path/to/data bash run_lens.sh
DATAPATH=${DATAPATH:-./data}

# 单类检测（把"镜头脏污"当作一个类别）
CLASS=lens

# -u 强制无缓冲输出，保证进度条实时刷新到终端。
exec $PYTHON -u main.py \
  --gpu 0 \
  --seed 0 \
  --log_group simplenet_lens \
  --log_project LensDirt_Results \
  --results_path results \
  --run_name run \
  net \
  -b wideresnet50 \
  -le layer2 \
  -le layer3 \
  --pretrain_embed_dimension 1536 \
  --target_embed_dimension 1536 \
  --patchsize 3 \
  --meta_epochs 10 \
  --embedding_size 256 \
  --gan_epochs 4 \
  --noise_std 0.015 \
  --dsc_hidden 1024 \
  --dsc_layers 2 \
  --dsc_margin .5 \
  --pre_proj 1 \
  dataset \
  --batch_size 4 \
  --num_workers 0 \
  --resize 256 \
  --imagesize 224 \
  --frames_per_segment 20 \
  -d $CLASS lens_dirt $DATAPATH
