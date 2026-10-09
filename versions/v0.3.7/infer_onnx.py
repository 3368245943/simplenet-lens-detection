#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ONNX 单帧/文件夹推理：SimpleNet 镜头脏污检测。

当前模型契约：
    输入：固定 batch=1，灰度复制为 3 通道，(1, 3, 180, 240)，宽 240、高 180
    预处理：RGB → 灰度 → 整图缩放，不裁剪，复制为 3 通道，ImageNet 归一化
    输出：单个异常分数，分数 >= THRESHOLD 判定为脏污

运行：
    ./venv/bin/python infer_onnx.py
"""

import os

import cv2
import numpy as np
import onnxruntime as ort

# ==================== 配置区 ====================
ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "simplenet_lens_B_gray240_stereo.onnx")

# 可以是单张图片，也可以是图片文件夹。
IMAGE_PATH = "/media/bai/D27A57537A573407/datatset/datasets/负样本/室内/光照良好/灰尘/大量/data_14/20260903_0923/cam0/"

# 由当前均衡测试集（每场景、每类别等量抽样）计算得到。
# 若重新训练或重新导出模型，应重新用 test_report.py 计算并更新。
THRESHOLD = 1.812

IN_H, IN_W = 180, 240
NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# ==============================================


def preprocess(image_bgr):
    """将 BGR 图像转灰度，再复制为固定输入 (1,3,180,240) 的 tensor。"""
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("输入必须是三通道 BGR 图像")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IN_W, IN_H), interpolation=cv2.INTER_LINEAR)
    arr = np.repeat(gray[..., None], 3, axis=2).astype(np.float32) / 255.0
    arr = (arr - NORM_MEAN) / NORM_STD
    return np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)


def annotate(image_bgr, score):
    """在原始尺寸图像上绘制判定结果。"""
    is_anomaly = score >= THRESHOLD
    label = "脏污" if is_anomaly else "正常"
    color = (0, 0, 255) if is_anomaly else (0, 180, 0)
    text = f"{label} {score:.3f}"

    h, w = image_bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(h, w) / 400.0)
    thickness = max(1, int(min(h, w) * 0.004))
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(image_bgr, (0, 0), (tw + 20, th + 20), color, -1)
    cv2.putText(image_bgr, text, (10, th + 10), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return image_bgr, label


def collect_images(path):
    if os.path.isdir(path):
        return sorted(
            os.path.join(path, name)
            for name in os.listdir(path)
            if name.lower().endswith(IMAGE_EXTS)
        )
    return [path]


def detect_one(session, input_name, path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        return None, None, None
    input_array = preprocess(image)
    output = session.run(None, {input_name: input_array})
    score = float(np.asarray(output[0]).reshape(-1)[0])
    annotated, label = annotate(image.copy(), score)
    return annotated, score, label


def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"[错误] 输入路径不存在: {IMAGE_PATH}")
        return
    if not os.path.isfile(MODEL_PATH):
        print(f"[错误] ONNX 模型不存在: {MODEL_PATH}")
        print("请先运行 export_onnx.py 导出当前 checkpoint。")
        return

    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    input_name = model_input.name
    input_shape = model_input.shape
    expected_shape = [1, 3, IN_H, IN_W]
    if input_shape != expected_shape:
        raise RuntimeError(
            f"ONNX 输入形状错误: {input_shape}，期望固定形状 {expected_shape}"
        )
    print(f"[信息] ONNX 输入: {input_name} {input_shape}")
    print(f"[信息] 判定阈值: {THRESHOLD:.3f}")

    paths = collect_images(IMAGE_PATH)
    if not paths:
        print(f"[错误] 文件夹中没有图片: {IMAGE_PATH}")
        return
    print(f"[信息] 待检测图片: {len(paths)} 张")

    idx = 0
    while idx < len(paths):
        path = paths[idx]
        image, score, label = detect_one(session, input_name, path)
        if image is None:
            print(f"[跳过] 无法读取: {path}")
            idx += 1
            continue

        print(f"[{idx + 1}/{len(paths)}] {label} score={score:.4f}  {os.path.basename(path)}")
        cv2.imshow("SimpleNet Lens Dirt Detection", image)

        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("b") and len(paths) > 1:
            idx = max(0, idx - 1)
        else:
            idx += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
