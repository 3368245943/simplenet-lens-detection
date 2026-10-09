import os
from enum import Enum

import PIL
from PIL import ImageFile
import torch
from torchvision import transforms

# 尽量读取截断/损坏的 JPEG（PIL 默认遇到截断会报错）。
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 当前灰度实验版：灰度图整图缩放到 ToF 尺寸 240×180，再复制到 3 通道。
CROP_H = 180
CROP_W = 240


def _uniform_sample(items, k):
    """从列表中均匀抽取 k 个元素（保持顺序），k 大于列表长度时返回全部。"""
    n = len(items)
    if n == 0:
        return items
    if k is None or k <= 0 or k >= n:
        return items
    step = (n - 1) / (k - 1) if k > 1 else 0
    idxs = [int(round(i * step)) for i in range(k)]
    # 去重并保持顺序
    seen = set()
    out = []
    for i in idxs:
        if i not in seen:
            seen.add(i)
            out.append(items[i])
    return out


def _stride_sample(items, stride):
    """按固定步长抽样：每隔 stride 帧取 1 帧（frames[::stride]）。

    stride=20 表示「20 帧抽 1 张」。stride 为 None/<=0/>=len 时返回全部。
    """
    if stride is None or stride <= 1:
        return items
    return items[::stride]


def _sample_equal_by_scene(items_by_scene):
    """按场景等量抽样，目标数量取所有非空场景的最小值。"""
    nonempty = [items for items in items_by_scene.values() if items]
    if not nonempty:
        return []
    target = min(len(items) for items in nonempty)
    sampled = []
    for scene in sorted(items_by_scene):
        sampled.extend(_uniform_sample(items_by_scene[scene], target))
    return sampled


def _preferred_segments(segment_dirs, prefer_second=True):
    """每个 data_x 选择一个时间段，默认优先第二段（通常为运动状态）。"""
    grouped = {}
    for seg_dir in segment_dirs:
        timestamp_dir = os.path.dirname(seg_dir)
        data_dir = os.path.dirname(timestamp_dir)
        grouped.setdefault(data_dir, []).append(seg_dir)

    selected = []
    for data_dir in sorted(grouped):
        segs = sorted(grouped[data_dir], key=lambda p: os.path.basename(os.path.dirname(p)))
        # 第二个时间目录通常对应运动状态；只有一个时回退到唯一目录。
        if len(segs) >= 2 and prefer_second:
            selected.append(segs[1])
        else:
            selected.append(segs[0])
    return selected


def _scene_of(path):
    """从路径提取场景标签（正样本/负样本 之后的两级目录）。

    例如 `.../正样本/室内/光照不良/data_0/...` -> `室内/光照不良`。
    用于按场景分层切分训练/测试，避免某个场景整体落入测试集导致 domain 泄漏。
    """
    parts = path.split(os.sep)
    for i, x in enumerate(parts):
        if x in ("正样本", "负样本"):
            return "/".join(parts[i + 1:i + 3])
    return path


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class LensDirtDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for lens dirt / blockage detection.

    Expected source layout (as produced by the user's recording tool):

        source/
            正样本/          <- normal / clean images (negative anomaly)
                <scene>/<lighting>/<data_x>/<timestamp>/cam0/*.jpg
            负样本/          <- dirty / blocked images (positive anomaly)
                <scene>/<lighting>/<data_x>/<timestamp>/cam0/*.jpg

    The RGB camera images are the ``*.jpg`` files living inside a ``cam0``
    directory. 训练只使用 RGB（cam0）图像；以下目录会被排除：
        - ``<timestamp>/ORIGIN/cam0``、``<timestamp>/ORIGIN/cam1``（原始帧序列）
        - ``depth_data/`` 下的 left/right 双目图与 tof/img ToF 灰度图（均作为独立样本）

    ``classname`` here is a single artificial class (``"lens"``) so the dataset
    plugs into the existing SimpleNet ``main.py`` dataset machinery.
    """

    NORMAL_DIR = "正样本"
    ANOMALY_DIR = "负样本"

    def __init__(
        self,
        source,
        classname=None,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        rotate_degrees=0,
        translate=0,
        brightness_factor=0,
        contrast_factor=0,
        saturation_factor=0,
        gray_p=0,
        h_flip_p=0,
        v_flip_p=0,
        scale=0,
        max_samples_per_class=None,
        frames_per_segment=None,
        frame_stride=5,
        normal_train_frac=0.7,
        per_scene_train=None,
        seed=0,
        balance_scenes=True,
        strict_test_split=False,
        **kwargs,
    ):
        super().__init__()
        self.source = source
        self.split = split
        self.classname = classname or "lens"
        self.train_val_split = train_val_split
        self.transform_std = IMAGENET_STD
        self.transform_mean = IMAGENET_MEAN
        self.frames_per_segment = frames_per_segment
        # 每 frame_stride 帧抽 1 张（默认 20 帧抽 1 张）。
        self.frame_stride = frame_stride
        # 训练集每个场景的平均帧数（None=自动取各场景训练帧数的最小值）。
        self.per_scene_train = per_scene_train
        self.balance_scenes = balance_scenes
        self.strict_test_split = strict_test_split

        # Collect normal / anomaly image paths (grouped by recording segment).
        normal_segments, anomaly_paths = self._collect_paths()

        # ---- 只保留与负样本场景对齐的正常样本 ----
        # 负样本（脏污图）只覆盖少数场景（毛毯、室内），而正样本（干净图）
        # 额外包含黑墙/灰墙/泡沫棉/棋盘格等场景。这些"负样本里没有对应
        # 脏污图"的场景若放进训练，会让模型学到无关的场景差异，反而干扰
        # 对脏污的判断。因此训练/测试都只保留负样本中存在的场景。
        anomaly_scenes = set(_scene_of(p) for p in anomaly_paths)
        normal_segments = {
            seg: frames
            for seg, frames in normal_segments.items()
            if _scene_of(seg) in anomaly_scenes
        }

        # ---- 按"录制片段"切分正常样本，避免训练/测试数据泄漏 ----
        # 关键：按场景（正样本/负样本 后两级目录）分层切分，保证每个场景
        # 都有训练和测试片段。否则片段数少的场景（如「毛毯」只有 2 个片段）
        # 会被纯随机切分整体分到测试集，训练集从未见过该场景，导致 AUROC 失效。
        import random
        rng = random.Random(seed)
        segments = sorted(normal_segments.keys())

        # 按场景分组
        scene_segments = {}
        for seg in segments:
            scene_segments.setdefault(_scene_of(seg), []).append(seg)

        train_segments = set()
        test_segments = set()
        for scene, segs in scene_segments.items():
            segs = sorted(segs)
            aux_train = [s for s in segs if "::aux_train" in s]
            aux_test = [s for s in segs if "::aux_test" in s]
            train_segments.update(aux_train)
            test_segments.update(aux_test)
            regular = [s for s in segs if s not in aux_train and s not in aux_test]
            if self.split == DatasetSplit.TEST and not self.strict_test_split:
                test_segments.update(regular)
                continue
            # 训练实例使用运动段，按场景划分训练片段。
            if len(regular) == 1:
                train_segments.update(regular)
            elif regular:
                n_train = max(1, int(len(regular) * normal_train_frac))
                n_train = min(n_train, len(regular) - 1)
                train_segments.update(regular[:n_train])
                test_segments.update(regular[n_train:])

        train_normal = []
        test_normal = []
        # 收集各场景的训练帧，用于后续「每个场景平均」。
        train_frames_by_scene = {}
        for seg in segments:
            frames = normal_segments[seg]
            # 过滤损坏/无法读取的图片（自动跳过）。
            frames = [p for p in frames if self._is_readable(p)]
            if seg in train_segments:
                train_frames_by_scene.setdefault(_scene_of(seg), []).extend(frames)
            else:
                test_normal.extend(frames)

        # ---- 训练集按场景等量抽样 ----
        if train_frames_by_scene:
            available = [len(v) for v in train_frames_by_scene.values() if v]
            target = self.per_scene_train
            if target is None:
                target = min(available) if available else 0
            self.per_scene_train = target
            for scene in sorted(train_frames_by_scene):
                train_normal.extend(
                    _uniform_sample(train_frames_by_scene[scene], target)
                )

        # 测试正常样本也按场景等量抽样，避免场景数量影响总 AUROC。
        normal_by_scene = {}
        for p in test_normal:
            normal_by_scene.setdefault(_scene_of(p), []).append(p)
        if self.balance_scenes:
            test_normal = _sample_equal_by_scene(normal_by_scene)
        normal_by_scene = {}
        for p in test_normal:
            normal_by_scene.setdefault(_scene_of(p), []).append(p)

        # 过滤损坏的异常样本（自动跳过），并按场景等量抽样。
        anomaly_paths = [p for p in anomaly_paths if self._is_readable(p)]
        anomaly_by_scene = {}
        for p in anomaly_paths:
            anomaly_by_scene.setdefault(_scene_of(p), []).append(p)
        if self.balance_scenes:
            anomaly_paths = _sample_equal_by_scene(anomaly_by_scene)

        # B 方案下异常场景可以没有对应的正常场景（例如正样本/光照不良），
        # 因此正常和异常分别按各自场景等量抽样，不能取场景交集。
        if self.split == DatasetSplit.TEST and self.balance_scenes:
            normal_target = min((len(v) for v in normal_by_scene.values()), default=0)
            anomaly_target = min(
                min((len(v) for v in anomaly_by_scene.values()), default=0),
                len(test_normal) // max(1, len(anomaly_by_scene)),
            )
            test_normal = []
            anomaly_paths = []
            for scene in sorted(normal_by_scene):
                test_normal.extend(_uniform_sample(normal_by_scene[scene], normal_target))
            for scene in sorted(anomaly_by_scene):
                anomaly_paths.extend(_uniform_sample(anomaly_by_scene[scene], anomaly_target))

        if self.split == DatasetSplit.TRAIN:
            self.normal_paths = train_normal
            self.anomaly_paths = []  # 训练只用正常样本（无监督）。
        else:
            self.normal_paths = test_normal
            self.anomaly_paths = anomaly_paths

        # SimpleNet trains only on normal images; testing uses both.
        if self.split == DatasetSplit.TRAIN:
            self.data_to_iterate = [
                (self.classname, "good", p, None) for p in self.normal_paths
            ]
        else:
            self.data_to_iterate = [
                (self.classname, "good", p, None) for p in self.normal_paths
            ] + [
                (self.classname, "defect", p, None) for p in self.anomaly_paths
            ]

        self.transform_img = transforms.Compose(
            [
                # 灰度实验版：不使用颜色信息，复制成 3 通道以兼容预训练 WideResNet-50。
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((CROP_H, CROP_W)),
                transforms.ColorJitter(
                    brightness_factor, contrast_factor, saturation_factor
                ),
                transforms.RandomHorizontalFlip(h_flip_p),
                transforms.RandomVerticalFlip(v_flip_p),
                transforms.RandomGrayscale(gray_p),
                transforms.RandomAffine(
                    rotate_degrees,
                    translate=(translate, translate),
                    scale=(1.0 - scale, 1.0 + scale),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        self.transform_mask = transforms.Compose(
            [
                transforms.Resize((CROP_H, CROP_W)),
                transforms.ToTensor(),
            ]
        )

        self.imagesize = (3, CROP_H, CROP_W)

    def _collect_paths(self):
        """返回 (normal_segments, anomaly_paths)。

        normal_segments: dict {片段目录路径: [帧路径, ...]}（按片段分组，未过滤）。
        anomaly_paths: list（已按片段均匀抽样，未过滤）。
        """
        normal_segments = {}
        anomaly_paths = []

        for label_dir, is_anomaly in (
            (self.NORMAL_DIR, False),
            (self.ANOMALY_DIR, True),
        ):
            base = os.path.join(self.source, label_dir)
            if not os.path.isdir(base):
                continue
            # 每个 cam0 目录 = 一个独立录制片段（一段连续视频的抽帧）。
            # 只用 RGB（cam0）训练：排除 ORIGIN/ 下的原始帧序列（cam0/cam1 原始图），
            # 以及 depth_data/（tof/left/right/disp 等深度/双目数据）。
            segment_dirs = []
            for root, dirs, files in os.walk(base):
                if os.path.basename(root) != "cam0":
                    continue
                # 排除 ORIGIN 原始帧目录与 depth_data 深度目录（只留抽帧后的 RGB cam0）。
                parts = root.split(os.sep)
                if "ORIGIN" in parts or "depth_data" in parts:
                    continue
                # 手部遮挡目录包含正常/遮挡混合帧，无法仅凭目录可靠标注，暂不使用。
                if "手部遮挡" in parts:
                    continue
                segment_dirs.append(root)
            for seg_dir in _preferred_segments(
                sorted(segment_dirs),
                prefer_second=(is_anomaly or self.split == DatasetSplit.TRAIN or self.strict_test_split),
            ):
                frames = sorted(
                    f for f in os.listdir(seg_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
                )
                frames = [os.path.join(seg_dir, f) for f in frames]
                # 降采样：每 frame_stride 帧抽 1 张（20 帧抽 1 张）。
                frames = _stride_sample(frames, self.frame_stride)
                # 可选：每个录制片段再限制最多 frames_per_segment 帧（均匀抽样）。
                if self.frames_per_segment is not None and self.frames_per_segment > 0:
                    frames = _uniform_sample(frames, self.frames_per_segment)
                if is_anomaly or "光照不良" in seg_dir.split(os.sep):
                    # B 方案：光照不良本身属于异常画面，即使原目录位于正样本。
                    anomaly_paths.extend(frames)
                else:
                    normal_segments[seg_dir] = frames

            # 同步加入双目与 ToF 灰度图：各作为独立样本，保持当前模型输入接口不变。
            stereo_dirs = []
            for root, dirs, files in os.walk(base):
                if root.endswith(os.path.join("depth_data", "img", "left")) or \
                   root.endswith(os.path.join("depth_data", "img", "right")) or \
                   root.endswith(os.path.join("depth_data", "tof", "img")):
                    parts = root.split(os.sep)
                    if "手部遮挡" in parts:
                        continue
                    stereo_dirs.append(root)
            # 辅助传感器片段会按帧拆分到训练/测试，保证两边都实际包含双目/ToF。
            for stereo_dir in sorted(stereo_dirs):
                frames = sorted(
                    f for f in os.listdir(stereo_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
                )
                frames = [os.path.join(stereo_dir, f) for f in frames]
                frames = _stride_sample(frames, self.frame_stride)
                if self.frames_per_segment is not None and self.frames_per_segment > 0:
                    frames = _uniform_sample(frames, self.frames_per_segment)
                # 光照不良只改变可见光相机/双目的标签；正样本 ToF 不受环境亮度影响。
                is_tof = stereo_dir.endswith(os.path.join("depth_data", "tof", "img"))
                if is_anomaly or ("光照不良" in stereo_dir.split(os.sep) and not is_tof):
                    anomaly_paths.extend(frames)
                else:
                    cut = max(1, int(len(frames) * 0.7))
                    if self.split == DatasetSplit.TRAIN:
                        normal_segments[stereo_dir + "::aux_train"] = frames[:cut]
                    else:
                        normal_segments[stereo_dir + "::aux_test"] = frames[cut:]

        return normal_segments, anomaly_paths

    @staticmethod
    def _is_readable(path):
        """Return True if the image can actually be opened (skip corrupted)."""
        try:
            with PIL.Image.open(path) as im:
                im.load()
            return True
        except Exception:
            return False

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        # 本项目只做图像级二分类，无像素标注，不需要 mask。
        # 返回标量占位（可被 DataLoader collate），避免每次生成 (1,180,240) zeros 拖慢。
        mask = torch.tensor(0.0)

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split(os.sep)[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)
