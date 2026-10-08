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
    directory (depth / ``*.png`` / ``*.yaml`` files are ignored).

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
        normal_train_frac=0.7,
        seed=0,
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

        # Collect normal / anomaly image paths (grouped by recording segment).
        normal_segments, anomaly_paths = self._collect_paths()

        # ---- 按"录制片段"切分正常样本，避免训练/测试数据泄漏 ----
        # 70% 的片段用于训练，30% 的片段整体留作测试（片段内帧高度相似，
        # 若同片段既训练又测试，AUROC 会虚高）。
        import random
        rng = random.Random(seed)
        segments = sorted(normal_segments.keys())
        rng.shuffle(segments)
        n_train_seg = max(1, int(len(segments) * normal_train_frac))
        train_segments = set(segments[:n_train_seg])
        test_segments = set(segments[n_train_seg:])

        train_normal = []
        test_normal = []
        for seg in segments:
            frames = normal_segments[seg]
            # 过滤损坏/无法读取的图片（自动跳过）。
            frames = [p for p in frames if self._is_readable(p)]
            if seg in train_segments:
                train_normal.extend(frames)
            else:
                test_normal.extend(frames)

        # 过滤损坏的异常样本（自动跳过）。
        anomaly_paths = [p for p in anomaly_paths if self._is_readable(p)]

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
                transforms.Resize(resize),
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
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        self.transform_mask = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
            ]
        )

        self.imagesize = (3, imagesize, imagesize)

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
            segment_dirs = []
            for root, dirs, files in os.walk(base):
                if os.path.basename(root) == "cam0":
                    segment_dirs.append(root)
            for seg_dir in sorted(segment_dirs):
                frames = sorted(
                    f for f in os.listdir(seg_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                )
                frames = [os.path.join(seg_dir, f) for f in frames]
                # 按片段均匀抽样：每个录制片段最多保留 frames_per_segment 帧。
                if self.frames_per_segment is not None and self.frames_per_segment > 0:
                    frames = _uniform_sample(frames, self.frames_per_segment)
                if is_anomaly:
                    anomaly_paths.extend(frames)
                else:
                    normal_segments[seg_dir] = frames

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

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

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
