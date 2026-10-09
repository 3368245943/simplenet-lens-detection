import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from datasets.lens_dirt import DatasetSplit, LensDirtDataset


class LightingLabelsTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.source = Path(self.temp_dir.name)
        self.clean = self.source / "正样本" / "室内" / "光照不良" / "data_0" / "recording"
        self.dirty = self.source / "负样本" / "室内" / "光照不良" / "data_1" / "recording"
        for base in (self.clean, self.dirty):
            for relative in ("cam0", "depth_data/img/left", "depth_data/img/right",
                             "depth_data/tof/img"):
                folder = base / relative
                folder.mkdir(parents=True)
                for index in range(4):
                    Image.new("RGB", (4, 4)).save(folder / f"{index}.png")

    def test_poor_light_clean_tof_stays_normal_in_train_and_test(self):
        clean_tof = str(self.clean / "depth_data/tof/img") + os.sep
        dirty_tof = str(self.dirty / "depth_data/tof/img") + os.sep
        for split in (DatasetSplit.TRAIN, DatasetSplit.TEST):
            with self.subTest(split=split):
                ds = LensDirtDataset(str(self.source), split=split, frame_stride=1)
                normal = [p for _, label, p, _ in ds.data_to_iterate if label == "good"]
                anomaly = [p for _, label, p, _ in ds.data_to_iterate if label == "defect"]
                self.assertTrue(any(p.startswith(clean_tof) for p in normal))
                self.assertFalse(any(p.startswith(clean_tof) for p in anomaly))
                raw_normal, raw_anomaly = ds._collect_paths()
                raw_normal_paths = [p for frames in raw_normal.values() for p in frames]
                self.assertTrue(any(p.startswith(clean_tof) for p in raw_normal_paths))
                self.assertFalse(any(p.startswith(clean_tof) for p in raw_anomaly))
                self.assertTrue(any(p.startswith(dirty_tof) for p in raw_anomaly))
                for relative in ("cam0", "depth_data/img/left", "depth_data/img/right"):
                    prefix = str(self.clean / relative) + os.sep
                    self.assertTrue(any(p.startswith(prefix) for p in raw_anomaly))
                    self.assertFalse(any(p.startswith(prefix) for p in raw_normal_paths))
                if split == DatasetSplit.TEST:
                    self.assertTrue(any(p.startswith(dirty_tof) for p in anomaly))

    def test_strict_unbalanced_test_has_no_training_file_overlap(self):
        train = LensDirtDataset(str(self.source), split=DatasetSplit.TRAIN, frame_stride=1)
        test = LensDirtDataset(str(self.source), split=DatasetSplit.TEST, frame_stride=1,
                               strict_test_split=True, balance_scenes=False)
        self.assertFalse(set(train.normal_paths) & set(test.normal_paths))
        self.assertTrue(test.normal_paths)
        self.assertTrue(test.anomaly_paths)


if __name__ == "__main__":
    unittest.main()
