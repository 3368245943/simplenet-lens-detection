import unittest

import numpy as np

from sensor_thresholds import evaluate_by_scene, sensor_from_path, youden_threshold


class SensorThresholdsTest(unittest.TestCase):
    def test_sensor_detection_uses_directory_not_filename(self):
        self.assertEqual(sensor_from_path("data/正样本/室内/光照不良/depth_data/tof/img/cam0.jpg"), "tof")
        self.assertEqual(sensor_from_path("data/正样本/室内/光照良好/depth_data/img/left/1.png"), "stereo")
        self.assertEqual(sensor_from_path("data/正样本/室内/光照良好/data_0/cam0/1.jpg"), "rgb")
        with self.assertRaises(ValueError):
            sensor_from_path("data/unknown/image.png")

    def test_each_scene_is_predicted_with_other_scenes_thresholds(self):
        paths, labels, scores = [], [], []
        for scene in ("室内/光照良好", "室内/强光", "毛毯/光照良好"):
            for sensor, directory, offset in (
                ("rgb", "cam0", 0.0),
                ("tof", "depth_data/tof/img", 10.0),
            ):
                for label, category, value in ((0, "正样本", 0.5), (1, "负样本", 1.5)):
                    paths.append(f"data/{category}/{scene}/{directory}/{sensor}_{label}.png")
                    labels.append(label)
                    scores.append(value + offset)
        result, global_pred, sensor_pred = evaluate_by_scene(paths, labels, scores)
        self.assertEqual(len(result["folds"]), 3)
        self.assertEqual(len(global_pred), len(sensor_pred))
        self.assertEqual(result["sensor_aware"]["confusion_matrix"], [[6, 0], [0, 6]])
        self.assertLess(result["global"]["accuracy"], result["sensor_aware"]["accuracy"])

    def test_threshold_needs_both_classes(self):
        with self.assertRaises(ValueError):
            youden_threshold(np.array([0, 0]), np.array([0.2, 0.3]))


if __name__ == "__main__":
    unittest.main()
