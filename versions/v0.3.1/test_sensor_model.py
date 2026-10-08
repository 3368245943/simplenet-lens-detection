import unittest

import torch

from export_sensor_model import SensorONNXWrapper


class TinySensorWrapper(SensorONNXWrapper):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.projections = torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])
        self.discriminators = torch.nn.ModuleList([
            torch.nn.Linear(1, 1, bias=False) for _ in range(3)
        ])
        for sensor, layer in enumerate(self.discriminators, 1):
            layer.weight.data.fill_(-sensor)
        self.register_buffer("thresholds", torch.tensor([2.0, 7.0, 8.0]))

    def _embed(self, image):
        return torch.tensor([[1.0], [3.0]])


class SensorModelTest(unittest.TestCase):
    def test_explicit_id_selects_matching_head_and_threshold(self):
        model = TinySensorWrapper()
        for sensor, expected_score, expected_alarm in ((0, 3.0, True),
                                                        (1, 6.0, False),
                                                        (2, 9.0, True)):
            with self.subTest(sensor=sensor):
                score, alarm = model(torch.zeros(1, 3, 180, 240), torch.tensor([sensor]))
                self.assertEqual(score.item(), expected_score)
                self.assertEqual(alarm.item(), expected_alarm)


if __name__ == "__main__":
    unittest.main()
