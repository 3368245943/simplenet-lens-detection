#!/usr/bin/env python
"""Export one self-contained ONNX with image + explicit sensor_id inputs."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision import models

from export_onnx import IN_H, IN_W, ONNXWrapper
from sensor_thresholds import SENSORS
import simplenet


class SensorONNXWrapper(ONNXWrapper):
    def __init__(self, net, projections, discriminators, thresholds, stereo_topk=1,
                 stereo_aggregation="max", stereo_patch_thresholds=None):
        super().__init__(net)
        self.stereo_topk = stereo_topk
        self.stereo_aggregation = stereo_aggregation
        if stereo_patch_thresholds is None:
            stereo_patch_thresholds = torch.empty(0, dtype=torch.float32)
        self.register_buffer("stereo_patch_thresholds", torch.as_tensor(stereo_patch_thresholds, dtype=torch.float32))
        del self.pre_projection
        del self.discriminator
        self.projections = torch.nn.ModuleList(projections)
        self.discriminators = torch.nn.ModuleList(discriminators)
        self.register_buffer("thresholds", torch.tensor(thresholds, dtype=torch.float32))

    def _aggregate(self, values, sensor_index):
        mode = getattr(self, "stereo_aggregation", "max")
        if sensor_index != 1 or mode == "max":
            return values.max()
        flat = values.reshape(-1)
        if mode == "topk_mean":
            k = min(getattr(self, "stereo_topk", 1), flat.numel())
            return torch.topk(flat, k).values.mean()
        if mode == "normal_q95_count":
            refs = getattr(self, "stereo_patch_thresholds", torch.empty(0, device=flat.device)).reshape(-1)
            if refs.numel() != flat.numel():
                raise ValueError("normal_q95_count requires one normal reference per patch")
            return (flat > refs).to(values.dtype).sum()
        raise ValueError("Unsupported stereo aggregation: " + str(mode))

    def forward(self, images, sensor_id):
        features = self._embed(images.float())
        scores = torch.stack([
            self._aggregate(-discriminator(projection(features)), index)
            for index, (projection, discriminator) in enumerate(zip(self.projections, self.discriminators))
        ])
        selected = torch.index_select(scores, 0, sensor_id).reshape(1)
        threshold = torch.index_select(self.thresholds, 0, sensor_id).reshape(1)
        valid = (sensor_id >= 0) & (sensor_id < len(SENSORS))
        selected = torch.where(valid, selected, torch.full_like(selected, float("nan")))
        return selected, valid & (selected >= threshold)


def load_wrapper(bundle_path, stereo_topk=1, stereo_aggregation="max"):
    bundle = torch.load(bundle_path, map_location="cpu")
    if bundle.get("format") != "lens_sensor_heads_v1" or tuple(bundle["sensor_names"]) != SENSORS:
        raise ValueError("Unsupported sensor model bundle")
    thresholds = bundle["thresholds"]
    if len(thresholds) != len(SENSORS) or not np.isfinite(thresholds).all():
        raise ValueError("Invalid embedded thresholds")
    patch_thresholds = bundle.get("stereo_patch_thresholds")
    if stereo_aggregation == "normal_q95_count":
        if patch_thresholds is None or len(patch_thresholds) != 690:
            raise ValueError("Bundle must contain 690 stereo normal patch thresholds")
    elif stereo_aggregation not in ("max", "topk_mean"):
        raise ValueError("Unsupported stereo aggregation: " + str(stereo_aggregation))
    backbone = models.wide_resnet50_2(weights=None)
    backbone.name = "wideresnet50"
    backbone.seed = None
    net = simplenet.SimpleNet(torch.device("cpu"))
    net.load(backbone=backbone, layers_to_extract_from=["layer2", "layer3"],
             device=torch.device("cpu"), input_shape=(3, IN_H, IN_W),
             pretrain_embed_dimension=1536, target_embed_dimension=1536,
             patchsize=3, embedding_size=256, meta_epochs=10,
             gan_epochs=4, noise_std=0.015, dsc_hidden=1024,
             dsc_layers=2, dsc_margin=0.5, pre_proj=1)
    net.backbone.load_state_dict(bundle["backbone"], strict=True)
    projections, discriminators = [], []
    for sensor in SENSORS:
        projection = simplenet.Projection(1536, 1536, 1)
        discriminator = simplenet.Discriminator(1536, n_layers=2, hidden=1024)
        projection.load_state_dict(bundle["heads"][sensor]["pre_projection"], strict=True)
        discriminator.load_state_dict(bundle["heads"][sensor]["discriminator"], strict=True)
        projections.append(projection)
        discriminators.append(discriminator)
    return SensorONNXWrapper(net, projections, discriminators, thresholds,
                             stereo_topk=stereo_topk,
                             stereo_aggregation=stereo_aggregation,
                             stereo_patch_thresholds=patch_thresholds).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default="results/LensDirt_Results/sensor_heads/run/sensor_model.pth")
    parser.add_argument("--out", default="results/LensDirt_Results/sensor_heads/run/deploy/sensor_model_opset11.onnx")
    parser.add_argument("--stereo-aggregation", choices=("max", "topk_mean", "normal_q95_count"), default="max")
    parser.add_argument("--stereo-topk", type=int, default=10, help="K for topk_mean aggregation")
    args = parser.parse_args()
    if args.stereo_topk < 1:
        parser.error("--stereo-topk must be positive")
    out = Path(args.out)
    if out.exists():
        parser.error(f"Output already exists; refusing to overwrite: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    wrapper = load_wrapper(args.bundle, stereo_topk=args.stereo_topk,
                           stereo_aggregation=args.stereo_aggregation)
    image = torch.randn(1, 3, IN_H, IN_W)
    sensor_id = torch.tensor([0], dtype=torch.long)
    with torch.no_grad():
        torch.onnx.export(wrapper, (image, sensor_id), str(out),
                          input_names=["image", "sensor_id"],
                          output_names=["anomaly_score", "is_anomaly"],
                          opset_version=11, do_constant_folding=True)
    import onnx
    import onnxruntime as ort
    model = onnx.load(str(out))
    onnx.checker.check_model(model)
    aggregation_name = (f"topk_mean_{args.stereo_topk}" if args.stereo_aggregation == "topk_mean"
                       else "normal_q95_count" if args.stereo_aggregation == "normal_q95_count" else "max")
    metadata = {"sensor_ids": "0=rgb,1=stereo,2=tof",
                "preprocessing": "grayscale_3ch_resize_180x240_imagenet_normalize",
                "stereo_aggregation": aggregation_name,
                "thresholds": ",".join(str(float(x)) for x in wrapper.thresholds)}
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save_model(model, str(out))
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 2 or len(session.get_outputs()) != 2:
        raise RuntimeError("ONNX interface is incomplete")
    for sensor in range(len(SENSORS)):
        sensor_id = torch.tensor([sensor], dtype=torch.long)
        with torch.no_grad():
            expected_score, expected_label = wrapper(image, sensor_id)
        actual_score, actual_label = session.run(None, {"image": image.numpy(),
                                                         "sensor_id": sensor_id.numpy()})
        if not np.allclose(expected_score.numpy(), actual_score, atol=1e-3) or not np.array_equal(
            expected_label.numpy(), actual_label
        ):
            raise RuntimeError(f"ONNX verification failed for {SENSORS[sensor]}")
    print(f"One ONNX model verified for RGB/stereo/ToF: {out}")


if __name__ == "__main__":
    main()
