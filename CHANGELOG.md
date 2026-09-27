# Changelog

## [0.3.0] - 2026-09-24

### Added

- Added the sensor-aware training, export, inference and evaluation path for RGB, stereo and ToF.
- Added a single-file ONNX deployment flow with embedded sensor-specific thresholds.
- Added release evidence and a repeatable update-cycle checklist.
- Added CI compile checks for every push and pull request.

### Validation

- Development set: 224 samples, overall F1 `0.9298`.
- Scene cross-validation: F1 `0.9107`.
- These metrics are not independent acceptance metrics; a fresh recording batch is required before production deployment.

### Candidate experiments

- Stereo top-10 patch aggregation reached test AUROC `0.9205` on the fixed split.
- It remains a candidate experiment and is not the default deployment model.

## [0.2.0] - 2026-09-17

- Published the RGB SimpleNet baseline and its reproducible evaluation report.
- RGB baseline image-level AUROC was approximately `0.741` on the fixed split.

## [0.1.0]

- Initial SimpleNet lens dirt detection implementation.
