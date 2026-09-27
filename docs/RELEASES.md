# Release Evidence

## Version timeline

| Version | Date | Route | Evidence | Status |
|---|---|---|---|---|
| `v0.1.0` | initial | SimpleNet adaptation | Initial implementation | historical |
| `v0.2.0` | 2026-09-17 | RGB baseline | AUROC `0.741`, F1 `0.679` | baseline |
| `v0.3.0` | 2026-09-24 | Sensor-aware RGB / stereo / ToF | User-confirmed usable version; development F1 `0.9298`, scene CV F1 `0.9107` | confirmed usable; fresh-batch acceptance still recommended |

## Why v0.3.0 is the current mainline

The sensor-aware route uses a sensor-specific head and threshold instead of applying one global score threshold to every acquisition source. On the recorded development set it reduces the cross-scene error substantially compared with the global-threshold comparison.

The reported development numbers are useful for comparing versions, but they are not a production acceptance gate. The same development data was involved in checkpoint or threshold selection. Before deployment, collect a new recording batch and keep it out of training, epoch selection, threshold calibration and model tuning.

## Candidate not promoted

The stereo patch aggregation experiment reached test AUROC `0.9205` with top-10 averaging and `0.9202` with normal-q95 counting on the fixed split. It remains under `report/stereo_aggregation` in the source workspace, but is not promoted to the default ONNX deployment path because the fixed test split was inspected in earlier experiments and is not a pristine independent acceptance set.

## Release gate

A version may be promoted to production only when all of the following are true:

- Python compilation and regression tests pass.
- A fresh recording batch is documented with capture date, sensor mix and sample counts.
- The fresh batch is excluded from training, checkpoint selection and threshold calibration.
- The release report includes confusion matrix, AUROC, PR-AUC, F1, false-positive and false-negative counts.
- The model interface and preprocessing are documented.
- The source commit is tagged and the large model is attached to the GitHub Release rather than committed to Git history.
