# Update Cycle

This repository follows a small, evidence-based release cycle. The cycle is intended to keep the best verified version visible without treating every experiment as a deployment.

## Weekly or milestone cycle

1. **Collect**: record new normal and dirty/occluded clips; preserve capture date, sensor ID, scene, lighting and clip ID.
2. **Freeze**: create a dataset manifest and split by recording clip. Keep the acceptance batch untouched.
3. **Train**: run the selected training command and save the commit, environment, checkpoint and random seed.
4. **Evaluate**: run development evaluation, sensor-aware threshold calibration and regression tests.
5. **Review**: compare against the current mainline using the same metrics and record false positives/negatives.
6. **Accept**: run the candidate against the untouched batch. Reject candidates that improve only the calibration or fixed test split.
7. **Publish**: update `CHANGELOG.md`, `docs/RELEASES.md`, the version tag and GitHub Release notes. Attach ONNX artifacts to the Release.
8. **Monitor**: record production false positives, false negatives and lighting/sensor distribution for the next cycle.

## Required release record

- Source commit and tag
- Dataset manifest hash and capture dates
- Python, PyTorch, CUDA and GPU versions
- Training command and checkpoint selection rule
- Threshold calibration data and rule
- Per-sensor sample counts and confusion matrices
- AUROC, PR-AUC, precision, recall and F1
- Known limitations and rollback version

## Commands

```bash
# Static check used by CI
python -m compileall -q .

# Mainline evaluation, after installing dependencies and providing data
./venv/bin/python evaluate_sensor_model.py --datapath /path/to/data
./venv/bin/python test_sensor_model.py

# Inspect the candidate before creating a tag
git diff main...HEAD
git status --short
```

## Branch and tag convention

- `main`: latest reviewed code.
- `experiment/<short-name>`: training and evaluation work in progress.
- `release/<version>`: release preparation only.
- Stable tags use `vMAJOR.MINOR.PATCH`, for example `v0.3.0`.

A release is published only after the release gate in [`RELEASES.md`](RELEASES.md) is satisfied.
