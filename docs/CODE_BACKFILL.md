# Code Backfill Notice

The original `v0.3.1` through `v0.3.6` tags were first created as chronological experiment notes. During cleanup, some later sensor scripts had already been included in `v0.3.0`, so the tags did not show every code delta clearly.

The following commits backfill the actual missing source files without rewriting the pushed history:

| Stage | Code commit | Added source |
|---|---|---|
| Stereo preprocessing | `e8e5210` | `analyze_stereo_preprocess.py`, `analyze_stereo_normal_calibration.py` |
| Stereo finetune comparison | `04bd51b` | `finetune_stereo_head.py`, `compare_stereo_finetune.py` |
| Stereo validation | `9a1ee68` | `finetune_stereo_validated.py`, `evaluate_stereo_final.py` |
| Stereo aggregation | `09ff57d` | `evaluate_patch_aggregation.py`, `evaluate_stereo_aggregation.py`, `train_stereo_sides.py` |

The existing tags are not force-moved because doing so would rewrite remote history. These commits are the authoritative source-code progression after the original chronological notes. The sensor-threshold code and `evaluate_candidate_all_sensors.py` were already present in the earlier `v0.3.0` commit and therefore have no later file addition.
