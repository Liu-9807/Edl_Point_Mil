# EDL Point MIL Tuning Experiments

This note describes a quick sweep workflow for the EDL Point MIL pipeline.

## Script

- `tools/analysis_tools/edl_point_mil_sweep.py`

## Predefined Experiments

- `baseline`
- `thr_0p01`
- `thr_0p03`
- `nms_0p30`
- `nms_0p70`
- `box_multiscale`
- `posbag_0p70`
- `anneal_5`
- `aux_w_0p25`

## Quick Commands

List all experiments:

```bash
python tools/analysis_tools/edl_point_mil_sweep.py --list
```

Run full quick sweep (1 epoch train + subset test):

```bash
python tools/analysis_tools/edl_point_mil_sweep.py \
  --mode both \
  --experiments all \
  --subset-size 50 \
  --max-epochs 1
```

Run only selected experiments:

```bash
python tools/analysis_tools/edl_point_mil_sweep.py \
  --mode both \
  --experiments baseline,thr_0p03,nms_0p30,box_multiscale
```

Test-only against a fixed checkpoint:

```bash
python tools/analysis_tools/edl_point_mil_sweep.py \
  --mode test \
  --checkpoint work_dirs/edl_r50_fpn_1x/epoch_10.pth \
  --experiments baseline,thr_0p01,thr_0p03
```

## Outputs

Each run creates:

- `work_dirs/edl_tuning/sweep_<timestamp>/<experiment>/train.log`
- `work_dirs/edl_tuning/sweep_<timestamp>/<experiment>/test.log`
- `work_dirs/edl_tuning/sweep_<timestamp>/summary.csv`

CSV fields:

- `experiment`
- `train_status`
- `test_status`
- `precision`
- `recall`
- `f1`
- `empty_results_warning`
- `work_dir`
