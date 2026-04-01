#!/usr/bin/env python3
"""Run quick tuning sweeps for EDL Point MIL experiments.

This script launches train/test runs with predefined cfg-options and collects
key metrics into a CSV summary for fast comparison.
"""

import argparse
import csv
import datetime as dt
import os
import re
import shlex
import subprocess
import sys
from typing import Dict, List


DEFAULT_CONFIG = "mmdet/configs/edl_point_mil/edl_point_mil_r50_fpn_1x.py"


EXPERIMENTS: Dict[str, List[str]] = {
    "baseline": [],
    "thr_0p01": [
        "model.roi_head.test_cfg.rcnn.score_thr=0.01",
    ],
    "thr_0p03": [
        "model.roi_head.test_cfg.rcnn.score_thr=0.03",
    ],
    "nms_0p30": [
        "model.roi_head.test_cfg.rcnn.nms.iou_threshold=0.3",
    ],
    "nms_0p70": [
        "model.roi_head.test_cfg.rcnn.nms.iou_threshold=0.7",
    ],
    "box_multiscale": [
        "model.roi_head.proposal_generator.box_sizes=[[96,96],[160,160],[224,224],[320,320]]",
    ],
    "posbag_0p70": [
        "model.roi_head.proposal_generator.pos_bag_prob=0.7",
    ],
    "anneal_5": [
        "model.roi_head.bbox_head.loss_edl.annealing_step=5",
        "model.roi_head.bbox_head.loss_aux.annealing_step=5",
    ],
    "aux_w_0p25": [
        "model.roi_head.bbox_head.loss_aux.loss_weight=0.25",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDL Point MIL tuning sweep")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config path")
    parser.add_argument(
        "--mode",
        choices=["train", "test", "both"],
        default="both",
        help="Run mode for each experiment",
    )
    parser.add_argument(
        "--experiments",
        default="all",
        help="Comma-separated experiment names, or 'all'",
    )
    parser.add_argument(
        "--base-work-dir",
        default="work_dirs/edl_tuning",
        help="Base directory for all experiment outputs",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=50,
        help="Subset size for quick validation/testing",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=1,
        help="Max epochs for quick train runs",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Checkpoint for test-only mode. If empty in both mode, uses epoch_1 from each experiment work_dir.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List predefined experiments and exit",
    )
    return parser.parse_args()


def run_cmd(cmd: List[str], cwd: str, log_path: str) -> int:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("CMD: " + " ".join(shlex.quote(x) for x in cmd) + "\n\n")
        proc = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def parse_metrics(log_path: str) -> Dict[str, str]:
    if not os.path.exists(log_path):
        return {
            "precision": "",
            "recall": "",
            "f1": "",
            "empty_results_warning": "",
        }
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    def _find(pattern: str) -> str:
        m = re.findall(pattern, text)
        return m[-1] if m else ""

    return {
        "precision": _find(r"val_detection_/val_detection_precision:\s*([0-9.]+)"),
        "recall": _find(r"val_detection_/val_detection_recall:\s*([0-9.]+)"),
        "f1": _find(r"val_detection_/val_detection_f1:\s*([0-9.]+)"),
        "empty_results_warning": "yes" if "PointMilMetric got empty `self.results`" in text else "no",
    }


def common_cfg_options(args: argparse.Namespace) -> List[str]:
    return [
        f"train_cfg.max_epochs={args.max_epochs}",
        "default_hooks.logger.interval=10",
        "train_dataloader.num_workers=0",
        "train_dataloader.persistent_workers=False",
        "val_dataloader.num_workers=0",
        "val_dataloader.persistent_workers=False",
        "test_dataloader.num_workers=0",
        "test_dataloader.persistent_workers=False",
        f"test_dataloader.dataset.indices={args.subset_size}",
    ]


def main() -> int:
    args = parse_args()

    if args.list:
        print("Available experiments:")
        for name in EXPERIMENTS:
            print(f"- {name}")
        return 0

    if args.experiments == "all":
        exp_names = list(EXPERIMENTS.keys())
    else:
        exp_names = [x.strip() for x in args.experiments.split(",") if x.strip()]
        invalid = [x for x in exp_names if x not in EXPERIMENTS]
        if invalid:
            raise ValueError(f"Unknown experiments: {invalid}")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = os.path.join(repo_root, args.base_work_dir, f"sweep_{stamp}")
    os.makedirs(summary_dir, exist_ok=True)

    results = []
    base_opts = common_cfg_options(args)

    for exp_name in exp_names:
        exp_dir = os.path.join(summary_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)

        exp_opts = base_opts + EXPERIMENTS[exp_name]
        run_status = {"train": "skipped", "test": "skipped"}

        ckpt_for_test = args.checkpoint
        if args.mode in ("train", "both"):
            train_cmd = [
                sys.executable,
                "tools/train.py",
                args.config,
                "--work-dir",
                exp_dir,
                "--cfg-options",
                *exp_opts,
            ]
            train_log = os.path.join(exp_dir, "train.log")
            train_rc = run_cmd(train_cmd, repo_root, train_log)
            run_status["train"] = "ok" if train_rc == 0 else f"fail({train_rc})"
            ckpt_for_test = os.path.join(exp_dir, f"epoch_{args.max_epochs}.pth")

        metrics = {
            "precision": "",
            "recall": "",
            "f1": "",
            "empty_results_warning": "",
        }
        if args.mode in ("test", "both"):
            if not ckpt_for_test or not os.path.exists(ckpt_for_test):
                run_status["test"] = "skip(no_ckpt)"
            else:
                test_cmd = [
                    sys.executable,
                    "tools/test.py",
                    args.config,
                    ckpt_for_test,
                    "--cfg-options",
                    *exp_opts,
                ]
                test_log = os.path.join(exp_dir, "test.log")
                test_rc = run_cmd(test_cmd, repo_root, test_log)
                run_status["test"] = "ok" if test_rc == 0 else f"fail({test_rc})"
                metrics = parse_metrics(test_log)

        row = {
            "experiment": exp_name,
            "train_status": run_status["train"],
            "test_status": run_status["test"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "empty_results_warning": metrics["empty_results_warning"],
            "work_dir": exp_dir,
        }
        results.append(row)
        print(row)

    summary_csv = os.path.join(summary_dir, "summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment",
                "train_status",
                "test_status",
                "precision",
                "recall",
                "f1",
                "empty_results_warning",
                "work_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved summary: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
