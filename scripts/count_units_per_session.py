#!/usr/bin/env python3
"""Report n_units (decoder channel count) per session with min/max/mean ± SD.

Uses the same SingleSessionDataModule path as decode_single_session_cv.py.
`base_path` must be the root that contains the `ibl_aligned` directory (same as --base_path
when running decoding).

Examples:
  python scripts/count_units_per_session.py \\
    --base_path /path/to/ibl-mouse_neural-activity \\
    --eids_file data/target_eids.txt \\
    --target lightning-pose-pawR-3d-speed \\
    --pose_model_name resnet50_median_new

  python scripts/count_units_per_session.py \\
    --base_path /path/to/ibl-mouse_neural-activity \\
    --eids uuid1,uuid2 \\
    --target lightning-pose-pawR-3d-speed \\
    --pose_model_name resnet50_median_new
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.config_utils import config_from_kwargs, update_config  # noqa: E402
from utils.data_loader_utils import SingleSessionDataModule  # noqa: E402

POSE_TARGETS = [
    "lightning-pose-left-pawL-speed",
    "lightning-pose-right-pawL-speed",
    "lightning-pose-left-pawR-speed",
    "lightning-pose-right-pawR-speed",
    "lightning-pose-pawL-3d-speed",
    "lightning-pose-pawR-3d-speed",
]


def load_eids(args: argparse.Namespace) -> list[str]:
    if args.eids:
        return [e.strip() for e in args.eids.split(",") if e.strip()]
    if args.eids_file:
        path = Path(args.eids_file)
        if not path.is_file():
            raise FileNotFoundError(f"eids file not found: {path}")
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    raise ValueError("Provide --eids or --eids_file")


def build_base_config(args: argparse.Namespace):
    kwargs = {"model": f"include:{REPO_ROOT / 'src/configs/decoder.yaml'}"}
    config = config_from_kwargs(kwargs)
    config = update_config(str(REPO_ROOT / "src/configs/decoder.yaml"), config)
    config = update_config(str(REPO_ROOT / "src/configs/reg_trainer.yaml"), config)
    config["dirs"]["data_dir"] = Path(args.base_path) / config["dirs"]["data_dir"]
    config["training"]["device"] = torch.device("cpu")
    config["data"]["use_nlb"] = args.use_nlb
    config["data"]["bin_size"] = args.bin_size
    config["data"]["fold_idx"] = args.fold_idx
    return config


def session_config(base_config, eid: str, target: str, region: str, pose_model_name: str | None):
    cfg = base_config.copy()
    cfg["eid"] = eid
    cfg["target"] = target
    cfg["region"] = region if region != "all" else "all"
    cfg["model"]["target"] = "reg"
    is_pose = target in POSE_TARGETS
    cfg["pose_model_name"] = pose_model_name if is_pose else None
    return cfg


def n_units_for_session(cfg) -> int:
    dm = SingleSessionDataModule(cfg)
    dm.update_config()
    return int(dm.config["n_units"])


def main():
    ap = argparse.ArgumentParser(description="Count units per session (same as decoding n_units).")
    ap.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Root containing ibl_aligned/ (same as decode --base_path).",
    )
    ap.add_argument("--eids", type=str, default=None, help="Comma-separated session UUIDs.")
    ap.add_argument("--eids_file", type=str, default=None, help="One eid per line.")
    ap.add_argument("--target", type=str, default="lightning-pose-pawR-3d-speed")
    ap.add_argument("--region", type=str, default="all")
    ap.add_argument("--bin_size", type=int, default=5)
    ap.add_argument("--fold_idx", type=int, default=0)
    ap.add_argument("--use_nlb", action="store_true")
    ap.add_argument(
        "--pose_model_name",
        type=str,
        default=None,
        help="Required for pose targets (loads pose files under pose_aligned/).",
    )
    ap.add_argument("--csv", type=str, default=None, help="Optional path to write eid,n_units CSV.")
    ap.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only aggregate statistics (no per-session lines).",
    )
    args = ap.parse_args()

    if args.target in POSE_TARGETS and not args.pose_model_name:
        ap.error(f"--pose_model_name is required for target {args.target!r}")

    eids = load_eids(args)
    base = build_base_config(args)

    rows: list[tuple[str, int]] = []
    failed: list[tuple[str, str]] = []

    for eid in eids:
        cfg = session_config(base, eid, args.target, args.region, args.pose_model_name)
        try:
            n = n_units_for_session(cfg)
            rows.append((eid, n))
            if not args.summary_only:
                print(f"{eid}  n_units={n}")
        except Exception as exc:  # noqa: BLE001
            failed.append((eid, str(exc)))
            print(f"{eid}  FAILED: {exc}", file=sys.stderr)

    if not rows:
        print("No sessions loaded successfully.", file=sys.stderr)
        sys.exit(1)

    counts = np.array([n for _, n in rows], dtype=float)
    if not args.summary_only:
        print()
    print(
        f"Sessions: {len(rows)} / {len(eids)}"
        + (f" ({len(failed)} failed)" if failed else "")
    )
    print(
        f"Neurons per session — min: {int(counts.min())}, max: {int(counts.max())}, "
        f"median: {float(np.median(counts)):.1f}"
    )
    mean = float(np.mean(counts))
    std = float(np.std(counts, ddof=1)) if len(counts) > 1 else 0.0
    print(f"Mean ± SD: {mean:.1f} ± {std:.1f}")
    sem = std / np.sqrt(len(counts)) if len(counts) > 1 else 0.0
    print(f"Mean ± SEM: {mean:.1f} ± {sem:.3f}")

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = ["eid,n_units"] + [f"{eid},{n}" for eid, n in rows]
        out.write_text("\n".join(lines) + "\n")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
