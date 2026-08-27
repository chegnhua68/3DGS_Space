#!/usr/bin/env python3
"""Collect common and thermal-oriented metrics for named experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from thermal3dgs_sparse_ir.metrics import collect_metrics  # noqa: E402
import torch  # noqa: E402


def _named_path(value: str, option: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "{} must use NAME=PATH syntax: {!r}".format(option, value)
        )
    name, raw_path = value.split("=", 1)
    name = name.strip()
    raw_path = raw_path.strip()
    if not name or not raw_path:
        raise argparse.ArgumentTypeError(
            "{} requires a non-empty name and path".format(option)
        )
    return name, Path(raw_path)


def _experiment(value: str) -> Tuple[str, Path]:
    return _named_path(value, "--experiment")


def _roi_mask(value: str) -> Tuple[str, Path]:
    return _named_path(value, "--roi-mask")


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requested but CUDA is unavailable")
    return torch.device(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate strictly same-named, lexically sorted lossless images from "
            "METHOD_DIR/renders and METHOD_DIR/gt."
        ),
        epilog=(
            "Images use a fixed [0,1] range (uint8/255, uint16/65535), never "
            "per-image min-max normalization. RGB is converted to BT.601 luma; "
            "LPIPS replicates that luma to three [0,1] channels. ROI masks are "
            "soft [0,1] weights and empty masks are reported as unavailable."
        ),
    )
    parser.add_argument(
        "--experiment",
        action="append",
        required=True,
        type=_experiment,
        metavar="NAME=METHOD_DIR",
        help="repeat for each method directory containing renders/ and gt/",
    )
    parser.add_argument(
        "--roi-mask",
        action="append",
        default=[],
        type=_roi_mask,
        metavar="NAME=MASK_DIR",
        help="optional exact-filename ROI masks for a named experiment",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="output directory (default: results)",
    )
    parser.add_argument(
        "--skip-lpips",
        action="store_true",
        help="do not initialize lpipsPyTorch; write unavailable for LPIPS",
    )
    parser.add_argument(
        "--lpips-net",
        choices=("alex", "squeeze", "vgg"),
        default="vgg",
        help="repository LPIPS backbone (default: vgg, matching upstream metrics.py)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="metric device (default: auto)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    experiments: List[Tuple[str, Path]] = args.experiment
    roi_masks: Dict[str, Path] = {}
    try:
        for name, path in args.roi_mask:
            if name in roi_masks:
                raise ValueError("duplicate --roi-mask name: {}".format(name))
            roi_masks[name] = path
        device = _resolve_device(args.device)
        outputs = collect_metrics(
            experiments=experiments,
            output_dir=args.output_dir,
            roi_mask_dirs=roi_masks,
            skip_lpips=args.skip_lpips,
            lpips_net=args.lpips_net,
            device=device,
        )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(2, "error: {}\n".format(exc))
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
