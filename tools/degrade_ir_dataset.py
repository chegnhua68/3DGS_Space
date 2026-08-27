#!/usr/bin/env python3
"""Generate deterministic infrared training degradations from manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from thermal3dgs_sparse_ir.degradations import (  # noqa: E402
    DependencyUnavailableError,
    ManifestError,
    degrade_dataset,
)


def _ordered_names(values: Optional[Sequence[str]]) -> Optional[List[str]]:
    if values is None:
        return None
    names: List[str] = []
    for value in values:
        names.extend(item.strip() for item in value.split(",") if item.strip())
    return names


def build_transforms(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Translate validated CLI options into an explicit transform sequence."""

    if args.degradation == "combined":
        names = _ordered_names(args.order)
        if not names:
            raise ValueError("combined degradation requires an explicit --order")
    else:
        if args.order:
            raise ValueError("--order is only valid with --degradation combined")
        names = [args.degradation]

    transforms: List[Dict[str, Any]] = []
    for name in names:
        if name == "gaussian-noise":
            transforms.append({"name": name, "sigma": args.noise_sigma})
        elif name == "contrast":
            transforms.append({"name": name, "alpha": args.contrast_alpha})
        elif name == "blur":
            transforms.append(
                {
                    "name": name,
                    "sigma": args.blur_sigma,
                    "kernel_size": args.blur_kernel_size,
                }
            )
        else:
            raise ValueError(
                f"Unsupported transform {name!r}; choose gaussian-noise, contrast, or blur"
            )
    return transforms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply deterministic degradation to sparse-split train_selected views only; "
            "validation and test views remain clean."
        ),
        epilog=(
            "dataset data_range must describe encoded source intensities, for example "
            "[0,255] for uint8 or [0,65535] for uint16; noise sigma is measured after "
            "that fixed range is mapped to [0,1]."
        ),
    )
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--degradation",
        required=True,
        choices=("gaussian-noise", "contrast", "blur", "combined"),
    )
    parser.add_argument(
        "--order",
        nargs="+",
        metavar="TRANSFORM",
        help=(
            "Explicit combined order, as space- or comma-separated names: "
            "gaussian-noise, contrast, blur"
        ),
    )
    parser.add_argument(
        "--noise-sigma",
        type=float,
        default=0.03,
        help="Gaussian noise standard deviation as a fraction of global data_range (default: 0.03)",
    )
    parser.add_argument(
        "--contrast-alpha",
        type=float,
        default=0.5,
        help="Contrast factor in [0, 1] around the image/channel mean (default: 0.5)",
    )
    parser.add_argument("--blur-sigma", type=float, default=1.0)
    parser.add_argument("--blur-kernel-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0, help="Global seed (default: 0)")
    parser.add_argument(
        "--manifest-name", default="degradation_manifest.v1.json"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated targets; never modifies source images",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        transforms = build_transforms(args)
        manifest_path = degrade_dataset(
            dataset_manifest_path=args.dataset_manifest,
            split_manifest_path=args.split_manifest,
            output_dir=args.output_dir,
            transforms=transforms,
            global_seed=args.seed,
            overwrite=args.overwrite,
            manifest_name=args.manifest_name,
        )
    except (DependencyUnavailableError, ManifestError, FileExistsError, FileNotFoundError, ValueError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
