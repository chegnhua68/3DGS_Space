#!/usr/bin/env python3
"""Create the frozen Step2 edge-filter ablation result package."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Optional, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from thermal3dgs_sparse_ir.edge_filter_ablation import (  # noqa: E402
    AblationAnalysisError,
    AblationDependencyError,
    CHECKPOINTS,
    METHODS,
    analyze_edge_filter_ablation,
)


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


def _metrics_dir(value: str) -> Tuple[int, Path]:
    name, path = _named_path(value, "--metrics-dir")
    try:
        iteration = int(name)
    except ValueError:
        raise argparse.ArgumentTypeError("--metrics-dir name must be an integer iteration")
    if str(iteration) != name or iteration not in CHECKPOINTS:
        raise argparse.ArgumentTypeError(
            "--metrics-dir iteration must be one of {}".format(
                ", ".join(str(item) for item in CHECKPOINTS)
            )
        )
    return iteration, path


def _run_dir(value: str) -> Tuple[str, Path]:
    name, path = _named_path(value, "--run-dir")
    if name not in METHODS:
        raise argparse.ArgumentTypeError(
            "--run-dir name must be one of {}".format(", ".join(METHODS))
        )
    return name, path


def _unique_mapping(values, option: str) -> Dict:
    result = {}
    for name, path in values:
        if name in result:
            raise AblationAnalysisError("duplicate {} name: {}".format(option, name))
        result[name] = path
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and summarize the frozen B0/E2-no-filter/E2 validation "
            "ablation without recomputing image metrics."
        )
    )
    parser.add_argument(
        "--metrics-dir",
        action="append",
        required=True,
        type=_metrics_dir,
        metavar="ITER=DIR",
        help="repeat exactly for iterations 2000 and 7000",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        type=_run_dir,
        metavar="METHOD=DIR",
        help="repeat exactly for B0, E2_no_filter, and E2",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only this analyzer's seven existing output files",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        metric_dirs = _unique_mapping(args.metrics_dir, "--metrics-dir")
        run_dirs = _unique_mapping(args.run_dir, "--run-dir")
        outputs = analyze_edge_filter_ablation(
            metric_dirs=metric_dirs,
            run_dirs=run_dirs,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except (
        AblationAnalysisError,
        AblationDependencyError,
        FileExistsError,
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        parser.exit(2, "error: {}\n".format(exc))
    for key in (
        "metrics_deltas",
        "per_view_deltas",
        "resource_summary",
        "qualitative_full",
        "absolute_error",
        "qualitative_zoom",
        "manifest",
    ):
        print("{}={}".format(key, outputs[key]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
