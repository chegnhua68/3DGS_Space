#!/usr/bin/env python3
"""Generate reproducible thermal comparison figures from a JSON spec."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from thermal3dgs_sparse_ir.figures import (  # noqa: E402
    FigureDependencyError,
    FigureSpecError,
    make_figures,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate qualitative, absolute-error, Sobel-edge, sparse-view, "
            "and degradation-robustness PNG figures from an explicit JSON spec."
        )
    )
    parser.add_argument("--spec", required=True, type=Path, help="thermal-figure-spec JSON file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="optional output directory override; relative paths use the current directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing figure PNGs and figure_manifest.json",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        outputs = make_figures(args.spec, output_dir=args.output_dir, overwrite=args.overwrite)
    except (
        FigureDependencyError,
        FigureSpecError,
        FileExistsError,
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        parser.exit(2, "error: {}\n".format(exc))
    for key in (
        "qualitative",
        "absolute_error",
        "edge",
        "sparse_view_curve",
        "degradation_curve",
        "manifest",
    ):
        print("{}={}".format(key, outputs[key]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
