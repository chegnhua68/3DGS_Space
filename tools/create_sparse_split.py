#!/usr/bin/env python3
"""Thin command-line entry point for sparse split generation."""

from pathlib import Path
import sys


_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from thermal3dgs_sparse_ir.splits import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
