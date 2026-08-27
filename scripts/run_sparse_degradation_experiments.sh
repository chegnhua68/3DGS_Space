#!/usr/bin/env bash
set -euo pipefail

matrix="${1:-configs/experiment_matrix.example.json}"
shift || true
python_bin="${THERMAL3DGS_PYTHON:-python3.11}"
"$python_bin" scripts/run_experiments.py --matrix "$matrix" "$@"
