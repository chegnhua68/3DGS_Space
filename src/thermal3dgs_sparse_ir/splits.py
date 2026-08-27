"""Deterministic nested sparse-view split generation."""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

from .contracts import (
    SCHEMA_VERSION,
    SPARSE_SPLIT_KIND,
    ManifestValidationError,
    load_base_split_manifest,
    load_dataset_manifest,
    manifest_sha256,
    validate_base_split_manifest,
    validate_dataset_manifest,
    validate_sparse_split_manifest,
    with_manifest_sha256,
    write_manifest,
)


DEFAULT_RATIOS = (1.0, 0.5, 0.25, 0.125)
SUPPORTED_METHODS = ("nested-random", "trajectory-uniform")
GENERATOR_VERSION = "1.0.0"


def _validated_ratios(ratios: Sequence[float]) -> Tuple[float, ...]:
    if isinstance(ratios, (str, bytes)) or not ratios:
        raise ManifestValidationError("ratios must be a non-empty sequence")
    result = []
    for index, value in enumerate(ratios):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ManifestValidationError("ratios[{}] must be a number".format(index))
        ratio = float(value)
        if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
            raise ManifestValidationError("ratios must be finite values in (0, 1]")
        result.append(ratio)
    if len(set(result)) != len(result):
        raise ManifestValidationError("ratios must not contain duplicates")
    return tuple(sorted(result, reverse=True))


def _target_count(pool_size: int, ratio: float) -> int:
    # Round halves upward, then keep every non-empty pool usable at low ratios.
    return min(pool_size, max(1, int(math.floor(pool_size * ratio + 0.5))))


def _nested_random_order(view_ids: Sequence[str], seed: int) -> List[str]:
    seed_bytes = str(seed).encode("ascii") + b"\0"

    def score(view_id: str) -> Tuple[bytes, str]:
        digest = hashlib.sha256(seed_bytes + view_id.encode("utf-8")).digest()
        return digest, view_id

    return sorted(view_ids, key=score)


def _trajectory_uniform_order(
    view_ids: Sequence[str], views_by_id: Mapping[str, Mapping[str, Any]]
) -> List[str]:
    trajectory = sorted(
        view_ids,
        key=lambda view_id: (views_by_id[view_id]["sequence_index"], view_id),
    )
    selected = [trajectory[0]]
    remaining = trajectory[1:]
    selected_positions = [views_by_id[trajectory[0]]["sequence_index"]]

    while remaining:
        best_index = 0
        best_distance = -1
        for index, view_id in enumerate(remaining):
            position = views_by_id[view_id]["sequence_index"]
            distance = min(abs(position - chosen) for chosen in selected_positions)
            if distance > best_distance:
                best_index = index
                best_distance = distance
        chosen_id = remaining.pop(best_index)
        selected.append(chosen_id)
        selected_positions.append(views_by_id[chosen_id]["sequence_index"])
    return selected


def build_selection_order(
    dataset_manifest: Mapping[str, Any],
    train_pool: Sequence[str],
    *,
    method: str = "nested-random",
    seed: int = 0,
) -> List[str]:
    """Build one stable ranking whose prefixes define every sparse ratio."""

    validate_dataset_manifest(dataset_manifest)
    if method not in SUPPORTED_METHODS:
        raise ManifestValidationError(
            "method must be one of {}".format(", ".join(SUPPORTED_METHODS))
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ManifestValidationError("seed must be an integer")
    if isinstance(train_pool, (str, bytes)) or not train_pool:
        raise ManifestValidationError("train_pool must be a non-empty sequence")
    sorted_pool = sorted(train_pool)
    if len(set(sorted_pool)) != len(sorted_pool):
        raise ManifestValidationError("train_pool contains duplicate view ids")
    views_by_id = {view["id"]: view for view in dataset_manifest["views"]}
    missing = set(sorted_pool) - set(views_by_id)
    if missing:
        raise ManifestValidationError(
            "train_pool references unknown view id(s): {}".format(
                ", ".join(sorted(missing))
            )
        )
    if method == "nested-random":
        return _nested_random_order(sorted_pool, seed)
    return _trajectory_uniform_order(sorted_pool, views_by_id)


def _ratio_token(ratio: float) -> str:
    percentage = format(ratio * 100.0, ".12g")
    return percentage.replace("-", "m").replace(".", "p")


def _split_id(base_split_id: str, method: str, ratio: float) -> str:
    return "{}:{}:{}".format(base_split_id, method, _ratio_token(ratio))


def generate_sparse_splits(
    dataset_manifest: Mapping[str, Any],
    base_split_manifest: Mapping[str, Any],
    *,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    method: str = "nested-random",
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Generate all ratios from one ranking so their train sets are nested."""

    validate_dataset_manifest(dataset_manifest)
    validate_base_split_manifest(base_split_manifest, dataset_manifest)
    normalized_ratios = _validated_ratios(ratios)
    order = build_selection_order(
        dataset_manifest,
        base_split_manifest["train_pool"],
        method=method,
        seed=seed,
    )
    pool = sorted(base_split_manifest["train_pool"])
    source_hash = manifest_sha256(base_split_manifest)
    generated = []
    previous_selected: Optional[Set[str]] = None
    # Generate from sparse to dense while checking the nesting invariant, then
    # return the conventional dense-to-sparse ratio order.
    sparse_to_dense = sorted(normalized_ratios)
    by_ratio: Dict[float, Dict[str, Any]] = {}
    for ratio in sparse_to_dense:
        count = _target_count(len(pool), ratio)
        selected = sorted(order[:count])
        selected_set = set(selected)
        if previous_selected is not None and not previous_selected <= selected_set:
            raise AssertionError("internal error: generated split sets are not nested")
        previous_selected = selected_set
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": SPARSE_SPLIT_KIND,
            "split_id": _split_id(base_split_manifest["split_id"], method, ratio),
            "dataset_id": base_split_manifest["dataset_id"],
            "dataset_sha256": base_split_manifest["dataset_sha256"],
            "train_pool": pool,
            "val": sorted(base_split_manifest["val"]),
            "test": sorted(base_split_manifest["test"]),
            "requested_ratio": ratio,
            "actual_ratio": count / len(pool),
            "train_selected": selected,
            "method": method,
            "seed": seed,
            "generator_version": GENERATOR_VERSION,
            "source_base_split_sha256": source_hash,
        }
        manifest = with_manifest_sha256(manifest)
        validate_sparse_split_manifest(
            manifest, dataset_manifest, base_split_manifest
        )
        by_ratio[ratio] = manifest
    generated.extend(by_ratio[ratio] for ratio in normalized_ratios)
    return generated


def generate_sparse_split(
    dataset_manifest: Mapping[str, Any],
    base_split_manifest: Mapping[str, Any],
    ratio: float,
    *,
    method: str = "nested-random",
    seed: int = 0,
) -> Dict[str, Any]:
    """Generate one sparse split using the same contract as the batch API."""

    return generate_sparse_splits(
        dataset_manifest,
        base_split_manifest,
        ratios=(ratio,),
        method=method,
        seed=seed,
    )[0]


def write_sparse_splits(
    output_dir: Union[Path, str],
    manifests: Sequence[Mapping[str, Any]],
) -> List[Path]:
    """Write generated manifests using deterministic, collision-free names."""

    destination = Path(output_dir)
    paths = []
    seen_names: Set[str] = set()
    for manifest in manifests:
        validate_sparse_split_manifest(manifest)
        name = "sparse_{}_{}.json".format(
            manifest["method"], _ratio_token(float(manifest["requested_ratio"]))
        )
        if name in seen_names:
            raise ManifestValidationError("output file name collision: " + name)
        seen_names.add(name)
        path = destination / name
        write_manifest(path, manifest)
        paths.append(path)
    return paths


def create_sparse_split_files(
    dataset_manifest_path: Union[Path, str],
    base_split_manifest_path: Union[Path, str],
    *,
    output_dir: Union[Path, str] = "splits",
    ratios: Sequence[float] = DEFAULT_RATIOS,
    method: str = "nested-random",
    seed: int = 0,
) -> List[Path]:
    dataset = load_dataset_manifest(dataset_manifest_path)
    base_split = load_base_split_manifest(base_split_manifest_path, dataset)
    manifests = generate_sparse_splits(
        dataset, base_split, ratios=ratios, method=method, seed=seed
    )
    return write_sparse_splits(output_dir, manifests)


def _parse_ratio_arguments(values: Sequence[str]) -> Tuple[float, ...]:
    result = []
    for value in values:
        for part in value.split(","):
            if not part.strip():
                raise argparse.ArgumentTypeError("ratio list contains an empty value")
            try:
                result.append(float(part))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    "invalid ratio: {!r}".format(part)
                ) from exc
    return tuple(result)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic nested sparse-view splits from explicit "
            "dataset and frozen base-split manifests."
        )
    )
    parser.add_argument(
        "--dataset-manifest",
        "--dataset",
        dest="dataset_manifest",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--base-split-manifest",
        "--base-split",
        dest="base_split_manifest",
        required=True,
        type=Path,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("splits"))
    parser.add_argument(
        "--ratios",
        nargs="+",
        default=[str(value) for value in DEFAULT_RATIOS],
        metavar="RATIO",
        help="space- or comma-separated ratios in (0, 1]",
    )
    parser.add_argument("--method", choices=SUPPORTED_METHODS, default="nested-random")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        ratios = _parse_ratio_arguments(args.ratios)
        paths = create_sparse_split_files(
            args.dataset_manifest,
            args.base_split_manifest,
            output_dir=args.output_dir,
            ratios=ratios,
            method=args.method,
            seed=args.seed,
        )
    except (ManifestValidationError, OSError, argparse.ArgumentTypeError) as exc:
        parser.error(str(exc))
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
