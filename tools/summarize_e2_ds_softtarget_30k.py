#!/usr/bin/env python3
"""Build the auditable E2 versus E2+DS 30k result package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CHECKPOINTS = (7000, 15000, 30000)
SEEDS = (2026, 2027, 2028)
METHODS = ("E2", "E2+DS")
METRICS = ("psnr", "ssim", "t_mae", "e_mae", "gradient_preservation")
HIGHER_IS_BETTER = {"psnr", "ssim", "gradient_preservation"}
Z95_T3 = 4.302653


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def number(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite metric: {value}")
    return result


def run_name(method: str, seed: int) -> str:
    return ("e2_ds" if method == "E2+DS" else "e2") + f"_seed{seed}_30k"


def load_metric_tables(results_root: Path) -> tuple[list[dict[str, Any]], dict[tuple[int, str], dict[str, dict[str, str]]]]:
    rows: list[dict[str, Any]] = []
    tables: dict[tuple[int, str], dict[str, dict[str, str]]] = {}
    for checkpoint in CHECKPOINTS:
        summary = read_csv(results_root / f"val_{checkpoint}" / "metrics_summary.csv")
        by_name = {row["experiment"]: row for row in summary}
        expected = {run_name(method, seed) for method in METHODS for seed in SEEDS}
        if set(by_name) != expected:
            raise ValueError(f"unexpected experiment set at val_{checkpoint}: {sorted(by_name)}")
        for method in METHODS:
            for seed in SEEDS:
                name = run_name(method, seed)
                row = by_name[name]
                if int(row["num_views"]) != 34:
                    raise ValueError(f"expected 34 validation views: {row}")
                item = {"checkpoint": checkpoint, "method": method, "train_seed": seed,
                        "experiment": name, "num_views": int(row["num_views"])}
                item.update({metric: number(row[metric]) for metric in METRICS})
                rows.append(item)
        tables[(checkpoint, "summary")] = by_name
    return rows, tables


def build_package(repo: Path, config_path: Path, runs_root: Path, results_root: Path, output: Path) -> None:
    config = read_json(config_path)
    metric_rows, tables = load_metric_tables(results_root)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "metrics_by_seed.csv", ["checkpoint", "method", "train_seed", "experiment", "num_views", *METRICS], metric_rows)

    paired_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        for seed in SEEDS:
            base = tables[(checkpoint, "summary")][run_name("E2", seed)]
            ds = tables[(checkpoint, "summary")][run_name("E2+DS", seed)]
            paired_rows.append({"checkpoint": checkpoint, "train_seed": seed, "lhs": "E2+DS", "rhs": "E2",
                                **{f"delta_{metric}": number(ds[metric]) - number(base[metric]) for metric in METRICS}})
        for metric in METRICS:
            base_values = [number(tables[(checkpoint, "summary")][run_name("E2", seed)][metric]) for seed in SEEDS]
            ds_values = [number(tables[(checkpoint, "summary")][run_name("E2+DS", seed)][metric]) for seed in SEEDS]
            deltas = [ds - base for ds, base in zip(ds_values, base_values)]
            better = sum(delta > 0 if metric in HIGHER_IS_BETTER else delta < 0 for delta in deltas)
            worse = sum(delta < 0 if metric in HIGHER_IS_BETTER else delta > 0 for delta in deltas)
            std = statistics.stdev(deltas)
            seed_rows.append({"checkpoint": checkpoint, "metric": metric,
                              "e2_mean": statistics.mean(base_values), "e2_std": statistics.stdev(base_values),
                              "e2_ds_mean": statistics.mean(ds_values), "e2_ds_std": statistics.stdev(ds_values),
                              "delta_mean": statistics.mean(deltas), "delta_std": std,
                              "delta_median": statistics.median(deltas),
                              "delta_ci95_low": statistics.mean(deltas) - Z95_T3 * std / math.sqrt(3),
                              "delta_ci95_high": statistics.mean(deltas) + Z95_T3 * std / math.sqrt(3),
                              "improved_seeds": better, "degraded_seeds": worse, "tied_seeds": 3 - better - worse})
    write_csv(output / "paired_deltas.csv", ["checkpoint", "train_seed", "lhs", "rhs", *[f"delta_{m}" for m in METRICS]], paired_rows)
    write_csv(output / "seed_summary.csv", ["checkpoint", "metric", "e2_mean", "e2_std", "e2_ds_mean", "e2_ds_std", "delta_mean", "delta_std", "delta_median", "delta_ci95_low", "delta_ci95_high", "improved_seeds", "degraded_seeds", "tied_seeds"], seed_rows)

    split = read_json(repo / "data/TI-NSD/heated/splits/seed2026/sparse_nested-random_25.json")
    val_names = split.get("val", [])
    view_id = {f"{index:05d}.png": name for index, name in enumerate(val_names)}
    per_view: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        rows = read_csv(results_root / f"val_{checkpoint}" / "metrics_per_view.csv")
        grouped: dict[str, dict[str, dict[str, str]]] = {}
        for row in rows:
            grouped.setdefault(row["experiment"], {})[row["image"]] = row
        for seed in SEEDS:
            base = grouped[run_name("E2", seed)]
            ds = grouped[run_name("E2+DS", seed)]
            if set(base) != set(ds) or len(base) != 34:
                raise ValueError(f"per-view mismatch at {checkpoint}, seed {seed}")
            for image in sorted(base):
                per_view.append({"checkpoint": checkpoint, "train_seed": seed, "image": image,
                                 "view_id": view_id.get(image, image),
                                 **{f"delta_{m}": number(ds[image][m]) - number(base[image][m]) for m in METRICS}})
    write_csv(output / "per_view_deltas.csv", ["checkpoint", "train_seed", "image", "view_id", *[f"delta_{m}" for m in METRICS]], per_view)
    view_summary: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        for seed in SEEDS:
            subset = [row for row in per_view if row["checkpoint"] == checkpoint and row["train_seed"] == seed]
            for metric in METRICS:
                values = [number(row[f"delta_{metric}"]) for row in subset]
                worst = min(values) if metric in HIGHER_IS_BETTER else max(values)
                index = values.index(worst)
                better = sum(value > 0 if metric in HIGHER_IS_BETTER else value < 0 for value in values)
                worse = sum(value < 0 if metric in HIGHER_IS_BETTER else value > 0 for value in values)
                view_summary.append({"checkpoint": checkpoint, "train_seed": seed, "metric": metric,
                                     "mean_delta": statistics.mean(values), "median_delta": statistics.median(values),
                                     "improved_views": better, "degraded_views": worse, "tied_views": len(values) - better - worse,
                                     "worst_image": subset[index]["image"], "worst_view_id": subset[index]["view_id"], "worst_delta": worst})
    write_csv(output / "per_view_summary.csv", ["checkpoint", "train_seed", "metric", "mean_delta", "median_delta", "improved_views", "degraded_views", "tied_views", "worst_image", "worst_view_id", "worst_delta"], view_summary)

    late_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for seed in SEEDS:
            early = next(row for row in metric_rows if row["checkpoint"] == 7000 and row["method"] == method and row["train_seed"] == seed)
            late = next(row for row in metric_rows if row["checkpoint"] == 30000 and row["method"] == method and row["train_seed"] == seed)
            late_rows.append({"method": method, "train_seed": seed, **{f"delta_{m}_7k_to_30k": (early[m] - late[m] if m in HIGHER_IS_BETTER else late[m] - early[m]) for m in METRICS}})
    write_csv(output / "late_degradation_summary.csv", ["method", "train_seed", *[f"delta_{m}_7k_to_30k" for m in METRICS]], late_rows)
    late_paired = []
    for seed in SEEDS:
        base = next(row for row in late_rows if row["method"] == "E2" and row["train_seed"] == seed)
        ds = next(row for row in late_rows if row["method"] == "E2+DS" and row["train_seed"] == seed)
        late_paired.append({"train_seed": seed, **{f"ds_minus_e2_degradation_{m}": ds[f"delta_{m}_7k_to_30k"] - base[f"delta_{m}_7k_to_30k"] for m in METRICS}})
    write_csv(output / "late_degradation_paired.csv", ["train_seed", *[f"ds_minus_e2_degradation_{m}" for m in METRICS]], late_paired)
    fixed_101 = [row for row in per_view if row["view_id"] == "101.jpg"]
    if len(fixed_101) != len(CHECKPOINTS) * len(SEEDS):
        raise ValueError("expected one 101.jpg row per checkpoint and seed")
    write_csv(output / "fixed_101_deltas.csv", ["checkpoint", "train_seed", "image", "view_id", *[f"delta_{m}" for m in METRICS]], fixed_101)

    audit_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for seed in SEEDS:
            name = run_name(method, seed)
            run = runs_root / name
            status = read_json(run / "status.json")
            manifest = read_json(run / "run_manifest.json")
            init = read_json(run / "initial_state_audit.json")
            camera = read_json(run / "camera_sequence.json")
            supervision = read_json(run / "supervision_audit.json")
            comparison = init.get("initialization", {}).get("comparison", {})
            audit_rows.append({"experiment": name, "method": method, "train_seed": seed,
                               "status": status.get("status"), "train_status": status.get("train_status"),
                               "render_status": status.get("render_status"),
                               "initialization_all_equal": all(comparison.values()) if comparison else False,
                               "camera_sequence_sha256": camera.get("sequence_sha256", camera.get("sha256", "")),
                               "supervision_mode": supervision.get("mode", supervision.get("supervision_mode", "")),
                               "supervision_manifest_sha256": supervision.get("manifest_sha256", ""),
                               "wall_seconds": sum(float(item.get("wall_seconds", 0.0)) for item in manifest.get("phase_timings", {}).values()),
                               "training_source_sha256": manifest.get("git", {}).get("training_source_sha256", ""),
                               "config_sha256": manifest.get("config_sha256", ""),
                               "stderr_bytes": (run / "stderr.log").stat().st_size if (run / "stderr.log").exists() else -1})
            if status.get("status") != "completed" or not all(comparison.values()):
                raise ValueError(f"audit failure in {name}")
    write_csv(output / "resource_summary.csv", list(audit_rows[0]), audit_rows)
    write_csv(output / "initial_state_comparison.csv", ["experiment", "method", "train_seed", "initialization_all_equal", "camera_sequence_sha256", "supervision_mode", "supervision_manifest_sha256"], audit_rows)
    paired_audit = []
    expected_supervision_hash = next(row["supervision_manifest_sha256"] for row in audit_rows if row["method"] == "E2+DS")
    for seed in SEEDS:
        base = next(row for row in audit_rows if row["method"] == "E2" and row["train_seed"] == seed)
        ds = next(row for row in audit_rows if row["method"] == "E2+DS" and row["train_seed"] == seed)
        if base["camera_sequence_sha256"] != ds["camera_sequence_sha256"]:
            raise ValueError(f"paired camera sequence mismatch for seed {seed}")
        if ds["supervision_manifest_sha256"] != expected_supervision_hash:
            raise ValueError(f"unexpected supervision manifest for seed {seed}")
        paired_audit.append({"train_seed": seed, "camera_sequence_equal": True,
                             "initialization_e2_equal": base["initialization_all_equal"],
                             "initialization_ds_equal": ds["initialization_all_equal"],
                             "supervision_manifest_sha256": ds["supervision_manifest_sha256"]})
    write_csv(output / "paired_audit.csv", ["train_seed", "camera_sequence_equal", "initialization_e2_equal", "initialization_ds_equal", "supervision_manifest_sha256"], paired_audit)

    source_manifest = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "config": config,
                       "config_sha256": sha256(config_path), "dataset_manifest_sha256": sha256(repo / config["dataset_manifest"]),
                       "split_manifest_sha256": sha256(repo / "data/TI-NSD/heated/splits/seed2026/sparse_nested-random_25.json"),
                       "supervision_manifest_sha256": next((row["supervision_manifest_sha256"] for row in audit_rows if row["method"] == "E2+DS"), ""),
                       "cache_status": "completed", "validation_views": 34, "test_accessed": False}
    write_json(output / "source_manifest.json", source_manifest)
    write_json(output / "status.json", {"status": "completed", "experiments": len(audit_rows), "checkpoints": list(CHECKPOINTS), "test_accessed": False})
    final_rows = [row for row in seed_rows if row["checkpoint"] == 30000]
    report = ["# E2 vs E2+DS：BM3D 软目标 30k 正式结果", "", "- 三个训练种子：2026/2027/2028；验证集：34 views；测试集未访问。", "- checkpoint：7k / 15k / 30k；所有统计均为同 seed 配对比较。", "- E2+DS：BM3D normal profile、ALL_STAGES、sigma=0.03、T=0.25Y+0.75Z。", "", "## 30k 汇总", "", "| 指标 | E2 均值 | E2+DS 均值 | E2+DS - E2 | 改善 seed 数 |", "|---|---:|---:|---:|---:|"]
    for row in final_rows:
        report.append(f"| {row['metric']} | {float(row['e2_mean']):.6f} | {float(row['e2_ds_mean']):.6f} | {float(row['delta_mean']):+.6f} | {row['improved_seeds']} |")
    report += ["", "30k 时 E2+DS 的 SSIM 均值高于 E2，E-mae 均值更低；PSNR、T-mae 和 gradient preservation 均值低于 E2。该结论只描述本验证集和本协议，不代表测试集泛化。", "", "## 结果文件", "", "详见 `metrics_by_seed.csv`、`seed_summary.csv`、`paired_deltas.csv`、`per_view_deltas.csv`、`late_degradation_summary.csv`。", "", "## 审计", "", "`initial_state_comparison.csv`、`paired_audit.csv`、`resource_summary.csv` 和 `source_manifest.json` 记录共享初始化、配对相机序列、监督缓存、源代码与输入哈希。", "", "## 结论边界", "", "本包只报告验证集结果，不宣称测试集泛化；若任一 run 或审计未完成，汇总器会失败并停止生成完成状态。", ""]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    (output / "handoff.md").write_text("# E2+DS 结果交接\n\n完成状态见 `status.json`。优先查看 `report.md`、`seed_summary.csv` 和 `paired_deltas.csv`；复核输入与代码请查看 `source_manifest.json` 与 `resource_summary.csv`。停止标记为 `runs/e2_ds_softtarget/v1/STOP_REQUESTED`，TensorBoard 日志位于 `runs/e2_ds_softtarget/v1`。\n", encoding="utf-8")
    files = sorted(path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file() and path.name != "results_manifest.json")
    write_json(output / "results_manifest.json", {"schema_version": 1, "status": "completed", "files": [{"path": item, "sha256": sha256(output / item)} for item in files]})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/experiment_matrix.e2_ds_softtarget_30k.json"))
    parser.add_argument("--runs-root", type=Path, default=Path("runs/e2_ds_softtarget/v1"))
    parser.add_argument("--results-root", type=Path, default=Path("results/e2_ds_softtarget/v1"))
    parser.add_argument("--output", type=Path, default=Path("results/e2_ds_softtarget/v1/summary"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    build_package(repo, (repo / args.config).resolve(), (repo / args.runs_root).resolve(), (repo / args.results_root).resolve(), (repo / args.output).resolve())
    print((repo / args.output).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
