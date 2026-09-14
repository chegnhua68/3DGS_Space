#!/usr/bin/env python3
"""Summarize the B0/E2 shared-initialization 30k validation package."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

CHECKPOINTS = (7000, 15000, 30000)
SEEDS = (2026, 2027, 2028)
METRICS = ("psnr", "ssim", "t_mae", "e_mae", "gradient_preservation")
HIGHER = {"psnr", "ssim", "gradient_preservation"}
RUNS = tuple(
    ("{}_seed{}_30k".format(method.lower(), seed), method, seed)
    for seed in SEEDS
    for method in ("B0", "E2")
)
RUN_NAMES = tuple(item[0] for item in RUNS)
Z95_T3 = 4.302653


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: {}".format(path))
    return value


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def number(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite metric: {}".format(value))
    return result


def _method_key(method: str, seed: int) -> str:
    return "{}_seed{}_30k".format(method.lower(), seed)


def collect_metric_tables(metrics_root: Path, output: Path) -> Dict[int, Dict[str, Dict[str, str]]]:
    summaries: Dict[int, Dict[str, Dict[str, str]]] = {}
    metric_rows = []
    for checkpoint in CHECKPOINTS:
        rows = read_csv(metrics_root / ("val_{}".format(checkpoint)) / "metrics_summary.csv")
        by_name = {row["experiment"]: row for row in rows}
        if set(by_name) != set(RUN_NAMES):
            raise ValueError("unexpected experiments at {}: {}".format(checkpoint, sorted(by_name)))
        summaries[checkpoint] = by_name
        for name, row in sorted(by_name.items()):
            if int(row["num_views"]) != 34:
                raise ValueError("expected 34 val views: {}".format(row))
            method = "B0" if name.startswith("b0_") else "E2"
            seed = int(name.split("seed", 1)[1].split("_", 1)[0])
            metric_rows.append({
                "checkpoint": checkpoint,
                "train_seed": seed,
                "method": method,
                "experiment": name,
                "num_views": row["num_views"],
                **{metric: number(row[metric]) for metric in METRICS},
            })
    write_csv(output / "metrics_by_seed.csv", ["checkpoint", "train_seed", "method", "experiment", "num_views", *METRICS], metric_rows)

    delta_rows = []
    for checkpoint in CHECKPOINTS:
        for seed in SEEDS:
            b0 = summaries[checkpoint][_method_key("B0", seed)]
            e2 = summaries[checkpoint][_method_key("E2", seed)]
            delta_rows.append({
                "checkpoint": checkpoint,
                "train_seed": seed,
                "lhs": "E2",
                "rhs": "B0",
                **{"delta_" + metric: number(e2[metric]) - number(b0[metric]) for metric in METRICS},
            })
    write_csv(output / "paired_deltas.csv", ["checkpoint", "train_seed", "lhs", "rhs", *["delta_" + metric for metric in METRICS]], delta_rows)

    seed_rows = []
    for checkpoint in CHECKPOINTS:
        for metric in METRICS:
            b0_values = [number(summaries[checkpoint][_method_key("B0", seed)][metric]) for seed in SEEDS]
            e2_values = [number(summaries[checkpoint][_method_key("E2", seed)][metric]) for seed in SEEDS]
            deltas = [e2 - b0 for e2, b0 in zip(e2_values, b0_values)]
            improved = sum(delta > 0 if metric in HIGHER else delta < 0 for delta in deltas)
            degraded = sum(delta < 0 if metric in HIGHER else delta > 0 for delta in deltas)
            seed_rows.append({
                "checkpoint": checkpoint,
                "metric": metric,
                "b0_mean": statistics.mean(b0_values),
                "b0_std": statistics.stdev(b0_values),
                "e2_mean": statistics.mean(e2_values),
                "e2_std": statistics.stdev(e2_values),
                "delta_mean": statistics.mean(deltas),
                "delta_std": statistics.stdev(deltas),
                "delta_median": statistics.median(deltas),
                "delta_ci95_low": statistics.mean(deltas) - Z95_T3 * statistics.stdev(deltas) / math.sqrt(3),
                "delta_ci95_high": statistics.mean(deltas) + Z95_T3 * statistics.stdev(deltas) / math.sqrt(3),
                "improved_seeds": improved,
                "degraded_seeds": degraded,
                "tied_seeds": 3 - improved - degraded,
            })
    write_csv(output / "seed_summary.csv", ["checkpoint", "metric", "b0_mean", "b0_std", "e2_mean", "e2_std", "delta_mean", "delta_std", "delta_median", "delta_ci95_low", "delta_ci95_high", "improved_seeds", "degraded_seeds", "tied_seeds"], seed_rows)

    split_path = Path("data/TI-NSD/heated/splits/seed2026/sparse_nested-random_25.json")
    split = read_json(split_path)
    val_names = split["val"]
    image_to_view = {"{:05d}.png".format(index): name for index, name in enumerate(val_names)}
    per_view_rows = []
    for checkpoint in CHECKPOINTS:
        all_rows = read_csv(metrics_root / ("val_{}".format(checkpoint)) / "metrics_per_view.csv")
        grouped: Dict[str, Dict[str, Dict[str, str]]] = {}
        for row in all_rows:
            grouped.setdefault(row["experiment"], {})[row["image"]] = row
        for seed in SEEDS:
            b0 = grouped[_method_key("B0", seed)]
            e2 = grouped[_method_key("E2", seed)]
            if set(b0) != set(e2) or len(b0) != 34:
                raise ValueError("per-view mismatch for seed {} checkpoint {}".format(seed, checkpoint))
            for image in sorted(b0):
                per_view_rows.append({
                    "checkpoint": checkpoint,
                    "train_seed": seed,
                    "image": image,
                    "view_id": image_to_view.get(image, image),
                    **{"delta_" + metric: number(e2[image][metric]) - number(b0[image][metric]) for metric in METRICS},
                })
    write_csv(output / "per_view_deltas.csv", ["checkpoint", "train_seed", "image", "view_id", *["delta_" + metric for metric in METRICS]], per_view_rows)

    view_summary = []
    for checkpoint in CHECKPOINTS:
        for seed in SEEDS:
            subset = [row for row in per_view_rows if int(row["checkpoint"]) == checkpoint and int(row["train_seed"]) == seed]
            for metric in METRICS:
                values = [number(row["delta_" + metric]) for row in subset]
                worst = min(values) if metric in HIGHER else max(values)
                worst_row = next(row for row in subset if number(row["delta_" + metric]) == worst)
                improved = sum(value > 0 if metric in HIGHER else value < 0 for value in values)
                degraded = sum(value < 0 if metric in HIGHER else value > 0 for value in values)
                view_summary.append({
                    "checkpoint": checkpoint,
                    "train_seed": seed,
                    "metric": metric,
                    "mean_delta": statistics.mean(values),
                    "median_delta": statistics.median(values),
                    "improved_views": improved,
                    "degraded_views": degraded,
                    "tied_views": len(values) - improved - degraded,
                    "worst_image": worst_row["image"],
                    "worst_view_id": worst_row["view_id"],
                    "worst_delta": worst,
                })
    write_csv(output / "per_view_summary.csv", ["checkpoint", "train_seed", "metric", "mean_delta", "median_delta", "improved_views", "degraded_views", "tied_views", "worst_image", "worst_view_id", "worst_delta"], view_summary)
    return summaries


def resource_summary(runs_root: Path, output: Path) -> List[Dict[str, Any]]:
    rows = []
    for run_name, method, seed in RUNS:
        run_dir = runs_root / run_name
        manifest = read_json(run_dir / "run_manifest.json")
        status = read_json(run_dir / "status.json")
        audits = {int(row["iteration"]): row for row in read_csv(run_dir / "loss_components.csv")}
        for checkpoint in CHECKPOINTS:
            timing = read_json(run_dir / "val" / ("ours_{}".format(checkpoint)) / "render_timing.json")
            files = [
                run_dir / "point_cloud" / ("iteration_{}".format(checkpoint)) / "point_cloud.ply",
                run_dir / "ATF" / ("iteration_{}".format(checkpoint)) / "ATF.pth",
                run_dir / "TCM" / ("iteration_{}".format(checkpoint)) / "TCM.pth",
            ]
            if not all(path.is_file() for path in files):
                raise FileNotFoundError("checkpoint artifact missing for {} {}".format(run_name, checkpoint))
            audit = audits[checkpoint]
            rows.append({
                "experiment": run_name,
                "method": method,
                "train_seed": seed,
                "iteration": checkpoint,
                "run_status": status["status"],
                "train_wall_seconds": manifest["phase_timings"]["train"]["wall_seconds"],
                "gaussian_count": int(audit["Gaussian_count"]),
                "core_avg_ms": number(audit["avg_ms"]),
                "cuda_allocated_peak_mib": number(audit["peak_cuda_mb"]),
                "model_total_bytes": sum(path.stat().st_size for path in files),
                "render_wall_seconds": timing["wall_seconds"],
                "render_forward_mean_ms": timing["forward_mean_ms"],
                "render_samples_total": timing["forward_samples_total"],
                "render_warmup_excluded": timing["warmup_excluded"],
            })
    write_csv(output / "resource_summary.csv", list(rows[0]), rows)
    return rows


def audit_initialization_and_pairing(runs_root: Path, output: Path) -> Dict[str, Any]:
    init_rows = []
    source_hashes = set()
    config_hashes = set()
    for run_name, method, seed in RUNS:
        run_dir = runs_root / run_name
        audit = read_json(run_dir / "initial_state_audit.json")
        comparison = audit["initialization"]["comparison"]
        init_rows.append({
            "experiment": run_name,
            "method": method,
            "train_seed": seed,
            "package_sha256": audit["initialization"]["package_sha256"],
            "gaussian": comparison["gaussian"],
            "atf": comparison["atf"],
            "tcm": comparison["tcm"],
            "optimizer": comparison["optimizer"],
            "rng": comparison["rng"],
        })
        manifest = read_json(run_dir / "run_manifest.json")
        source_hashes.add(manifest.get("git", {}).get("training_source_sha256"))
        config_hashes.add(manifest.get("config_sha256"))
    write_csv(output / "initial_state_comparison.csv", list(init_rows[0]), init_rows)

    pair_rows = []
    for seed in SEEDS:
        b0 = read_json(runs_root / _method_key("B0", seed) / "camera_sequence.json")
        e2 = read_json(runs_root / _method_key("E2", seed) / "camera_sequence.json")
        pair_rows.append({
            "train_seed": seed,
            "same_sequence": b0["sequence"] == e2["sequence"],
            "b0_count": b0["count"],
            "e2_count": e2["count"],
            "b0_sequence_sha256": b0["sequence_sha256"],
            "e2_sequence_sha256": e2["sequence_sha256"],
            **{"prefix_{}_same".format(checkpoint): b0["prefix_sha256"].get(str(checkpoint)) == e2["prefix_sha256"].get(str(checkpoint)) for checkpoint in CHECKPOINTS},
        })
    write_csv(output / "camera_pairing.csv", list(pair_rows[0]), pair_rows)
    return {
        "initialization_rows": init_rows,
        "camera_rows": pair_rows,
        "training_source_sha256": sorted(value for value in source_hashes if value),
        "config_sha256": sorted(value for value in config_hashes if value),
    }


def _load_luma(path: Path):
    import numpy as np
    from PIL import Image
    with Image.open(str(path)) as image:
        rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    return rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114


def _sobel(image):
    import numpy as np
    padded = np.pad(image, ((1, 1), (1, 1)), mode="edge")
    gx = (-padded[:-2, :-2] + padded[:-2, 2:] - 2 * padded[1:-1, :-2] + 2 * padded[1:-1, 2:] - padded[2:, :-2] + padded[2:, 2:]) / 8.0
    gy = (-padded[:-2, :-2] - 2 * padded[:-2, 1:-1] - padded[:-2, 2:] + padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:]) / 8.0
    return np.sqrt(gx * gx + gy * gy)


def make_figures(runs_root: Path, output: Path) -> Dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fixed = ("00000.png", "00011.png", "00022.png", "00033.png")
    records = []
    for seed in SEEDS:
        b0_dir = runs_root / _method_key("B0", seed) / "val" / "ours_30000"
        e2_dir = runs_root / _method_key("E2", seed) / "val" / "ours_30000"
        values = {"qualitative": [], "absolute_error": [], "edge": []}
        for name in fixed:
            gt = _load_luma(b0_dir / "gt" / name)
            b0 = _load_luma(b0_dir / "renders" / name)
            e2 = _load_luma(e2_dir / "renders" / name)
            values["qualitative"].extend((gt, b0, e2))
            values["absolute_error"].extend((abs(b0 - gt), abs(e2 - gt)))
            values["edge"].extend((_sobel(gt), _sobel(b0), _sobel(e2)))
        for kind, columns, cmap, vmin, vmax in (("qualitative", 3, "gray", 0.0, 1.0), ("absolute_error", 2, "magma", 0.0, 0.25), ("edge", 3, "viridis", 0.0, 0.25)):
            fig, axes = plt.subplots(4, columns, figsize=(3.0 * columns, 10.0), squeeze=False, constrained_layout=True)
            for index, image in enumerate(values[kind]):
                axis = axes[index // columns][index % columns]
                axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
                axis.set_axis_off()
            path = figure_dir / "{}_seed{}_30000.png".format(kind, seed)
            fig.savefig(str(path), dpi=140, facecolor="white")
            plt.close(fig)
            records.append({"path": str(path.relative_to(output)), "sha256": sha256(path), "kind": kind, "seed": seed})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    metric_rows = {row["experiment"]: row for row in read_csv(output / "metrics_by_seed.csv") if int(row["checkpoint"]) == 30000}
    for method, style in (("B0", "-"), ("E2", "--")):
        xs = list(SEEDS)
        axes[0].plot(xs, [number(metric_rows[_method_key(method, seed)]["psnr"]) for seed in xs], style, label=method, marker="o")
        axes[1].plot(xs, [number(metric_rows[_method_key(method, seed)]["t_mae"]) for seed in xs], style, label=method, marker="o")
    for axis, title, ylabel in ((axes[0], "PSNR at 30000", "PSNR"), (axes[1], "T-MAE at 30000", "T-MAE")):
        axis.set_title(title); axis.set_xlabel("training seed"); axis.set_ylabel(ylabel); axis.grid(True, alpha=0.25); axis.legend()
    path = figure_dir / "metric_curves_30000.png"
    fig.savefig(str(path), dpi=140, facecolor="white"); plt.close(fig)
    records.append({"path": str(path.relative_to(output)), "sha256": sha256(path), "kind": "metric_curves"})
    return {"outputs": records}


def tensorboard_figure(runs_root: Path, output: Path) -> Dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    tags = ("train/loss_total", "train/edge_weighted", "val_monitor/psnr", "model/gaussian_count")
    titles = ("Training total loss", "Weighted edge loss", "Internal val PSNR", "Gaussian count")
    for run_name, method, seed in RUNS:
        event_path = next((path for path in (runs_root / run_name).glob("events.out.tfevents.*")), None)
        if event_path is None:
            raise FileNotFoundError("event missing: {}".format(run_name))
        accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0}); accumulator.Reload()
        available = set(accumulator.Tags().get("scalars", []))
        for axis, tag, title in zip(axes.ravel(), tags, titles):
            if tag not in available:
                continue
            events = accumulator.Scalars(tag)
            axis.plot([item.step for item in events], [item.value for item in events], label="{} {}".format(method, seed), linestyle="-" if method == "B0" else "--", linewidth=1.0)
            axis.set_title(title); axis.set_xlabel("iteration"); axis.grid(True, alpha=0.25)
    for axis in axes.ravel():
        axis.legend(fontsize=7, ncol=2)
    path = output / "figures" / "tensorboard_training_summary.png"
    figure.savefig(str(path), dpi=140, facecolor="white"); plt.close(figure)
    return {"path": str(path.relative_to(output)), "sha256": sha256(path), "tags": list(tags)}


def write_report(output: Path, summaries: Mapping[int, Mapping[str, Mapping[str, str]]], resources: Sequence[Mapping[str, Any]], audits: Mapping[str, Any], figure_info: Mapping[str, Any], tb_info: Mapping[str, Any]) -> None:
    seed_summary = read_csv(output / "seed_summary.csv")
    lines = [
        "# Thermal3DGS B0 / E2 共同初始化三 seed 30k 报告", "", "日期：2026-09-14  ",
        "状态：六组从共同 step-0 包开始的正式 run 均已完成；18 组独立 val 评价均已完成；未访问 test。", "",
        "## 1. 协议与审计", "",
        "本轮只比较 B0 (`lambda_edge=0`) 与 E2 (`lambda_edge=0.001`)；其余辅助损失、Detail、GD 和 dropout 均关闭。每个 seed 使用同一未经训练初始化包，训练预算为 30000，固定观察点为 7000/15000/30000。",
        "", "初始化逐张量/优化器/RNG 比较：{}。相机序列配对：{}。".format(
            "通过" if all(all(row[key] for key in ("gaussian", "atf", "tcm", "optimizer", "rng")) for row in audits["initialization_rows"]) else "失败",
            "通过" if all(row["same_sequence"] and all(row["prefix_{}_same".format(checkpoint)] for checkpoint in CHECKPOINTS) for row in audits["camera_rows"]) else "失败",
        ),
        "", "`training_source_sha256` 唯一值数量：{}；配置 hash 唯一值数量：{}。源码摘要排除 runs/results/data/.venv/.git，运行日志增长不会改变该摘要。".format(len(audits["training_source_sha256"]), len(audits["config_sha256"])),
        "", "## 2. 跨 seed 统计", "", "delta 统一为 E2 - B0；PSNR/SSIM/Gradient preservation 的 delta>0 为改善，T-MAE/E-MAE 的 delta<0 为改善。30k 的 CI 为 n=3 描述性 t 区间，不代表显著性检验。", "",
        "| checkpoint | metric | B0 mean±std | E2 mean±std | delta mean±std | 95% CI (30k only) | improved/degraded/tied |", "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in seed_summary:
        ci = "{} .. {}".format(row["delta_ci95_low"], row["delta_ci95_high"]) if int(row["checkpoint"]) == 30000 else "unavailable"
        lines.append("| {} | {} | {} ± {} | {} ± {} | {} ± {} | {} | {}/{}/{} |".format(row["checkpoint"], row["metric"], row["b0_mean"], row["b0_std"], row["e2_mean"], row["e2_std"], row["delta_mean"], row["delta_std"], ci, row["improved_seeds"], row["degraded_seeds"], row["tied_seeds"]))
    lines += ["", "### 2.1 每个 seed 的五指标", "", "| checkpoint | seed | method | PSNR | SSIM | T-MAE | E-MAE | Gradient preservation |", "|---:|---:|---|---:|---:|---:|---:|---:|"]
    for checkpoint in CHECKPOINTS:
        for seed in SEEDS:
            for method in ("B0", "E2"):
                row = summaries[checkpoint][_method_key(method, seed)]
                lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(checkpoint, seed, method, row["psnr"], row["ssim"], row["t_mae"], row["e_mae"], row["gradient_preservation"]))
    lines += ["", "### 2.2 101.jpg 与逐视角", "", "固定映射 `00011.png -> 101.jpg`；完整 34-view 方向计数、中位数、最差真实 ID 见 `per_view_summary.csv`。", "", "| checkpoint | seed | ΔPSNR | ΔSSIM | ΔT-MAE | ΔE-MAE | ΔGradient preservation |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in read_csv(output / "per_view_deltas.csv"):
        if row["image"] == "00011.png":
            lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(row["checkpoint"], row["train_seed"], row["delta_psnr"], row["delta_ssim"], row["delta_t_mae"], row["delta_e_mae"], row["delta_gradient_preservation"]))
    lines += ["", "## 3. 资源与训练轨迹", "", "| checkpoint | method | train wall mean(s) | core mean(ms) | Gaussian mean | model bytes mean | render forward mean(ms) |", "|---:|---|---:|---:|---:|---:|---:|"]
    for checkpoint in CHECKPOINTS:
        for method in ("B0", "E2"):
            selected = [row for row in resources if int(row["iteration"]) == checkpoint and row["method"] == method]
            lines.append("| {} | {} | {:.3f} | {:.3f} | {:.1f} | {:.1f} | {:.3f} |".format(checkpoint, method, statistics.mean(number(row["train_wall_seconds"]) for row in selected), statistics.mean(number(row["core_avg_ms"]) for row in selected), statistics.mean(number(row["gaussian_count"]) for row in selected), statistics.mean(number(row["model_total_bytes"]) for row in selected), statistics.mean(number(row["render_forward_mean_ms"]) for row in selected)))
    lines += ["", "TensorBoard 静态汇总图：`figures/tensorboard_training_summary.png`。固定视角、绝对误差和 Sobel 诊断图位于 `figures/`。", "", "## 4. 限制与未执行", "", "本轮为单场景、固定 Sparse-25/noise03、完整 COLMAP 先验和三个训练 seed；三个 seed 不是三套独立数据。LPIPS/ROI-MAE 仍 unavailable；未读取、渲染或评分最终 test；未执行渐进权重、随机性来源分离、额外场景或任何追加调参。", "", "停止接口保留为 `runs/b0_e2_30k/v1/STOP_REQUESTED`；完整命令、stdout/stderr、CSV、event 和状态文件均在对应目录。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("runs/b0_e2_30k/v1"))
    parser.add_argument("--metrics-root", type=Path, default=Path("results/b0_e2_30k/v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/b0_e2_30k/v1"))
    args = parser.parse_args()
    runs_root = args.runs_root.resolve(); output = args.output_dir.resolve(); metrics_root = args.metrics_root.resolve(); output.mkdir(parents=True, exist_ok=True)
    summary_data = collect_metric_tables(metrics_root, output)
    resources = resource_summary(runs_root, output)
    audits = audit_initialization_and_pairing(runs_root, output)
    figure_info = make_figures(runs_root, output)
    tb_info = tensorboard_figure(runs_root, output)
    figure_manifest = {"schema": "thermal3dgs.b0_e2_shared_init_30k_figure_manifest", "schema_version": 1, "fixed_images": ["00000.png", "00011.png", "00022.png", "00033.png"], "fixed_view_ids": ["001.jpg", "101.jpg", "202.jpg", "302.jpg"], "outputs": figure_info["outputs"] + [tb_info]}
    (output / "figures" / "figure_manifest.json").write_text(json.dumps(figure_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(output, summary_data, resources, audits, figure_info, tb_info)
    (output / "protocol_snapshot.md").write_text("# B0 / E2 shared-init 30k frozen protocol\n\nSix from-scratch runs use the same per-seed step-0 package and compare B0 lambda_edge=0 with E2 lambda_edge=0.001 at 7000/15000/30000. Evaluation is val-only with 34 views.\n", encoding="utf-8")
    (output / "handoff.md").write_text("# B0 / E2 shared-init 30k handoff\n\nFormal queue and independent val evaluation completed. See report.md, CSVs, figures and initial_state_comparison.csv. No test pixels were accessed and no follow-up experiments were started.\n", encoding="utf-8")
    run_records = []
    for run_name, method, seed in RUNS:
        manifest_path = runs_root / run_name / "run_manifest.json"
        run_records.append({"name": run_name, "method": method, "train_seed": seed, "status": read_json(runs_root / run_name / "status.json"), "run_manifest_sha256": sha256(manifest_path)})
    source_manifest = read_json(runs_root / RUNS[0][0] / "run_manifest.json").get("git", {})
    (output / "source_manifest.json").write_text(json.dumps({
        "schema": "thermal3dgs.training_source_manifest",
        "schema_version": 1,
        "training_source_sha256": source_manifest.get("training_source_sha256"),
        "training_source_files": source_manifest.get("training_source_files", []),
        "note": "Source files exclude runs/results/data/.venv/.git; configuration hashes are stored per run manifest.",
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs = [path for path in output.rglob("*") if path.is_file() and path.name != "results_manifest.json"]
    results_manifest = {"schema": "thermal3dgs.b0_e2_shared_init_30k_results_manifest", "schema_version": 1, "protocol": {"methods": ["B0", "E2"], "seeds": list(SEEDS), "checkpoints": list(CHECKPOINTS), "evaluation_partition": "val", "delta": "E2-B0"}, "runs_root": str(runs_root), "results_root": str(output), "runs": run_records, "audits": audits, "tensorboard": tb_info, "outputs": {str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(outputs)}}
    (output / "results_manifest.json").write_text(json.dumps(results_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
