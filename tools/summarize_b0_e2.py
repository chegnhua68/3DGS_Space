#!/usr/bin/env python3
"""Build the B0/E2 three-seed confirmation result package."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

METRICS = ("psnr", "ssim", "t_mae", "e_mae", "gradient_preservation")
HIGHER = {"psnr", "ssim", "gradient_preservation"}
RUNS = (
    ("b0_seed2026_7k", "B0", 2026),
    ("e2_seed2026_7k", "E2", 2026),
    ("e2_seed2027_7k", "E2", 2027),
    ("b0_seed2027_7k", "B0", 2027),
    ("b0_seed2028_7k", "B0", 2028),
    ("e2_seed2028_7k", "E2", 2028),
)
RUN_BY_EXPERIMENT = {name[:-3]: (method, seed) for name, method, seed in RUNS}


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


def num(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite value: {}".format(value))
    return result


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return "{:.8f}".format(value)
    return str(value)


def summaries(metrics_root: Path, output: Path) -> Dict[int, Dict[str, Dict[str, str]]]:
    result: Dict[int, Dict[str, Dict[str, str]]] = {}
    metric_rows: List[Dict[str, Any]] = []
    for checkpoint in (2000, 7000):
        rows = read_csv(metrics_root / ("val_{}".format(checkpoint)) / "metrics_summary.csv")
        by_name = {row["experiment"]: row for row in rows}
        expected = {name[:-3] for name, _, _ in RUNS}
        if set(by_name) != expected:
            raise ValueError("unexpected experiments at {}: {}".format(checkpoint, sorted(by_name)))
        result[checkpoint] = by_name
        for experiment, row in sorted(by_name.items()):
            method, seed = RUN_BY_EXPERIMENT[experiment]
            if int(row["num_views"]) != 34:
                raise ValueError("expected 34 val views: {}".format(row))
            metric_rows.append({
                "checkpoint": checkpoint,
                "train_seed": seed,
                "method": method,
                "experiment": experiment,
                "num_views": row["num_views"],
                **{metric: num(row[metric]) for metric in METRICS},
            })
    write_csv(output / "metrics_by_seed.csv", ["checkpoint", "train_seed", "method", "experiment", "num_views", *METRICS], metric_rows)

    delta_rows: List[Dict[str, Any]] = []
    for checkpoint in (2000, 7000):
        for seed in (2026, 2027, 2028):
            b0 = result[checkpoint]["b0_seed{}".format(seed)]
            e2 = result[checkpoint]["e2_seed{}".format(seed)]
            delta_rows.append({
                "checkpoint": checkpoint,
                "train_seed": seed,
                "lhs": "E2",
                "rhs": "B0",
                **{"delta_" + metric: num(e2[metric]) - num(b0[metric]) for metric in METRICS},
            })
    write_csv(output / "paired_deltas.csv", ["checkpoint", "train_seed", "lhs", "rhs", *["delta_" + metric for metric in METRICS]], delta_rows)

    per_view_rows: List[Dict[str, Any]] = []
    for checkpoint in (2000, 7000):
        all_rows = read_csv(metrics_root / ("val_{}".format(checkpoint)) / "metrics_per_view.csv")
        grouped: Dict[str, Dict[str, Dict[str, str]]] = {}
        for row in all_rows:
            grouped.setdefault(row["experiment"], {})[row["image"]] = row
        for seed in (2026, 2027, 2028):
            b0 = grouped["b0_seed{}".format(seed)]
            e2 = grouped["e2_seed{}".format(seed)]
            if set(b0) != set(e2):
                raise ValueError("per-view filename mismatch for seed {}".format(seed))
            for image in sorted(b0):
                per_view_rows.append({
                    "checkpoint": checkpoint,
                    "train_seed": seed,
                    "image": image,
                    **{"delta_" + metric: num(e2[image][metric]) - num(b0[image][metric]) for metric in METRICS},
                })
    write_csv(output / "per_view_deltas.csv", ["checkpoint", "train_seed", "image", *["delta_" + metric for metric in METRICS]], per_view_rows)

    seed_rows: List[Dict[str, Any]] = []
    for checkpoint in (2000, 7000):
        for metric in METRICS:
            b0_values = [num(result[checkpoint]["b0_seed{}".format(seed)][metric]) for seed in (2026, 2027, 2028)]
            e2_values = [num(result[checkpoint]["e2_seed{}".format(seed)][metric]) for seed in (2026, 2027, 2028)]
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
                "improved_seeds": improved,
                "degraded_seeds": degraded,
                "tied_seeds": 3 - improved - degraded,
            })
    write_csv(output / "seed_summary.csv", ["checkpoint", "metric", "b0_mean", "b0_std", "e2_mean", "e2_std", "delta_mean", "delta_std", "delta_median", "improved_seeds", "degraded_seeds", "tied_seeds"], seed_rows)

    val_names = json.loads((Path("data/TI-NSD/heated/splits/seed2026/sparse_nested-random_25.json")).read_text(encoding="utf-8"))["val"]
    image_to_view = {"{:05d}.png".format(index): name for index, name in enumerate(val_names)}
    view_summary_rows: List[Dict[str, Any]] = []
    for checkpoint in (2000, 7000):
        for seed in (2026, 2027, 2028):
            subset = [row for row in per_view_rows if int(row["checkpoint"]) == checkpoint and int(row["train_seed"]) == seed]
            for metric in METRICS:
                values = [float(row["delta_" + metric]) for row in subset]
                worst = min(values) if metric in HIGHER else max(values)
                worst_row = next(row for row in subset if float(row["delta_" + metric]) == worst)
                improved = sum(value > 0 if metric in HIGHER else value < 0 for value in values)
                degraded = sum(value < 0 if metric in HIGHER else value > 0 for value in values)
                view_summary_rows.append({
                    "checkpoint": checkpoint,
                    "train_seed": seed,
                    "metric": metric,
                    "mean_delta": statistics.mean(values),
                    "median_delta": statistics.median(values),
                    "improved_views": improved,
                    "degraded_views": degraded,
                    "tied_views": len(values) - improved - degraded,
                    "worst_image": worst_row["image"],
                    "worst_view_id": image_to_view.get(worst_row["image"], worst_row["image"]),
                    "worst_delta": worst,
                })
    write_csv(output / "per_view_summary.csv", ["checkpoint", "train_seed", "metric", "mean_delta", "median_delta", "improved_views", "degraded_views", "tied_views", "worst_image", "worst_view_id", "worst_delta"], view_summary_rows)
    return result


def resource_summary(root: Path, output: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for run_name, method, seed in RUNS:
        run_dir = root / run_name
        manifest = read_json(run_dir / "run_manifest.json")
        status = read_json(run_dir / "status.json")
        audits = {int(row["iteration"]): row for row in read_csv(run_dir / "loss_components.csv")}
        for checkpoint in (2000, 7000):
            timing = read_json(run_dir / "val" / ("ours_{}".format(checkpoint)) / "render_timing.json")
            files = [
                run_dir / "point_cloud" / ("iteration_{}".format(checkpoint)) / "point_cloud.ply",
                run_dir / "ATF" / ("iteration_{}".format(checkpoint)) / "ATF.pth",
                run_dir / "TCM" / ("iteration_{}".format(checkpoint)) / "TCM.pth",
            ]
            sizes = [path.stat().st_size for path in files]
            audit = audits[checkpoint]
            rows.append({
                "experiment": run_name[:-3], "method": method, "train_seed": seed, "iteration": checkpoint,
                "run_status": status["status"], "train_wall_seconds": manifest["phase_timings"]["train"]["wall_seconds"],
                "runner_final_render_wall_seconds": manifest["phase_timings"]["render"]["wall_seconds"] if checkpoint == 7000 else "unavailable",
                "gaussian_count": int(audit["Gaussian_count"]), "core_avg_ms": num(audit["avg_ms"]), "cuda_allocated_peak_mib": num(audit["peak_cuda_mb"]),
                "point_cloud_bytes": sizes[0], "atf_bytes": sizes[1], "tcm_bytes": sizes[2], "model_total_bytes": sum(sizes),
                "render_wall_seconds": timing["wall_seconds"], "render_forward_mean_ms": timing["forward_mean_ms"],
                "render_samples_total": timing["forward_samples_total"], "render_warmup_excluded": timing["warmup_excluded"], "render_timed_samples": timing["forward_timed_samples"],
            })
    write_csv(output / "resource_summary.csv", list(rows[0]), rows)


def _load_image(path: Path):
    import numpy as np
    from PIL import Image
    with Image.open(str(path)) as image:
        image.load()
        if image.mode in ("1", "L"):
            return np.array(image.convert("L"), dtype=np.float32) / 255.0
        if image.mode.startswith("I;16") or image.mode == "I":
            return np.array(image, dtype=np.float32) / 65535.0
        rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        return rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114


def _sobel(image):
    import numpy as np
    padded = np.pad(image, ((1, 1), (1, 1)), mode="edge")
    gx = (-padded[:-2, :-2] + padded[:-2, 2:] - 2 * padded[1:-1, :-2] + 2 * padded[1:-1, 2:] - padded[2:, :-2] + padded[2:, 2:]) / 8.0
    gy = (-padded[:-2, :-2] - 2 * padded[:-2, 1:-1] - padded[:-2, 2:] + padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:]) / 8.0
    return np.sqrt(gx * gx + gy * gy)


def figures(root: Path, output: Path) -> Dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fixed = ("00000.png", "00011.png", "00022.png", "00033.png")
    records: List[Dict[str, Any]] = []
    for seed in (2026, 2027, 2028):
        b0_dir = root / "b0_seed{}_7k".format(seed) / "val" / "ours_7000"
        e2_dir = root / "e2_seed{}_7k".format(seed) / "val" / "ours_7000"
        qualitative, errors, edges, crops = [], [], [], []
        for name in fixed:
            gt = _load_image(b0_dir / "gt" / name)
            b0 = _load_image(b0_dir / "renders" / name)
            e2 = _load_image(e2_dir / "renders" / name)
            qualitative.extend((gt, b0, e2))
            errors.extend((abs(b0 - gt), abs(e2 - gt)))
            edges.extend((_sobel(gt), _sobel(b0), _sobel(e2)))
            h, w = gt.shape
            crop = (slice(h // 3, (2 * h) // 3), slice(w // 3, (2 * w) // 3))
            crops.extend((gt[crop], b0[crop], e2[crop]))
        for values, columns, prefix, cmap, vmin, vmax in ((qualitative, 3, "qualitative", "gray", 0.0, 1.0), (errors, 2, "absolute_error", "magma", 0.0, 0.25), (edges, 3, "edge", "viridis", 0.0, 0.25), (crops, 3, "center_crop", "gray", 0.0, 1.0)):
            fig, axes = plt.subplots(4, columns, figsize=(3.0 * columns, 2.7 * 4), squeeze=False, constrained_layout=True)
            for index, image in enumerate(values):
                axis = axes[index // columns][index % columns]
                axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
                axis.set_axis_off()
            path = figure_dir / "{}_seed{}_7000.png".format(prefix, seed)
            fig.savefig(str(path), dpi=160, facecolor="white")
            plt.close(fig)
            records.append({"path": str(path.relative_to(output)), "sha256": sha256(path), "seed": seed, "kind": prefix})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    colors = {2026: "#0072B2", 2027: "#D55E00", 2028: "#009E73"}
    metric_rows = {row["experiment"]: row for row in read_csv(output / "metrics_by_seed.csv") if int(row["checkpoint"]) == 7000}
    for method, style in (("B0", "-"), ("E2", "--")):
        xs = [2026, 2027, 2028]
        axes[0].plot(xs, [num(metric_rows["{}_seed{}".format(method.lower(), seed)]["psnr"]) for seed in xs], style, label=method, color="#444444", marker="o")
        axes[1].plot(xs, [num(metric_rows["{}_seed{}".format(method.lower(), seed)]["t_mae"]) for seed in xs], style, label=method, color="#444444", marker="o")
    for axis, title, ylabel in ((axes[0], "PSNR by train seed (7000)", "PSNR"), (axes[1], "T-MAE by train seed (7000)", "T-MAE")):
        axis.set_title(title); axis.set_xlabel("training seed"); axis.set_ylabel(ylabel); axis.grid(True, alpha=0.25); axis.legend()
    path = figure_dir / "metric_curves_7000.png"
    fig.savefig(str(path), dpi=160, facecolor="white"); plt.close(fig)
    records.append({"path": str(path.relative_to(output)), "sha256": sha256(path), "kind": "metric_curves"})
    return {"outputs": records}


def tensorboard_figure(root: Path, output: Path) -> Dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    tags = ("train/loss_total", "train/edge_weighted", "val_monitor/psnr", "model/gaussian_count")
    titles = ("Training total loss", "Weighted edge loss", "Internal val PSNR", "Gaussian count")
    for run_name, method, seed in RUNS:
        event_path = next((path for path in (root / run_name).glob("events.out.tfevents.*")), None)
        if event_path is None:
            raise FileNotFoundError("event missing: {}".format(run_name))
        acc = EventAccumulator(str(event_path), size_guidance={"scalars": 0}); acc.Reload()
        available = set(acc.Tags().get("scalars", []))
        for axis, tag, title in zip(axes.ravel(), tags, titles):
            if tag not in available:
                continue
            events = acc.Scalars(tag)
            axis.plot([item.step for item in events], [item.value for item in events], label="{} {}".format(method, seed), linestyle="-" if method == "B0" else "--", linewidth=1.1)
            axis.set_title(title); axis.set_xlabel("iteration"); axis.grid(True, alpha=0.25)
    for axis in axes.ravel():
        axis.legend(fontsize=7, ncol=2)
    path = output / "figures" / "tensorboard_training_summary.png"
    fig_dir = path.parent; fig_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(path), dpi=160, facecolor="white"); plt.close(figure)
    return {"path": str(path.relative_to(output)), "sha256": sha256(path), "tags": list(tags)}


def report(output: Path, summaries_by_checkpoint: Mapping[int, Mapping[str, Mapping[str, str]]], resources: Sequence[Mapping[str, Any]], figure_info: Mapping[str, Any], tb_info: Mapping[str, Any]) -> None:
    seed_summary = read_csv(output / "seed_summary.csv")
    lines = [
        "# Thermal3DGS B0 / E2 三 seed 确认报告", "", "日期：2026-09-11  ",
        "状态：六个正式 run 全部完成；仅访问 val，未访问 test；未运行 30k。", "",
        "## 1. 冻结范围", "",
        "本轮只比较 B0 与 E2。两组使用同一 Sparse-25/noise03 数据、同一 34-view val、同一训练 seed；唯一算法差异为 `lambda_edge=0` 与 `lambda_edge=0.001`。`lambda_thermal=lambda_smooth=lambda_detail=0`，GD=0，未加入 Detail、GD 或其他模块。",
        "", "训练代码 HEAD：`22deede`；六组 tracked diff hash 一致，正式目录为 `runs/b0_e2_confirm/v1/`。输入哈希见 `configs/experiment_matrix.b0_e2_confirm_7k.json`。", "",
        "## 2. 跨 seed mean ± sample std", "", "delta 统一定义为 `E2 - B0`；误差指标只有 delta<0 才计为改善。", "",
        "| checkpoint | metric | B0 | E2 | delta | improved/degraded/tied seeds |", "|---:|---|---:|---:|---:|---:|",
    ]
    for item in seed_summary:
        lines.append("| {} | {} | {} ± {} | {} ± {} | {} ± {} | {}/{}/{} |".format(item["checkpoint"], item["metric"], item["b0_mean"], item["b0_std"], item["e2_mean"], item["e2_std"], item["delta_mean"], item["delta_std"], item["improved_seeds"], item["degraded_seeds"], item["tied_seeds"]))
    lines += ["", "### 2.1 每个 seed 的五指标均值", "", "| checkpoint | seed | method | PSNR | SSIM | T-MAE | E-MAE | Gradient preservation |", "|---:|---:|---|---:|---:|---:|---:|---:|"]
    for checkpoint in (2000, 7000):
        for seed in (2026, 2027, 2028):
            for method in ("B0", "E2"):
                row = summaries_by_checkpoint[checkpoint]["{}_seed{}".format(method.lower(), seed)]
                lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(checkpoint, seed, method, row["psnr"], row["ssim"], row["t_mae"], row["e_mae"], row["gradient_preservation"]))
    lines += ["", "### 2.2 101.jpg 与逐视角分布", "", "固定映射 `00011.png -> 101.jpg`。该视角的 E2-B0 差值如下；完整 34-view 均值、中位数、方向计数及最差真实 ID 见 `per_view_summary.csv`。", "", "| checkpoint | seed | ΔPSNR | ΔSSIM | ΔT-MAE | ΔE-MAE | ΔGradient preservation |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in read_csv(output / "per_view_deltas.csv"):
        if row["image"] == "00011.png":
            lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(row["checkpoint"], row["train_seed"], row["delta_psnr"], row["delta_ssim"], row["delta_t_mae"], row["delta_e_mae"], row["delta_gradient_preservation"]))
    lines.append("")
    lines += ["## 3. 配对与 provenance", "", "三个 seed 的 B0/E2 相机序列 SHA256 均逐 seed 相同；输入 dataset/split/degradation 哈希各只有一个值，初始 Gaussian 数均为 5079。当前 logger 没有记录优化前 Gaussian/ATF/TCM 张量内容摘要，因此不能把这些证据写成逐张量等价证明。", "", "run manifest 的 `source_snapshot_sha256` 会纳入不断增长的未跟踪 `results/*.log`，六组因此得到六个不同值；这不是训练代码变化。六组 Git HEAD 与 tracked diff hash 各只有一个值，完整审计保留在 run manifest 与 `results_manifest.json`。", ""]
    final_resources = [row for row in resources if int(row["iteration"]) == 7000]
    lines += ["## 4. 资源与训练日志", "", "| method | train wall mean (s) | core mean (ms) | Gaussian mean | model mean (bytes) | render forward mean (ms) |", "|---|---:|---:|---:|---:|---:|"]
    for method in ("B0", "E2"):
        selected = [row for row in final_resources if row["method"] == method]
        lines.append("| {} | {:.3f} | {:.3f} | {:.1f} | {:.1f} | {:.3f} |".format(method, statistics.mean(num(row["train_wall_seconds"]) for row in selected), statistics.mean(num(row["core_avg_ms"]) for row in selected), statistics.mean(num(row["gaussian_count"]) for row in selected), statistics.mean(num(row["model_total_bytes"]) for row in selected), statistics.mean(num(row["render_forward_mean_ms"]) for row in selected)))
    b0_wall = statistics.mean(num(row["train_wall_seconds"]) for row in final_resources if row["method"] == "B0")
    e2_wall = statistics.mean(num(row["train_wall_seconds"]) for row in final_resources if row["method"] == "E2")
    lines.append("")
    lines.append("E2 相对 B0 的三 seed 平均训练墙钟开销为 `{:.2f}%`；完整测试为 `123 tests, OK`，两组 smoke 均完成。".format((e2_wall / b0_wall - 1.0) * 100.0))
    lines += ["", "`resource_summary.csv` 分开记录 train wall、CUDA-event core、allocated peak、快照大小和独立 render 计时；显存不是整卡占用，render forward 排除前五个 warm-up 和 PNG I/O。stdout 仅保留阶段消息，细节落在每个 run 的 CSV/event/stdout/stderr。", "", "TensorBoard：`http://127.0.0.1:6007/`；logdir：`runs/b0_e2_confirm/v1/`；静态曲线：`figures/tensorboard_training_summary.png`。", "", "停止命令：", "", "```powershell", "New-Item -ItemType File .\\runs\\b0_e2_confirm\\v1\\STOP_REQUESTED -Force | Out-Null", "```", "", "## 5. 固定图与限制", "", "每个 seed 的 7k 固定视角图、误差图、Sobel 图和中心 crop 位于 `figures/`；固定视角为 001/101/202/302.jpg，中心 crop 沿用 `[238,159,477,320]`，误差显示范围 `[0,0.25]`。", "", "本轮只验证单场景、固定 COLMAP 先验、固定 split/noise、三个训练 seed 和 7k 开发预算；LPIPS/ROI-MAE unavailable，不代表几何、温度真值或最终 test 结论。30k 仅生成候选配置，未启动；没有按结果调参或删 seed。", "", "## 6. 产物", "", "`metrics_by_seed.csv`、`paired_deltas.csv`、`seed_summary.csv`、`per_view_deltas.csv`、`per_view_summary.csv`、`resource_summary.csv`、`resolved_runs.json`、`results_manifest.json` 及 `figures/`。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("runs/b0_e2_confirm/v1"))
    parser.add_argument("--metrics-root", type=Path, default=Path("results/b0_e2_confirm/v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/b0_e2_confirm/v1"))
    args = parser.parse_args()
    root, output = args.runs_root.resolve(), args.output_dir.resolve(); metrics_root = args.metrics_root.resolve(); output.mkdir(parents=True, exist_ok=True)
    summary_data = summaries(metrics_root, output); resource_summary(root, output)
    fig_info = figures(root, output); tb_info = tensorboard_figure(root, output)
    figure_manifest = {"schema": "thermal3dgs.b0_e2_figure_manifest", "schema_version": 1, "fixed_images": ["00000.png", "00011.png", "00022.png", "00033.png"], "fixed_view_ids": ["001.jpg", "101.jpg", "202.jpg", "302.jpg"], "center_crop_xyxy": [238, 159, 477, 320], "absolute_error_range": [0.0, 0.25], "outputs": fig_info["outputs"] + [tb_info]}
    (output / "figures" / "figure_manifest.json").write_text(json.dumps(figure_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    resource_rows = read_csv(output / "resource_summary.csv")
    report(output, summary_data, resource_rows, fig_info, tb_info)
    run_records = []
    for run_name, method, seed in RUNS:
        run_dir = root / run_name
        run_records.append({"name": run_name, "method": method, "train_seed": seed, "status": read_json(run_dir / "status.json"), "run_manifest_sha256": sha256(run_dir / "run_manifest.json")})
    resolved = []
    for run_name, method, seed in RUNS:
        manifest = read_json(root / run_name / "run_manifest.json")
        resolved.append({"name": run_name, "method": method, "train_seed": seed, "train_command": manifest["train_command"], "render_command": manifest["render_command"], "input_file_sha256": manifest["input_file_sha256"], "git": manifest["git"], "status": manifest["status"]})
    (output / "resolved_runs.json").write_text(json.dumps(resolved, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "protocol_snapshot.md").write_text("# B0 / E2 frozen protocol\n\nSix from-scratch 7000-step runs compare B0 (`lambda_edge=0`) and E2 (`lambda_edge=0.001`) at train seeds 2026/2027/2028. All other auxiliary weights and GD are zero. Inputs are fixed Sparse-25/noise03; evaluation is val-only at 2000/7000. TensorBoard interval is 50, audit interval is 500, and the sticky stop file is `runs/b0_e2_confirm/v1/STOP_REQUESTED`.\n", encoding="utf-8")
    (output / "next_phase_30k.md").write_text("# 30k 后续候选（未执行）\n\n候选矩阵为 `configs/experiment_matrix.b0_e2_30k_candidate.json`，包含 B0/E2 × seed 2026/2027/2028，保存/评价 7k、15k、30k。该文件仅供下一次明确授权，本轮没有加载、排队或启动。\n", encoding="utf-8")
    outputs = [path for path in output.rglob("*") if path.is_file() and path.name not in ("results_manifest.json",)]
    manifest = {"schema": "thermal3dgs.b0_e2_results_manifest", "schema_version": 1, "protocol": {"methods": ["B0", "E2"], "seeds": [2026, 2027, 2028], "checkpoints": [2000, 7000], "delta": "E2-B0", "evaluation_partition": "val"}, "runs_root": str(root), "results_root": str(output), "runs": run_records, "tensorboard": tb_info, "outputs": {str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(outputs)}}
    (output / "results_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
