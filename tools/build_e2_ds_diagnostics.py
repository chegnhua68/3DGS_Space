#!/usr/bin/env python3
"""Create fixed-view diagnostics, cost tables, and the root handoff package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

SEEDS = (2026, 2027, 2028)
METHODS = ("E2", "E2+DS")
CHECKPOINTS = (7000, 15000, 30000)
FIXED_TRAIN = ("004.jpg", "121.jpg", "207.jpg", "295.jpg")
FIXED_VAL = ("001.jpg", "101.jpg", "202.jpg", "302.jpg")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def save_teacher_diagnostics(repo: Path, cache: Path, output: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    degradation_manifest_path = (repo / manifest["degradation_manifest"]).resolve()
    train_root = degradation_manifest_path.parent
    diag_root = output / "teacher_diagnostics"
    diag_root.mkdir(parents=True, exist_ok=True)
    by_id = {item["id"]: item for item in manifest["views"]}
    if set(FIXED_TRAIN) - set(by_id):
        raise ValueError("fixed training diagnostic id missing from supervision manifest")
    rows = []
    for image_id, item in by_id.items():
        observed = load_rgb(train_root / item["observed_relative_path"])
        teacher = np.load(cache / item["teacher_relative_path"], allow_pickle=False).astype(np.float32)
        target = np.load(cache / item["target_relative_path"], allow_pickle=False).astype(np.float32)
        rows.append({"id": image_id, "observed_mean": float(observed.mean()), "teacher_mean": float(teacher.mean()),
                     "target_mean": float(target.mean()), "mean_abs_y_minus_z": float(np.abs(observed - teacher).mean()),
                     "mean_abs_y_minus_t": float(np.abs(observed - target).mean()), "teacher_min": float(teacher.min()),
                     "teacher_max": float(teacher.max()), "target_min": float(target.min()), "target_max": float(target.max()),
                     "teacher_out_of_range_fraction": float(item.get("teacher_raw_out_of_range_fraction", 0.0))})
        if image_id not in FIXED_TRAIN:
            continue
        residual_yz = np.abs(observed - teacher).mean(axis=2)
        residual_yt = np.abs(observed - target).mean(axis=2)
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        panels = [(observed, "Y degraded", "image"), (teacher, "Z BM3D", "image"), (target, "T=0.25Y+0.75Z", "image"),
                  (residual_yz, "abs(Y-Z)", "residual"), (residual_yt, "abs(Y-T)", "residual")]
        for axis, (data, title, kind) in zip(axes.flat, panels):
            axis.imshow(data, cmap="magma" if kind == "residual" else None, vmin=0 if kind == "residual" else None, vmax=0.25 if kind == "residual" else None)
            axis.set_title(title)
            axis.axis("off")
        axes.flat[-1].axis("off")
        fig.suptitle(image_id)
        fig.savefig(diag_root / f"{Path(image_id).stem}.png", dpi=140)
        plt.close(fig)
    write_csv(diag_root / "teacher_diagnostics_summary.csv", list(rows[0]), rows)
    return {"fixed_ids": list(FIXED_TRAIN), "all_train_selected": len(rows), "mean_abs_y_minus_z": float(np.mean([row["mean_abs_y_minus_z"] for row in rows])),
            "mean_abs_y_minus_t": float(np.mean([row["mean_abs_y_minus_t"] for row in rows])), "teacher_min": min(row["teacher_min"] for row in rows),
            "teacher_max": max(row["teacher_max"] for row in rows), "target_min": min(row["target_min"] for row in rows), "target_max": max(row["target_max"] for row in rows)}


def save_fixed_validation(repo: Path, output: Path) -> None:
    figure_root = output / "figures" / "fixed_validation"
    figure_root.mkdir(parents=True, exist_ok=True)
    split = read_json(repo / "data/TI-NSD/heated/splits/seed2026/sparse_nested-random_25.json")
    val = split["val"]
    mapping = {name: f"{index:05d}.png" for index, name in enumerate(val)}
    for seed in SEEDS:
        for checkpoint in CHECKPOINTS:
            overview, overview_axes = plt.subplots(len(FIXED_VAL), 4, figsize=(12, 10), constrained_layout=True)
            for row_index, view_id in enumerate(FIXED_VAL):
                mapped = mapping[view_id]
                full_e2 = load_rgb(repo / "runs/e2_ds_softtarget/v1" / f"e2_seed{seed}_30k/val/ours_{checkpoint}/renders/{mapped}")
                full_ds = load_rgb(repo / "runs/e2_ds_softtarget/v1" / f"e2_ds_seed{seed}_30k/val/ours_{checkpoint}/renders/{mapped}")
                full_gt = load_rgb(repo / "runs/e2_ds_softtarget/v1" / f"e2_seed{seed}_30k/val/ours_{checkpoint}/gt/{mapped}")
                full_error = np.abs(full_ds - full_gt).mean(axis=2)
                for axis, data, title in zip(overview_axes[row_index], (full_gt, full_e2, full_ds, full_error), (f"{view_id} GT", "E2", "E2+DS", "abs(DS-GT)")):
                    axis.imshow(data, cmap="magma" if data.ndim == 2 else None, vmin=0 if data.ndim == 2 else None, vmax=0.25 if data.ndim == 2 else None)
                    axis.set_title(title)
                    axis.axis("off")
            overview.suptitle(f"Fixed validation views | seed {seed} | {checkpoint}")
            overview.savefig(figure_root / f"fixed_views_seed{seed}_{checkpoint}.png", dpi=140)
            plt.close(overview)
            image_name = mapping["101.jpg"]
            e2 = load_rgb(repo / "runs/e2_ds_softtarget/v1" / f"e2_seed{seed}_30k/val/ours_{checkpoint}/renders/{image_name}")
            ds = load_rgb(repo / "runs/e2_ds_softtarget/v1" / f"e2_ds_seed{seed}_30k/val/ours_{checkpoint}/renders/{image_name}")
            gt = load_rgb(repo / "runs/e2_ds_softtarget/v1" / f"e2_seed{seed}_30k/val/ours_{checkpoint}/gt/{image_name}")
            x1, y1, x2, y2 = (238, 159, 477, 320)
            gt, e2, ds = (array[y1:y2, x1:x2] for array in (gt, e2, ds))
            luma = lambda array: 0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]
            error = np.abs(ds - gt).mean(axis=2)
            sobel_gt = np.hypot(ndimage.sobel(luma(gt), axis=0), ndimage.sobel(luma(gt), axis=1))
            sobel_ds = np.hypot(ndimage.sobel(luma(ds), axis=0), ndimage.sobel(luma(ds), axis=1))
            fig, axes = plt.subplots(1, 6, figsize=(16, 3.2), constrained_layout=True)
            panels = [(gt, "GT"), (e2, "E2"), (ds, "E2+DS"), (error, "abs(DS-GT)"), (sobel_gt, "Sobel GT"), (sobel_ds, "Sobel DS")]
            for axis, (data, title) in zip(axes, panels):
                axis.imshow(data, cmap="magma" if data.ndim == 2 else None, vmin=0 if data.ndim == 2 else None, vmax=0.25 if title.startswith("abs") else None)
                axis.set_title(title)
                axis.axis("off")
            fig.suptitle(f"101.jpg | seed {seed} | {checkpoint}")
            fig.savefig(figure_root / f"101_seed{seed}_{checkpoint}.png", dpi=150)
            plt.close(fig)


def save_costs(repo: Path, output: Path, cache_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for method in ("E2", "E2+DS"):
        for seed in SEEDS:
            name = ("e2_ds" if method == "E2+DS" else "e2") + f"_seed{seed}_30k"
            run = repo / "runs/e2_ds_softtarget/v1" / name
            manifest = read_json(run / "run_manifest.json")
            train_seconds = manifest.get("phase_timings", {}).get("train", {}).get("wall_seconds", "")
            render_seconds = manifest.get("phase_timings", {}).get("render", {}).get("wall_seconds", "")
            ply = run / "point_cloud/iteration_30000/point_cloud.ply"
            header = ply.read_bytes()[:8192].decode("ascii", errors="ignore") if ply.exists() else ""
            match = re.search(r"element vertex (\d+)", header)
            timing = read_json(run / "val/ours_30000/render_timing.json")
            loss_rows = read_csv(run / "loss_components.csv")
            final_loss = next((row for row in reversed(loss_rows) if row.get("iteration") == "30000"), {})
            rows.append({"experiment": name, "method": method, "train_seed": seed, "teacher_preparation_wall_once_s": cache_manifest.get("preparation_wall_seconds", "") if method == "E2+DS" and seed == SEEDS[0] else ("not_applicable" if method == "E2" else "amortized_once_only"), "teacher_preparation_per_view_s": cache_manifest.get("preparation_per_view_seconds", "") if method == "E2+DS" and seed == SEEDS[0] else ("not_applicable" if method == "E2" else "amortized_once_only"), "cache_write_and_check_s": "included_in_teacher_preparation" if method == "E2+DS" else "not_applicable", "run_total_train_wall_s": train_seconds, "checkpoint_elapsed_wall_s": "unavailable", "core_forward_backward_ms": final_loss.get("avg_ms", "unavailable"), "gaussian_count": match.group(1) if match else "", "model_bytes": ply.stat().st_size if ply.exists() else "", "cuda_allocated_peak_mb": final_loss.get("peak_cuda_mb", "unavailable"), "cuda_reserved_peak_mb": "unavailable", "independent_render_forward_ms": timing.get("forward_mean_ms", ""), "independent_render_wall_s": timing.get("wall_seconds", ""), "training_source_sha256": manifest.get("git", {}).get("training_source_sha256", ""), "config_sha256": manifest.get("config_sha256", "")})
    write_csv(output / "resource_summary.csv", list(rows[0]), rows)
    return rows


def build_root_package(repo: Path, output: Path) -> None:
    summary = output / "summary"
    seed_rows = read_csv(summary / "seed_summary.csv")
    cache_manifest_path = repo / "data/TI-NSD/heated/supervision/ds_bm3d_rgb_s003_r075_v1/supervision_manifest.v1.json"
    cache_manifest = read_json(cache_manifest_path)
    teacher_summary = save_teacher_diagnostics(repo, cache_manifest_path.parent, output, cache_manifest)
    save_fixed_validation(repo, output)
    costs = save_costs(repo, output, cache_manifest)
    write_json(output / "teacher_preparation_manifest.json", {"cache_manifest": str(cache_manifest_path.relative_to(repo)), "cache_manifest_sha256": sha256(cache_manifest_path), "cache_status": cache_manifest["cache_status"], "dependencies": cache_manifest.get("dependencies", {}), "teacher_profile": cache_manifest.get("teacher_profile_name", cache_manifest.get("teacher_profile")), "teacher_stages": cache_manifest.get("teacher_stages"), "teacher_sigma": cache_manifest.get("teacher_sigma"), "teacher_channel_policy": cache_manifest.get("teacher_channel_policy"), "teacher_range_policy": cache_manifest.get("teacher_range_policy"), "rho": cache_manifest.get("rho"), "dtype": cache_manifest.get("dtype"), "layout": cache_manifest.get("layout"), "data_range": cache_manifest.get("data_range"), "repeat_check_first_view_max_abs": cache_manifest.get("repeat_check_first_view_max_abs"), "views": len(cache_manifest.get("views", [])), "diagnostics": teacher_summary})
    resolved = {"runs_root": "runs/e2_ds_softtarget/v1", "results_root": "results/e2_ds_softtarget/v1", "experiments": [f"{prefix}_seed{seed}_30k" for seed in SEEDS for prefix in ("e2", "e2_ds")], "interrupted_archive": "runs/e2_ds_softtarget/v1/archive/e2_ds_seed2027_30k_interrupted_26800"}
    write_json(output / "resolved_runs.json", resolved)
    source = read_json(summary / "source_manifest.json")
    source.update({"cache_manifest_sha256": sha256(cache_manifest_path), "training_source_sha256": sorted({row["training_source_sha256"] for row in costs}), "test_accessed": False, "user_stop_then_manual_continuation": True})
    write_json(output / "source_manifest.json", source)
    (output / "protocol_snapshot.md").write_text("# Protocol snapshot\n\nE2 vs E2+DS only; three seeds (2026/2027/2028); 30k with 7k/15k/30k evaluation; 59 degraded train views and 34 validation views; test not accessed. E2+DS uses BM3D normal profile, ALL_STAGES, sigma=0.03, T=0.25Y+0.75Z, rho=0.75. E2 edge term remains on Y; target-dependent main terms use T.\n", encoding="utf-8")
    rows30 = [row for row in seed_rows if row["checkpoint"] == "30000"]
    report = ["# E2 vs E2+DS：BM3D 软目标 30k 正式报告", "", "## 完成状态", "", "六个 run、18 组独立 val 已完成；每组 30k，checkpoint 为 7k/15k/30k，验证集 34 views。测试集未访问。seed2027 的 E2+DS 曾按用户请求在 26,800 步停止，原目录保留在 `runs/e2_ds_softtarget/v1/archive/`，随后从共享初始化重新完成；没有自动重试或调参。", "", "## 30k 五指标", "", "| 指标 | E2 mean±std | E2+DS mean±std | delta (DS-E2) |", "|---|---:|---:|---:|"]
    for row in rows30:
        report.append(f"| {row['metric']} | {float(row['e2_mean']):.6f} ± {float(row['e2_std']):.6f} | {float(row['e2_ds_mean']):.6f} ± {float(row['e2_ds_std']):.6f} | {float(row['delta_mean']):+.6f} |")
    e2_train = sum(float(row["run_total_train_wall_s"]) for row in costs if row["method"] == "E2")
    ds_train = sum(float(row["run_total_train_wall_s"]) for row in costs if row["method"] == "E2+DS")
    teacher_wall = float(cache_manifest.get("preparation_wall_seconds", 0.0))
    report += ["", "## 7k→30k", "", "详见 `summary/late_degradation_summary.csv` 和 `summary/late_degradation_paired.csv`。E2+DS 的 PSNR/SSIM/T-MAE/E-MAE/gradient 退化分别按协议定义计算；绝对质量和配对差值需同时解读，退化较轻不单独等于成功。固定 101.jpg 的量化差值见 `summary/fixed_101_deltas.csv`。", "", "## 监督与审计", "", "教师只读取 59 张既有含噪训练图；固定缓存 manifest、BM3D 参数、目标数组哈希见 `teacher_preparation_manifest.json`。主 L1、SSIM、corner 项从 Y 切换到 T，E2 边缘项仍对 Y，不保留额外的完整含噪主损失。同 seed 的 Gaussian/ATF/TCM、优化器/RNG 和相机序列审计全部通过，配对记录见 `summary/paired_audit.csv`。", "", "## 成本", "", f"BM3D 一次准备墙钟 {teacher_wall:.3f} s，单 view {float(cache_manifest.get('preparation_per_view_seconds', 0.0)):.3f} s；三次 E2 训练合计 {e2_train:.3f} s，三次 E2+DS 训练合计 {ds_train:.3f} s，E2+DS 整批冷启动成本为 {teacher_wall + ds_train:.3f} s。每个 run 的完整训练墙钟、模型大小、高斯数量和 30k 独立渲染 forward 见 `resource_summary.csv`。checkpoint 墙钟、TCM 中间图和 CUDA allocated/reserved peak 未记录，明确标记 unavailable。", "", "## 限制", "", "单场景、固定 split/噪声和 COLMAP 先验；仅三个训练 seed；BM3D 伪目标不是真值；未分离教师、rho 和各主损失替换的贡献；无真实温度/几何真值；LPIPS/ROI 不可用；未使用 test。", ""]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    (output / "handoff.md").write_text("# E2+DS 结果交接\n\n状态：六个 30k run 与 18 组 val 已完成，测试集未访问。主报告：`report.md`；数值表：`summary/seed_summary.csv`、`summary/paired_deltas.csv`；教师诊断：`teacher_diagnostics/`；固定 101.jpg：`figures/fixed_validation/`；成本：`resource_summary.csv`；完整哈希：`results_manifest.json`。TensorBoard：`http://127.0.0.1:6009/`。\n\n停止/继续：当前没有 STOP_REQUESTED。重新启动前先确认是否需要新版本；不要覆盖现有 run。\n", encoding="utf-8")
    write_json(output / "status.json", {"status": "completed", "experiments": 6, "validation_groups": 18, "validation_views_per_group": 34, "test_accessed": False, "user_interruption": "e2_ds_seed2027 at 26800; archived and completed from shared initialization", "updated_at_utc": datetime.now(timezone.utc).isoformat()})
    files = sorted(path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file() and path.name != "results_manifest.json")
    write_json(output / "results_manifest.json", {"schema_version": 1, "status": "completed", "files": [{"path": item, "sha256": sha256(output / item)} for item in files]})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("results/e2_ds_softtarget/v1"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    build_root_package(repo, (repo / args.output).resolve())
    print((repo / args.output).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
