"""Audit three historical validation views without changing training or metrics."""

import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import torch
import torchvision
from argparse import ArgumentParser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import ATFModel, GaussianModel, Scene, TCMModel
from utils.general_utils import safe_state


VIEW_NAMES = ("092.jpg", "101.jpg", "110.jpg")


def _stats(tensor):
    values = tensor.detach().float()
    return {
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "nonfinite_count": int((~torch.isfinite(values)).sum().item()),
        "below_zero_fraction": float((values < 0).float().mean().item()),
        "above_one_fraction": float((values > 1).float().mean().item()),
    }


def _load_run(run_path, iteration):
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    sys.argv = ["audit_legacy_failure", "-m", str(run_path)]
    args = get_combined_args(parser)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    atf = ATFModel(dataset.is_blender)
    atf.load_weights(dataset.model_path, scene.loaded_iter)
    tcm = TCMModel()
    tcm.load_weights(dataset.model_path, scene.loaded_iter)
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )
    return dataset, pipe, scene, atf, tcm, background


def audit_method(label, run_path, output_root, iteration):
    dataset, pipe, scene, atf, tcm, background = _load_run(run_path, iteration)
    cameras = {view.image_name: view for view in scene.getTestCameras()}
    missing = [name for name in VIEW_NAMES if name not in cameras]
    if missing:
        raise RuntimeError("missing historical validation views: {}".format(missing))
    method_root = Path(output_root) / label
    method_root.mkdir(parents=True, exist_ok=True)
    records = []
    with torch.no_grad():
        for view_name in VIEW_NAMES:
            view = cameras[view_name]
            if dataset.load2gpu_on_the_fly:
                view.load2device()
            xyz = scene.gaussians.get_xyz
            time_input = view.fid.unsqueeze(0).expand(xyz.shape[0], -1)
            abs_value, scale_value, direction = atf.step(xyz.detach(), time_input)
            d_rgb = torch.exp((abs_value + scale_value) * direction)
            rendered = render(view, scene.gaussians, pipe, background, d_rgb)["render"]
            residual = tcm.step(rendered)
            final_float = rendered + residual
            final_png_path = method_root / (view_name.replace(".jpg", "") + "_final.png")
            torchvision.utils.save_image(final_float, str(final_png_path))
            png = torch.from_numpy(imageio.imread(final_png_path)).float() / 255.0
            records.append(
                {
                    "view": view_name,
                    "fid": float(view.fid.item()),
                    "raw_gaussian": _stats(rendered),
                    "tcm_residual": _stats(residual),
                    "final_float": _stats(final_float),
                    "independent_png": _stats(png),
                    "final_png": str(final_png_path),
                }
            )
            if dataset.load2gpu_on_the_fly:
                view.load2device("cpu")
    return {
        "label": label,
        "run_path": str(Path(run_path).resolve()),
        "iteration": iteration,
        "views": records,
        "gaussian_count": int(scene.gaussians.get_xyz.shape[0]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2", required=True, type=Path)
    parser.add_argument("--detail", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iteration", type=int, default=7000)
    args = parser.parse_args()
    safe_state(True, seed=0)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "views": list(VIEW_NAMES),
        "methods": [
            audit_method("e2", args.e2, args.output, args.iteration),
            audit_method("detail", args.detail, args.output, args.iteration),
        ],
        "scope": "historical 00010/00011/00012 only; no training or metric selection",
    }
    with (args.output / "audit_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
