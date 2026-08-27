"""Deterministic image metrics for thermal-infrared experiments.

Metric inputs use one fixed intensity convention: encoded 8-bit images are
divided by 255 and encoded 16-bit images by 65535, producing values in [0, 1].
Floating-point images must already be in [0, 1].  No image-wise min/max
normalization is performed.  RGB inputs are converted to thermal intensity
with BT.601 luma weights (0.299, 0.587, 0.114).  All metrics operate on that
single channel; only LPIPS replicates it three times, without another range
conversion, to satisfy the repository LPIPS network's RGB input contract.

The E-MAE edge support follows ``losses.thermal_physics_loss``: Gaussian
smoothing (5x5, sigma 1.0), replicate padding, and /8 Sobel kernels.  Its
component error is weighted by the smoothed GT Sobel magnitude and divided by
the sum of those weights rather than by the image area.  Edge weights are not
min/max stretched per image.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

try:
    from utils.loss_utils import ssim as _official_ssim
except Exception:  # Installed src package may not include the upstream helper.
    _official_ssim = None


LOSSLESS_IMAGE_SUFFIXES = frozenset(
    (".png", ".tif", ".tiff", ".bmp", ".pgm", ".ppm", ".pbm", ".pnm")
)
LOSSY_IMAGE_SUFFIXES = frozenset((".jpg", ".jpeg", ".jpe", ".webp"))
METRIC_KEYS = (
    "psnr",
    "ssim",
    "lpips",
    "t_mae",
    "e_mae",
    "gradient_preservation",
    "roi_mae",
)
SUMMARY_FIELDS = ("experiment", "num_views") + METRIC_KEYS
PER_VIEW_FIELDS = ("experiment", "image") + METRIC_KEYS


def _as_bchw(image: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError("{} must be a torch.Tensor".format(name))
    if image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.ndim == 3:
        image = image.unsqueeze(0)
    elif image.ndim != 4:
        raise ValueError("{} must have shape HxW, CxHxW, or BxCxHxW".format(name))
    if image.shape[0] < 1 or image.shape[2] < 1 or image.shape[3] < 1:
        raise ValueError("{} must have non-empty dimensions".format(name))
    if image.shape[1] not in (1, 3, 4):
        raise ValueError("{} must contain 1, 3, or 4 channels".format(name))
    if not image.is_floating_point():
        raise TypeError("{} must be floating point and normalized to [0, 1]".format(name))
    if not bool(torch.isfinite(image).all().item()):
        raise ValueError("{} contains a non-finite value".format(name))
    minimum = float(image.detach().amin().item())
    maximum = float(image.detach().amax().item())
    if minimum < -1e-6 or maximum > 1.0 + 1e-6:
        raise ValueError(
            "{} must use the fixed [0, 1] data range; got [{:.8g}, {:.8g}]".format(
                name, minimum, maximum
            )
        )
    return image


def to_grayscale(image: torch.Tensor, name: str = "image") -> torch.Tensor:
    """Return a BCHW BT.601 intensity tensor in the fixed [0, 1] range."""

    image = _as_bchw(image, name)
    if image.shape[1] == 1:
        return image
    rgb = image[:, :3]
    weights = rgb.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
    return (rgb * weights).sum(dim=1, keepdim=True)


def _prepare_pair(
    prediction: torch.Tensor, ground_truth: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    prediction = to_grayscale(prediction, "prediction")
    ground_truth = to_grayscale(ground_truth, "ground_truth")
    if prediction.shape != ground_truth.shape:
        raise ValueError("prediction and ground_truth must have matching shapes")
    if prediction.device != ground_truth.device:
        raise ValueError("prediction and ground_truth must be on the same device")
    if prediction.dtype != ground_truth.dtype:
        ground_truth = ground_truth.to(dtype=prediction.dtype)
    return prediction, ground_truth


def _prepare_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    mask = to_grayscale(mask, "roi_mask")
    if mask.shape[0] == 1 and reference.shape[0] != 1:
        mask = mask.expand(reference.shape[0], -1, -1, -1)
    if mask.shape != reference.shape:
        raise ValueError("roi_mask must match the image batch and spatial dimensions")
    if mask.device != reference.device:
        mask = mask.to(reference.device)
    return mask.to(dtype=reference.dtype)


def psnr(
    prediction: torch.Tensor, ground_truth: torch.Tensor, eps: float = 0.0
) -> float:
    """Peak signal-to-noise ratio with a fixed data range of exactly 1."""

    if eps < 0:
        raise ValueError("eps must be non-negative")
    prediction, ground_truth = _prepare_pair(prediction, ground_truth)
    mse = float((prediction - ground_truth).square().mean().item())
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(1.0 / max(mse, eps))


def _fallback_ssim(
    prediction: torch.Tensor, ground_truth: torch.Tensor, window_size: int = 11
) -> torch.Tensor:
    coordinates = torch.arange(
        window_size, dtype=prediction.dtype, device=prediction.device
    )
    coordinates = coordinates - window_size // 2
    kernel_1d = torch.exp(-(coordinates.square()) / (2.0 * 1.5 * 1.5))
    kernel_1d = kernel_1d / kernel_1d.sum()
    window = (kernel_1d[:, None] * kernel_1d[None, :]).view(
        1, 1, window_size, window_size
    )
    mu_pred = F.conv2d(prediction, window, padding=window_size // 2)
    mu_gt = F.conv2d(ground_truth, window, padding=window_size // 2)
    mu_pred_sq = mu_pred.square()
    mu_gt_sq = mu_gt.square()
    mu_product = mu_pred * mu_gt
    variance_pred = (
        F.conv2d(prediction.square(), window, padding=window_size // 2)
        - mu_pred_sq
    )
    variance_gt = (
        F.conv2d(ground_truth.square(), window, padding=window_size // 2)
        - mu_gt_sq
    )
    covariance = (
        F.conv2d(prediction * ground_truth, window, padding=window_size // 2)
        - mu_product
    )
    numerator = (2.0 * mu_product + 0.01 ** 2) * (
        2.0 * covariance + 0.03 ** 2
    )
    denominator = (mu_pred_sq + mu_gt_sq + 0.01 ** 2) * (
        variance_pred + variance_gt + 0.03 ** 2
    )
    return (numerator / denominator).mean()


def ssim(prediction: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """SSIM using the upstream 3D-GS implementation when it is importable."""

    prediction, ground_truth = _prepare_pair(prediction, ground_truth)
    if _official_ssim is not None:
        value = _official_ssim(prediction, ground_truth)
    else:
        value = _fallback_ssim(prediction, ground_truth)
    return float(value.item())


def thermal_mae(prediction: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """Mean absolute error of normalized thermal intensity (T-MAE)."""

    prediction, ground_truth = _prepare_pair(prediction, ground_truth)
    return float((prediction - ground_truth).abs().mean().item())


def _gaussian_smooth(
    image: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0
) -> torch.Tensor:
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    radius = kernel_size // 2
    coordinates = torch.arange(
        kernel_size, dtype=image.dtype, device=image.device
    ) - radius
    kernel_1d = torch.exp(-coordinates.square() / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = (kernel_1d[:, None] * kernel_1d[None, :]).view(
        1, 1, kernel_size, kernel_size
    )
    padded = F.pad(image, (radius, radius, radius, radius), mode="replicate")
    return F.conv2d(padded, kernel)


def _sobel_components(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    sobel_x = image.new_tensor(
        ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))
    ) / 8.0
    sobel_y = sobel_x.transpose(0, 1)
    padded = F.pad(image, (1, 1, 1, 1), mode="replicate")
    return (
        F.conv2d(padded, sobel_x.view(1, 1, 3, 3)),
        F.conv2d(padded, sobel_y.view(1, 1, 3, 3)),
    )


def _gt_edge_weights(ground_truth: torch.Tensor) -> torch.Tensor:
    smooth_gt = _gaussian_smooth(ground_truth)
    grad_x, grad_y = _sobel_components(smooth_gt)
    return torch.sqrt(grad_x.square() + grad_y.square())


def edge_mae(
    prediction: torch.Tensor, ground_truth: torch.Tensor, eps: float = 1e-6
) -> float:
    """GT-edge-weighted Sobel component MAE, normalized by weight support."""

    if eps <= 0:
        raise ValueError("eps must be positive")
    prediction, ground_truth = _prepare_pair(prediction, ground_truth)
    weights = _gt_edge_weights(ground_truth)
    pred_x, pred_y = _sobel_components(prediction)
    gt_x, gt_y = _sobel_components(ground_truth)
    component_error = 0.5 * ((pred_x - gt_x).abs() + (pred_y - gt_y).abs())
    value = (weights * component_error).sum() / weights.sum().clamp_min(eps)
    return float(value.item())


def gradient_preservation_score(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    stability_constant: float = 1e-6,
) -> float:
    """Mean gradient-magnitude similarity in [0, 1] using the project Sobel."""

    if stability_constant <= 0:
        raise ValueError("stability_constant must be positive")
    prediction, ground_truth = _prepare_pair(prediction, ground_truth)
    pred_x, pred_y = _sobel_components(prediction)
    gt_x, gt_y = _sobel_components(ground_truth)
    pred_magnitude = torch.sqrt(pred_x.square() + pred_y.square())
    gt_magnitude = torch.sqrt(gt_x.square() + gt_y.square())
    similarity = (
        2.0 * pred_magnitude * gt_magnitude + stability_constant
    ) / (
        pred_magnitude.square()
        + gt_magnitude.square()
        + stability_constant
    )
    return float(similarity.mean().item())


def roi_mae(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    roi_mask: Optional[torch.Tensor],
    eps: float = 1e-12,
) -> Optional[float]:
    """Soft-mask normalized ROI MAE; return ``None`` for absent/empty ROI."""

    if roi_mask is None:
        return None
    if eps <= 0:
        raise ValueError("eps must be positive")
    prediction, ground_truth = _prepare_pair(prediction, ground_truth)
    mask = _prepare_mask(roi_mask, prediction)
    support = float(mask.sum().item())
    if support <= eps:
        return None
    value = (mask * (prediction - ground_truth).abs()).sum() / mask.sum()
    return float(value.item())


def lpips_score(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    lpips_model: Optional[torch.nn.Module],
) -> Optional[float]:
    """LPIPS on three replicated [0, 1] intensity channels."""

    if lpips_model is None:
        return None
    prediction, ground_truth = _prepare_pair(prediction, ground_truth)
    prediction_rgb = prediction.repeat(1, 3, 1, 1)
    ground_truth_rgb = ground_truth.repeat(1, 3, 1, 1)
    with torch.no_grad():
        value = lpips_model(prediction_rgb, ground_truth_rgb)
    return float(value.detach().mean().item())


def evaluate_pair(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    roi_mask: Optional[torch.Tensor] = None,
    lpips_model: Optional[torch.nn.Module] = None,
) -> Dict[str, Optional[float]]:
    """Compute all per-view metrics using one shared preprocessing contract."""

    return {
        "psnr": psnr(prediction, ground_truth),
        "ssim": ssim(prediction, ground_truth),
        "lpips": lpips_score(prediction, ground_truth, lpips_model),
        "t_mae": thermal_mae(prediction, ground_truth),
        "e_mae": edge_mae(prediction, ground_truth),
        "gradient_preservation": gradient_preservation_score(
            prediction, ground_truth
        ),
        "roi_mae": roi_mae(prediction, ground_truth, roi_mask),
    }


def load_image(path: Path) -> torch.Tensor:
    """Load a supported lossless image as CHW float32 with a fixed encoding range."""

    path = Path(path)
    if path.suffix.lower() not in LOSSLESS_IMAGE_SUFFIXES:
        raise ValueError("unsupported or lossy image extension: {}".format(path.suffix))
    with Image.open(str(path)) as image:
        image.load()
        mode = image.mode
        if mode in ("1", "L"):
            array = np.array(image.convert("L"), dtype=np.float32, copy=True)
            scale = 255.0
            array = array[:, :, None]
        elif mode == "LA":
            array = np.array(image.getchannel("L"), dtype=np.float32, copy=True)
            scale = 255.0
            array = array[:, :, None]
        elif mode.startswith("I;16") or mode == "I":
            array = np.array(image, dtype=np.float32, copy=True)
            scale = 65535.0
            array = array[:, :, None]
        elif mode == "F":
            array = np.array(image, dtype=np.float32, copy=True)
            scale = 1.0
            array = array[:, :, None]
        else:
            array = np.array(image.convert("RGB"), dtype=np.float32, copy=True)
            scale = 255.0
    if not bool(np.isfinite(array).all()):
        raise ValueError("image contains non-finite values: {}".format(path))
    minimum = float(array.min())
    maximum = float(array.max())
    if minimum < 0.0 or maximum > scale:
        raise ValueError(
            "{} mode {} must stay inside fixed [0, {}] encoding range".format(
                path, mode, int(scale)
            )
        )
    normalized = np.ascontiguousarray(array / scale, dtype=np.float32)
    return torch.from_numpy(normalized).permute(2, 0, 1).contiguous()


def _list_lossless_images(directory: Path) -> Dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError("image directory does not exist: {}".format(directory))
    result: Dict[str, Path] = {}
    lossy: List[str] = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in LOSSY_IMAGE_SUFFIXES:
            lossy.append(path.name)
        elif suffix in LOSSLESS_IMAGE_SUFFIXES:
            result[path.name] = path
    if lossy:
        raise ValueError(
            "lossy images are not accepted in {}: {}".format(
                directory, ", ".join(sorted(lossy))
            )
        )
    if not result:
        raise ValueError("no supported lossless images found in {}".format(directory))
    return result


def pair_experiment_images(
    method_dir: Path, roi_mask_dir: Optional[Path] = None
) -> List[Tuple[str, Path, Path, Optional[Path]]]:
    """Strictly pair sorted, exact filenames from ``renders`` and ``gt``."""

    method_dir = Path(method_dir)
    renders = _list_lossless_images(method_dir / "renders")
    ground_truth = _list_lossless_images(method_dir / "gt")
    render_names = set(renders)
    gt_names = set(ground_truth)
    if render_names != gt_names:
        missing_renders = sorted(gt_names - render_names)
        missing_gt = sorted(render_names - gt_names)
        details: List[str] = []
        if missing_renders:
            details.append("missing renders: {}".format(", ".join(missing_renders)))
        if missing_gt:
            details.append("missing gt: {}".format(", ".join(missing_gt)))
        raise ValueError("render/gt filename mismatch ({})".format("; ".join(details)))

    masks: Optional[Dict[str, Path]] = None
    if roi_mask_dir is not None:
        masks = _list_lossless_images(Path(roi_mask_dir))
        mask_names = set(masks)
        if mask_names != render_names:
            missing_masks = sorted(render_names - mask_names)
            extra_masks = sorted(mask_names - render_names)
            details = []
            if missing_masks:
                details.append("missing masks: {}".format(", ".join(missing_masks)))
            if extra_masks:
                details.append("extra masks: {}".format(", ".join(extra_masks)))
            raise ValueError("ROI filename mismatch ({})".format("; ".join(details)))

    pairs: List[Tuple[str, Path, Path, Optional[Path]]] = []
    for name in sorted(render_names):
        mask_path = None if masks is None else masks[name]
        pairs.append((name, renders[name], ground_truth[name], mask_path))
    return pairs


def create_lpips_model(
    device: torch.device, net_type: str = "vgg"
) -> torch.nn.Module:
    """Construct the repository-local LPIPS implementation once per run."""

    try:
        from lpipsPyTorch.modules.lpips import LPIPS

        model = LPIPS(net_type=net_type, version="0.1")
        model = model.to(device)
        model.eval()
        return model
    except Exception as exc:
        raise RuntimeError(
            "unable to initialize repository lpipsPyTorch; install/cache its "
            "torchvision weights or rerun with --skip-lpips: {}".format(exc)
        )


def evaluate_experiment(
    experiment: str,
    method_dir: Path,
    device: torch.device,
    lpips_model: Optional[torch.nn.Module] = None,
    roi_mask_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate one ``renders``/``gt`` method directory."""

    pairs = pair_experiment_images(method_dir, roi_mask_dir)
    rows: List[Dict[str, Any]] = []
    for image_name, render_path, gt_path, mask_path in pairs:
        prediction = load_image(render_path).to(device)
        ground_truth = load_image(gt_path).to(device)
        mask = None if mask_path is None else load_image(mask_path).to(device)
        values = evaluate_pair(prediction, ground_truth, mask, lpips_model)
        row: Dict[str, Any] = {"experiment": experiment, "image": image_name}
        row.update(values)
        rows.append(row)

    summary: Dict[str, Any] = {
        "experiment": experiment,
        "num_views": len(rows),
    }
    for key in METRIC_KEYS:
        available = [row[key] for row in rows if row[key] is not None]
        summary[key] = (
            None
            if not available
            else float(sum(float(value) for value in available) / len(available))
        )
    return summary, rows


def _format_metric(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, (str, Path)):
        return str(value)
    if isinstance(value, int):
        return str(value)
    numeric = float(value)
    if math.isnan(numeric):
        return "unavailable"
    if math.isinf(numeric):
        return "inf" if numeric > 0 else "-inf"
    return "{:.8f}".format(numeric)


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format_metric(row.get(field)) for field in fields})


def _markdown_cell(value: Any) -> str:
    return _format_metric(value).replace("|", "\\|")


def write_metric_outputs(
    output_dir: Path,
    summaries: Sequence[Mapping[str, Any]],
    per_view_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Path]:
    """Write deterministic CSV and Markdown result artifacts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "metrics_summary.csv"
    summary_md = output_dir / "metrics_summary.md"
    per_view_csv = output_dir / "metrics_per_view.csv"
    _write_csv(summary_csv, SUMMARY_FIELDS, summaries)
    _write_csv(per_view_csv, PER_VIEW_FIELDS, per_view_rows)

    labels = (
        "Experiment",
        "Views",
        "PSNR",
        "SSIM",
        "LPIPS",
        "T-MAE",
        "E-MAE",
        "Gradient preservation",
        "ROI-MAE",
    )
    with summary_md.open("w", encoding="utf-8", newline="\n") as output:
        output.write("# Thermal IR metric summary\n\n")
        output.write(
            "All source encodings use a fixed mapping to [0,1] (uint8/255, "
            "uint16/65535); no per-image min-max normalization is used for "
            "intensities or edge weights. RGB is converted with BT.601 luma. "
            "LPIPS alone receives three replicated luma channels. E-MAE is "
            "normalized by GT edge-weight support. "
            "`unavailable` denotes skipped LPIPS or an absent/empty ROI.\n\n"
        )
        output.write("| " + " | ".join(labels) + " |\n")
        output.write("| " + " | ".join("---" for _ in labels) + " |\n")
        for row in summaries:
            values = [row.get(field) for field in SUMMARY_FIELDS]
            output.write(
                "| " + " | ".join(_markdown_cell(value) for value in values) + " |\n"
            )
    return {
        "summary_csv": summary_csv,
        "summary_md": summary_md,
        "per_view_csv": per_view_csv,
    }


def collect_metrics(
    experiments: Sequence[Tuple[str, Path]],
    output_dir: Path = Path("results"),
    roi_mask_dirs: Optional[Mapping[str, Path]] = None,
    skip_lpips: bool = False,
    lpips_net: str = "vgg",
    device: Optional[torch.device] = None,
) -> Dict[str, Path]:
    """Evaluate named experiments and write summary plus per-view tables."""

    if not experiments:
        raise ValueError("at least one experiment is required")
    names = [name for name, _ in experiments]
    if any(not name for name in names):
        raise ValueError("experiment names must not be empty")
    if len(names) != len(set(names)):
        raise ValueError("experiment names must be unique")
    if roi_mask_dirs is None:
        roi_mask_dirs = {}
    unknown_roi = sorted(set(roi_mask_dirs) - set(names))
    if unknown_roi:
        raise ValueError(
            "ROI masks reference unknown experiments: {}".format(", ".join(unknown_roi))
        )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = None if skip_lpips else create_lpips_model(device, lpips_net)

    summaries: List[Mapping[str, Any]] = []
    per_view_rows: List[Mapping[str, Any]] = []
    for name, method_dir in sorted(experiments, key=lambda item: item[0]):
        summary, rows = evaluate_experiment(
            name,
            Path(method_dir),
            device,
            lpips_model=lpips_model,
            roi_mask_dir=roi_mask_dirs.get(name),
        )
        summaries.append(summary)
        per_view_rows.extend(rows)
    return write_metric_outputs(output_dir, summaries, per_view_rows)
