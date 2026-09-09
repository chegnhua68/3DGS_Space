"""Noise-aware, edge-preserving losses for normalized thermal images.

Inputs are expected to use a fixed, dataset-level intensity mapping. The module
does not perform image-wise intensity normalization; only the derived edge and
noise proxy maps are normalized independently per image.
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _require_nonnegative_finite(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("{} must be a real scalar".format(name)) from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError("{} must be finite and non-negative".format(name))
    return result


def _as_batched_image(image: torch.Tensor, name: str) -> Tuple[torch.Tensor, bool]:
    if not isinstance(image, torch.Tensor):
        raise TypeError("{} must be a torch.Tensor".format(name))
    if not image.is_floating_point():
        raise TypeError("{} must have a floating-point dtype".format(name))
    if image.ndim == 3:
        image = image.unsqueeze(0)
        was_unbatched = True
    elif image.ndim == 4:
        was_unbatched = False
    else:
        raise ValueError("{} must have shape (C,H,W) or (B,C,H,W)".format(name))
    if image.shape[1] < 1 or image.shape[2] < 1 or image.shape[3] < 1:
        raise ValueError("{} must have non-empty channel and spatial dimensions".format(name))
    return image, was_unbatched


def _restore_batch(image: torch.Tensor, was_unbatched: bool) -> torch.Tensor:
    return image.squeeze(0) if was_unbatched else image


def _validate_odd_kernel(kernel_size: int) -> None:
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")


def _gaussian_smooth(
    image: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0
) -> torch.Tensor:
    _validate_odd_kernel(kernel_size)
    sigma = _require_nonnegative_finite(sigma, "sigma")
    if sigma == 0:
        raise ValueError("sigma must be positive")

    radius = kernel_size // 2
    coordinates = torch.arange(
        kernel_size, dtype=image.dtype, device=image.device
    ) - radius
    kernel_1d = torch.exp(-(coordinates * coordinates) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel = kernel_2d.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.expand(image.shape[1], 1, kernel_size, kernel_size)
    padded = F.pad(image, (radius, radius, radius, radius), mode="replicate")
    return F.conv2d(padded, kernel, groups=image.shape[1])


def _sobel_components(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    dtype = image.dtype
    device = image.device
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    ) / 8.0
    sobel_y = sobel_x.transpose(0, 1)
    channels = image.shape[1]
    kernel_x = sobel_x.view(1, 1, 3, 3).expand(channels, 1, 3, 3)
    kernel_y = sobel_y.view(1, 1, 3, 3).expand(channels, 1, 3, 3)
    padded = F.pad(image, (1, 1, 1, 1), mode="replicate")
    return (
        F.conv2d(padded, kernel_x, groups=channels),
        F.conv2d(padded, kernel_y, groups=channels),
    )


def _filtered_luminance(image: torch.Tensor, name: str) -> torch.Tensor:
    image_4d, _ = _as_batched_image(image, name)
    channels = image_4d.shape[1]
    if channels == 1:
        return image_4d
    if channels != 3:
        raise ValueError("{} must have exactly 1 or 3 channels".format(name))
    weights = image_4d.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
    return (image_4d * weights).sum(dim=1, keepdim=True)


def _require_reflect_padding(
    image: torch.Tensor, padding: int, operator_name: str
) -> None:
    if image.shape[-2] <= padding or image.shape[-1] <= padding:
        raise ValueError(
            "{} requires height and width greater than reflect padding {}".format(
                operator_name, padding
            )
        )


def _validate_edge_filter_mode(edge_filter_mode: str) -> str:
    if edge_filter_mode not in ("gaussian", "identity"):
        raise ValueError("edge_filter_mode must be 'gaussian' or 'identity'")
    return edge_filter_mode


def _filtered_gaussian_smooth(
    image: torch.Tensor, kernel_size: int, sigma: float
) -> torch.Tensor:
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, int):
        raise TypeError("kernel_size must be an integer")
    _validate_odd_kernel(kernel_size)
    sigma = _require_nonnegative_finite(sigma, "sigma")
    if sigma == 0:
        raise ValueError("sigma must be positive")
    radius = kernel_size // 2
    _require_reflect_padding(image, radius, "Gaussian filter")
    coordinates = torch.arange(
        kernel_size, dtype=image.dtype, device=image.device
    ) - radius
    squared_radius = (
        coordinates[:, None].square() + coordinates[None, :].square()
    )
    kernel = torch.exp(-squared_radius / (2.0 * sigma * sigma))
    kernel = (kernel / kernel.sum()).view(1, 1, kernel_size, kernel_size)
    padded = F.pad(image, (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(padded, kernel)


def _filtered_sobel_components(
    image: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _require_reflect_padding(image, 1, "Sobel filter")
    sobel_x = image.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ) / 8.0
    sobel_y = sobel_x.transpose(0, 1)
    padded = F.pad(image, (1, 1, 1, 1), mode="reflect")
    return (
        F.conv2d(padded, sobel_x.view(1, 1, 3, 3)),
        F.conv2d(padded, sobel_y.view(1, 1, 3, 3)),
    )


def filtered_edge_consistency_loss(
    prediction: torch.Tensor,
    observation: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 1.0,
    edge_filter_mode: str = "gaussian",
) -> torch.Tensor:
    """Match signed Sobel gradients after an identical fixed filter.

    Inputs must have shape ``(C,H,W)`` or ``(B,C,H,W)`` with one or three
    channels. RGB inputs use fixed BT.601 luminance. The observation branch is
    detached while the complete prediction branch remains differentiable.
    ``edge_filter_mode='identity'`` skips only Gaussian filtering; both modes
    share the same luminance and signed Sobel implementation.
    """

    edge_filter_mode = _validate_edge_filter_mode(edge_filter_mode)
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, int):
        raise TypeError("kernel_size must be an integer")
    _validate_odd_kernel(kernel_size)
    sigma = _require_nonnegative_finite(sigma, "sigma")
    if sigma == 0:
        raise ValueError("sigma must be positive")
    prediction_4d, _ = _as_batched_image(prediction, "prediction")
    observation_4d, _ = _as_batched_image(observation, "observation")
    if prediction_4d.shape != observation_4d.shape:
        raise ValueError("prediction and observation must have the same shape")
    if prediction_4d.device != observation_4d.device:
        raise ValueError("prediction and observation must be on the same device")
    if prediction_4d.dtype != observation_4d.dtype:
        raise ValueError("prediction and observation must have the same dtype")

    prediction_luma = _filtered_luminance(prediction_4d, "prediction")
    observation_luma = _filtered_luminance(
        observation_4d, "observation"
    ).detach()
    if edge_filter_mode == "gaussian":
        prediction_filtered = _filtered_gaussian_smooth(
            prediction_luma, kernel_size, sigma
        )
        observation_filtered = _filtered_gaussian_smooth(
            observation_luma, kernel_size, sigma
        )
    else:
        prediction_filtered = prediction_luma
        observation_filtered = observation_luma
    pred_x, pred_y = _filtered_sobel_components(prediction_filtered)
    obs_x, obs_y = _filtered_sobel_components(observation_filtered)
    return 0.5 * ((pred_x - obs_x).abs().mean() + (pred_y - obs_y).abs().mean())


def _normalize_per_image(value: torch.Tensor, eps: float) -> torch.Tensor:
    eps = _require_nonnegative_finite(eps, "eps")
    if eps == 0:
        raise ValueError("eps must be positive")
    flat = value.flatten(start_dim=1)
    minimum = flat.min(dim=1).values.view(-1, 1, 1, 1)
    maximum = flat.max(dim=1).values.view(-1, 1, 1, 1)
    span = maximum - minimum
    normalized = (value - minimum) / span.clamp_min(eps)
    return torch.where(span > eps, normalized, torch.zeros_like(value))


def compute_edge_map(
    image: torch.Tensor,
    gaussian_kernel_size: int = 5,
    gaussian_sigma: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return a per-image normalized edge map with shape ``(B,1,H,W)``.

    A ``(C,H,W)`` input produces ``(1,H,W)``. Constant images produce an
    all-zero map instead of NaNs.
    """

    image_4d, was_unbatched = _as_batched_image(image, "image")
    smoothed = _gaussian_smooth(image_4d, gaussian_kernel_size, gaussian_sigma)
    grad_x, grad_y = _sobel_components(smoothed)
    magnitude = torch.sqrt(grad_x.square() + grad_y.square() + eps * eps)
    magnitude = magnitude.mean(dim=1, keepdim=True)
    edge_map = _normalize_per_image(magnitude, eps)
    return _restore_batch(edge_map, was_unbatched)


def compute_noise_reliability_map(
    image: torch.Tensor,
    beta: float = 5.0,
    gaussian_kernel_size: int = 5,
    gaussian_sigma: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Estimate reliability from high-frequency residual magnitude.

    This is an image-derived heuristic, not a calibrated sensor noise model.
    Values are in ``[exp(-beta), 1]`` for non-negative ``beta``.
    """

    beta = _require_nonnegative_finite(beta, "beta")
    image_4d, was_unbatched = _as_batched_image(image, "image")
    smoothed = _gaussian_smooth(image_4d, gaussian_kernel_size, gaussian_sigma)
    residual = (image_4d - smoothed).abs().mean(dim=1, keepdim=True)
    noise_proxy = _normalize_per_image(residual, eps)
    reliability = torch.exp(-beta * noise_proxy)
    return _restore_batch(reliability, was_unbatched)


def _prepare_pair(pred: torch.Tensor, gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pred_4d, _ = _as_batched_image(pred, "pred")
    gt_4d, _ = _as_batched_image(gt, "gt")
    if pred_4d.shape != gt_4d.shape:
        raise ValueError("pred and gt must have the same shape")
    if pred_4d.device != gt_4d.device:
        raise ValueError("pred and gt must be on the same device")
    return pred_4d, gt_4d


def _prepare_map(
    value: torch.Tensor, reference: torch.Tensor, name: str
) -> torch.Tensor:
    value_4d, _ = _as_batched_image(value, name)
    if value_4d.shape[0] not in (1, reference.shape[0]):
        raise ValueError("{} batch dimension is not broadcastable".format(name))
    if value_4d.shape[1] not in (1, reference.shape[1]):
        raise ValueError("{} channel dimension is not broadcastable".format(name))
    if value_4d.shape[2:] != reference.shape[2:]:
        raise ValueError("{} spatial dimensions must match pred and gt".format(name))
    if value_4d.device != reference.device:
        raise ValueError("{} must be on the same device as pred and gt".format(name))
    return value_4d


def thermal_reconstruction_loss(
    pred: torch.Tensor, gt: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Return ``mean(weight * abs(pred - gt))`` from the project formula."""

    pred_4d, gt_4d = _prepare_pair(pred, gt)
    weight_4d = _prepare_map(weight, pred_4d, "weight")
    if not bool(torch.isfinite(weight_4d).all().item()):
        raise ValueError("weight must contain only finite values")
    if torch.any(weight_4d < 0):
        raise ValueError("weight must be non-negative")
    expanded_weight = weight_4d.expand_as(pred_4d)
    return (expanded_weight * (pred_4d - gt_4d).abs()).mean()


def edge_preserving_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    edge_map: torch.Tensor,
    gamma: float = 3.0,
) -> torch.Tensor:
    """Match Sobel components while emphasizing reliable target boundaries."""

    gamma = _require_nonnegative_finite(gamma, "gamma")
    pred_4d, gt_4d = _prepare_pair(pred, gt)
    edge_4d = _prepare_map(edge_map, pred_4d, "edge_map")
    pred_x, pred_y = _sobel_components(pred_4d)
    gt_x, gt_y = _sobel_components(gt_4d)
    component_error = 0.5 * ((pred_x - gt_x).abs() + (pred_y - gt_y).abs())
    component_error = component_error.mean(dim=1, keepdim=True)
    return ((1.0 + gamma * edge_4d) * component_error).mean()


def non_edge_smoothness_loss(
    pred: torch.Tensor, edge_map: torch.Tensor
) -> torch.Tensor:
    """Penalize predicted gradients away from target thermal boundaries."""

    pred_4d, _ = _as_batched_image(pred, "pred")
    edge_4d = _prepare_map(edge_map, pred_4d, "edge_map")
    pred_x, pred_y = _sobel_components(pred_4d)
    gradient = 0.5 * (pred_x.abs() + pred_y.abs())
    gradient = gradient.mean(dim=1, keepdim=True)
    non_edge_weight = (1.0 - edge_4d).clamp(0.0, 1.0)
    return (non_edge_weight * gradient).mean()


def thermal_physics_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    lambda_thermal: float = 0.0,
    lambda_edge: float = 0.0,
    lambda_smooth: float = 0.0,
    noise_beta: float = 5.0,
    edge_gamma: float = 3.0,
    aux_loss_version: str = "legacy",
    edge_filter_kernel: int = 5,
    edge_filter_sigma: float = 1.0,
    edge_filter_mode: str = "gaussian",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compose enabled loss terms and return ``(weighted_total, terms)``.

    When all lambdas are zero, this function returns scalar zeros without
    constructing Gaussian or Sobel kernels. The training loop also guards the
    call so the new-loss path remains untouched by default.
    """

    lambdas = tuple(
        _require_nonnegative_finite(value, name)
        for value, name in zip(
            (lambda_thermal, lambda_edge, lambda_smooth),
            ("lambda_thermal", "lambda_edge", "lambda_smooth"),
        )
    )
    lambda_thermal, lambda_edge, lambda_smooth = lambdas
    noise_beta = _require_nonnegative_finite(noise_beta, "noise_beta")
    edge_gamma = _require_nonnegative_finite(edge_gamma, "edge_gamma")
    if aux_loss_version not in ("legacy", "filtered_edge"):
        raise ValueError("aux_loss_version must be 'legacy' or 'filtered_edge'")
    edge_filter_mode = _validate_edge_filter_mode(edge_filter_mode)
    if aux_loss_version == "filtered_edge":
        if lambda_thermal != 0 or lambda_smooth != 0:
            raise ValueError(
                "filtered_edge requires lambda_thermal == 0 and lambda_smooth == 0"
            )
        if isinstance(edge_filter_kernel, bool) or not isinstance(
            edge_filter_kernel, int
        ):
            raise TypeError("edge_filter_kernel must be an integer")
        _validate_odd_kernel(edge_filter_kernel)
        edge_filter_sigma = _require_nonnegative_finite(
            edge_filter_sigma, "edge_filter_sigma"
        )
        if edge_filter_sigma == 0:
            raise ValueError("edge_filter_sigma must be positive")
    pred_4d, gt_4d = _prepare_pair(pred, gt)
    zero = pred_4d.new_zeros(())
    terms = {"thermal": zero, "edge": zero, "smooth": zero}
    if not any(value > 0 for value in lambdas):
        return zero, terms

    if aux_loss_version == "filtered_edge":
        terms["edge"] = filtered_edge_consistency_loss(
            pred_4d,
            gt_4d,
            kernel_size=edge_filter_kernel,
            sigma=edge_filter_sigma,
            edge_filter_mode=edge_filter_mode,
        )
        return lambda_edge * terms["edge"], terms

    if lambda_thermal > 0:
        reliability = compute_noise_reliability_map(gt_4d, beta=noise_beta)
        terms["thermal"] = thermal_reconstruction_loss(
            pred_4d, gt_4d, reliability
        )

    if lambda_edge > 0 or lambda_smooth > 0:
        edge_map = compute_edge_map(gt_4d)
        if lambda_edge > 0:
            terms["edge"] = edge_preserving_loss(
                pred_4d, gt_4d, edge_map, gamma=edge_gamma
            )
        if lambda_smooth > 0:
            terms["smooth"] = non_edge_smoothness_loss(pred_4d, edge_map)

    total = (
        lambda_thermal * terms["thermal"]
        + lambda_edge * terms["edge"]
        + lambda_smooth * terms["smooth"]
    )
    return total, terms
