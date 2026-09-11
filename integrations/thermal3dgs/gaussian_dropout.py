"""Low-rate Gaussian opacity dropout with an isolated random stream."""

from dataclasses import dataclass
import hashlib
import math
from collections.abc import Mapping
from typing import Optional

import torch


def _option_value(options, name, default):
    if isinstance(options, Mapping):
        return options.get(name, default)
    return getattr(options, name, default)


def validate_gaussian_dropout_options(options) -> None:
    max_rate = _option_value(options, "gd_max_rate", 0.0)
    if isinstance(max_rate, bool) or not isinstance(max_rate, (int, float)):
        raise TypeError("gd_max_rate must be a real scalar")
    max_rate = float(max_rate)
    if not math.isfinite(max_rate) or not 0.0 <= max_rate < 1.0:
        raise ValueError("gd_max_rate must be finite and in [0, 1)")

    warmup = _option_value(options, "gd_warmup_iterations", 1000)
    ramp_end = _option_value(options, "gd_ramp_end", 3000)
    seed = _option_value(options, "gd_seed", 104729)
    for name, value in (
        ("gd_warmup_iterations", warmup),
        ("gd_ramp_end", ramp_end),
        ("gd_seed", seed),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("{} must be an integer".format(name))
    if warmup < 0:
        raise ValueError("gd_warmup_iterations must be non-negative")
    if ramp_end <= warmup:
        raise ValueError("gd_ramp_end must be greater than gd_warmup_iterations")
    if seed < 0:
        raise ValueError("gd_seed must be non-negative")


def gaussian_dropout_rate(iteration, max_rate, warmup_iterations, ramp_end) -> float:
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 1:
        raise ValueError("iteration must be a positive integer")
    validate_gaussian_dropout_options(
        {
            "gd_max_rate": max_rate,
            "gd_warmup_iterations": warmup_iterations,
            "gd_ramp_end": ramp_end,
            "gd_seed": 0,
        }
    )
    progress = (iteration - warmup_iterations) / float(ramp_end - warmup_iterations)
    return float(max_rate) * min(max(progress, 0.0), 1.0)


@dataclass(frozen=True)
class GaussianDropoutSample:
    opacity: torch.Tensor
    keep_mask: Optional[torch.Tensor]
    target_p: float
    compensation: float

    @property
    def applied(self) -> bool:
        return self.keep_mask is not None


class GaussianDropoutController:
    """Apply inverted-opacity dropout without consuming the default RNG."""

    def __init__(self, max_rate, warmup_iterations, ramp_end, seed, device):
        validate_gaussian_dropout_options(
            {
                "gd_max_rate": max_rate,
                "gd_warmup_iterations": warmup_iterations,
                "gd_ramp_end": ramp_end,
                "gd_seed": seed,
            }
        )
        self.max_rate = float(max_rate)
        self.warmup_iterations = int(warmup_iterations)
        self.ramp_end = int(ramp_end)
        self.seed = int(seed)
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.seed)

    def rate(self, iteration: int) -> float:
        return gaussian_dropout_rate(
            iteration,
            self.max_rate,
            self.warmup_iterations,
            self.ramp_end,
        )

    def apply(
        self,
        opacity: torch.Tensor,
        iteration: int,
        *,
        is_training_render: bool,
        keep_mask: Optional[torch.Tensor] = None,
    ) -> GaussianDropoutSample:
        target_p = self.rate(iteration) if is_training_render else 0.0
        if target_p == 0.0:
            return GaussianDropoutSample(opacity, None, 0.0, 1.0)
        if opacity.ndim != 2 or opacity.shape[1] != 1:
            raise ValueError("opacity must have shape [N, 1]")
        if opacity.device != self.device:
            raise ValueError("opacity and Gaussian dropout generator must use the same device")

        if keep_mask is None:
            random_values = torch.rand(
                (opacity.shape[0], 1),
                dtype=torch.float32,
                device=opacity.device,
                generator=self.generator,
            )
            keep_mask = random_values >= target_p
        else:
            if keep_mask.shape != opacity.shape:
                raise ValueError("keep_mask must have the same [N, 1] shape as opacity")
            if keep_mask.device != opacity.device:
                raise ValueError("keep_mask and opacity must use the same device")
            keep_mask = keep_mask.to(dtype=torch.bool)

        compensation = 1.0 / (1.0 - target_p)
        render_opacity = opacity * keep_mask.to(dtype=opacity.dtype) * compensation
        return GaussianDropoutSample(
            render_opacity,
            keep_mask,
            target_p,
            compensation,
        )

    def state_sha256(self) -> str:
        state = self.generator.get_state().detach().cpu().numpy().tobytes()
        return hashlib.sha256(state).hexdigest()

    def audit_config(self):
        return {
            "gd_max_rate": self.max_rate,
            "gd_warmup_iterations": self.warmup_iterations,
            "gd_ramp_end": self.ramp_end,
            "gd_seed": self.seed,
            "compensation": "inverted_opacity",
            "densification_stats_policy": "upstream_visibility",
        }
