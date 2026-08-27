"""Losses introduced by the sparse-view infrared extension."""

from .thermal_physics_loss import (
    compute_edge_map,
    compute_noise_reliability_map,
    edge_preserving_loss,
    non_edge_smoothness_loss,
    thermal_physics_loss,
    thermal_reconstruction_loss,
)

__all__ = [
    "compute_edge_map",
    "compute_noise_reliability_map",
    "edge_preserving_loss",
    "non_edge_smoothness_loss",
    "thermal_physics_loss",
    "thermal_reconstruction_loss",
]
