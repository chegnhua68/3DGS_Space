import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from losses.thermal_physics_loss import (
    compute_edge_map,
    compute_noise_reliability_map,
    edge_preserving_loss,
    non_edge_smoothness_loss,
    thermal_physics_loss,
    thermal_reconstruction_loss,
)


class ThermalPhysicsLossTests(unittest.TestCase):
    def test_constant_image_has_no_edges_and_full_reliability(self):
        image = torch.full((1, 8, 8), 0.4)
        self.assertTrue(torch.equal(compute_edge_map(image), torch.zeros(1, 8, 8)))
        self.assertTrue(
            torch.equal(compute_noise_reliability_map(image), torch.ones(1, 8, 8))
        )

    def test_edge_map_is_per_image_normalized(self):
        image = torch.zeros((2, 1, 8, 8))
        image[0, :, :, 4:] = 1.0
        image[1, :, 4:, :] = 0.5
        edge = compute_edge_map(image)
        self.assertEqual(tuple(edge.shape), (2, 1, 8, 8))
        self.assertTrue(torch.all(edge >= 0))
        self.assertTrue(torch.all(edge <= 1))
        self.assertTrue(torch.allclose(edge.flatten(1).amax(1), torch.ones(2)))

    def test_weighted_l1_matches_l1_for_unit_weights(self):
        pred = torch.tensor([[[0.0, 1.0], [0.5, 0.25]]])
        gt = torch.zeros_like(pred)
        weight = torch.ones((1, 2, 2))
        actual = thermal_reconstruction_loss(pred, gt, weight)
        self.assertTrue(torch.allclose(actual, (pred - gt).abs().mean()))

    def test_weighted_l1_matches_explicit_nonuniform_mean(self):
        pred = torch.tensor([[[[0.0, 1.0]]]])
        gt = torch.zeros_like(pred)
        weight = torch.tensor([[[[1.0, 0.5]]]])
        actual = thermal_reconstruction_loss(pred, gt, weight)
        self.assertAlmostEqual(actual.item(), 0.25)

    def test_nonfinite_hyperparameters_are_rejected(self):
        image = torch.zeros((1, 1, 3, 3))
        with self.assertRaisesRegex(ValueError, "finite"):
            thermal_physics_loss(image, image, lambda_thermal=float("nan"))
        with self.assertRaisesRegex(ValueError, "finite"):
            compute_noise_reliability_map(image, beta=float("inf"))
        with self.assertRaisesRegex(ValueError, "finite"):
            thermal_reconstruction_loss(
                image, image, torch.full_like(image, float("nan"))
            )

    def test_identical_images_have_zero_data_and_edge_loss(self):
        image = torch.rand((3, 12, 9))
        edge = compute_edge_map(image)
        reliability = compute_noise_reliability_map(image)
        self.assertEqual(thermal_reconstruction_loss(image, image, reliability).item(), 0.0)
        self.assertEqual(edge_preserving_loss(image, image, edge).item(), 0.0)

    def test_constant_prediction_has_zero_smoothness(self):
        pred = torch.full((3, 7, 5), 0.2)
        edge = torch.zeros((1, 7, 5))
        self.assertAlmostEqual(
            non_edge_smoothness_loss(pred, edge).item(), 0.0, places=8
        )

    def test_zero_lambdas_short_circuit_to_zero(self):
        pred = torch.rand((3, 6, 6), requires_grad=True)
        gt = torch.rand_like(pred)
        total, terms = thermal_physics_loss(pred, gt)
        self.assertEqual(total.item(), 0.0)
        self.assertFalse(total.requires_grad)
        self.assertEqual(set(terms), {"thermal", "edge", "smooth"})

    def test_enabled_terms_backpropagate(self):
        pred = torch.rand((1, 3, 10, 10), requires_grad=True)
        gt = torch.rand_like(pred)
        total, terms = thermal_physics_loss(
            pred,
            gt,
            lambda_thermal=0.5,
            lambda_edge=0.05,
            lambda_smooth=0.005,
        )
        total.backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertTrue(all(torch.isfinite(term) for term in terms.values()))

    def test_invalid_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            thermal_reconstruction_loss(
                torch.zeros(1, 3, 4, 4),
                torch.zeros(1, 3, 5, 4),
                torch.ones(1, 1, 4, 4),
            )


if __name__ == "__main__":
    unittest.main()
