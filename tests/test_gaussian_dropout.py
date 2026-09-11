import random
import unittest

import numpy as np
import torch

from integrations.thermal3dgs.gaussian_dropout import (
    GaussianDropoutController,
    gaussian_dropout_rate,
    validate_gaussian_dropout_options,
)


class GaussianDropoutTests(unittest.TestCase):
    def test_schedule_matches_frozen_protocol(self):
        values = [
            gaussian_dropout_rate(t, 0.05, 1000, 3000)
            for t in (1, 1000, 2000, 3000, 7000)
        ]
        self.assertEqual(values, [0.0, 0.0, 0.025, 0.05, 0.05])

    def test_zero_rate_is_identity_and_does_not_consume_default_rng(self):
        controller = GaussianDropoutController(0.0, 1000, 3000, 106755, "cpu")
        opacity = torch.tensor([[0.2], [0.8]], requires_grad=True)
        torch.manual_seed(41)
        expected = torch.rand(4)
        torch.manual_seed(41)
        sample = controller.apply(opacity, 7000, is_training_render=True)
        actual = torch.rand(4)
        self.assertIs(sample.opacity, opacity)
        self.assertIsNone(sample.keep_mask)
        self.assertTrue(torch.equal(expected, actual))

    def test_manual_mask_uses_inverted_opacity_without_mutation(self):
        controller = GaussianDropoutController(0.05, 0, 1, 7, "cpu")
        opacity = torch.tensor([[0.2], [0.8]], requires_grad=True)
        original = opacity.detach().clone()
        mask = torch.tensor([[True], [False]])
        sample = controller.apply(opacity, 1, is_training_render=True, keep_mask=mask)
        expected = torch.tensor([[0.2 / 0.95], [0.0]])
        self.assertTrue(torch.allclose(sample.opacity, expected))
        self.assertTrue(torch.equal(opacity.detach(), original))
        sample.opacity.sum().backward()
        self.assertTrue(torch.allclose(opacity.grad, torch.tensor([[1.0 / 0.95], [0.0]])))

    def test_zero_mask_is_finite_and_dynamic_point_count_is_supported(self):
        controller = GaussianDropoutController(0.05, 0, 1, 7, "cpu")
        opacity = torch.ones((5, 1))
        mask = torch.zeros((5, 1), dtype=torch.bool)
        sample = controller.apply(opacity, 1, is_training_render=True, keep_mask=mask)
        self.assertEqual(tuple(sample.opacity.shape), (5, 1))
        self.assertTrue(torch.isfinite(sample.opacity).all())
        self.assertTrue(torch.equal(sample.opacity, torch.zeros_like(opacity)))

    def test_eval_path_does_not_advance_gd_generator(self):
        controller = GaussianDropoutController(0.05, 0, 1, 7, "cpu")
        opacity = torch.ones((4, 1))
        before = controller.state_sha256()
        sample = controller.apply(opacity, 1, is_training_render=False)
        self.assertIs(sample.opacity, opacity)
        self.assertIsNone(sample.keep_mask)
        self.assertEqual(before, controller.state_sha256())

    def test_generator_isolated_and_reproducible(self):
        first = GaussianDropoutController(0.05, 0, 1, 99, "cpu")
        second = GaussianDropoutController(0.05, 0, 1, 99, "cpu")
        opacity = torch.ones((1000, 1))
        a = first.apply(opacity, 1, is_training_render=True).keep_mask
        b = second.apply(opacity, 1, is_training_render=True).keep_mask
        self.assertTrue(torch.equal(a, b))
        self.assertAlmostEqual(float((~a).float().mean()), 0.05, delta=0.02)

    def test_validation_rejects_invalid_schedule(self):
        with self.assertRaises(ValueError):
            validate_gaussian_dropout_options(
                {"gd_max_rate": 1.0, "gd_warmup_iterations": 0, "gd_ramp_end": 1, "gd_seed": 1}
            )
        with self.assertRaises(ValueError):
            validate_gaussian_dropout_options(
                {"gd_max_rate": 0.05, "gd_warmup_iterations": 2, "gd_ramp_end": 2, "gd_seed": 1}
            )


if __name__ == "__main__":
    unittest.main()
