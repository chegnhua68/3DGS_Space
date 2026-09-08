import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch
import torch.nn.functional as F

from losses.thermal_physics_loss import (
    compute_edge_map,
    compute_noise_reliability_map,
    edge_preserving_loss,
    filtered_edge_consistency_loss,
    non_edge_smoothness_loss,
    thermal_physics_loss,
    thermal_reconstruction_loss,
)


thermal_loss_module = importlib.import_module("losses.thermal_physics_loss")


def _reference_filtered_edge_loss(prediction, observation, kernel_size=5, sigma=1.0):
    def as_luminance(image):
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.shape[1] == 1:
            return image
        weights = image.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
        return (image * weights).sum(dim=1, keepdim=True)

    radius = kernel_size // 2
    coordinates = torch.arange(
        -radius,
        radius + 1,
        dtype=prediction.dtype,
        device=prediction.device,
    )
    gaussian = torch.exp(
        -(
            coordinates[:, None].square()
            + coordinates[None, :].square()
        )
        / (2.0 * sigma * sigma)
    )
    gaussian = (gaussian / gaussian.sum()).view(1, 1, kernel_size, kernel_size)
    sobel_x = prediction.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3) / 8.0
    sobel_y = sobel_x.transpose(-1, -2)

    def components(image):
        luminance = as_luminance(image)
        smoothed = F.conv2d(
            F.pad(
                luminance,
                (radius, radius, radius, radius),
                mode="reflect",
            ),
            gaussian,
        )
        padded = F.pad(smoothed, (1, 1, 1, 1), mode="reflect")
        return F.conv2d(padded, sobel_x), F.conv2d(padded, sobel_y)

    pred_x, pred_y = components(prediction)
    obs_x, obs_y = components(observation.detach())
    return 0.5 * (
        (pred_x - obs_x).abs().mean() + (pred_y - obs_y).abs().mean()
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

    def test_filtered_edge_matches_independent_explicit_reference(self):
        values = torch.arange(2 * 3 * 7 * 8, dtype=torch.float64)
        prediction = ((values % 17) / 19.0).reshape(2, 3, 7, 8)
        observation = ((values.flip(0) % 13) / 15.0).reshape(2, 3, 7, 8)

        actual = filtered_edge_consistency_loss(prediction, observation)
        expected = _reference_filtered_edge_loss(prediction, observation)

        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)

    def test_filtered_edge_identical_high_frequency_images_have_zero_loss(self):
        rows = torch.arange(7).view(7, 1)
        columns = torch.arange(8).view(1, 8)
        checkerboard = ((rows + columns) % 2).to(torch.float64).unsqueeze(0)

        actual = filtered_edge_consistency_loss(checkerboard, checkerboard)

        torch.testing.assert_close(
            actual, torch.zeros_like(actual), atol=0.0, rtol=0.0
        )

    def test_filtered_edge_ignores_constant_intensity_offset(self):
        prediction = torch.full((1, 7, 8), 0.2, dtype=torch.float64)
        observation = torch.full_like(prediction, 0.8)

        actual = filtered_edge_consistency_loss(prediction, observation)

        torch.testing.assert_close(
            actual, torch.zeros_like(actual), atol=1e-12, rtol=0.0
        )

    def test_filtered_edge_matches_signed_components_not_only_magnitude(self):
        ramp = torch.linspace(0.0, 1.0, 9, dtype=torch.float64)
        prediction = ramp.view(1, 1, 9).expand(1, 7, 9).clone()
        observation = 1.0 - prediction

        actual = filtered_edge_consistency_loss(prediction, observation)

        self.assertGreater(actual.item(), 0.05)

    def test_filtered_edge_backpropagates_only_to_prediction(self):
        values = torch.arange(56, dtype=torch.float64).reshape(1, 1, 7, 8)
        prediction = (torch.sin(values / 7.0) * 0.25 + 0.5).requires_grad_()
        observation = (torch.cos(values / 5.0) * 0.2 + 0.45).requires_grad_()

        loss = filtered_edge_consistency_loss(prediction, observation)
        loss.backward()

        self.assertEqual(loss.ndim, 0)
        self.assertEqual(loss.dtype, prediction.dtype)
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(prediction.grad.abs().sum().item(), 0.0)
        self.assertIsNone(observation.grad)

    def test_filtered_edge_passes_double_precision_gradcheck(self):
        torch.manual_seed(17)
        prediction = torch.rand((1, 1, 5, 6), dtype=torch.float64, requires_grad=True)
        observation = torch.rand_like(prediction)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda value: filtered_edge_consistency_loss(value, observation),
                (prediction,),
                eps=1e-6,
                atol=1e-5,
                rtol=1e-3,
            )
        )

    def test_filtered_edge_supports_chw_bchw_and_bt601_luminance(self):
        values = torch.arange(3 * 7 * 8, dtype=torch.float64).reshape(3, 7, 8)
        prediction = (values % 23) / 23.0
        observation = (values.flip(-1) % 19) / 19.0
        weights = prediction.new_tensor((0.299, 0.587, 0.114)).view(3, 1, 1)
        prediction_luma = (prediction * weights).sum(dim=0, keepdim=True)
        observation_luma = (observation * weights).sum(dim=0, keepdim=True)

        chw = filtered_edge_consistency_loss(prediction, observation)
        bchw = filtered_edge_consistency_loss(
            prediction.unsqueeze(0), observation.unsqueeze(0)
        )
        luminance = filtered_edge_consistency_loss(
            prediction_luma, observation_luma
        )

        torch.testing.assert_close(chw, bchw, atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(chw, luminance, atol=1e-12, rtol=1e-10)

    def test_filtered_edge_rejects_invalid_shapes_channels_and_dtype(self):
        valid = torch.zeros((1, 1, 7, 8))
        with self.assertRaisesRegex(ValueError, "shape"):
            filtered_edge_consistency_loss(torch.zeros(7, 8), torch.zeros(7, 8))
        with self.assertRaisesRegex(ValueError, "same shape"):
            filtered_edge_consistency_loss(valid, torch.zeros(1, 1, 8, 8))
        with self.assertRaisesRegex(ValueError, "exactly 1 or 3 channels"):
            filtered_edge_consistency_loss(
                torch.zeros(1, 2, 7, 8), torch.zeros(1, 2, 7, 8)
            )
        with self.assertRaisesRegex(ValueError, "same dtype"):
            filtered_edge_consistency_loss(valid, valid.to(torch.float64))
        with self.assertRaisesRegex(TypeError, "floating-point"):
            filtered_edge_consistency_loss(
                torch.zeros(1, 1, 7, 8, dtype=torch.int64),
                torch.zeros(1, 1, 7, 8, dtype=torch.int64),
            )

    def test_filtered_edge_rejects_invalid_reflect_padding(self):
        too_short = torch.zeros((1, 1, 2, 8))
        with self.assertRaisesRegex(
            ValueError, "Gaussian filter requires height and width greater"
        ):
            filtered_edge_consistency_loss(too_short, too_short)

        sobel_too_short = torch.zeros((1, 1, 1, 4))
        with self.assertRaisesRegex(
            ValueError, "Sobel filter requires height and width greater"
        ):
            filtered_edge_consistency_loss(
                sobel_too_short,
                sobel_too_short,
                kernel_size=1,
            )

    def test_filtered_edge_rejects_invalid_kernel_and_sigma(self):
        image = torch.zeros((1, 1, 7, 8))
        for kernel in (0, 4):
            with self.subTest(kernel=kernel):
                with self.assertRaisesRegex(ValueError, "positive odd"):
                    filtered_edge_consistency_loss(
                        image, image, kernel_size=kernel
                    )
        for kernel in (True, 5.0):
            with self.subTest(kernel=kernel):
                with self.assertRaisesRegex(TypeError, "integer"):
                    filtered_edge_consistency_loss(
                        image, image, kernel_size=kernel
                    )
        with self.assertRaisesRegex(ValueError, "positive"):
            filtered_edge_consistency_loss(image, image, sigma=0.0)
        for sigma in (-1.0, float("nan"), float("inf")):
            with self.subTest(sigma=sigma):
                with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                    filtered_edge_consistency_loss(image, image, sigma=sigma)

    def test_dispatcher_default_is_legacy_and_preserves_legacy_values(self):
        prediction_values = [
            [0.00, 0.15, 0.30, 0.45],
            [0.60, 0.75, 0.90, 0.20],
            [0.35, 0.50, 0.65, 0.80],
            [0.95, 0.10, 0.25, 0.40],
        ]
        observation_values = [
            [0.50, 0.40, 0.30, 0.20],
            [0.10, 0.20, 0.30, 0.40],
            [0.50, 0.60, 0.70, 0.80],
            [0.90, 0.80, 0.70, 0.60],
        ]
        prediction = torch.tensor(
            [[prediction_values]], dtype=torch.float32, requires_grad=True
        )
        observation = torch.tensor([[observation_values]], dtype=torch.float32)
        kwargs = {
            "lambda_thermal": 0.1,
            "lambda_edge": 0.01,
            "lambda_smooth": 0.001,
            "noise_beta": 5.0,
            "edge_gamma": 3.0,
        }

        default_total, default_terms = thermal_physics_loss(
            prediction, observation, **kwargs
        )
        explicit_total, explicit_terms = thermal_physics_loss(
            prediction.detach().clone(),
            observation,
            aux_loss_version="legacy",
            **kwargs,
        )

        torch.testing.assert_close(default_total, explicit_total)
        for name in ("thermal", "edge", "smooth"):
            torch.testing.assert_close(default_terms[name], explicit_terms[name])
        expected_terms = {
            "thermal": 0.09248720109462738,
            "edge": 0.3815741240978241,
            "smooth": 0.07600954174995422,
        }
        for name, expected in expected_terms.items():
            torch.testing.assert_close(
                default_terms[name],
                default_terms[name].new_tensor(expected),
                atol=1e-7,
                rtol=1e-5,
            )
        torch.testing.assert_close(
            default_total,
            default_total.new_tensor(0.013140471652150154),
            atol=1e-7,
            rtol=1e-5,
        )
        default_total.backward()
        expected_gradient = prediction.new_tensor(
            [[[
                [-0.0020322236, -0.0044218944, 0.0007929697, 0.0017740424],
                [0.0006563069, 0.0013516113, 0.0020792747, -0.0047607394],
                [-0.0035074803, -0.0031103706, -0.0007744891, -0.0008509933],
                [0.0002999397, -0.0032370295, -0.0075373896, -0.0038189678],
            ]]]
        )
        torch.testing.assert_close(
            prediction.grad, expected_gradient, atol=1e-7, rtol=1e-5
        )

    def test_dispatcher_filtered_edge_returns_only_weighted_edge(self):
        values = torch.arange(56, dtype=torch.float64).reshape(1, 1, 7, 8)
        prediction = torch.sin(values / 7.0) * 0.25 + 0.5
        observation = torch.cos(values / 5.0) * 0.2 + 0.45
        edge_weight = 0.001

        total, terms = thermal_physics_loss(
            prediction,
            observation,
            lambda_edge=edge_weight,
            aux_loss_version="filtered_edge",
            edge_filter_kernel=5,
            edge_filter_sigma=1.0,
            noise_beta=0.0,
            edge_gamma=0.0,
        )
        direct = filtered_edge_consistency_loss(prediction, observation)
        changed_legacy_parameters, changed_terms = thermal_physics_loss(
            prediction,
            observation,
            lambda_edge=edge_weight,
            aux_loss_version="filtered_edge",
            noise_beta=11.0,
            edge_gamma=9.0,
        )

        torch.testing.assert_close(terms["edge"], direct)
        torch.testing.assert_close(total, edge_weight * direct)
        torch.testing.assert_close(changed_legacy_parameters, total)
        torch.testing.assert_close(changed_terms["edge"], terms["edge"])
        self.assertEqual(terms["thermal"].item(), 0.0)
        self.assertEqual(terms["smooth"].item(), 0.0)

    def test_dispatcher_rejects_invalid_mode_and_filtered_combinations(self):
        image = torch.zeros((1, 1, 7, 8))
        with self.assertRaisesRegex(ValueError, "aux_loss_version"):
            thermal_physics_loss(image, image, aux_loss_version="unknown")
        with self.assertRaisesRegex(ValueError, "lambda_thermal == 0"):
            thermal_physics_loss(
                image,
                image,
                lambda_thermal=0.1,
                lambda_edge=0.001,
                aux_loss_version="filtered_edge",
            )
        with self.assertRaisesRegex(ValueError, "lambda_smooth == 0"):
            thermal_physics_loss(
                image,
                image,
                lambda_edge=0.001,
                lambda_smooth=0.1,
                aux_loss_version="filtered_edge",
            )

    def test_zero_lambdas_do_not_call_legacy_or_filtered_helpers(self):
        prediction = torch.rand((1, 1, 7, 8), requires_grad=True)
        observation = torch.rand_like(prediction)
        with patch.object(
            thermal_loss_module, "compute_edge_map"
        ) as edge_map, patch.object(
            thermal_loss_module, "compute_noise_reliability_map"
        ) as reliability, patch.object(
            thermal_loss_module, "filtered_edge_consistency_loss"
        ) as filtered:
            legacy_total, legacy_terms = thermal_physics_loss(
                prediction, observation
            )
            filtered_total, filtered_terms = thermal_physics_loss(
                prediction,
                observation,
                aux_loss_version="filtered_edge",
            )

        edge_map.assert_not_called()
        reliability.assert_not_called()
        filtered.assert_not_called()
        for total, terms in (
            (legacy_total, legacy_terms),
            (filtered_total, filtered_terms),
        ):
            self.assertEqual(total.item(), 0.0)
            self.assertFalse(total.requires_grad)
            self.assertTrue(all(term.item() == 0.0 for term in terms.values()))


if __name__ == "__main__":
    unittest.main()
