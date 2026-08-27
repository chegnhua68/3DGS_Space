import math
import unittest

import torch


try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
    from simple_knn._C import distCUDA2

    CUDA_EXTENSIONS_AVAILABLE = True
except (ImportError, OSError):
    CUDA_EXTENSIONS_AVAILABLE = False


@unittest.skipUnless(
    torch.cuda.is_available() and CUDA_EXTENSIONS_AVAILABLE,
    "CUDA and the compiled project extensions are required",
)
class CudaExtensionTests(unittest.TestCase):
    def test_simple_knn_matches_torch_reference(self):
        generator = torch.Generator(device="cuda").manual_seed(7)
        points = torch.rand((64, 3), generator=generator, device="cuda")

        actual = distCUDA2(points)
        pairwise_squared = torch.cdist(points, points).square()
        pairwise_squared.fill_diagonal_(float("inf"))
        expected = pairwise_squared.topk(3, largest=False).values.mean(dim=1)

        self.assertEqual(tuple(actual.shape), (64,))
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue(torch.allclose(actual, expected, rtol=2e-5, atol=2e-6))

    def test_rasterizer_forward_and_backward_are_finite(self):
        device = torch.device("cuda")
        image_size = 32
        znear, zfar = 0.01, 100.0
        projection = torch.zeros((4, 4), device=device)
        projection[0, 0] = 1.0
        projection[1, 1] = 1.0
        projection[2, 2] = zfar / (zfar - znear)
        projection[2, 3] = -(zfar * znear) / (zfar - znear)
        projection[3, 2] = 1.0

        settings = GaussianRasterizationSettings(
            image_height=image_size,
            image_width=image_size,
            tanfovx=math.tan(math.pi / 4.0),
            tanfovy=math.tan(math.pi / 4.0),
            bg=torch.zeros(3, device=device),
            scale_modifier=1.0,
            viewmatrix=torch.eye(4, device=device),
            projmatrix=projection.transpose(0, 1).contiguous(),
            sh_degree=0,
            campos=torch.zeros(3, device=device),
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        means3d = torch.tensor(
            [[0.12, -0.08, 2.0]], device=device, requires_grad=True
        )
        means2d = torch.zeros_like(means3d, requires_grad=True)
        colors = torch.tensor(
            [[0.2, 0.5, 0.8]], device=device, requires_grad=True
        )
        opacities = torch.tensor([[0.8]], device=device, requires_grad=True)
        scales = torch.tensor(
            [[0.15, 0.15, 0.15]], device=device, requires_grad=True
        )
        rotations = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], device=device, requires_grad=True
        )

        image, radii, depth = rasterizer(
            means3D=means3d,
            means2D=means2d,
            colors_precomp=colors,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
        )
        torch.cuda.synchronize()

        self.assertEqual(tuple(image.shape), (3, image_size, image_size))
        self.assertEqual(tuple(radii.shape), (1,))
        self.assertEqual(tuple(depth.shape), (1, image_size, image_size))
        self.assertGreater(radii.item(), 0)
        self.assertGreater(image.max().item(), 0.0)
        self.assertTrue(torch.isfinite(image).all())
        self.assertTrue(torch.isfinite(depth).all())

        image.sum().backward()
        torch.cuda.synchronize()
        for value in (means3d, means2d, colors, opacities, scales, rotations):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
        self.assertGreater(colors.grad.abs().sum().item(), 0.0)
        self.assertGreater(opacities.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
