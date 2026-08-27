import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from utils.loss_utils import corners_loss, find_corners


class CornerLossRegressionTests(unittest.TestCase):
    def test_harris_response_detects_a_square_corner(self):
        image = torch.zeros((3, 11, 11), dtype=torch.float32)
        image[:, 4:8, 4:8] = 1.0
        response = find_corners(image)
        self.assertTrue(torch.isfinite(response).all())
        self.assertGreater(response.max().item(), 0.0)

    def test_corner_loss_backward_is_finite(self):
        prediction = torch.rand((3, 13, 9), requires_grad=True)
        target = torch.rand_like(prediction)
        loss = corners_loss(prediction, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(prediction.grad).all())


if __name__ == "__main__":
    unittest.main()
