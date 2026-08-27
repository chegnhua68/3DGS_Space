#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

# cq:
def find_corners(img):
    if img.ndim != 3:
        raise ValueError("find_corners expects an image with shape (C,H,W)")
    inputs = img.mean(dim=0, keepdim=True).unsqueeze(0) * 255
    sobel_x = torch.tensor([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    I_x = F.conv2d(inputs, sobel_x, stride=1, padding=1,)
    I_y = F.conv2d(inputs, sobel_y, stride=1, padding=1,)
    k = 0.04
    I_x_squared = F.avg_pool2d(I_x * I_x, kernel_size=3, stride=1, padding=1)
    I_y_squared = F.avg_pool2d(I_y * I_y, kernel_size=3, stride=1, padding=1)
    I_x_y = F.avg_pool2d(I_x * I_y, kernel_size=3, stride=1, padding=1)
    
    det_M = I_x_squared * I_y_squared - I_x_y* I_x_y
    trace_M = I_x_squared + I_y_squared
    R = det_M - k * (trace_M*trace_M)
    return R
    
# cq:
def corners_loss(network_output, gt):
    if network_output.shape != gt.shape:
        raise ValueError("network_output and gt must have identical shapes")
    R = find_corners(network_output)
    # exp(R) / max(exp(R)) is equivalent to exp(R - max(R)), but the
    # latter cannot underflow to an all-zero map before normalization.
    weights = torch.exp(R - R.max()).squeeze(0)
    weights = weights.expand_as(network_output)
    return (torch.abs(network_output - gt) * weights).mean()

def kl_divergence(rho, rho_hat):
    rho_hat = torch.mean(torch.sigmoid(rho_hat), 0)
    rho = torch.tensor([rho] * len(rho_hat)).cuda()
    return torch.mean(
        rho * torch.log(rho / (rho_hat + 1e-5)) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat + 1e-5)))


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
