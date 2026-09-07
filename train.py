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

import os
import math
import torch
from random import randint
# cq:import corners_loss
from utils.loss_utils import l1_loss, ssim, kl_divergence, corners_loss
from losses.thermal_physics_loss import thermal_physics_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, ATFModel, TCMModel
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    run_config=None,
    log_interval=1000,
    quiet=False,
):
    if isinstance(log_interval, bool) or not isinstance(log_interval, int) or log_interval < 1:
        raise ValueError("log_interval must be a positive integer")
    thermal_weights = (opt.lambda_thermal, opt.lambda_edge, opt.lambda_smooth)
    thermal_parameters = thermal_weights + (opt.noise_beta, opt.edge_gamma)
    if not all(math.isfinite(value) for value in thermal_parameters):
        raise ValueError("thermal loss parameters must be finite")
    if any(weight < 0 for weight in thermal_parameters):
        raise ValueError("thermal loss parameters must be non-negative")
    tb_writer = prepare_output_and_logger(dataset, run_config)
    gaussians = GaussianModel(dataset.sh_degree)
    ATF = ATFModel(dataset.is_blender)
    ATF.train_setting(opt)
    #modified
    TCM = TCMModel()
    TCM.train_setting(opt)

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    total_elapsed_ms = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(
        range(opt.iterations),
        desc="Training",
        disable=quiet,
        mininterval=5.0,
    )
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
    thermal_loss_enabled = any(
        weight > 0 for weight in thermal_weights
    )
    for iteration in range(1, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        total_frame = len(viewpoint_stack)
        time_interval = 1 / total_frame

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        if iteration < opt.warm_up:
            d_rgb = 1.0
        else:
            
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)

            ast_noise = 0 
            # cq: start
            abs, sca, d = ATF.step(gaussians.get_xyz.detach(), time_input + ast_noise)
            d_rgb = torch.exp((abs + sca) * d)
            # cq: end

        # Render
        render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_rgb)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        
#        #modified
#        if iteration >= 2*opt.warm_up:
##            if iteration >= 40000:
##                k = 1
##            else:
#            k = 1/(50000 - 2*opt.warm_up)*(iteration - 2*opt.warm_up)
#        else:
#            k = 0
#        #image = image + k*TCM.step(image)
#        #if iteration >= 10000:
##            k = 1/(opt.iterations - 10000)*(iteration - 10000)
##            image = image + k*TCM.step(image)
##        elif iteration >20000:
##            k = 1
        image = image + TCM.step(image)
        # cq:
        c_loss = corners_loss(image, gt_image)
        #import pdb;pdb.set_trace()
        Ll1 = l1_loss(image, gt_image)
        if iteration<30000000:
        #if iteration<0:
            loss = (1.0 - opt.lambda_dssim -0.2) * (Ll1) + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + 0.2*c_loss*max(1-iteration/5000,0)
        else:
            loss = (1.0 - opt.lambda_dssim) * (Ll1) + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        thermal_terms = None
        if thermal_loss_enabled:
            thermal_extra, thermal_terms = thermal_physics_loss(
                image,
                gt_image,
                lambda_thermal=opt.lambda_thermal,
                lambda_edge=opt.lambda_edge,
                lambda_smooth=opt.lambda_smooth,
                noise_beta=opt.noise_beta,
                edge_gamma=opt.edge_gamma,
            )
            loss = loss + thermal_extra
        loss.backward()

        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            elapsed_ms = iter_start.elapsed_time(iter_end)
            total_elapsed_ms += elapsed_ms
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            progress_bar.update(1)
            if iteration % log_interval == 0 or iteration == opt.iterations:
                progress_bar.set_postfix({"loss": f"{ema_loss_for_log:.6f}"})
                peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                average_ms = total_elapsed_ms / iteration
                print(
                    "[ITER {}] loss={:.6f} points={} avg_ms={:.2f} peak_cuda_mb={:.1f}".format(
                        iteration,
                        ema_loss_for_log,
                        gaussians.get_xyz.shape[0],
                        average_ms,
                        peak_memory_mb,
                    ),
                    flush=True,
                )

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])

            # Log and save
            cur_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed_ms,
                                       testing_iterations, scene, render, (pipe, background), ATF,
                                       TCM, dataset.load2gpu_on_the_fly)
            if tb_writer and thermal_terms is not None:
                tb_writer.add_scalar("train_loss/thermal_weighted", thermal_extra.item(), iteration)
                for name, value in thermal_terms.items():
                    tb_writer.add_scalar("train_loss/" + name, value.item(), iteration)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("[ITER {}] saving model".format(iteration), flush=True)
                scene.save(iteration)
                ATF.save_weights(scene.model_path, iteration)
                TCM.save_weights(scene.model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                ATF.optimizer.step()
                TCM.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                ATF.optimizer.zero_grad()
                ATF.update_learning_rate(iteration)
                TCM.optimizer.zero_grad()
                TCM.update_learning_rate(iteration)

    progress_bar.close()
    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration), flush=True)


def prepare_output_and_logger(args, run_config=None):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    recorded_args = vars(run_config).copy() if run_config is not None else {}
    recorded_args.update(vars(args))
    with open(
        os.path.join(args.model_path, "cfg_args"), 'w', encoding="utf-8"
    ) as cfg_log_f:
        cfg_log_f.write(str(Namespace(**recorded_args)))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, ATF, TCM, load2gpu_on_the_fly):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                for idx, viewpoint in enumerate(config['cameras']):
                
                
                
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                        
                        
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    # cq: start
                    abs, sca, d = ATF.step(xyz.detach(), time_input)
                    d_rgb = torch.exp((abs + sca) * d)
                    #d_rgb = 1.0
                    # cq: end
                    
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_rgb)["render"],
                        0.0, 1.0)
#                    #modified
##                    if iteration >= 5000:
##                        k = 1/(50000 - 5000)*(iteration - 5000)
##                        image = image + k*TCM.step(image)
#                    if iteration >= 0:
##            if iteration >= 40000:
##                k = 1
##            else:
#                        k = 1/(50000 - 0)*(iteration - 0)
#                    else:
#                        k = 0
#                    #image = image + k*TCM.step(image)
                    image = image + TCM.step(image)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5000, 6000, 7_000] + list(range(10000, 60001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000] + list(range(25000, 30001, 1000)))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--log_interval",
        type=int,
        default=1000,
        help="update the training summary at this iteration interval (default: 1000)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path, flush=True)

    # Initialize system state (RNG)
    # Keep milestone messages visible; --quiet only disables the live progress bar.
    safe_state(False, seed=args.seed)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        run_config=args,
        log_interval=args.log_interval,
        quiet=args.quiet,
    )

    # All done
    print("Training complete.", flush=True)
