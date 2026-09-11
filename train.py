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
import csv
import hashlib
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
from arguments import (
    ModelParams,
    PipelineParams,
    OptimizationParams,
    validate_aux_loss_options,
)
from integrations.thermal3dgs.gaussian_dropout import GaussianDropoutController
from integrations.thermal3dgs.run_control import (
    INTERRUPTED_EXIT_CODE,
    UserStopRequested,
    raise_if_stop_requested,
    resolve_stop_file,
    utc_now,
    write_json_atomic,
    write_run_status,
)
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
LOSS_COMPONENT_FIELDS = (
    "iteration",
    "train_seed",
    "gd_seed",
    "aux_loss_version",
    "edge_filter_mode",
    "filter_active",
    "edge_filter_kernel",
    "edge_filter_sigma",
    "lambda_thermal",
    "lambda_edge",
    "lambda_smooth",
    "lambda_detail",
    "detail_status",
    "L_baseline",
    "L_thermal_raw",
    "L_edge_raw",
    "L_smooth_raw",
    "weighted_L_thermal",
    "weighted_L_edge",
    "weighted_L_smooth",
    "L_total",
    "gd_target_p",
    "gd_realized_drop_fraction",
    "gd_compensation",
    "opacity_base_max",
    "compensated_opacity_gt1_fraction",
    "Gaussian_count",
    "avg_ms",
    "peak_cuda_mb",
)

LIVE_LOSS_FIELDS = (
    "iteration",
    "timestamp_utc",
    "phase",
    "train_seed",
    "loss_total",
    "loss_base",
    "edge_raw",
    "edge_weighted",
    "lambda_edge",
    "gd_target_p",
)


def _initialize_loss_component_log(model_path):
    path = os.path.join(model_path, "loss_components.csv")
    with open(path, "w", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=LOSS_COMPONENT_FIELDS).writeheader()
    return path


def _append_loss_component_log(path, row):
    with open(path, "a", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=LOSS_COMPONENT_FIELDS).writerow(row)


def _initialize_live_loss_log(model_path):
    path = os.path.join(model_path, "live_loss.csv")
    with open(path, "w", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=LIVE_LOSS_FIELDS).writeheader()
    return path


def _append_live_loss_log(path, row):
    with open(path, "a", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=LIVE_LOSS_FIELDS).writerow(row)


def _raw_and_weighted_term(terms, name, weight):
    if weight == 0:
        return "disabled", 0.0
    raw = float(terms[name].detach().item())
    return raw, weight * raw


def _camera_sequence_digest(audit_state):
    return audit_state["camera_hasher"].copy().hexdigest()


def _write_gd_rng_audit(model_path, gd_controller, audit_state, status):
    write_json_atomic(
        os.path.join(model_path, "gd_rng_audit.json"),
        {
            "schema_version": 1,
            "status": status,
            "last_completed_iteration": audit_state["last_completed_iteration"],
            "camera_sequence_sha256": _camera_sequence_digest(audit_state),
            "gd_generator_state_sha256": gd_controller.state_sha256(),
            "gd": gd_controller.audit_config(),
            "updated_at_utc": utc_now(),
            "resume_capability": "audit_only_not_optimizer_resume",
        },
    )


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    run_config=None,
    log_interval=500,
    tb_log_interval=50,
    tb_flush_secs=5,
    stop_file=None,
    quiet=False,
):
    for name, value in (
        ("log_interval", log_interval),
        ("tb_log_interval", tb_log_interval),
        ("tb_flush_secs", tb_flush_secs),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("{} must be a positive integer".format(name))
    validate_aux_loss_options(opt)
    stop_file = resolve_stop_file(stop_file)
    audit_state = {
        "last_completed_iteration": 0,
        "camera_hasher": hashlib.sha256(),
    }
    write_run_status(
        dataset.model_path,
        status="running",
        phase="initializing",
        last_completed_iteration=0,
    )
    tb_writer = prepare_output_and_logger(dataset, run_config, tb_flush_secs)
    gd_controller = GaussianDropoutController(
        getattr(opt, "gd_max_rate", 0.0),
        getattr(opt, "gd_warmup_iterations", 1000),
        getattr(opt, "gd_ramp_end", 3000),
        getattr(opt, "gd_seed", 104729),
        "cuda",
    )
    try:
        raise_if_stop_requested(stop_file, "initializing", 0)
        result = _training_impl(
            dataset,
            opt,
            pipe,
            testing_iterations,
            saving_iterations,
            tb_writer=tb_writer,
            gd_controller=gd_controller,
            audit_state=audit_state,
            train_seed=getattr(run_config, "seed", 0),
            log_interval=log_interval,
            tb_log_interval=tb_log_interval,
            stop_file=stop_file,
            quiet=quiet,
        )
        _write_gd_rng_audit(dataset.model_path, gd_controller, audit_state, "completed")
        write_run_status(
            dataset.model_path,
            status="completed",
            phase="completed",
            last_completed_iteration=audit_state["last_completed_iteration"],
        )
        return result
    except UserStopRequested as exc:
        audit_state["last_completed_iteration"] = exc.last_completed_iteration
        _write_gd_rng_audit(dataset.model_path, gd_controller, audit_state, "interrupted")
        write_run_status(
            dataset.model_path,
            status="interrupted",
            phase=exc.phase,
            last_completed_iteration=exc.last_completed_iteration,
            stop_reason="user_request",
        )
        raise
    except Exception as exc:
        _write_gd_rng_audit(dataset.model_path, gd_controller, audit_state, "failed")
        write_run_status(
            dataset.model_path,
            status="failed",
            phase="training",
            last_completed_iteration=audit_state["last_completed_iteration"],
            error="{}: {}".format(type(exc).__name__, exc),
        )
        raise
    finally:
        if tb_writer:
            try:
                tb_writer.flush()
            finally:
                tb_writer.close()


def _training_impl(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    tb_writer,
    gd_controller,
    audit_state,
    train_seed,
    log_interval=500,
    tb_log_interval=50,
    stop_file=None,
    quiet=False,
):
    thermal_weights = (opt.lambda_thermal, opt.lambda_edge, opt.lambda_smooth)
    aux_loss_version = getattr(opt, "aux_loss_version", "legacy")
    edge_filter_mode = getattr(opt, "edge_filter_mode", "gaussian")
    edge_filter_kernel = getattr(opt, "edge_filter_kernel", 5)
    edge_filter_sigma = getattr(opt, "edge_filter_sigma", 1.0)
    loss_component_path = _initialize_loss_component_log(dataset.model_path)
    live_loss_path = _initialize_live_loss_log(dataset.model_path)
    auxiliary_enabled = any(weight > 0 for weight in thermal_weights)
    filter_active = bool(
        auxiliary_enabled
        and aux_loss_version == "filtered_edge"
        and opt.lambda_edge > 0
        and edge_filter_mode == "gaussian"
    )
    filter_kernel_log = edge_filter_kernel if filter_active else "inactive"
    filter_sigma_log = edge_filter_sigma if filter_active else "inactive"
    aux_summary_parts = [
        "version={}".format(aux_loss_version),
        "enabled={}".format(str(auxiliary_enabled).lower()),
        "edge_filter_mode={}".format(edge_filter_mode),
        "filter_active={}".format(str(filter_active).lower()),
        "edge_filter_kernel={}".format(filter_kernel_log),
        "edge_filter_sigma={}".format(filter_sigma_log),
        "lambda_thermal={:g}".format(opt.lambda_thermal),
        "lambda_edge={:g}".format(opt.lambda_edge),
        "lambda_smooth={:g}".format(opt.lambda_smooth),
    ]
    if aux_loss_version == "legacy" and auxiliary_enabled:
        aux_summary_parts.extend(
            (
                "noise_beta={:g}".format(opt.noise_beta),
                "edge_gamma={:g}".format(opt.edge_gamma),
            )
        )
    aux_summary = " ".join(aux_summary_parts)
    print("[AUX] " + aux_summary, flush=True)
    if tb_writer:
        tb_writer.add_text("config/auxiliary_loss", aux_summary, 0)
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
    thermal_loss_enabled = auxiliary_enabled
    for iteration in range(1, opt.iterations + 1):
        if iteration == 1:
            raise_if_stop_requested(
                stop_file, "training", audit_state["last_completed_iteration"]
            )
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
        audit_state["camera_hasher"].update(
            str(viewpoint_cam.image_name).encode("utf-8") + b"\0"
        )
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
        gd_target_p = gd_controller.rate(iteration)
        gd_sample = None
        gd_base_opacity = None
        training_opacity_override = None
        if gd_target_p > 0.0:
            gd_base_opacity = gaussians.get_opacity
            gd_sample = gd_controller.apply(
                gd_base_opacity,
                iteration,
                is_training_render=True,
            )
            training_opacity_override = gd_sample.opacity
        render_pkg_re = render(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            d_rgb,
            training_opacity_override=training_opacity_override,
        )
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
            baseline_loss = (1.0 - opt.lambda_dssim -0.2) * (Ll1) + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + 0.2*c_loss*max(1-iteration/5000,0)
        else:
            baseline_loss = (1.0 - opt.lambda_dssim) * (Ll1) + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss = baseline_loss
        thermal_terms = None
        thermal_extra = None
        if thermal_loss_enabled:
            thermal_extra, thermal_terms = thermal_physics_loss(
                image,
                gt_image,
                lambda_thermal=opt.lambda_thermal,
                lambda_edge=opt.lambda_edge,
                lambda_smooth=opt.lambda_smooth,
                noise_beta=opt.noise_beta,
                edge_gamma=opt.edge_gamma,
                aux_loss_version=aux_loss_version,
                edge_filter_mode=edge_filter_mode,
                edge_filter_kernel=edge_filter_kernel,
                edge_filter_sigma=edge_filter_sigma,
            )
            loss = loss + thermal_extra
        loss.backward()

        iter_end.record()
        finite_loss = torch.isfinite(loss.detach())
        if hasattr(torch, "_assert_async"):
            torch._assert_async(
                finite_loss,
                "non-finite training loss at iteration {}".format(iteration),
            )
        elif not bool(finite_loss.item()):
            raise FloatingPointError(
                "non-finite training loss at iteration {}".format(iteration)
            )

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            elapsed_ms = iter_start.elapsed_time(iter_end)
            total_elapsed_ms += elapsed_ms
            progress_bar.update(1)
            live_due = (
                iteration == 1
                or iteration % tb_log_interval == 0
                or iteration == opt.iterations
            )
            audit_due = iteration % log_interval == 0 or iteration == opt.iterations
            scalar_values = None
            if live_due or audit_due:
                edge_tensor = (
                    thermal_terms["edge"]
                    if opt.lambda_edge > 0 and thermal_terms is not None
                    else baseline_loss.new_zeros(())
                )
                packed = torch.stack(
                    (
                        loss.detach(),
                        baseline_loss.detach(),
                        edge_tensor.detach(),
                        (edge_tensor * opt.lambda_edge).detach(),
                    )
                ).cpu().tolist()
                scalar_values = {
                    "loss_total": packed[0],
                    "loss_base": packed[1],
                    "edge_raw": packed[2],
                    "edge_weighted": packed[3],
                }

            if live_due:
                live_row = {
                    "iteration": iteration,
                    "timestamp_utc": utc_now(),
                    "phase": "train",
                    "train_seed": train_seed,
                    **scalar_values,
                    "lambda_edge": opt.lambda_edge,
                    "gd_target_p": gd_target_p,
                }
                _append_live_loss_log(live_loss_path, live_row)
                if tb_writer:
                    for tag, field in (
                        ("train/loss_total", "loss_total"),
                        ("train/loss_base", "loss_base"),
                        ("train/edge_raw", "edge_raw"),
                        ("train/edge_weighted", "edge_weighted"),
                    ):
                        tb_writer.add_scalar(tag, scalar_values[field], iteration)
                    tb_writer.add_scalar("train/lambda_edge", opt.lambda_edge, iteration)
                    tb_writer.add_scalar("gd/target_p", gd_target_p, iteration)
                    if iteration == 1:
                        tb_writer.flush()

            if audit_due:
                peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                average_ms = total_elapsed_ms / iteration
                terms = thermal_terms or {}
                raw_thermal, weighted_thermal = _raw_and_weighted_term(
                    terms, "thermal", opt.lambda_thermal
                )
                raw_edge = scalar_values["edge_raw"] if opt.lambda_edge > 0 else "disabled"
                weighted_edge = scalar_values["edge_weighted"]
                raw_smooth, weighted_smooth = _raw_and_weighted_term(
                    terms, "smooth", opt.lambda_smooth
                )
                if gd_sample is None:
                    gd_realized_drop_fraction = 0.0
                    gd_compensation = 1.0
                    opacity_base_max = "disabled"
                    compensated_opacity_gt1_fraction = "disabled"
                else:
                    gd_stats = torch.stack(
                        (
                            1.0 - gd_sample.keep_mask.float().mean(),
                            gd_base_opacity.detach().amax(),
                            (gd_sample.opacity.detach() > 1.0).float().mean(),
                        )
                    ).cpu().tolist()
                    gd_realized_drop_fraction = gd_stats[0]
                    opacity_base_max = gd_stats[1]
                    compensated_opacity_gt1_fraction = gd_stats[2]
                    gd_compensation = gd_sample.compensation
                component_row = {
                    "iteration": iteration,
                    "train_seed": train_seed,
                    "gd_seed": gd_controller.seed,
                    "aux_loss_version": aux_loss_version,
                    "edge_filter_mode": edge_filter_mode,
                    "filter_active": str(filter_active).lower(),
                    "edge_filter_kernel": filter_kernel_log,
                    "edge_filter_sigma": filter_sigma_log,
                    "lambda_thermal": opt.lambda_thermal,
                    "lambda_edge": opt.lambda_edge,
                    "lambda_smooth": opt.lambda_smooth,
                    "lambda_detail": 0.0,
                    "detail_status": "disabled",
                    "L_baseline": scalar_values["loss_base"],
                    "L_thermal_raw": raw_thermal,
                    "L_edge_raw": raw_edge,
                    "L_smooth_raw": raw_smooth,
                    "weighted_L_thermal": weighted_thermal,
                    "weighted_L_edge": weighted_edge,
                    "weighted_L_smooth": weighted_smooth,
                    "L_total": scalar_values["loss_total"],
                    "gd_target_p": gd_target_p,
                    "gd_realized_drop_fraction": gd_realized_drop_fraction,
                    "gd_compensation": gd_compensation,
                    "opacity_base_max": opacity_base_max,
                    "compensated_opacity_gt1_fraction": compensated_opacity_gt1_fraction,
                    "Gaussian_count": gaussians.get_xyz.shape[0],
                    "avg_ms": average_ms,
                    "peak_cuda_mb": peak_memory_mb,
                }
                _append_loss_component_log(loss_component_path, component_row)
                progress_bar.set_postfix(
                    {"loss": "{:.6f}".format(component_row["L_total"])}
                )
                print(
                    "[ITER {}] total={:.6f} edge={} gd_p={:.4f} points={} avg_ms={:.2f} peak_mib={:.1f}".format(
                        iteration,
                        component_row["L_total"],
                        raw_edge if raw_edge == "disabled" else "{:.6f}".format(raw_edge),
                        gd_target_p,
                        gaussians.get_xyz.shape[0],
                        average_ms,
                        peak_memory_mb,
                    ),
                    flush=True,
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        "gd/realized_drop_fraction",
                        gd_realized_drop_fraction,
                        iteration,
                    )
                    tb_writer.add_scalar("gd/compensation", gd_compensation, iteration)
                    tb_writer.add_scalar(
                        "model/gaussian_count", gaussians.get_xyz.shape[0], iteration
                    )
                    tb_writer.add_scalar("system/core_avg_ms", average_ms, iteration)
                    tb_writer.add_scalar(
                        "system/cuda_allocated_peak_mib", peak_memory_mb, iteration
                    )

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])

            # Log and save
            if iteration in testing_iterations:
                write_run_status(
                    dataset.model_path,
                    status="running",
                    phase="val_monitor",
                    last_completed_iteration=audit_state["last_completed_iteration"],
                )
            cur_psnr = training_report(tb_writer, iteration,
                                       testing_iterations, scene, render, (pipe, background), ATF,
                                       TCM, dataset.load2gpu_on_the_fly,
                                       dataset.evaluation_partition if dataset.dataset_manifest else "test",
                                       stop_file=stop_file,
                                       last_completed_iteration=audit_state["last_completed_iteration"])
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("[ITER {}] saving model".format(iteration), flush=True)
                if tb_writer:
                    tb_writer.flush()
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

            audit_state["last_completed_iteration"] = iteration
            if iteration in saving_iterations:
                _write_gd_rng_audit(
                    dataset.model_path, gd_controller, audit_state, "running"
                )
                if tb_writer:
                    tb_writer.flush()
            if iteration % 50 == 0 or iteration == opt.iterations:
                write_run_status(
                    dataset.model_path,
                    status="running",
                    phase="training",
                    last_completed_iteration=iteration,
                )
                raise_if_stop_requested(stop_file, "training", iteration)

    progress_bar.close()
    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration), flush=True)
    return {"best_psnr": best_psnr, "best_iteration": best_iteration}


def prepare_output_and_logger(args, run_config=None, flush_secs=5):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
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
        tb_writer = SummaryWriter(args.model_path, max_queue=100, flush_secs=flush_secs)
    return tb_writer


def training_report(tb_writer, iteration, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, ATF, TCM, load2gpu_on_the_fly, evaluation_partition="test",
                    stop_file=None, last_completed_iteration=0):
    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        raise_if_stop_requested(
            stop_file, "val_monitor", last_completed_iteration
        )
        torch.cuda.empty_cache()
        validation_configs = ({'name': evaluation_partition, 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                for idx, viewpoint in enumerate(config['cameras']):
                    raise_if_stop_requested(
                        stop_file, "val_monitor", last_completed_iteration
                    )
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
                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == evaluation_partition or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer and config['name'] == evaluation_partition:
                    tb_writer.add_scalar('val_monitor/l1', l1_test, iteration)
                    tb_writer.add_scalar('val_monitor/psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.flush()
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
        default=500,
        help="write the detailed audit log at this iteration interval (default: 500)",
    )
    parser.add_argument(
        "--tb_log_interval",
        type=int,
        default=50,
        help="write live TensorBoard scalars at this iteration interval (default: 50)",
    )
    parser.add_argument(
        "--tb_flush_secs",
        type=int,
        default=5,
        help="SummaryWriter asynchronous flush interval in seconds",
    )
    parser.add_argument(
        "--stop_file",
        type=str,
        default=None,
        help="absolute or project-relative sticky stop request file",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    dataset = lp.extract(args)
    optimization = op.extract(args)
    pipeline = pp.extract(args)

    print("Optimizing " + args.model_path, flush=True)

    # Initialize system state (RNG)
    # Keep milestone messages visible; --quiet only disables the live progress bar.
    safe_state(False, seed=args.seed)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    try:
        training(
            dataset,
            optimization,
            pipeline,
            args.test_iterations,
            args.save_iterations,
            run_config=args,
            log_interval=args.log_interval,
            tb_log_interval=args.tb_log_interval,
            tb_flush_secs=args.tb_flush_secs,
            stop_file=args.stop_file,
            quiet=args.quiet,
        )
    except UserStopRequested as exc:
        print(
            "[INTERRUPTED] phase={} last_completed_iteration={} stop_reason=user_request".format(
                exc.phase, exc.last_completed_iteration
            ),
            flush=True,
        )
        sys.exit(INTERRUPTED_EXIT_CODE)

    # All done
    print("Training complete.", flush=True)
