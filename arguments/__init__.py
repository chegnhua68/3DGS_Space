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

from argparse import ArgumentParser, Namespace
import ast
import math
import sys
import os
from collections.abc import Mapping


class GroupParams:
    pass


def _option_value(options, name, default):
    if isinstance(options, Mapping):
        return options.get(name, default)
    return getattr(options, name, default)


def validate_aux_loss_options(options) -> None:
    """Validate resolved auxiliary-loss options without touching CUDA state."""

    version = _option_value(options, "aux_loss_version", "legacy")
    if version not in ("legacy", "filtered_edge"):
        raise ValueError("aux_loss_version must be 'legacy' or 'filtered_edge'")
    edge_filter_mode = _option_value(options, "edge_filter_mode", "gaussian")
    if edge_filter_mode not in ("gaussian", "identity"):
        raise ValueError("edge_filter_mode must be 'gaussian' or 'identity'")

    numeric_defaults = {
        "lambda_thermal": 0.0,
        "lambda_edge": 0.0,
        "lambda_smooth": 0.0,
        "noise_beta": 5.0,
        "edge_gamma": 3.0,
    }
    resolved = {}
    for name, default in numeric_defaults.items():
        value = _option_value(options, name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("{} must be a real scalar".format(name))
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("{} must be finite and non-negative".format(name))
        resolved[name] = value

    kernel_size = _option_value(options, "edge_filter_kernel", 5)
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, int):
        raise TypeError("edge_filter_kernel must be an integer")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("edge_filter_kernel must be a positive odd integer")

    sigma = _option_value(options, "edge_filter_sigma", 1.0)
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float)):
        raise TypeError("edge_filter_sigma must be a real scalar")
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("edge_filter_sigma must be finite and positive")

    if version == "filtered_edge" and (
        resolved["lambda_thermal"] != 0 or resolved["lambda_smooth"] != 0
    ):
        raise ValueError(
            "filtered_edge requires lambda_thermal == 0 and lambda_smooth == 0"
        )


def evaluation_output_name(dataset_manifest, evaluation_partition):
    """Return the on-disk name for the selected held-out partition."""

    if not dataset_manifest:
        return "test"
    if evaluation_partition not in ("val", "test"):
        raise ValueError("evaluation_partition must be 'val' or 'test'")
    return evaluation_partition


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self.dataset_manifest = ""
        self.split_manifest = ""
        self.degradation_manifest = ""
        self.evaluation_partition = "test"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.load2gpu_on_the_fly = False
        self.is_blender = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        for name in ("dataset_manifest", "split_manifest", "degradation_manifest"):
            value = getattr(g, name, None)
            if value:
                setattr(g, name, os.path.abspath(value))
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = True
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.warm_up = 0
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.TCM_lr_max_steps = 30_000
        self.ATF_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.001
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_thermal = 0.0
        self.lambda_edge = 0.0
        self.lambda_smooth = 0.0
        self.noise_beta = 5.0
        self.edge_gamma = 3.0
        self.aux_loss_version = "legacy"
        self.edge_filter_mode = "gaussian"
        self.edge_filter_kernel = 5
        self.edge_filter_sigma = 1.0
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        super().__init__(parser, "Optimization Parameters")

    def extract(self, args):
        group = super().extract(args)
        validate_aux_loss_options(group)
        return group


def _parse_cfg_namespace(value):
    """Parse the project's Namespace(...) config format without executing it."""
    try:
        expression = ast.parse(value, mode="eval").body
    except SyntaxError as exc:
        raise ValueError("cfg_args is not valid Python syntax") from exc
    if not (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Name)
        and expression.func.id == "Namespace"
        and not expression.args
    ):
        raise ValueError("cfg_args must contain a single Namespace(...) expression")
    values = {}
    for keyword in expression.keywords:
        if keyword.arg is None or keyword.arg in values:
            raise ValueError("cfg_args contains invalid or duplicate fields")
        try:
            values[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "cfg_args field {!r} is not a literal".format(keyword.arg)
            ) from exc
    return Namespace(**values)


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath, encoding="utf-8") as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = _parse_cfg_namespace(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
