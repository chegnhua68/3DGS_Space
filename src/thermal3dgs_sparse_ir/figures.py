"""Reproducible paper figures driven by an explicit JSON specification.

The module intentionally imports plotting dependencies only when figure
generation starts.  This keeps manifest and CLI validation usable in minimal
training environments while providing a clear error when plotting support is
missing.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union


FIGURE_SPEC_VERSION = 1
FIGURE_SPEC_KIND = "thermal-figure-spec"
FIGURE_MANIFEST_KIND = "thermal-figure-manifest"
GENERATOR_VERSION = "1"
LOSSLESS_IMAGE_SUFFIXES = frozenset(
    (".png", ".tif", ".tiff", ".bmp", ".pgm", ".ppm", ".pbm", ".pnm")
)
LOSSY_IMAGE_SUFFIXES = frozenset((".jpg", ".jpeg", ".jpe", ".webp"))
METRIC_FIELDS = frozenset(
    ("psnr", "ssim", "lpips", "t_mae", "e_mae", "gradient_preservation", "roi_mae")
)
_DEFAULT_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
)
_DEFAULT_MARKERS = ("o", "s", "^", "D", "v", "P", "X")
_LEGEND_LOCATIONS = frozenset(
    (
        "best",
        "upper right",
        "upper left",
        "lower left",
        "lower right",
        "right",
        "center left",
        "center right",
        "lower center",
        "upper center",
        "center",
    )
)


class FigureSpecError(ValueError):
    """Raised when a figure specification or one of its inputs is invalid."""


class FigureDependencyError(RuntimeError):
    """Raised when optional plotting dependencies cannot be imported."""


def _duplicate_rejecting_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FigureSpecError("duplicate JSON object key: {!r}".format(key))
        result[key] = value
    return result


def read_figure_spec(path: Union[os.PathLike, str]) -> Dict[str, Any]:
    """Read, strictly validate, and resolve defaults in a figure spec."""

    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as stream:
            raw = json.load(stream, object_pairs_hook=_duplicate_rejecting_object)
    except FigureSpecError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FigureSpecError("could not read figure spec {}: {}".format(source, exc))
    return validate_figure_spec(raw)


def _object(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FigureSpecError("{} must be an object".format(location))
    return value


def _keys(
    value: Mapping[str, Any],
    required: Set[str],
    optional: Set[str],
    location: str,
) -> None:
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing:
        raise FigureSpecError(
            "{} is missing field(s): {}".format(location, ", ".join(sorted(missing)))
        )
    if unknown:
        raise FigureSpecError(
            "{} has unknown field(s): {}".format(location, ", ".join(sorted(unknown)))
        )


def _string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FigureSpecError("{} must be a non-empty string".format(location))
    return value


def _number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FigureSpecError("{} must be a number".format(location))
    result = float(value)
    if not math.isfinite(result):
        raise FigureSpecError("{} must be finite".format(location))
    return result


def _positive_integer(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise FigureSpecError("{} must be a positive integer".format(location))
    return value


def _figsize(value: Any, default: Sequence[float], location: str) -> List[float]:
    if value is None:
        return [float(default[0]), float(default[1])]
    if not isinstance(value, list) or len(value) != 2:
        raise FigureSpecError("{} must contain [width, height]".format(location))
    size = [_number(value[0], location + "[0]"), _number(value[1], location + "[1]")]
    if size[0] <= 0 or size[1] <= 0:
        raise FigureSpecError("{} values must be positive".format(location))
    return size


def _limits(value: Any, location: str) -> Optional[List[float]]:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise FigureSpecError("{} must contain [minimum, maximum]".format(location))
    limits = [_number(value[0], location + "[0]"), _number(value[1], location + "[1]")]
    if limits[0] >= limits[1]:
        raise FigureSpecError("{} minimum must be less than maximum".format(location))
    return limits


def _output_name(value: Any, location: str) -> str:
    text = _string(value, location)
    if "\\" in text:
        raise FigureSpecError("{} must use forward slashes".format(location))
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or ".." in posix.parts
        or posix.as_posix() != text
        or posix.suffix.lower() != ".png"
    ):
        raise FigureSpecError(
            "{} must be a normalized relative .png path".format(location)
        )
    return text


def _filename(value: Any, location: str) -> str:
    text = _string(value, location)
    if Path(text).name != text or "\\" in text or text in (".", ".."):
        raise FigureSpecError("{} must be a plain filename".format(location))
    if Path(text).suffix.lower() not in LOSSLESS_IMAGE_SUFFIXES:
        raise FigureSpecError("{} must name a supported lossless image".format(location))
    return text


def _validate_image_section(raw: Any) -> Dict[str, Any]:
    section = _object(raw, "image_comparison")
    _keys(
        section,
        {"filenames", "ground_truth_label", "methods", "qualitative", "absolute_error", "edge"},
        set(),
        "image_comparison",
    )
    filenames_raw = section["filenames"]
    if not isinstance(filenames_raw, list) or not filenames_raw:
        raise FigureSpecError("image_comparison.filenames must be a non-empty array")
    filenames = [
        _filename(value, "image_comparison.filenames[{}]".format(index))
        for index, value in enumerate(filenames_raw)
    ]
    if len(filenames) != len(set(filenames)):
        raise FigureSpecError("image_comparison.filenames must be unique")

    methods_raw = section["methods"]
    if not isinstance(methods_raw, list) or not methods_raw:
        raise FigureSpecError("image_comparison.methods must be a non-empty array")
    methods: List[Dict[str, Any]] = []
    labels: Set[str] = set()
    for index, raw_method in enumerate(methods_raw):
        location = "image_comparison.methods[{}]".format(index)
        method = _object(raw_method, location)
        _keys(method, {"label", "method_dir"}, set(), location)
        label = _string(method["label"], location + ".label")
        if label in labels:
            raise FigureSpecError("image comparison method labels must be unique")
        labels.add(label)
        methods.append(
            {"label": label, "method_dir": _string(method["method_dir"], location + ".method_dir")}
        )

    qualitative = _object(section["qualitative"], "image_comparison.qualitative")
    _keys(qualitative, {"output"}, {"figsize"}, "image_comparison.qualitative")
    qualitative_normalized = {
        "output": _output_name(qualitative["output"], "image_comparison.qualitative.output"),
        "figsize": _figsize(
            qualitative.get("figsize"),
            (3.0 * (len(methods) + 1), 2.7 * len(filenames)),
            "image_comparison.qualitative.figsize",
        ),
    }

    absolute_error = _object(section["absolute_error"], "image_comparison.absolute_error")
    _keys(
        absolute_error,
        {"output", "cmap", "vmin", "vmax", "colorbar_label"},
        {"figsize"},
        "image_comparison.absolute_error",
    )
    error_vmin = _number(absolute_error["vmin"], "image_comparison.absolute_error.vmin")
    error_vmax = _number(absolute_error["vmax"], "image_comparison.absolute_error.vmax")
    if error_vmin >= error_vmax:
        raise FigureSpecError("absolute_error.vmin must be less than vmax")
    absolute_error_normalized = {
        "output": _output_name(absolute_error["output"], "image_comparison.absolute_error.output"),
        "cmap": _string(absolute_error["cmap"], "image_comparison.absolute_error.cmap"),
        "vmin": error_vmin,
        "vmax": error_vmax,
        "colorbar_label": _string(
            absolute_error["colorbar_label"], "image_comparison.absolute_error.colorbar_label"
        ),
        "figsize": _figsize(
            absolute_error.get("figsize"),
            (3.0 * len(methods), 2.7 * len(filenames)),
            "image_comparison.absolute_error.figsize",
        ),
    }

    edge = _object(section["edge"], "image_comparison.edge")
    _keys(
        edge,
        {"output", "operator", "cmap", "vmin", "vmax", "colorbar_label"},
        {"figsize"},
        "image_comparison.edge",
    )
    if edge["operator"] != "sobel":
        raise FigureSpecError("image_comparison.edge.operator must be 'sobel'")
    edge_vmin = _number(edge["vmin"], "image_comparison.edge.vmin")
    edge_vmax = _number(edge["vmax"], "image_comparison.edge.vmax")
    if edge_vmin >= edge_vmax:
        raise FigureSpecError("edge.vmin must be less than vmax")
    edge_normalized = {
        "output": _output_name(edge["output"], "image_comparison.edge.output"),
        "operator": "sobel",
        "cmap": _string(edge["cmap"], "image_comparison.edge.cmap"),
        "vmin": edge_vmin,
        "vmax": edge_vmax,
        "colorbar_label": _string(edge["colorbar_label"], "image_comparison.edge.colorbar_label"),
        "figsize": _figsize(
            edge.get("figsize"),
            (3.0 * (len(methods) + 1), 2.7 * len(filenames)),
            "image_comparison.edge.figsize",
        ),
    }
    return {
        "filenames": filenames,
        "ground_truth_label": _string(
            section["ground_truth_label"], "image_comparison.ground_truth_label"
        ),
        "methods": methods,
        "qualitative": qualitative_normalized,
        "absolute_error": absolute_error_normalized,
        "edge": edge_normalized,
    }


def _validate_curve(raw: Any, location: str) -> Dict[str, Any]:
    curve = _object(raw, location)
    _keys(
        curve,
        {"output", "title", "metrics_csv", "y_field", "x_label", "y_label", "series"},
        {"x_limits", "y_limits", "figsize", "legend_location"},
        location,
    )
    y_field = _string(curve["y_field"], location + ".y_field")
    if y_field not in METRIC_FIELDS:
        raise FigureSpecError(
            "{}.y_field must be one of {}".format(location, ", ".join(sorted(METRIC_FIELDS)))
        )
    series_raw = curve["series"]
    if not isinstance(series_raw, list) or not series_raw:
        raise FigureSpecError("{}.series must be a non-empty array".format(location))
    normalized_series: List[Dict[str, Any]] = []
    labels: Set[str] = set()
    for series_index, raw_series in enumerate(series_raw):
        series_location = "{}.series[{}]".format(location, series_index)
        series = _object(raw_series, series_location)
        _keys(
            series,
            {"label", "points"},
            {"color", "marker", "linestyle"},
            series_location,
        )
        label = _string(series["label"], series_location + ".label")
        if label in labels:
            raise FigureSpecError("{}.series labels must be unique".format(location))
        labels.add(label)
        points_raw = series["points"]
        if not isinstance(points_raw, list) or len(points_raw) < 2:
            raise FigureSpecError("{}.points must contain at least two points".format(series_location))
        points: List[Dict[str, Any]] = []
        previous_x: Optional[float] = None
        experiments: Set[str] = set()
        for point_index, raw_point in enumerate(points_raw):
            point_location = "{}.points[{}]".format(series_location, point_index)
            point = _object(raw_point, point_location)
            _keys(point, {"x", "experiment"}, set(), point_location)
            x_value = _number(point["x"], point_location + ".x")
            experiment = _string(point["experiment"], point_location + ".experiment")
            if previous_x is not None and x_value <= previous_x:
                raise FigureSpecError(
                    "{}.points x values must be strictly increasing in spec order".format(series_location)
                )
            if experiment in experiments:
                raise FigureSpecError("{}.points experiments must be unique".format(series_location))
            previous_x = x_value
            experiments.add(experiment)
            points.append({"x": x_value, "experiment": experiment})
        normalized_series.append(
            {
                "label": label,
                "color": _string(
                    series.get("color", _DEFAULT_COLORS[series_index % len(_DEFAULT_COLORS)]),
                    series_location + ".color",
                ),
                "marker": _string(
                    series.get("marker", _DEFAULT_MARKERS[series_index % len(_DEFAULT_MARKERS)]),
                    series_location + ".marker",
                ),
                "linestyle": _string(series.get("linestyle", "-"), series_location + ".linestyle"),
                "points": points,
            }
        )
    return {
        "output": _output_name(curve["output"], location + ".output"),
        "title": _string(curve["title"], location + ".title"),
        "metrics_csv": _string(curve["metrics_csv"], location + ".metrics_csv"),
        "y_field": y_field,
        "x_label": _string(curve["x_label"], location + ".x_label"),
        "y_label": _string(curve["y_label"], location + ".y_label"),
        "series": normalized_series,
        "x_limits": _limits(curve.get("x_limits"), location + ".x_limits"),
        "y_limits": _limits(curve.get("y_limits"), location + ".y_limits"),
        "figsize": _figsize(curve.get("figsize"), (6.4, 4.2), location + ".figsize"),
        "legend_location": _string(curve.get("legend_location", "best"), location + ".legend_location"),
    }


def validate_figure_spec(value: Any) -> Dict[str, Any]:
    """Validate a JSON-compatible object and return a defaults-resolved copy."""

    spec = _object(value, "figure spec")
    _keys(
        spec,
        {
            "schema_version",
            "kind",
            "output_dir",
            "image_comparison",
            "sparse_view_curve",
            "degradation_curve",
        },
        {"dpi"},
        "figure spec",
    )
    if (
        isinstance(spec["schema_version"], bool)
        or not isinstance(spec["schema_version"], int)
        or spec["schema_version"] != FIGURE_SPEC_VERSION
    ):
        raise FigureSpecError(
            "schema_version must be {}, got {!r}".format(FIGURE_SPEC_VERSION, spec["schema_version"])
        )
    if spec["kind"] != FIGURE_SPEC_KIND:
        raise FigureSpecError("kind must be {!r}".format(FIGURE_SPEC_KIND))
    dpi = _positive_integer(spec.get("dpi", 160), "dpi")
    if dpi < 72 or dpi > 600:
        raise FigureSpecError("dpi must be between 72 and 600")
    normalized = {
        "schema_version": FIGURE_SPEC_VERSION,
        "kind": FIGURE_SPEC_KIND,
        "output_dir": _string(spec["output_dir"], "output_dir"),
        "dpi": dpi,
        "image_comparison": _validate_image_section(spec["image_comparison"]),
        "sparse_view_curve": _validate_curve(spec["sparse_view_curve"], "sparse_view_curve"),
        "degradation_curve": _validate_curve(spec["degradation_curve"], "degradation_curve"),
    }
    for curve_name in ("sparse_view_curve", "degradation_curve"):
        legend_location = normalized[curve_name]["legend_location"]
        if legend_location not in _LEGEND_LOCATIONS:
            raise FigureSpecError(
                "{}.legend_location must be one of {}".format(
                    curve_name, ", ".join(sorted(_LEGEND_LOCATIONS))
                )
            )
    outputs = [
        normalized["image_comparison"]["qualitative"]["output"],
        normalized["image_comparison"]["absolute_error"]["output"],
        normalized["image_comparison"]["edge"]["output"],
        normalized["sparse_view_curve"]["output"],
        normalized["degradation_curve"]["output"],
    ]
    if len(outputs) != len(set(outputs)):
        raise FigureSpecError("all five figure output paths must be unique")
    return normalized


def _plotting_dependencies() -> Tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
        from PIL import Image
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except (ImportError, OSError) as exc:
        raise FigureDependencyError(
            "figure generation requires NumPy, Pillow, and Matplotlib; "
            "install them in the active Python environment: {}".format(exc)
        )
    return np, Image, matplotlib, plt


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _resolve(base_dir: Path, text: str) -> Path:
    path = Path(text)
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _list_images(directory: Path) -> Dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError("image directory does not exist: {}".format(directory))
    result: Dict[str, Path] = {}
    lossy: List[str] = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in LOSSY_IMAGE_SUFFIXES:
            lossy.append(path.name)
        elif suffix in LOSSLESS_IMAGE_SUFFIXES:
            result[path.name] = path
    if lossy:
        raise FigureSpecError(
            "lossy images are not accepted in {}: {}".format(directory, ", ".join(sorted(lossy)))
        )
    if not result:
        raise FigureSpecError("no supported lossless images found in {}".format(directory))
    return result


def _inventory_digest(entries: Sequence[Tuple[str, str]]) -> str:
    payload = json.dumps(
        [{"path": name, "sha256": digest} for name, digest in sorted(entries)],
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_method_inputs(
    methods: Sequence[Mapping[str, Any]], base_dir: Path, selected_names: Sequence[str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Set[Path]]:
    prepared: List[Dict[str, Any]] = []
    manifest_inputs: List[Dict[str, Any]] = []
    input_paths: Set[Path] = set()
    reference_names: Optional[Set[str]] = None
    reference_gt_hashes: Optional[Dict[str, str]] = None
    for method in methods:
        method_dir = _resolve(base_dir, method["method_dir"])
        renders = _list_images(method_dir / "renders")
        ground_truth = _list_images(method_dir / "gt")
        render_names = set(renders)
        gt_names = set(ground_truth)
        if render_names != gt_names:
            missing_renders = sorted(gt_names - render_names)
            missing_gt = sorted(render_names - gt_names)
            raise FigureSpecError(
                "render/gt filename mismatch for {!r} (missing renders: {}; missing gt: {})".format(
                    method["label"], ", ".join(missing_renders) or "none", ", ".join(missing_gt) or "none"
                )
            )
        missing_selected = sorted(set(selected_names) - render_names)
        if missing_selected:
            raise FigureSpecError(
                "method {!r} is missing selected image(s): {}".format(
                    method["label"], ", ".join(missing_selected)
                )
            )
        if reference_names is not None and render_names != reference_names:
            raise FigureSpecError(
                "method {!r} has a different paired filename set".format(method["label"])
            )
        file_hashes: Dict[Path, str] = {}
        entries: List[Tuple[str, str]] = []
        for role, images in (("renders", renders), ("gt", ground_truth)):
            for name in sorted(images):
                path = images[name].resolve()
                digest = _sha256_file(path)
                file_hashes[path] = digest
                input_paths.add(path)
                entries.append((role + "/" + name, digest))
        gt_hashes = {name: file_hashes[path.resolve()] for name, path in ground_truth.items()}
        if reference_gt_hashes is not None and gt_hashes != reference_gt_hashes:
            differing = sorted(
                name for name in render_names if gt_hashes.get(name) != reference_gt_hashes.get(name)
            )
            raise FigureSpecError(
                "ground truth content differs across methods: {}".format(", ".join(differing))
            )
        reference_names = render_names
        reference_gt_hashes = gt_hashes
        prepared.append(
            {
                "label": method["label"],
                "method_dir": method_dir,
                "renders": renders,
                "gt": ground_truth,
            }
        )
        manifest_inputs.append(
            {
                "kind": "paired-image-inventory",
                "label": method["label"],
                "path": method_dir.as_posix(),
                "file_count": len(entries),
                "sha256": _inventory_digest(entries),
            }
        )
    return prepared, manifest_inputs, input_paths


def _read_metric_rows(path: Path, y_field: str) -> Dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError("metrics CSV does not exist: {}".format(path))
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            if fields is None:
                raise FigureSpecError("metrics CSV has no header: {}".format(path))
            if len(fields) != len(set(fields)):
                raise FigureSpecError("metrics CSV has duplicate columns: {}".format(path))
            missing = {"experiment", y_field} - set(fields)
            if missing:
                raise FigureSpecError(
                    "metrics CSV {} is missing column(s): {}".format(path, ", ".join(sorted(missing)))
                )
            rows: Dict[str, float] = {}
            for line_number, row in enumerate(reader, start=2):
                if None in row:
                    raise FigureSpecError(
                        "metrics CSV {} line {} has more fields than the header".format(
                            path, line_number
                        )
                    )
                experiment = (row.get("experiment") or "").strip()
                if not experiment:
                    raise FigureSpecError(
                        "metrics CSV {} line {} has an empty experiment".format(path, line_number)
                    )
                if experiment in rows:
                    raise FigureSpecError(
                        "metrics CSV {} repeats experiment {!r}".format(path, experiment)
                    )
                raw_value = (row.get(y_field) or "").strip()
                try:
                    value = float(raw_value)
                except ValueError:
                    raise FigureSpecError(
                        "metrics CSV {} experiment {!r} has non-numeric {}={!r}".format(
                            path, experiment, y_field, raw_value
                        )
                    )
                if not math.isfinite(value):
                    raise FigureSpecError(
                        "metrics CSV {} experiment {!r} has non-finite {}".format(
                            path, experiment, y_field
                        )
                    )
                rows[experiment] = value
    except UnicodeError as exc:
        raise FigureSpecError("metrics CSV must be UTF-8: {} ({})".format(path, exc))
    return rows


class _ImageCache(object):
    def __init__(self, np_module: Any, image_module: Any):
        self._np = np_module
        self._Image = image_module
        self._cache: Dict[Path, Tuple[Any, Any]] = {}

    def load(self, path: Path) -> Tuple[Any, Any]:
        path = path.resolve()
        if path in self._cache:
            return self._cache[path]
        try:
            with self._Image.open(str(path)) as image:
                image.load()
                mode = image.mode
                if mode in ("1", "L"):
                    display = self._np.array(image.convert("L"), dtype=self._np.float32, copy=True) / 255.0
                    gray = display
                elif mode == "LA":
                    display = self._np.array(image.getchannel("L"), dtype=self._np.float32, copy=True) / 255.0
                    gray = display
                elif mode.startswith("I;16") or mode == "I":
                    display = self._np.array(image, dtype=self._np.float32, copy=True) / 65535.0
                    gray = display
                elif mode == "F":
                    display = self._np.array(image, dtype=self._np.float32, copy=True)
                    gray = display
                else:
                    display = self._np.array(image.convert("RGB"), dtype=self._np.float32, copy=True) / 255.0
                    gray = (
                        display[:, :, 0] * 0.299
                        + display[:, :, 1] * 0.587
                        + display[:, :, 2] * 0.114
                    )
        except (OSError, ValueError) as exc:
            raise FigureSpecError("could not load image {}: {}".format(path, exc))
        if display.ndim not in (2, 3) or display.shape[0] < 1 or display.shape[1] < 1:
            raise FigureSpecError("image has invalid dimensions: {}".format(path))
        if not bool(self._np.isfinite(display).all()):
            raise FigureSpecError("image contains non-finite values: {}".format(path))
        minimum = float(display.min())
        maximum = float(display.max())
        if minimum < -1e-6 or maximum > 1.0 + 1e-6:
            raise FigureSpecError(
                "image {} must stay in its fixed encoded range, got [{:.8g}, {:.8g}]".format(
                    path, minimum, maximum
                )
            )
        result = (display, gray)
        self._cache[path] = result
        return result


def _sobel_magnitude(gray: Any, np_module: Any) -> Any:
    padded = np_module.pad(gray, ((1, 1), (1, 1)), mode="edge")
    grad_x = (
        -padded[:-2, :-2]
        + padded[:-2, 2:]
        - 2.0 * padded[1:-1, :-2]
        + 2.0 * padded[1:-1, 2:]
        - padded[2:, :-2]
        + padded[2:, 2:]
    ) / 8.0
    grad_y = (
        -padded[:-2, :-2]
        - 2.0 * padded[:-2, 1:-1]
        - padded[:-2, 2:]
        + padded[2:, :-2]
        + 2.0 * padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 8.0
    return np_module.sqrt(grad_x * grad_x + grad_y * grad_y)


def _row_label(axis: Any, filename: str) -> None:
    axis.text(
        -0.04,
        0.5,
        filename,
        transform=axis.transAxes,
        fontsize=8,
        rotation=90,
        horizontalalignment="right",
        verticalalignment="center",
    )


def _save_figure(figure: Any, destination: Path, dpi: int, plt: Any) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + destination.name + ".", suffix=".tmp", dir=str(destination.parent)
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        figure.savefig(
            str(temporary_path),
            format="png",
            dpi=dpi,
            facecolor="white",
            metadata={"Software": "thermal3dgs_sparse_ir.figures/{}".format(GENERATOR_VERSION)},
        )
    except Exception:
        try:
            temporary_path.unlink()
        except OSError:
            pass
        raise
    finally:
        plt.close(figure)
    return temporary_path


def _render_qualitative(
    section: Mapping[str, Any], methods: Sequence[Mapping[str, Any]], cache: _ImageCache, plt: Any,
    destination: Path, dpi: int
) -> Path:
    names = section["filenames"]
    config = section["qualitative"]
    figure, axes = plt.subplots(
        len(names), len(methods) + 1, squeeze=False, figsize=config["figsize"], constrained_layout=True
    )
    for row, name in enumerate(names):
        gt_display, gt_gray = cache.load(methods[0]["gt"][name])
        axes[row, 0].imshow(gt_display, cmap="gray" if gt_display.ndim == 2 else None, vmin=0.0, vmax=1.0)
        axes[row, 0].set_axis_off()
        _row_label(axes[row, 0], name)
        for column, method in enumerate(methods, start=1):
            render_display, render_gray = cache.load(method["renders"][name])
            if render_gray.shape != gt_gray.shape:
                raise FigureSpecError(
                    "render/gt dimensions differ for method {!r}, image {!r}".format(method["label"], name)
                )
            axes[row, column].imshow(
                render_display, cmap="gray" if render_display.ndim == 2 else None, vmin=0.0, vmax=1.0
            )
            axes[row, column].set_axis_off()
    titles = [section["ground_truth_label"]] + [method["label"] for method in methods]
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=10)
    return _save_figure(figure, destination, dpi, plt)


def _render_error_maps(
    section: Mapping[str, Any], methods: Sequence[Mapping[str, Any]], cache: _ImageCache,
    matplotlib: Any, plt: Any, destination: Path, dpi: int
) -> Path:
    names = section["filenames"]
    config = section["absolute_error"]
    figure, axes = plt.subplots(
        len(names), len(methods), squeeze=False, figsize=config["figsize"], constrained_layout=True
    )
    image_artist = None
    for row, name in enumerate(names):
        _, gt_gray = cache.load(methods[0]["gt"][name])
        for column, method in enumerate(methods):
            _, render_gray = cache.load(method["renders"][name])
            if render_gray.shape != gt_gray.shape:
                raise FigureSpecError(
                    "render/gt dimensions differ for method {!r}, image {!r}".format(method["label"], name)
                )
            image_artist = axes[row, column].imshow(
                abs(render_gray - gt_gray), cmap=config["cmap"], vmin=config["vmin"], vmax=config["vmax"]
            )
            axes[row, column].set_axis_off()
        _row_label(axes[row, 0], name)
    for column, method in enumerate(methods):
        axes[0, column].set_title(method["label"], fontsize=10)
    if image_artist is None:
        raise FigureSpecError("absolute error figure has no images")
    figure.colorbar(
        image_artist,
        ax=axes.ravel().tolist(),
        label=config["colorbar_label"],
        fraction=0.025,
        pad=0.02,
    )
    return _save_figure(figure, destination, dpi, plt)


def _render_edges(
    section: Mapping[str, Any], methods: Sequence[Mapping[str, Any]], cache: _ImageCache,
    np_module: Any, plt: Any, destination: Path, dpi: int
) -> Path:
    names = section["filenames"]
    config = section["edge"]
    figure, axes = plt.subplots(
        len(names), len(methods) + 1, squeeze=False, figsize=config["figsize"], constrained_layout=True
    )
    image_artist = None
    for row, name in enumerate(names):
        _, gt_gray = cache.load(methods[0]["gt"][name])
        maps = [_sobel_magnitude(gt_gray, np_module)]
        for method in methods:
            _, render_gray = cache.load(method["renders"][name])
            if render_gray.shape != gt_gray.shape:
                raise FigureSpecError(
                    "render/gt dimensions differ for method {!r}, image {!r}".format(method["label"], name)
                )
            maps.append(_sobel_magnitude(render_gray, np_module))
        for column, edge_map in enumerate(maps):
            image_artist = axes[row, column].imshow(
                edge_map, cmap=config["cmap"], vmin=config["vmin"], vmax=config["vmax"]
            )
            axes[row, column].set_axis_off()
        _row_label(axes[row, 0], name)
    titles = [section["ground_truth_label"]] + [method["label"] for method in methods]
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=10)
    if image_artist is None:
        raise FigureSpecError("edge figure has no images")
    figure.colorbar(
        image_artist,
        ax=axes.ravel().tolist(),
        label=config["colorbar_label"],
        fraction=0.025,
        pad=0.02,
    )
    return _save_figure(figure, destination, dpi, plt)


def _render_curve(
    config: Mapping[str, Any], rows: Mapping[str, float], plt: Any, destination: Path, dpi: int
) -> Path:
    figure, axis = plt.subplots(1, 1, figsize=config["figsize"], constrained_layout=True)
    try:
        for series in config["series"]:
            missing = [
                point["experiment"] for point in series["points"] if point["experiment"] not in rows
            ]
            if missing:
                raise FigureSpecError(
                    "curve {!r} references missing experiment(s): {}".format(
                        config["title"], ", ".join(missing)
                    )
                )
            x_values = [point["x"] for point in series["points"]]
            y_values = [rows[point["experiment"]] for point in series["points"]]
            axis.plot(
                x_values,
                y_values,
                label=series["label"],
                color=series["color"],
                marker=series["marker"],
                linestyle=series["linestyle"],
                linewidth=1.8,
                markersize=5.5,
            )
        axis.set_title(config["title"])
        axis.set_xlabel(config["x_label"])
        axis.set_ylabel(config["y_label"])
        if config["x_limits"] is not None:
            axis.set_xlim(config["x_limits"])
        if config["y_limits"] is not None:
            axis.set_ylim(config["y_limits"])
        axis.grid(True, color="#D0D0D0", linewidth=0.7, alpha=0.8)
        axis.legend(loc=config["legend_location"], frameon=True)
    except Exception:
        plt.close(figure)
        raise
    return _save_figure(figure, destination, dpi, plt)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    serialized = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
        os.replace(temporary_name, str(path))
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            Path(temporary_name).unlink()
        except OSError:
            pass
        raise


def make_figures(
    spec_path: Union[os.PathLike, str],
    output_dir: Optional[Union[os.PathLike, str]] = None,
    overwrite: bool = False,
) -> Dict[str, Path]:
    """Generate five PNG figures and a hash-bearing figure manifest."""

    source_path = Path(spec_path).resolve()
    spec = read_figure_spec(source_path)
    base_dir = source_path.parent
    if output_dir is None:
        destination_dir = _resolve(base_dir, spec["output_dir"])
    else:
        override = Path(output_dir)
        destination_dir = override.resolve() if override.is_absolute() else (Path.cwd() / override).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)

    image_section = spec["image_comparison"]
    methods, manifest_inputs, input_image_paths = _validate_method_inputs(
        image_section["methods"], base_dir, image_section["filenames"]
    )
    curve_names = ("sparse_view_curve", "degradation_curve")
    curve_rows: Dict[str, Dict[str, float]] = {}
    csv_paths: Set[Path] = set()
    for curve_name in curve_names:
        config = spec[curve_name]
        csv_path = _resolve(base_dir, config["metrics_csv"])
        curve_rows[curve_name] = _read_metric_rows(csv_path, config["y_field"])
        csv_paths.add(csv_path)
        manifest_inputs.append(
            {
                "kind": "metrics-csv",
                "role": curve_name,
                "path": csv_path.as_posix(),
                "sha256": _sha256_file(csv_path),
            }
        )

    output_names = {
        "qualitative": image_section["qualitative"]["output"],
        "absolute_error": image_section["absolute_error"]["output"],
        "edge": image_section["edge"]["output"],
        "sparse_view_curve": spec["sparse_view_curve"]["output"],
        "degradation_curve": spec["degradation_curve"]["output"],
    }
    destinations = {key: destination_dir / PurePosixPath(name) for key, name in output_names.items()}
    manifest_path = destination_dir / "figure_manifest.json"
    escaped_outputs: List[Path] = []
    for path in list(destinations.values()) + [manifest_path]:
        try:
            path.resolve().relative_to(destination_dir)
        except ValueError:
            escaped_outputs.append(path)
    if escaped_outputs:
        raise FigureSpecError(
            "figure output resolves outside output_dir: {}".format(
                ", ".join(str(path) for path in escaped_outputs)
            )
        )
    input_paths = input_image_paths | csv_paths | {source_path}
    conflicts = [path for path in list(destinations.values()) + [manifest_path] if path.resolve() in input_paths]
    if conflicts:
        raise FigureSpecError(
            "figure output would overwrite an input: {}".format(", ".join(str(path) for path in conflicts))
        )
    existing = [path for path in list(destinations.values()) + [manifest_path] if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing figure output(s): {}".format(", ".join(str(path) for path in existing))
        )

    np_module, image_module, matplotlib, plt = _plotting_dependencies()
    for cmap_location, cmap_name in (
        ("image_comparison.absolute_error.cmap", image_section["absolute_error"]["cmap"]),
        ("image_comparison.edge.cmap", image_section["edge"]["cmap"]),
    ):
        try:
            matplotlib.colormaps[cmap_name]
        except KeyError:
            raise FigureSpecError("{} names an unknown Matplotlib colormap".format(cmap_location))
    for curve_name in curve_names:
        for index, series in enumerate(spec[curve_name]["series"]):
            try:
                matplotlib.lines.Line2D(
                    [],
                    [],
                    color=series["color"],
                    marker=series["marker"],
                    linestyle=series["linestyle"],
                )
            except (TypeError, ValueError) as exc:
                raise FigureSpecError(
                    "{}.series[{}] has an invalid color, marker, or linestyle: {}".format(
                        curve_name, index, exc
                    )
                )

    cache = _ImageCache(np_module, image_module)
    temporary_outputs: Dict[str, Path] = {}
    try:
        temporary_outputs["qualitative"] = _render_qualitative(
            image_section, methods, cache, plt, destinations["qualitative"], spec["dpi"]
        )
        temporary_outputs["absolute_error"] = _render_error_maps(
            image_section,
            methods,
            cache,
            matplotlib,
            plt,
            destinations["absolute_error"],
            spec["dpi"],
        )
        temporary_outputs["edge"] = _render_edges(
            image_section, methods, cache, np_module, plt, destinations["edge"], spec["dpi"]
        )
        temporary_outputs["sparse_view_curve"] = _render_curve(
            spec["sparse_view_curve"],
            curve_rows["sparse_view_curve"],
            plt,
            destinations["sparse_view_curve"],
            spec["dpi"],
        )
        temporary_outputs["degradation_curve"] = _render_curve(
            spec["degradation_curve"],
            curve_rows["degradation_curve"],
            plt,
            destinations["degradation_curve"],
            spec["dpi"],
        )
    except Exception:
        for path in temporary_outputs.values():
            try:
                path.unlink()
            except OSError:
                pass
        raise

    for key, temporary_path in temporary_outputs.items():
        os.replace(str(temporary_path), str(destinations[key]))

    manifest_inputs.insert(
        0,
        {
            "kind": "figure-spec",
            "path": source_path.as_posix(),
            "sha256": _sha256_file(source_path),
        },
    )
    output_records = [
        {
            "figure": key,
            "path": output_names[key],
            "sha256": _sha256_file(destinations[key]),
        }
        for key in ("qualitative", "absolute_error", "edge", "sparse_view_curve", "degradation_curve")
    ]
    manifest = {
        "schema_version": FIGURE_SPEC_VERSION,
        "kind": FIGURE_MANIFEST_KIND,
        "generator_version": GENERATOR_VERSION,
        "effective_output_dir": destination_dir.as_posix(),
        "parameters": spec,
        "dependencies": {
            "numpy": np_module.__version__,
            "pillow": image_module.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "inputs": manifest_inputs,
        "outputs": output_records,
    }
    _atomic_write_json(manifest_path, manifest)
    result = dict(destinations)
    result["manifest"] = manifest_path
    return result


__all__ = [
    "FIGURE_MANIFEST_KIND",
    "FIGURE_SPEC_KIND",
    "FIGURE_SPEC_VERSION",
    "FigureDependencyError",
    "FigureSpecError",
    "make_figures",
    "read_figure_spec",
    "validate_figure_spec",
]
