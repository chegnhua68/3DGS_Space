# Upstream Provenance

## Pinned baseline

- Repository: <https://github.com/mzzcdf/Thermal3DGS>
- Upstream branch: `master`
- Pinned commit: `03366b2a350ac5db6690dfd7fca51a56ba9e89a7`
- Local remote name: `upstream`
- Fetch date: 2026-08-27

The sparse-view work is developed on top of this exact tree. Results must record
this commit and the local diff so that later upstream updates do not silently
change the baseline.

## License status

The pinned repository does not contain a root `LICENSE` or `LICENSE.md` file.
Several inherited source headers refer to an absent `LICENSE.md` and state that
the code is for non-commercial research and evaluation. That notice is not a
complete repository-level license grant. Confirm redistribution and publication
terms with the upstream authors before distributing a modified source archive.

The vendored CUDA rasterizer includes its own license under
`submodules/depth-diff-gaussian-rasterization/LICENSE.md`; that file does not by
itself license the rest of Thermal3D-GS.

## Baseline behavior that affects reproduction

- The official environment pins Python 3.7.13, PyTorch 1.12.1, torchvision
  0.13.1, and CUDA Toolkit 11.6.
- Training is launched with `python train.py -s <COLMAP scene> --eval`.
- COLMAP evaluation uses every eighth sorted camera as test data.
- COLMAP thermal frame names are assumed to be numeric, or to follow the
  repository-specific `s1_...` / `s2_...` convention when computing frame time.
- The upstream training script forced `CUDA_VISIBLE_DEVICES=1`. This extension
  removes that machine-specific assignment; device visibility is now controlled
  by the caller's environment.
- The vendored rasterizer returns `(color, radii, depth)`, while the pinned
  renderer unpacked only two values. This extension restores the three-value
  contract; otherwise training fails on its first render call.
- The README describes checkpoint flags that are not exposed by the pinned
  `train.py` argument parser.
- Without an explicit `-m`, the pinned trainer saved ATF/TCM weights outside the
  generated model directory. The extension saves them through
  `scene.model_path`, matching the Gaussian checkpoint location.
- The pinned corner loss computes its Harris determinant from unsmoothed
  per-pixel products, making `Ix^2 Iy^2 - (Ix Iy)^2` algebraically zero. Its
  exponential normalization can also underflow to `0/0`. The extension uses
  mean thermal intensity, a 3x3 local second-moment window, channel broadcasting,
  and the stable form `exp(R - max(R))`.
- The pinned renderer calls a nonexistent `load2gpu()` method for on-demand
  camera loading, writes timing data to `ours_30000` regardless of the loaded
  iteration, and loads the latest ATF/TCM weights independently of the Gaussian
  iteration. These paths now use `load2device()` and one shared iteration.
- The pinned renderer advertises `view` and `pose` modes backed by undefined
  functions. The extension exposes only the four implemented modes.
- The upstream `cfg_args` loader executes the file with `eval`. The extension
  accepts only a literal `Namespace(...)` expression and rejects executable
  values.

These facts are part of the baseline audit. Because the corner response and
renderer contract require compatibility repairs, results from this branch must
be labeled as the pinned upstream revision plus the documented compatibility
patch, not as byte-for-byte execution of the broken upstream tree.
