# Experiment Log

## 2026-08-27 - Baseline acquisition and host audit

- Read `Thermal3DGS_sparse_IR_project_plan.md` in full.
- Located the official repository through the ECCV/arXiv project references.
- Fetched and pinned upstream commit
  `03366b2a350ac5db6690dfd7fca51a56ba9e89a7` as remote `upstream`.
- Read the upstream README, environment, training loop, argument definitions,
  COLMAP loader, scene construction, renderer contract, and loss implementation.
- No TI-NSD scene or pretrained checkpoint has been downloaded yet.

Host audit:

- GPU: NVIDIA GeForce RTX 2060 Max-Q, 6 GB VRAM, compute capability 7.5.
- Driver is installed, but `nvcc` and a CUDA Toolkit are absent.
- MSVC `cl` is not available in the current shell.
- No conda installation is on `PATH`.
- Default Python is 3.14.7 with CPU-only PyTorch 2.9.1.
- The current NumPy and PyTorch imports terminate with a Windows access
  violation, so this global Python environment is not usable for experiments.

Conclusion: source-level and standard-library checks can proceed, but the
official CUDA extensions cannot be compiled and no baseline training can run
until an isolated compatible CUDA/Python toolchain is provisioned. With 6 GB
VRAM, initial smoke tests should use one small scene and reduced resolution.

## 2026-08-27 - Python 3.11 retry after network recovery

- Selected interpreter:
  `E:\Software\Scoop\Apps\apps\python311\current\python.exe` (Python 3.11.9).
- Created `.venv` with `--system-site-packages` so it inherits PyTorch
  2.5.1+cu121 and torchvision 0.20.1+cu121.
- Confirmed CUDA is available on the NVIDIA GeForce RTX 2060 Max-Q.
- Installed `tqdm`, `plyfile`, `imageio`, and `imageio-ffmpeg` successfully
  through the restored network.
- Confirmed that both vendored CUDA extension source trees are present.
- `CUDA_HOME`, `nvcc`, and MSVC `cl` remain unavailable, so
  `diff_gaussian_rasterization` and `simple_knn` cannot yet be built/imported.
- No public thermal scene or pretrained checkpoint has been downloaded. A real
  training command, checkpoint, render, or paper result is therefore not yet
  recorded.

Python 3.11 regression checks completed at this stage: config parser 3/3,
manifest adapter 6/6, upstream corner loss 2/2, and proposed loss 10/10. The
complete repository test suite passes 63/63; all Python targets compile, both
example JSON files validate, and `git diff --check` reports no errors.

## 2026-08-27 - CUDA build and TI-NSD heated smoke runs

This entry supersedes the earlier host-blocker status while preserving it above
as a chronological snapshot.

Native toolchain and extension status:

- CUDA Toolkit 12.1 is available at
  `E:\Software\Scoop\Apps\apps\cuda-12.1\current`; the extension build sets
  `CUDA_HOME` and `CUDA_PATH` to this location.
- The existing Visual Studio 2026 Build Tools instance is at
  `E:\Software\Visual Studio Build Tools\Tools`. Its v142 MSVC toolset,
  compiler version 14.29, is selected with `vcvars64.bat -vcvars_ver=14.29`.
- Both vendored CUDA extensions, `simple_knn` and
  `diff_gaussian_rasterization`, built successfully and import in the selected
  Python 3.11 environment.
- `tests/test_cuda_extensions.py` passes 2/2 on CUDA: the `simple_knn`
  nearest-neighbor distances match a PyTorch reference, and the rasterizer
  forward/backward smoke test produces finite outputs and gradients.

Dataset and split status:

- The local TI-NSD `heated` scene contains 307 registered views, numbered
  `000.jpg` through `306.jpg`, with COLMAP data under `sparse/0`.
- The frozen base split contains 234 train-pool views, 34 validation views, and
  39 test views. Validation and test views are disjoint from the train pool.
- The nested-random 12.5% split selected 29 of the 234 train-pool views with
  seed 2026 (`actual_ratio = 0.12393162393162394`).
- Matching seed-2026 nested splits are also materialized for 100%, 50%, and
  25%, selecting 234, 117, and 59 training views respectively. All four sets
  are nested.
- Gaussian noise with normalized sigma `0.03` was materialized only for those
  29 selected training views. Validation and test image bytes remained clean.

Two initial end-to-end runs completed for 10 training iterations at resolution
factor 4, using CPU image storage with on-demand GPU transfer. Both were
evaluated on the same 34-view validation partition. These runs changed both
the training data and the loss, so the following values are diagnostics rather
than a controlled method comparison.

| Run | PSNR | SSIM | T-MAE | E-MAE | Gradient preservation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline, clean selected views, all added loss weights zero | 17.37825888 | 0.81121168 | 0.10964358 | 0.01910923 | 0.68269374 |
| Proposed, train-only noise sigma 0.03 | 17.38013316 | 0.81109118 | 0.10956914 | 0.01911008 | 0.68300996 |

The renderer initially stored both validation runs under `test/ours_10`. Its
output contract now uses `val/ours_10` or `test/ours_10` according to the
manifest-selected partition, while legacy runs without manifests keep the
upstream `test` name. Both checkpoints were re-rendered into `val/ours_10`;
all 34 corrected render hashes match the earlier files exactly. The old
misnamed directories remain for audit purposes.

A matched noise-sigma-0.03 pair was then run through
`configs/experiment_matrix.heated_smoke.json`. Dataset, split, degradation,
resolution, seed, optimizer schedule, and 10-iteration budget are identical;
only the three proposed-loss weights differ. Both `run_manifest.json` files
report `completed`, contain identical hashes for all three input manifests,
and each run contains exactly 34 clean validation render/ground-truth pairs
under `val/ours_10`. Ground-truth hashes match across methods for every view.

| Matched noise03 run | PSNR | SSIM | T-MAE | E-MAE | Gradient preservation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline, all added loss weights zero | 17.37711321 | 0.81107783 | 0.10965925 | 0.01910995 | 0.68279362 |
| Proposed, full added loss | 17.38011804 | 0.81109057 | 0.10956990 | 0.01911006 | 0.68301739 |

A separate capacity calibration used the full 234-view training pool at native
resolution for 100 baseline iterations. Its runner manifest reports
`completed` after 38 seconds, including provenance collection, checkpointing,
scene reload, and 34-view validation rendering. Validation metrics were PSNR
20.88558902, SSIM 0.92215234, T-MAE 0.05813449, E-MAE 0.00886292, and gradient
preservation 0.42490913. This confirms the native-resolution path but not peak
7k/30k memory use, because densification begins at iteration 500.

The proposed smoke run used `lambda_thermal = 0.1`, `lambda_edge = 0.01`, and
`lambda_smooth = 0.001`. LPIPS was not evaluated because the optional model was
not enabled, and ROI-MAE was not evaluated because no fixed ROI mask was
provided.

These 10-iteration runs are pipeline smoke tests only. They demonstrate that
the compiled renderer, sparse split, train-only degradation, optional loss,
checkpoint save/load, partition-safe rendering, experiment provenance, and
metric collection execute end to end.
They are not converged experiments, and their small metric differences do not
support a scientific effectiveness claim.

No matching upstream pretrained checkpoint was used or downloaded; both smoke
runs started locally from the scene initialization. The upstream repository,
TI-NSD release, and published weight folders do not provide a complete,
unambiguous repository-level redistribution license. Public download access is
not treated as permission to redistribute them, so the dataset, upstream
weights, and full modified upstream source will not be redistributed without
explicit authorization.

Final regression verification after these fixes passes 68/68 unittest cases,
including both CUDA extension tests. Project Python modules compile, all four
configuration JSON files parse, and `git diff --check` reports no whitespace
errors.
