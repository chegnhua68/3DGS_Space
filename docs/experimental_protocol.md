# Experimental Protocol

This document freezes the experimental decisions that are needed before running
large hyperparameter sweeps. It supplements the high-level project plan.

## Data contract

All views are identified by stable manifest IDs, never by directory enumeration
order. Paths in manifests are relative to the dataset root. A dataset declares a
single fixed intensity range shared by all views. Per-image min-max normalization
is prohibited because it destroys cross-view thermal-intensity consistency.

Use `thermal_intensity` for uncalibrated grayscale data. Use `temperature` only
when the source provides a documented radiometric calibration. Pseudocolor RGB
images are not treated as radiance or temperature measurements.

## View protocol

The official or preregistered test set is frozen first. Hyperparameters are tuned
only on a disjoint validation set. Sparse training sets are sampled from the
remaining train pool and are nested:

```text
Sparse-12.5 subset Sparse-25 subset Sparse-50 subset Full-train-pool
```

The main table uses the same split manifests for every method. Report at least
three preregistered sampling seeds with per-view paired results and mean/std when
compute permits. `100%` always means all views in the train pool, not validation
or test views.

Two pose protocols must not be mixed in one table:

1. Fixed-pose protocol: use official/full-view camera calibration and evaluate
   appearance reconstruction. Disclose that geometry is privileged information.
2. End-to-end protocol: estimate poses using only the sparse/degraded training
   inputs and report pose failures separately from rendering quality.

The first implementation targets the fixed-pose protocol.

The manifest adapter maps frame time as
`(sequence_index - min_index) / (max_index - min_index)` across the complete
dataset manifest. This avoids sparse-ratio-dependent time scaling. It is equal
to the upstream numeric-name rule only for contiguous zero-based sequences;
legacy reproduction without manifests retains the original filename rule.

## Degradation protocol

Only selected training images are degraded. Validation and test image bytes stay
unchanged. Every derived view records its source hash, output hash, transform
order, parameters, global seed, and stable per-view seed.

Noise sigma is expressed in the dataset's fixed normalized range, not the raw
integer range. Contrast, blur, and noise are independent transforms. Combined
experiments use the declared order `contrast -> blur -> gaussian-noise` unless a
configuration explicitly says otherwise. Output is clipped to the declared data
range. JPEG is accepted as source data, but every derived degraded image is
saved losslessly as PNG or TIFF so the experiment does not add an uncontrolled
second JPEG encoding step.

## Optimization fairness

All variants share initialization, view split, degraded bytes, optimizer,
densification schedule, iteration budget, checkpoint rule, and render/evaluator
versions. Test metrics are never used to select a checkpoint or loss weight.

The proposed weighted reconstruction term partially duplicates the baseline L1
term. Ablations therefore include a matched-scale unweighted-L1 control so that
an improvement cannot be attributed only to a larger reconstruction coefficient.
The reliability map is described as an image-derived noise heuristic unless a
calibrated sensor noise model is introduced; it is not presented as direct
thermal physics.

When all three new lambda values are zero, training skips construction and
evaluation of all new maps and losses.

## Metrics

- PSNR and SSIM use the manifest data range and identical border/mask handling.
- LPIPS on thermal grayscale is secondary; record the model version and whether
  the single channel was replicated to RGB.
- T-MAE is the mean absolute error in the declared thermal-intensity domain.
- E-MAE is normalized by the sum of the fixed ground-truth edge weights, with an
  epsilon-defined empty-edge case.
- ROI-MAE is reported only with a mask rule fixed without test-set tuning; empty
masks are reported as unavailable rather than zero.

The offline degradation contract supports uint8 JPEG/PNG/TIFF sources and
uint16 PNG/TIFF sources, always writing derived images losslessly. The pinned
Thermal3D-GS camera loader divides by 255. The current training adapter therefore
rejects any encoded `data_range` other than `[0,255]` and accepts only explicit
8-bit `L`, `LA`, `RGB`, or `RGBA` image modes. It converts thermal grayscale to
replicated RGB. Supporting uint16 or floating-point training requires a
separate loader change and must not be claimed by these experiments.

Completing the preregistered protocol is a valid outcome even when the proposed
loss does not improve every condition. Negative results must not be removed by
selecting favorable scenes or seeds.
