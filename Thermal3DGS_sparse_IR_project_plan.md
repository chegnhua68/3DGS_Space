# Thermal3D-GS Sparse-View Infrared Reconstruction Project Plan

## 1. Project Goal

This project focuses on **robust infrared novel-view synthesis and 3D thermal radiance field reconstruction** under sparse-view and degraded infrared imaging conditions.

The planned route is:

> Thermal3D-GS reproduction + sparse-view infrared novel-view synthesis + low-SNR / weak-texture degradation experiments + a lightweight physics-guided improvement.

The work does **not** rely on spaceborne data at this stage. The paper should be positioned as an infrared optoelectronic imaging and 3D reconstruction study.

---

## 2. Tentative Title

**Lightweight Physics-Guided 3D Gaussian Thermal Radiance Field Reconstruction for Sparse-View Degraded Infrared Imaging**

Alternative title:

**Robust Sparse-View Infrared Novel-View Synthesis via Physics-Guided 3D Gaussian Thermal Radiance Fields**

---

## 3. Research Motivation

Thermal infrared images often suffer from:

- sparse observation views;
- weak scene texture;
- low signal-to-noise ratio;
- blurred thermal boundaries;
- unstable geometry reconstruction under degraded imaging conditions.

Thermal3D-GS provides a strong baseline for thermal infrared novel-view synthesis. However, its robustness under **sparse-view**, **low-SNR**, and **weak-texture** conditions still needs further evaluation. This project aims to build a clear evaluation protocol and introduce a lightweight constraint to improve reconstruction stability and thermal boundary preservation.

---

## 4. Baseline

### Main baseline

- **Thermal3D-GS**

### Optional comparison baseline

- Standard 3D Gaussian Splatting, if it can be run on the same thermal infrared data.

The first priority is to reproduce Thermal3D-GS successfully and keep its original training pipeline unchanged as the baseline.

---

## 5. Proposed Improvement

### Name

**Noise-Aware Edge-Preserving Thermal Consistency Constraint**

### Core idea

In low-SNR infrared images, pixel-wise reconstruction losses may overfit noisy pixels. In weak-texture infrared scenes, thermal target boundaries may become blurred under sparse-view observations. Therefore, the optimization should:

1. preserve reliable thermal boundaries;
2. reduce the influence of noisy pixels;
3. encourage smooth thermal radiance in non-edge regions.

### Suggested loss design

Let:

- `I_gt` be the target thermal infrared image;
- `I_pred` be the rendered thermal infrared image;
- `G(.)` be a Gaussian smoothing operator;
- `S(.)` be a Sobel gradient operator;
- `E` be the thermal edge map;
- `W` be the noise-aware reliability map.

#### 5.1 Thermal edge map

```text
E = normalize(|S(G(I_gt))|)
```

Use a smoothed target image before Sobel filtering to avoid treating random noise as true structure.

#### 5.2 Noise-aware reliability map

A simple implementation can be:

```text
N = normalize(|I_gt - G(I_gt)|)
W = exp(-beta * N)
```

where `N` is a noise proxy and `beta` controls how strongly noisy pixels are down-weighted.

#### 5.3 Weighted thermal reconstruction loss

```text
L_thermal = mean(W * |I_pred - I_gt|)
```

#### 5.4 Edge-preserving gradient loss

```text
L_edge = mean((1 + gamma * E) * |S(I_pred) - S(I_gt)|)
```

This term gives more importance to thermal boundaries.

#### 5.5 Non-edge smoothness loss

```text
L_smooth = mean((1 - E) * |S(I_pred)|)
```

This term suppresses noise-like fluctuations in smooth thermal regions.

#### 5.6 Total loss

```text
L_total = L_base + lambda_t * L_thermal + lambda_e * L_edge + lambda_s * L_smooth
```

where `L_base` is the original Thermal3D-GS loss.

The new terms should be optional. When all lambda values are set to zero, the code should behave exactly like the original Thermal3D-GS baseline.

---

## 6. Experimental Design

### 6.1 Reproduction experiment

Goal: reproduce Thermal3D-GS on at least one public thermal infrared scene.

Required outputs:

- rendered novel-view thermal images;
- PSNR / SSIM / LPIPS;
- qualitative comparison figures;
- training log and checkpoint.

### 6.2 Sparse-view experiment

Train with different view ratios and evaluate on held-out views.

Suggested settings:

| Setting | Training Views | Testing Views |
|---|---:|---:|
| Full-view | 100% | held-out test views |
| Sparse-50 | 50% | same test views |
| Sparse-25 | 25% | same test views |
| Sparse-12.5 | 12.5% | same test views |

Use deterministic sampling, such as every `k`-th frame or a fixed random seed.

### 6.3 Low-SNR degradation experiment

Add synthetic noise to training images only. Test on clean held-out images.

Suggested noise levels:

```text
sigma = 0.01, 0.03, 0.05
```

Assume thermal images are normalized to `[0, 1]`.

### 6.4 Weak-texture / low-contrast degradation experiment

Simulate weak thermal texture by reducing local contrast or applying mild blur to training images only.

Possible degradation:

```text
I_deg = mean(I) + alpha * (GaussianBlur(I) - mean(I))
```

Suggested contrast factors:

```text
alpha = 0.75, 0.50, 0.25
```

Possible blur kernels:

```text
kernel size = 3, 5
```

### 6.5 Combined degradation experiment

Combine sparse-view and low-SNR conditions:

```text
Sparse-25 + sigma = 0.03
Sparse-12.5 + sigma = 0.03
Sparse-25 + weak texture alpha = 0.50
```

This experiment is useful for showing robustness.

---

## 7. Evaluation Metrics

### Common image quality metrics

- PSNR;
- SSIM;
- LPIPS.

### Infrared-oriented metrics

Use thermal intensity rather than true temperature unless the dataset provides calibrated temperature.

Recommended metrics:

1. **Thermal Intensity MAE**

```text
T-MAE = mean(|I_pred - I_gt|)
```

2. **Thermal Edge Error**

```text
E-MAE = mean(E_gt * |S(I_pred) - S(I_gt)|)
```

3. **Thermal Target Region MAE**

If a thermal target mask is available or can be obtained by thresholding:

```text
ROI-MAE = mean(M_roi * |I_pred - I_gt|)
```

4. **Gradient Preservation Score**

Compare gradient magnitude similarity between prediction and ground truth.

---

## 8. Ablation Study

At minimum, compare:

| Variant | Description |
|---|---|
| Baseline | Original Thermal3D-GS |
| + Thermal | Add weighted thermal reconstruction loss only |
| + Edge | Add edge-preserving gradient loss |
| + Smooth | Add non-edge smoothness loss |
| Full | Add all proposed terms |

Suggested hyperparameter search:

```text
lambda_t = 0.1, 0.5, 1.0
lambda_e = 0.01, 0.05, 0.1
lambda_s = 0.001, 0.005, 0.01
beta = 2, 5, 10
gamma = 1, 3, 5
```

Start with small weights to avoid over-constraining the baseline.

---

## 9. Suggested Code Tasks for Codex

### Task 1: Inspect and run the original repository

- Read the original `README.md`.
- Set up the environment according to the official instructions.
- Run one small scene with the original Thermal3D-GS pipeline.
- Save the exact command, log, checkpoint path, and output images.

### Task 2: Add sparse-view split generation

Create:

```text
tools/create_sparse_split.py
```

Required functions:

- read original camera/view list;
- generate fixed train/test splits;
- support view ratios: `1.0`, `0.5`, `0.25`, `0.125`;
- support deterministic random seed;
- save split files to `splits/`.

### Task 3: Add infrared degradation generation

Create:

```text
tools/degrade_ir_dataset.py
```

Supported degradation types:

- Gaussian noise;
- blur;
- contrast reduction;
- combined degradation.

Important rule:

- Apply degradation to training views only.
- Keep test views clean for fair evaluation.

### Task 4: Implement proposed loss

Create or modify a module such as:

```text
losses/thermal_physics_loss.py
```

Required functions:

- `compute_edge_map(image)`;
- `compute_noise_reliability_map(image, beta)`;
- `thermal_reconstruction_loss(pred, gt, weight)`;
- `edge_preserving_loss(pred, gt, edge_map, gamma)`;
- `non_edge_smoothness_loss(pred, edge_map)`.

Add command-line flags:

```text
--lambda_thermal
--lambda_edge
--lambda_smooth
--noise_beta
--edge_gamma
```

Default values must preserve original behavior:

```text
lambda_thermal = 0
lambda_edge = 0
lambda_smooth = 0
```

### Task 5: Batch experiment runner

Create:

```text
scripts/run_sparse_degradation_experiments.sh
```

It should run:

- baseline full-view;
- baseline sparse-view;
- proposed sparse-view;
- baseline degraded;
- proposed degraded;
- combined sparse + degraded.

### Task 6: Collect metrics

Create:

```text
tools/collect_metrics.py
```

It should collect:

- PSNR;
- SSIM;
- LPIPS;
- T-MAE;
- E-MAE;
- ROI-MAE if available.

Output:

```text
results/metrics_summary.csv
results/metrics_summary.md
```

### Task 7: Generate paper figures

Create:

```text
tools/make_figures.py
```

Generate:

- qualitative comparison grid;
- error map;
- thermal edge comparison;
- sparse-view performance curve;
- degradation robustness curve.

---

## 10. Recommended Project Structure

```text
thermal3dgs_sparse_ir/
├── README.md
├── configs/
│   ├── baseline.yaml
│   ├── sparse_25.yaml
│   ├── degraded_noise.yaml
│   └── ours_full.yaml
├── tools/
│   ├── create_sparse_split.py
│   ├── degrade_ir_dataset.py
│   ├── collect_metrics.py
│   └── make_figures.py
├── losses/
│   └── thermal_physics_loss.py
├── scripts/
│   ├── run_reproduction.sh
│   ├── run_sparse_experiments.sh
│   └── run_degradation_experiments.sh
├── results/
│   ├── metrics_summary.csv
│   ├── metrics_summary.md
│   └── figures/
└── docs/
    ├── experiment_log.md
    └── paper_notes.md
```

If modifying the original Thermal3D-GS repository directly, keep changes modular and add clear comments.

---

## 11. Expected Paper Contributions

1. A systematic evaluation of Thermal3D-GS under sparse-view infrared imaging conditions.
2. A degradation benchmark for low-SNR and weak-texture infrared novel-view synthesis.
3. A lightweight noise-aware edge-preserving thermal consistency constraint.
4. Improved robustness of 3D thermal radiance field reconstruction under sparse-view and degraded infrared conditions.

---

## 12. Draft Abstract

Infrared novel-view synthesis is challenging under sparse-view and degraded imaging conditions. Thermal infrared images often contain weak textures, low signal-to-noise ratios, and blurred thermal boundaries, which make 3D thermal scene reconstruction unstable. This paper proposes a lightweight physics-guided 3D Gaussian thermal radiance field reconstruction method for sparse-view degraded infrared imaging. Based on Thermal3D-GS, the proposed method introduces a noise-aware edge-preserving thermal consistency constraint. It enhances thermal boundary preservation while reducing the influence of unreliable noisy pixels during optimization. Sparse-view settings and infrared degradation scenarios are constructed on public thermal infrared datasets to evaluate reconstruction robustness. The proposed method is compared with standard 3D Gaussian Splatting and Thermal3D-GS in terms of novel-view synthesis quality, thermal target structure preservation, and thermal-radiance representation stability. Experimental results show that the proposed method improves infrared reconstruction robustness under sparse-view and low-SNR conditions.

---

## 13. Keywords

Infrared imaging; 3D Gaussian Splatting; thermal radiance field; sparse-view reconstruction; novel-view synthesis; physics-guided imaging.

---

## 14. Important Writing Notes

- Do not claim true temperature reconstruction unless the dataset provides calibrated temperature.
- Use terms such as `thermal intensity`, `infrared radiance representation`, or `apparent thermal distribution` when only grayscale thermal images are available.
- Do not emphasize spaceborne imaging in the main experiments unless real spaceborne infrared data are used.
- The application background can mention infrared optoelectronic imaging, thermal anomaly detection, and robust 3D thermal scene representation.
- Keep the method lightweight. The contribution should be easy to implement and easy to ablate.

---

## 15. Minimum Success Criteria

The project is considered successful if it produces:

1. a working reproduction of Thermal3D-GS on at least one public thermal scene;
2. sparse-view results at 50%, 25%, and 12.5% training views;
3. at least one low-SNR or weak-texture degradation experiment;
4. a working implementation of the proposed loss;
5. one quantitative table and one qualitative figure showing improvement over the baseline.
