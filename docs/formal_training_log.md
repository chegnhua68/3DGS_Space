# 正式训练计划执行记录

本文记录 `Thermal3D-GS 稀疏视角退化红外重建：正式训练计划.md` 的实际执行状态。这里的
`calibration` 运行用于确认训练预算、分辨率和显存是否可行；它们不替代正式的三 seed、
30,000 iteration 结果，也不用于宣称 proposed 方法优于 baseline。

## 1. 执行环境与可复现设置

- Python：3.11.9，解释器为 `.venv\Scripts\python.exe`。
- GPU：NVIDIA GeForce RTX 2060 Max-Q，6 GB。
- CUDA：12.1；项目 CUDA 扩展已编译并通过 CUDA 单元测试。
- 数据：本地 `data/TI-NSD/heated`，307 个注册视角。
- 冻结划分：34 个 validation、39 个 test，二者与训练池互斥。
- 校准条件：Sparse-25，训练图加入归一化高斯噪声 `sigma=0.03`，`resolution=1`，
  `data_device=cpu`，启用 `load2gpu_on_the_fly`，评估分区为 `val`。
- 随机种子：`--seed 2026`。该参数同时设置 Python、NumPy、PyTorch CPU 和 CUDA 随机状态。
- 方法参数：baseline 的三个新增损失权重为 0；proposed 使用
  `lambda_thermal=0.1`、`lambda_edge=0.01`、`lambda_smooth=0.001`、
  `edge_gamma=3`、`noise_beta=5`。

输入清单 SHA256：

| 文件 | SHA256 |
| --- | --- |
| `dataset_manifest.v1.json` | `d4fe9c97ed110fa1ecdf0636a19f7a7062169d3cfbd719d5594158f9d615e919` |
| `sparse_nested-random_25.json` | `c9a1e0aa65e54e1558779b557539cb2eaba4d1faab765b786a4437aa519ba79e` |
| `degradation_manifest.v1.json` | `06d68ec82b7faa8d65168b9bd2b35f013207331f4683a2ec0ad883837be3f03b` |

退化清单内部记录 `global_seed=2026`、59 个 train-selected 视角、34 个 validation 视角和
39 个 test 视角。验证集和测试集保持干净，噪声只写入选中的训练图。

## 2. Phase 1：Sparse-25 + noise03 容量校准

### 2.1 2k 运行

配置：`configs/experiment_matrix.calibration_sparse25_noise03.json`

输出：`runs/calibration/sparse25_noise03_seed2026_r1_2k/`

两种方法均完成训练、checkpoint 保存和 34-view validation 渲染，未发现 NaN、CUDA error
或 out-of-memory。指标由 `tools/collect_metrics.py --skip-lpips --device cuda` 生成：

| 方法 | Views | PSNR | SSIM | T-MAE | E-MAE | Gradient preservation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 34 | 29.15209567 | 0.94266845 | 0.02854849 | 0.00674837 | 0.54341700 |
| proposed | 34 | 28.91107187 | 0.94250396 | 0.03046912 | 0.00680100 | 0.54608331 |

### 2.2 7k 运行

配置：`configs/experiment_matrix.calibration_sparse25_noise03_7k.json`

输出：`runs/calibration/sparse25_noise03_seed2026_r1_7k/`

运行清单 `run_manifest.json` 对 baseline 和 proposed 均报告 `status=completed`；每个方法
均生成 34 张 `val/ours_7000` 渲染图及 34 张对应 ground truth，子进程 `stderr.log` 行数为 0。
指标如下：

| 方法 | Views | PSNR | SSIM | T-MAE | E-MAE | Gradient preservation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 34 | 30.38355241 | 0.94700242 | 0.02484987 | 0.00650183 | 0.53549927 |
| proposed | 34 | 30.22725793 | 0.94543467 | 0.02732344 | 0.00660346 | 0.53730123 |

### 2.3 稳定性观察

训练摘要每 500 次迭代输出一次。点数和 PyTorch allocator peak 的关键轨迹为：

| 方法 | 初始点数 | 7000 次点数 | 峰值 CUDA allocated |
| --- | ---: | ---: | ---: |
| baseline | 5,079 | 24,998 | 597.1 MB |
| proposed | 5,079 | 27,913 | 597.9 MB |

两条曲线都穿过 densification 阶段并完成到 7k；在这次校准中没有 OOM、NaN 或 stderr 错误。
因此，按照计划的退出条件，后续正式矩阵可以继续采用 `resolution=1`。这里的显存数值是
训练器报告的 CUDA allocator peak，不是系统任务管理器的完整显存占用。

## 3. 日志输出策略

训练器现在只在以下时刻输出关键信息：

- 每 `--log_interval` 次迭代：loss、Gaussian 数量、平均迭代耗时、CUDA 峰值显存；
- `--test_iterations`：验证评估结果；
- `--save_iterations`：checkpoint 保存；
- 训练结束：最佳 PSNR 和完成状态。

传入 `--quiet` 会关闭 tqdm 实时进度条，但不会隐藏上述摘要、评估、保存和错误信息。批量
runner 将每个子进程分别写入实验目录的 `stdout.log` 与 `stderr.log`，主终端只显示开始、
完成或失败状态。相机读取阶段也只保留总数和完成数，不再逐视角打印。

## 4. 下一步：正式 30k 矩阵

Phase 1 只完成了 seed 2026、Sparse-25、noise03 的 2k/7k 校准。正式实验仍需：

1. 按计划生成全部条件与 seed `2026/2027/2028` 的输入清单；
2. 对 baseline 和 proposed 使用完全相同的训练视角、噪声、初始化和优化预算；
3. 每个运行执行 30,000 iterations，并保存/验证 `7,000/15,000/30,000`；
4. 冻结超参数后单独渲染 39-view test，禁止训练期间查看 test 指标；
5. 汇总每个条件的 mean/std 和逐视角配对结果。

启动正式矩阵前，先执行：

```powershell
.\.venv\Scripts\python.exe scripts\run_experiments.py `
  --matrix configs\experiment_matrix.example.json `
  --dry-run
```

确认输入路径、seed、迭代次数和输出目录后，再去掉 `--dry-run`。每次运行完成后，优先检查
`run_manifest.json`、`stdout.log`、`stderr.log` 和对应的 `metrics_summary.md`。
