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

## 4. Priority 1：损失修订验证

### 4.1 实现验收

配置：`configs/experiment_matrix.priority1_loss_revision.json`

输出：`runs/priority1_loss_revision/` 与 `results/priority1_loss_revision/`

本阶段在原 Gaussian、ATF、TCM 和 baseline 损失之外保留 legacy 辅助损失，并新增
`filtered_edge` 模式。实现和运行固定在干净 commit
`dd5d6b139c2d3e899d0521240e87041ea622a155`（短写 `dd5d6b1`）。执行前后的验收事实为：

- 完整测试 **91/91 通过**，包含两项实际 CUDA 扩展测试；新损失专项测试 24/24 通过。
- B0 与 E2 的 2-iteration CUDA smoke 均完成前向、反向、保存和 34-view val 渲染。
- TensorBoard 2.21.0 的 1-iteration E2 检查生成有效 event；检查时本地服务返回 HTTP 200。
- matrix dry-run 通过，四组命令的 source、三个 manifest、seed、分辨率、训练预算和 val
  分区一致，仅损失参数与输出目录不同。
- 四个 runner 清单记录的实际/预期输入 SHA256 均一致，Git status 为空，运行状态均为
  `completed`；四个 `stderr.log` 均为 0 字节。
- B0、O0、E1、E2 均从头训练至 7k，并保存 2k/7k checkpoint。每组在两个 checkpoint
  上都生成 34 张 render 和 34 张同名 ground truth，配对检查全部为 34/34。

本阶段沿用第 1 节的三个输入哈希。runner 在建输出目录前复核 manifest 文件哈希，adapter
在加载时继续复核清单内部引用与逐图哈希。

### 4.2 运行时与资源记录

| 实验 | 模式与权重 | 训练耗时 | 7000 Gaussian 数 | `avg_ms` 范围 | 7000 `avg_ms` | CUDA allocator peak |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| B0 | legacy；0/0/0 | 7:40 | 24,876 | 40.62-53.99 ms | 53.99 ms | 597.1 MB |
| O0 | legacy；0.1/0.01/0.001 | 8:41 | 27,325 | 47.92-61.69 ms | 61.69 ms | 597.3 MB |
| E1 | legacy；0/0.001/0 | 8:00 | 27,253 | 43.19-56.85 ms | 56.85 ms | 598.3 MB |
| E2 | filtered_edge；0/0.001/0 | 7:49 | 26,548 | 42.35-55.46 ms | 55.46 ms | 597.0 MB |

权重列依次为 `lambda_thermal/lambda_edge/lambda_smooth`。训练耗时由日志中的 `[AUX]`
到 `Training complete` 计算，包含相机加载、训练、内部 validation 和 checkpoint 保存，不包含
独立 `render.py`。`avg_ms` 是每 500 iterations 记录一次的 CUDA event 累计平均，
表中范围覆盖 500-7000 的 14 个日志点；它不是单步延迟分布，也不包含 validation、保存和
独立渲染。峰值是 PyTorch allocator 统计，不是整卡显存占用。

### 4.3 7000 iteration 主比较

主表来自独立 `render.py` 生成的 val PNG 和冻结的 `tools/collect_metrics.py`，不是训练循环
控制台的内部 validation 数值。LPIPS 按协议跳过，ROI mask 未冻结，二者均为
`unavailable`。

| 方法 | Views | PSNR ↑ | SSIM ↑ | T-MAE ↓ | E-MAE ↓ | Gradient preservation ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 34 | 30.55578541 | 0.94624436 | 0.02609596 | 0.00655253 | **0.53585431** |
| O0 | 34 | 30.02708431 | 0.94623453 | 0.02639369 | 0.00647433 | 0.53198543 |
| E1 | 34 | 30.39146361 | 0.94625547 | 0.02625460 | 0.00653443 | 0.52984605 |
| E2 | 34 | **30.64555695** | **0.94726418** | **0.02550185** | **0.00642407** | 0.53200635 |

相对 B0 的 7k 变化为：

| 方法 | ΔPSNR | ΔSSIM | ΔT-MAE | ΔE-MAE | ΔGradient preservation |
| --- | ---: | ---: | ---: | ---: | ---: |
| O0 | -0.52870110 | -0.00000983 | +0.00029773 | -0.00007820 | -0.00386888 |
| E1 | -0.16432180 | +0.00001111 | +0.00015864 | -0.00001810 | -0.00600826 |
| E2 | +0.08977154 | +0.00101982 | -0.00059411 | -0.00012846 | -0.00384796 |

E2 在本次 val 上相对 B0 提高 PSNR/SSIM，并降低 T-MAE/E-MAE，但 Gradient preservation
下降。O0 和 E1 没有形成综合优势。E2 的幅度较小且存在梯度指标权衡，这一结果只支持继续
评估该候选项，不证明它已经解决泛化问题。

### 4.4 2000 iteration 次要检查

| 方法 | Views | PSNR ↑ | SSIM ↑ | T-MAE ↓ | E-MAE ↓ | Gradient preservation ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 34 | 29.60748353 | 0.94464562 | 0.02637973 | 0.00675459 | **0.55646594** |
| O0 | 34 | 28.70891908 | 0.94206068 | 0.03090710 | 0.00674758 | 0.55075741 |
| E1 | 34 | 29.62049671 | 0.94400735 | 0.02703862 | 0.00675309 | 0.54747425 |
| E2 | 34 | **29.62912333** | **0.94483947** | 0.02679957 | **0.00669644** | 0.54170621 |

2k 只作为早期训练检查，预先指定的主比较仍为 7k。

### 4.5 评价路径与证据边界

训练循环内部 validation 的路径是 `renderer -> clamp(0,1) -> +TCM -> metric`，而独立
`render.py` 主路径是 `renderer -> +TCM -> PNG 保存`。这是既有差异，本轮没有同时修改；
因此内部控制台指标与上述独立渲染主表可能略有不同，不能混合比较。

本阶段只有 heated 单场景、Sparse-25 + noise03、seed 2026。没有多 seed 均值/标准差，
也没有其他稀疏率、退化强度或场景证据。39-view test 始终未渲染、未计算且未人工查看，
validation 趋势不能写成最终 test 结论。

## 5. 当前停止点

执行已按要求停止在 Priority 1 验收完成处，没有自动启动正式 30k、多 seed、超参数搜索或
test。原正式计划仍未完成的部分包括：

1. 生成并核验全部条件与 seed `2026/2027/2028` 的冻结输入；
2. 对最终冻结方法和 B0 使用完全相同的 30,000-iteration 预算；
3. 保存并只用 validation 检查 `7,000/15,000/30,000`；
4. 方法冻结后才单独访问 39-view test；
5. 汇总 mean/std 和逐视角配对统计。

继续上述阶段需要新的明确执行决定；本记录不把 Priority 1 的单次 validation 结果冒充正式
训练计划的最终复现结果。
