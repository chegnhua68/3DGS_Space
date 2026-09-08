# Priority 1 损失修订结果报告

> 执行日期：2026-09-08<br>
> 代码提交：`dd5d6b139c2d3e899d0521240e87041ea622a155`<br>
> 数据条件：TI-NSD `heated`，Sparse-25，noise03，seed 2026，resolution 1<br>
> 结论范围：单场景、单 seed、7000 iterations 的开发验证，不代表统计显著性或跨场景泛化。

## 1. 代码完成情况

本轮已经完成规格要求的 Priority 1 修改，并在此停止；没有启动 Priority 2 权重调度，也没有启动 30k 正式矩阵。

### 1.1 损失与训练接线

- 保留原 Gaussian、ATF、TCM、L1、SSIM、角点项和完整 `legacy` 辅助损失。
- 新增可切换的 `filtered_edge`。预测图使用 TCM 后、未 clamp 的当前训练预测；观测图使用 degradation manifest 指向的含噪训练观测。
- 两个分支先统一到单通道亮度，RGB 使用固定 BT.601 权重；随后使用相同的 `5 x 5, sigma=1` Gaussian、reflect padding 和除以 8 的有符号 Sobel x/y。
- 观测分支显式 stop-gradient，预测分支保持可微；损失是两个方向绝对误差均值的 `0.5` 加权和。
- `filtered_edge` 不做逐图 min-max，不调用 legacy 的 E/N/W，不使用 `noise_beta` 或 `edge_gamma`。
- 三项权重全零时训练循环和 dispatcher 均短路，不构造辅助算子，不改变 B0 的随机状态或反向路径。
- `filtered_edge` 搭配非零 `lambda_thermal` 或 `lambda_smooth` 会在场景加载前报错。

### 1.2 参数、runner 与日志

- 新参数：`--aux_loss_version {legacy,filtered_edge}`、`--edge_filter_kernel`、`--edge_filter_sigma`；默认仍为 legacy 兼容行为。
- Priority 1 矩阵为每组固定 dataset、split、degradation 三份预期 SHA256；runner 在创建输出目录前比较实际哈希。
- runner 保留输出目录防覆盖，并在 `run_manifest.json` 记录命令、哈希、Git 状态和运行状态。
- `--quiet` 关闭 tqdm；每 500 步只输出一条必要摘要，并写入 `loss_components.csv`。关闭的 raw 项记录为 `disabled`，不会为了日志额外计算。
- TensorBoard 每 5 秒刷新 baseline、total、辅助项 raw/weighted、迭代耗时和 validation 标量；本轮服务地址为 `http://127.0.0.1:6006/`。

### 1.3 实际改动文件

提交 `dd5d6b1` 包含：

```text
.gitignore
README_zh-CN.md
arguments/__init__.py
configs/experiment_matrix.priority1_loss_revision.json
docs/priority1_loss_revision.md
losses/thermal_physics_loss.py
md/Thermal3DGS_priority1_loss_revision_zh-CN.md
pyproject.toml
scripts/run_experiments.py
tests/test_arguments.py
tests/test_experiment_runner.py
tests/test_thermal_physics_loss.py
train.py
```

README 已同步新模式、参数约束、哈希锁定、精简日志和 TensorBoard 用法。源码审计还记录了一项既有差异：训练内部 validation 在 Gaussian renderer 输出后先 clamp 再加 TCM，而训练损失路径是直接加 TCM；本轮为避免改变评价口径，没有顺带修改它。

## 2. 配置与输入冻结

| ID | 辅助损失版本 | `lambda_thermal` | `lambda_edge` | `lambda_smooth` | Gaussian 滤波 |
| --- | --- | ---: | ---: | ---: | --- |
| B0 | legacy | 0 | 0 | 0 | disabled |
| O0 | legacy | 0.1 | 0.01 | 0.001 | legacy 行为 |
| E1 | legacy | 0 | 0.001 | 0 | legacy 行为 |
| E2 | filtered_edge | 0 | 0.001 | 0 | 5 x 5，sigma=1 |

四组共同使用 59 个 Sparse-25 训练视角、34 个未退化 validation 视角、seed 2026、7000 iterations、`data_device=cpu` 和 `load2gpu_on_the_fly=true`。test 的 39 个视角没有被渲染或评分，四个输出目录均不存在 `test/`。

| 输入文件 | 冻结 SHA256 |
| --- | --- |
| `dataset_manifest.v1.json` | `d4fe9c97ed110fa1ecdf0636a19f7a7062169d3cfbd719d5594158f9d615e919` |
| `sparse_nested-random_25.json` | `c9a1e0aa65e54e1558779b557539cb2eaba4d1faab765b786a4437aa519ba79e` |
| `degradation_manifest.v1.json` | `06d68ec82b7faa8d65168b9bd2b35f013207331f4683a2ec0ad883837be3f03b` |

四个 `run_manifest.json` 中的预期值和实际值逐项一致，Git HEAD 均为 `dd5d6b1`，运行开始时工作树为空。

## 3. 验证与运行完整性

- 完整测试：`91/91` 通过，其中包括两项实际 CUDA 扩展测试；新损失专项测试 `24/24` 通过。
- B0/E2 CUDA 短跑：2 iterations 的前向、反向、保存及 34-view val 渲染均通过。
- TensorBoard 冒烟检查：E2 事件文件包含 baseline、total、edge raw/weighted 和 iter-time，HTTP 检查为 200。
- 四组 matrix dry-run：source、数据、split、退化、seed、resolution、迭代数、val 分区和损失参数均一致。
- 四组 7k：状态均为 `completed`，`stderr.log` 均为 0 字节，`loss_components.csv` 均为 14 条。
- 四组 2000/7000：每个 checkpoint 都有 34 张 render 与 34 张同名 GT；逐视角指标均已保留。

| ID | 7k 状态 | 2k render/GT | 7k render/GT | 训练时长 | 最终 Gaussian | 核心 `avg_ms` | 峰值 CUDA allocated |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | completed | 34/34 | 34/34 | 7:40 | 24,876 | 53.9940 | 597.1001 MB |
| O0 | completed | 34/34 | 34/34 | 8:41 | 27,325 | 61.6916 | 597.2949 MB |
| E1 | completed | 34/34 | 34/34 | 8:00 | 27,253 | 56.8499 | 598.2993 MB |
| E2 | completed | 34/34 | 34/34 | 7:49 | 26,548 | 55.4622 | 597.0278 MB |

训练时长按 `[AUX]` 到 `Training complete` 的日志时间戳计算，包含场景加载、训练、内部 validation 和保存，不包含独立 `render.py`。`avg_ms` 是训练器 CUDA event 覆盖的核心前向/反向累计均值，不包含优化器、增密、validation 或日志 I/O，因此不能当作墙钟时间。相对 B0，O0/E1/E2 的该核心均值分别增加约 14.26%、5.29% 和 2.72%。峰值记录是 PyTorch CUDA allocator 的 allocated peak，不是任务管理器中的整卡占用。

## 4. 2000 步辅助观察

LPIPS 按规格使用 `--skip-lpips`，结果明确为 `unavailable`；没有冻结 ROI mask，因此 ROI-MAE 同样为 `unavailable`。

| ID | PSNR ↑ | SSIM ↑ | T-MAE ↓ | E-MAE ↓ | Gradient preservation ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 29.60748353 | 0.94464562 | 0.02637973 | 0.00675459 | 0.55646594 |
| O0 | 28.70891908 | 0.94206068 | 0.03090710 | 0.00674758 | 0.55075741 |
| E1 | 29.62049671 | 0.94400735 | 0.02703862 | 0.00675309 | 0.54747425 |
| E2 | 29.62912333 | 0.94483947 | 0.02679957 | 0.00669644 | 0.54170621 |

2000 步不是主比较点。E2 此时相对 B0 的 PSNR、SSIM 和 E-MAE 略好，但 T-MAE 和 Gradient preservation 较差，说明短程指标存在冲突，不能据此提前接受方案。

## 5. 7000 步主比较

| ID | PSNR ↑ | SSIM ↑ | T-MAE ↓ | E-MAE ↓ | Gradient preservation ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 30.55578541 | 0.94624436 | 0.02609596 | 0.00655253 | 0.53585431 |
| O0 | 30.02708431 | 0.94623453 | 0.02639369 | 0.00647433 | 0.53198543 |
| E1 | 30.39146361 | 0.94625547 | 0.02625460 | 0.00653443 | 0.52984605 |
| E2 | **30.64555695** | **0.94726418** | **0.02550185** | **0.00642407** | 0.53200635 |

相对本轮 B0 的有符号差值如下。PSNR、SSIM、Gradient preservation 为正更好；T-MAE、E-MAE 为负更好。

| ID | ΔPSNR | ΔSSIM | ΔT-MAE | ΔE-MAE | ΔGradient preservation |
| --- | ---: | ---: | ---: | ---: | ---: |
| O0 | -0.52870110 | -0.00000983 | +0.00029773 | -0.00007820 | -0.00386888 |
| E1 | -0.16432180 | +0.00001111 | +0.00015864 | -0.00001810 | -0.00600826 |
| E2 | **+0.08977154** | **+0.00101982** | **-0.00059411** | **-0.00012846** | -0.00384796 |

## 6. 性能观察

1. E2 在主比较点相对 B0 同时提高 PSNR 和 SSIM，并降低 T-MAE 与 E-MAE。规格要求优先联合观察的 PSNR、T-MAE、E-MAE 三项方向一致；其中 T-MAE 约下降 2.28%，E-MAE 约下降 1.96%。
2. E2 的 Gradient preservation 相对 B0 下降 `0.00384796`，因此结果不是所有指标一致占优，必须保留这一冲突。
3. E2 在 7000 步的五个指标都优于 E1，但 E1/E2 同时改变了滤波域、padding、通道处理、归一化和有效梯度尺度，不能把差值单独归因于“对观测先滤波”。
4. O0 的 PSNR 比 B0 低 `0.52870110 dB`，T-MAE 也更差；这与前一轮 calibration 中完整 legacy 辅助项未优于 baseline 的现象一致，但仍不能据单次运行诊断为过拟合或确定具体因果。
5. E2 的 7000 步结果构成继续研究的正向开发证据，不构成方法优越性的最终证明。至少还需要预注册的多 seed、更多场景/稀疏度/退化强度、冻结后 30k 训练以及最终 test 评测。

## 7. 输出位置与复核命令

```text
runs/priority1_loss_revision/<experiment>/
  run_manifest.json
  stdout.log
  stderr.log
  loss_components.csv
  val/ours_2000/{renders,gt}/
  val/ours_7000/{renders,gt}/

results/priority1_loss_revision/val_2000/
  metrics_summary.csv
  metrics_summary.md
  metrics_per_view.csv

results/priority1_loss_revision/val_7000/
  metrics_summary.csv
  metrics_summary.md
  metrics_per_view.csv
```

复核配置但不重跑：

```powershell
.\.venv\Scripts\python.exe scripts\run_experiments.py `
  --matrix configs\experiment_matrix.priority1_loss_revision.json `
  --dry-run
```

实时查看已保存曲线：

```powershell
.\.venv\Scripts\python.exe -m tensorboard.main `
  --logdir runs\priority1_loss_revision `
  --host 127.0.0.1 `
  --port 6006 `
  --reload_interval 5
```

当前输出目录已存在，runner 的防覆盖行为会拒绝直接重跑同名矩阵。需要复现实验时应使用新的 `output_root`，不要覆盖本轮结果。

## 8. 本轮停止条件

Priority 1 的实现、测试、四组 7k 运行、2k/7k validation 渲染、独立指标与逐视角结果均已完成。本轮在此停止，不查看 test，不自动调整 lambda，不加入 mask、调度或新网络，也不启动 30k。
