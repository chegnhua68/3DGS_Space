# Priority 1 损失修订实施记录

> 记录日期：2026-09-08<br>
> 当前状态：源码审计、实现接线、91/91 回归测试、短步数兼容检查、TensorBoard 检查、dry-run、四组 7k 训练以及 2k/7k validation 渲染和指标收集均已完成。执行到 Priority 1 为止，未访问 test，也未启动 30k 或多 seed 实验。

## 1. 本轮范围

本轮保留 Gaussian、ATF、TCM、原 L1/SSIM/角点项以及 legacy 辅助损失，仅新增可切换的 `filtered_edge` 边缘辅助项。没有加入可靠边缘掩膜、权重调度、增密策略修改、网络结构修改、推理后处理或干净训练参考监督。

四组实验为 B0、O0、E1、E2，统一使用 Sparse-25、noise03、seed 2026、resolution 1，并从头训练到 7000 iterations。四组均已完成；本文仅在这一冻结条件内比较 validation 结果，不将单场景、单 seed 观察写成泛化结论。

## 2. 源码接线审计

### 2.1 训练预测图

训练中的辅助损失接收 TCM 之后的预测：

```text
Gaussian renderer 输出
  -> image
  -> image + TCM.step(image)
  -> 原 baseline 损失和可选辅助损失
```

这一位置没有对 TCM 后的 `image` 执行 `clamp(0, 1)`。新损失沿用相同预测位置并保留完整计算图，不为 `filtered_edge` 单独 clamp，也不 detach 预测分支。

### 2.2 训练观测图

训练观测的实际来源链路为：

```text
degradation_manifest.v1.json
  -> manifest adapter 校验数据集、划分、角色、清单哈希和逐图 output_sha256
  -> Sparse-25 的 train_selected 视角解析到 degradation_output
  -> PIL 图像加载并由 PILtoTorch 转为张量
  -> Camera.original_image = image.clamp(0, 1)
  -> gt_image = viewpoint_cam.original_image.cuda()
  -> 辅助损失的 observation
```

因此本轮 `I_obs` 是退化清单指定且通过哈希验证的含噪训练观测，不是未退化训练图。`val`/`test` 视角由 adapter 强制保持未退化，并且不会被读入训练损失。若相机图像带 alpha，仍由既有 Camera 路径处理；新损失不重新合成背景。

### 2.3 legacy 辅助损失的实际算子

legacy 行为保持不变：

- Gaussian 默认是 `5 x 5`、`sigma=1.0`，使用 `replicate` padding；卷积按通道分组，1 通道和多通道分别处理。
- Sobel 核为标准 3 x 3 核除以 8，使用 `replicate` padding；同样按通道分组。
- 边缘图先对各通道梯度幅值求值，再对通道取均值，并进行逐图 min-max 归一化。
- 噪声可靠性图来自观测与 Gaussian 平滑观测的绝对残差，先对通道取均值，再逐图归一化。
- legacy 边缘残差仍比较 `Sobel(I_pred)` 与未经 Gaussian 滤波的 `Sobel(I_obs)`；Gaussian 平滑只用于生成边缘权重。这正是本轮需要隔离检查的监督差异。

以上 legacy 约定与新模式的 padding、通道域和归一化不同，因此不能把相同的 `lambda_edge` 解释为相同的梯度影响强度。

## 3. `filtered_edge` 定义

新项先把输入统一为 `[B, 1, H, W]`：1 通道直接使用，3 通道用固定 BT.601 权重 `(0.299, 0.587, 0.114)` 转亮度；其他通道数或非 CHW/BCHW 形状报错。预测与观测必须具有相同形状、dtype 和 device。

观测亮度分支显式 `detach`，预测分支保持可微。两条分支应用完全相同的固定算子：

```text
Gaussian: 5 x 5, sigma=1.0, reflect padding=2, 核权重和为 1
Sobel Kx: [[-1,0,1],[-2,0,2],[-1,0,1]] / 8
Sobel Ky: transpose(Kx), reflect padding=1
```

默认损失为：

```text
P  = luminance(I_pred)
Y  = stop_gradient(luminance(I_obs))
Ps = Gaussian(P)
Ys = Gaussian(Y)

L_edge_filtered = 0.5 * (
    mean(abs(Sobel_x(Ps) - Sobel_x(Ys)))
  + mean(abs(Sobel_y(Ps) - Sobel_y(Ys)))
)
```

这里匹配两个有符号方向的梯度，不匹配单一梯度幅值，也不做逐图 min-max 或其他动态范围归一化。图像尺寸不足以进行 reflect padding 时直接报错，不回退到 replicate 或零填充。

## 4. 参数语义与兼容性

| 参数 | 默认值 | 生效语义 |
| --- | ---: | --- |
| `aux_loss_version` | `legacy` | 可选 `legacy` / `filtered_edge`；旧配置缺省时保持 legacy |
| `edge_filter_kernel` | `5` | `filtered_edge` Gaussian 奇数核尺寸 |
| `edge_filter_sigma` | `1.0` | `filtered_edge` Gaussian 正 sigma |
| `lambda_thermal` | `0` | legacy 热强度项权重；filtered_edge 必须为 0 |
| `lambda_edge` | `0` | 当前模式边缘项权重 |
| `lambda_smooth` | `0` | legacy 非边缘平滑项权重；filtered_edge 必须为 0 |
| `noise_beta` | `5` | 仅 legacy 使用；filtered_edge 标记为 disabled |
| `edge_gamma` | `3` | 仅 legacy 使用；filtered_edge 标记为 disabled |

`filtered_edge` 若搭配非零 `lambda_thermal` 或 `lambda_smooth`，会在训练启动前报错，不会静默改写权重。所有权重和滤波参数都要求有限且满足非负/正值约束。

三项 lambda 全为零时存在双层短路：训练循环不会调用 `thermal_physics_loss`，损失 dispatcher 本身也会在构造 Gaussian/Sobel/可靠性图之前返回标量零。这保证 B0 不会为了计算或日志而进入新增辅助路径，也不会引入额外随机数、可学习参数或第二次 backward。

## 5. Runner 输入锁定

`configs/experiment_matrix.priority1_loss_revision.json` 为四个实验逐项声明以下预期文件 SHA256：

```text
dataset_manifest.v1.json
  d4fe9c97ed110fa1ecdf0636a19f7a7062169d3cfbd719d5594158f9d615e919

sparse_nested-random_25.json
  c9a1e0aa65e54e1558779b557539cb2eaba4d1faab765b786a4437aa519ba79e

degradation_manifest.v1.json
  06d68ec82b7faa8d65168b9bd2b35f013207331f4683a2ec0ad883837be3f03b
```

runner 在创建实验输出目录和启动子进程之前计算实际文件哈希；任一不一致即停止。成功进入运行后，`run_manifest.json` 同时记录实际哈希、预期哈希、完整训练/渲染命令、Git 状态和日志文件位置。adapter 随后还会校验 degradation manifest 内部引用哈希与逐视角输出哈希，这两层检查职责不同，均不能跳过。

四份正式 `run_manifest.json` 的实际哈希与上述预期值逐项一致，状态均为 `completed`。运行代码固定在干净 commit `dd5d6b139c2d3e899d0521240e87041ea622a155`（短写 `dd5d6b1`），四份清单记录的 Git status 均为空、diff SHA256 均为空 diff 的标准摘要。

## 6. 最小训练日志

Priority 1 矩阵设置 `quiet=true`、`log_interval=500`。runner 控制台只保留每组 train/render 的 started、completed 或 failed 生命周期消息；子进程的完整输出写入实验目录的 `stdout.log` 和 `stderr.log`，避免终端被 tqdm 和逐步输出淹没。

训练启动时只打印一次 `[AUX]` resolved 摘要，说明实际模式、三项 lambda、滤波设置，以及 `noise_beta` / `edge_gamma` 是否 disabled。之后每 500 iterations 和最后一次迭代输出一条可审计摘要，并同步追加到 `loss_components.csv`。CSV 字段为：

```text
iteration, aux_loss_version,
lambda_thermal, lambda_edge, lambda_smooth,
L_baseline, L_thermal_raw, L_edge_raw, L_smooth_raw,
weighted_L_thermal, weighted_L_edge, weighted_L_smooth,
L_total, Gaussian_count, avg_ms, peak_cuda_mb
```

关闭的项以 `disabled` 记录 raw 值、加权贡献记为 0，不会为了日志强制计算。记录前只 detach 当次已有标量，不保留计算图。模型保存、validation 评估、最终最佳 PSNR 和异常仍保留输出；这些属于必要的里程碑或故障信息。

安装 TensorBoard 时，`SummaryWriter` 以 5 秒间隔刷新。实时曲线包括 baseline/total、辅助项
raw/weighted、迭代耗时以及里程碑 validation 指标；disabled 的 raw 项不会被伪装成已计算
的零值。服务入口为 `http://127.0.0.1:6006/`，日志根目录是
`runs/priority1_loss_revision/`。

实际四组训练的 `loss_components.csv` 均包含 14 行数据，对应 500 至 7000、间隔 500 的完整摘要；四个 `stderr.log` 均为 0 字节。TensorBoard 的 1-iteration E2 集成检查已生成 event 文件，并在检查时得到本地 HTTP 200。这里记录的是已完成的集成验证，不表示 TensorBoard 服务需要在训练结束后持续运行。

## 7. 已知的验证路径差异

训练循环内部的 validation renderer 当前先对 Gaussian renderer 输出执行 `clamp(0, 1)`，再加 `TCM.step(image)`；训练损失路径则是 renderer 输出直接加 TCM，且 TCM 后不 clamp：

```text
训练：     renderer -> + TCM -> loss
内部验证： renderer -> clamp(0,1) -> + TCM -> metric
```

这是本轮开始前已存在的差异，本轮不修改，以避免在损失修订实验中同时改变评价路径。解释 validation 数值时必须保留这一限制。独立 `render.py` 的主 render_set 路径是 renderer 后加 TCM，并由保存图像的既有流程处理；它与训练内部 validation 的 clamp 顺序也不完全相同。

下文主表来自独立 `render.py` 写出的 PNG，再由冻结的 `tools/collect_metrics.py` 汇总；不是训练控制台内的 validation 数值。两条路径的 clamp 顺序不同，因此控制台 PSNR 与独立渲染指标可能有小幅差异，二者不能混作同一评价口径。

## 8. 第一批实验矩阵

| ID | 模式 | `lambda_thermal` | `lambda_edge` | `lambda_smooth` | 当前状态 |
| --- | --- | ---: | ---: | ---: | --- |
| B0 | legacy | 0 | 0 | 0 | 7k 完成 |
| O0 | legacy | 0.1 | 0.01 | 0.001 | 7k 完成 |
| E1 | legacy | 0 | 0.001 | 0 | 7k 完成 |
| E2 | filtered_edge | 0 | 0.001 | 0 | 7k 完成 |

共同条件：59 个 Sparse-25 训练视角、固定 noise03 退化、34 个未退化 validation 视角、seed 2026、resolution 1、`data_device=cpu`、`load2gpu_on_the_fly=true`、保存/验证 iterations 2000 和 7000。test 分区本轮不运行、不渲染、不查看。

E1 与 E2 同时存在滤波域、边缘加权、padding、通道处理和归一化差异，因此后续结果只能比较两套完整辅助项方案，不能把差值单独归因于“滤波观测”。

## 9. 运行完整性与耗时

四组均保存 2k、7k checkpoint，并在两个 checkpoint 上各生成 34 张 render 和 34 张同名 ground truth；配对检查全部为 34/34。7k 是 runner 完成后的最终渲染，2k 使用同一独立 renderer 补充渲染。整个过程只选择 `evaluation_partition=val`，没有渲染、计算或查看 39-view test。

从各组 `stdout.log` 的 `[AUX]` 启动记录到 `Training complete` 计算，训练进程墙钟耗时为：

| 实验 | 训练耗时 | 7000 最终 Gaussian 数 | `avg_ms` 记录范围 | 7000 `avg_ms` | CUDA allocator peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 7:40 | 24,876 | 40.62-53.99 ms | 53.99 ms | 597.1 MB |
| O0 | 8:41 | 27,325 | 47.92-61.69 ms | 61.69 ms | 597.3 MB |
| E1 | 8:00 | 27,253 | 43.19-56.85 ms | 56.85 ms | 598.3 MB |
| E2 | 7:49 | 26,548 | 42.35-55.46 ms | 55.46 ms | 597.0 MB |

墙钟耗时包含相机加载、训练、内部 validation 和 checkpoint 保存，不包含独立 `render.py`。
`avg_ms` 是训练器在 14 个日志点记录的、截至该 iteration 的 CUDA event 累计平均值；上表范围是这些累计平均值的最小到最大，不是单步延迟分布，也不含数据准备、validation、保存和独立渲染。因此它不能直接替代端到端墙钟耗时。显存同样是 PyTorch CUDA allocator 的峰值，不是系统层面的整卡占用。

## 10. Validation 指标

### 10.1 7000 iteration 主结果

主比较使用 `results/priority1_loss_revision/val_7000/metrics_summary.csv`。LPIPS 按本轮约定跳过，ROI mask 未冻结，因此二者均为 `unavailable`。

| 实验 | Views | PSNR ↑ | SSIM ↑ | T-MAE ↓ | E-MAE ↓ | Gradient preservation ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 34 | 30.55578541 | 0.94624436 | 0.02609596 | 0.00655253 | **0.53585431** |
| O0 | 34 | 30.02708431 | 0.94623453 | 0.02639369 | 0.00647433 | 0.53198543 |
| E1 | 34 | 30.39146361 | 0.94625547 | 0.02625460 | 0.00653443 | 0.52984605 |
| E2 | 34 | **30.64555695** | **0.94726418** | **0.02550185** | **0.00642407** | 0.53200635 |

相对 B0 的变化如下；PSNR、SSIM、Gradient preservation 为正更好，T-MAE、E-MAE 为负更好：

| 实验 | ΔPSNR | ΔSSIM | ΔT-MAE | ΔE-MAE | ΔGradient preservation |
| --- | ---: | ---: | ---: | ---: | ---: |
| O0 | -0.52870110 | -0.00000983 | +0.00029773 | -0.00007820 | -0.00386888 |
| E1 | -0.16432180 | +0.00001111 | +0.00015864 | -0.00001810 | -0.00600826 |
| E2 | +0.08977154 | +0.00101982 | -0.00059411 | -0.00012846 | -0.00384796 |

在这一次验证中，E2 相对 B0 提高 PSNR/SSIM，并降低 T-MAE/E-MAE，但 Gradient preservation 下降；O0 的 PSNR 和 T-MAE 明显差于 B0，E1 也没有形成综合优势。E2 的优势幅度较小，且梯度指标存在权衡，不能据此宣称已经解决泛化问题或确定收益来自某个单一算子差异。

### 10.2 2000 iteration 次要结果

| 实验 | Views | PSNR ↑ | SSIM ↑ | T-MAE ↓ | E-MAE ↓ | Gradient preservation ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 34 | 29.60748353 | 0.94464562 | 0.02637973 | 0.00675459 | **0.55646594** |
| O0 | 34 | 28.70891908 | 0.94206068 | 0.03090710 | 0.00674758 | 0.55075741 |
| E1 | 34 | 29.62049671 | 0.94400735 | 0.02703862 | 0.00675309 | 0.54747425 |
| E2 | 34 | **29.62912333** | **0.94483947** | 0.02679957 | **0.00669644** | 0.54170621 |

2k 仅用于观察早期行为；预先指定的主比较仍是 7k，不能改用 2k 单项最优结果选择结论。

## 11. 验证状态与停止边界

截至本文记录时间：

- 单元与回归测试：**91/91 通过**，包含两项实际 CUDA 扩展测试；新损失专项测试 24/24 通过。
- CUDA 短步数兼容检查：**通过**。B0 与 E2 均完成 2 iterations 的前向、反向、保存和 34-view val 渲染；输出位于 `runs/priority1_loss_revision_smoke/`。
- 最小日志检查：**通过**。两组 `loss_components.csv` 均有两条数据；B0 三个 raw 项均为 `disabled`，E2 仅计算 `L_edge_raw`。
- TensorBoard 集成检查：**通过**。Python 3.11 环境使用 TensorBoard 2.21.0，1-iteration E2 event 文件包含 baseline、total、edge raw/weighted、thermal/smooth weighted 和 iter-time 标量；本地服务已在 `127.0.0.1:6006` 验证 HTTP 200。
- Priority 1 matrix dry-run 与命令核对：**通过**。四组使用相同 source/dataset/split/degradation、seed 2026、resolution 1、7000 iterations 和 val 分区。
- 输入文件 SHA256：**三项均与规格一致**；正式 runner 还会在每组创建目录前再次校验。
- B0、O0、E1、E2 四组 7k 训练：**全部完成**，四份 run manifest 均为 `completed`，`stderr.log` 均为 0 字节。
- 2k/7k validation 渲染与指标收集：**全部完成**，每组每个 checkpoint 均为 34/34 配对。
- test：**未访问**；本轮没有 test 渲染、指标或人工查看。

本轮在 Priority 1 验收后停止，没有自动追加 30k、多 seed、其他场景或超参数搜索。现有结果只覆盖 heated 单场景、Sparse-25 + noise03、seed 2026；尚不能报告均值/标准差，也不能把 validation 趋势当作最终 test 结论。
