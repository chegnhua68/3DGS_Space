# Thermal3D-GS 稀疏视角退化红外重建：正式训练计划

> 版本：v1.0<br>
> 适用项目：`Thermal3DGS_sparse`<br>
> 当前主场景：TI-NSD `heated`<br>
> 研究主线：**Thermal3D-GS 复现 + 稀疏视角红外成像 + 低信噪比/弱纹理退化 + 噪声感知边缘保持热一致性约束**

---

## 1. 训练目标与最终结论边界

本计划用于从工程 smoke test 进入可用于论文的正式训练。最终需要回答四个问题：

1. 原始 Thermal3D-GS 在红外视角逐渐稀疏时，性能如何变化？
2. 低信噪比和弱纹理退化会怎样影响红外新视角合成？
3. 新增的噪声感知边缘保持热一致性损失能否稳定改善结果？
4. 改进是否在多个预注册随机种子、相同训练数据和相同优化预算下成立？

当前 10-iteration 结果只用于确认流程可运行，不能作为性能结论。论文结论必须基于：

- 30,000 次完整训练迭代；
- baseline 与 proposed 严格配对；
- 至少 3 个预注册 seed；
- 独立 validation 调参；
- 超参数冻结后的 39-view test；
- 均值、标准差和逐视角配对结果。

---

## 2. 冻结的实验协议

### 2.1 数据集与划分

当前使用：

```text
data/TI-NSD/heated/
```

已注册视角：

| 分区 | 数量 | 用途 |
|---|---:|---|
| `train_pool` | 234 | 生成不同稀疏比例的训练子集 |
| `val` | 34 | 调整损失权重、选择实验配置 |
| `test` | 39 | 超参数冻结后的最终报告 |
| 总计 | 307 | 已注册 COLMAP 视角 |

采用**固定相机位姿协议**：

- 稀疏实验只减少训练图像数量；
- 所有设置复用完整 COLMAP 标定；
- 不在当前论文中研究稀疏红外图像的位姿重估；
- `val` 和 `test` 始终读取原始干净图像。

### 2.2 稀疏视角比例

正式实验使用：

| 名称 | 比例 | 约对应训练视角数 |
|---|---:|---:|
| Full | 100% | 234 |
| Sparse-50 | 50% | 117 |
| Sparse-25 | 25% | 约 59 |
| Sparse-12.5 | 12.5% | 约 29 |

每个 seed 内使用嵌套划分：

```text
12.5% subset ⊂ 25% subset ⊂ 50% subset ⊂ 100%
```

### 2.3 预注册 seed

正式论文实验固定使用：

```text
2026, 2027, 2028
```

每个 seed 应同时控制：

- 稀疏视角选择；
- 退化噪声的逐图随机数；
- 模型初始化和训练随机过程。

**正式训练前的阻塞检查：**当前 README 没有明确给出训练随机 seed 的命令行参数。必须先确认训练器和 runner 是否真实记录并设置了 Python、NumPy、PyTorch 与 CUDA 随机种子。若没有，应先增加一个可审计的训练 seed 参数，再启动三 seed 正式矩阵。

### 2.4 Baseline 与 Proposed

#### Baseline

保留原始 Thermal3D-GS：

- 3D Gaussian 表示；
- ATF；
- TCM；
- L1 损失；
- SSIM 损失；
- 角点损失。

关闭全部新增损失：

```text
lambda_thermal = 0
lambda_edge    = 0
lambda_smooth  = 0
```

#### Proposed

在完全相同的 Thermal3D-GS 主干上增加：

```text
L_thermal：噪声可靠性加权的热强度重建损失
L_edge：热边缘梯度保持损失
L_smooth：非边缘区域平滑损失
```

初始候选参数：

```text
lambda_thermal = 0.1
lambda_edge    = 0.01
lambda_smooth  = 0.001
noise_beta     = 5
edge_gamma     = 3
```

Baseline 和 Proposed 必须共享：

- 相同 dataset manifest；
- 相同 split manifest；
- 相同 degradation manifest；
- 相同图像字节；
- 相同相机位姿；
- 相同训练迭代；
- 相同 optimizer 与 densification 设置；
- 相同分辨率；
- 相同评估器；
- 相同 seed。

---

## 3. 全局训练设置

| 项目 | 正式设置 |
|---|---|
| 总迭代次数 | `30000` |
| 中间检查点 | `7000`, `15000`, `30000` |
| 最终论文检查点 | 默认固定为 `30000` |
| 训练图存储 | `--data_device cpu` |
| 当前视角传入 GPU | `--load2gpu_on_the_fly` |
| 验证分区 | `--evaluation_partition val` |
| 测试分区 | 超参数冻结后单独渲染 `test` |
| 正式指标 | PSNR、SSIM、LPIPS、T-MAE、E-MAE、Gradient preservation |
| 条件指标 | ROI-MAE，仅在固定 ROI mask 已冻结时报告 |

### 3.1 分辨率决策

目标优先使用：

```text
--resolution 1
```

由于当前 GPU 为 6 GB，必须先做 densification 后的容量校准。规则如下：

1. 先用 `resolution 1` 跑过第 500 次迭代，并继续到至少 2,000 次；
2. 若稳定，再跑到 7,000 次确认 densification 阶段显存；
3. 若发生稳定可复现的 OOM，则统一改为 `resolution 2`；
4. 一旦决定正式分辨率，所有 baseline、proposed、seed 和条件必须统一；
5. 禁止在同一论文表格中混用 `resolution 1/2/4`。

### 3.2 最终检查点规则

为避免根据验证集反复挑选有利迭代，正式主结果默认统一使用：

```text
iteration_30000
```

`7000` 和 `15000` 只用于：

- 观察收敛曲线；
- 检查是否出现发散或退化；
- 估计训练时间和显存；
- 排查方法实现问题。

若后续发现 30k 在所有方法和条件下均发生系统性过拟合，才允许在**查看 test 之前**重新制定统一的 checkpoint 选择规则，并从头冻结。

---

## 4. 目录与命名规范

### 4.1 数据目录

```text
data/TI-NSD/heated/
├── images/
├── sparse/0/
├── dataset_manifest.v1.json
├── base_split.v1.json
├── splits/
│   ├── seed2026/
│   ├── seed2027/
│   └── seed2028/
└── derived/
    ├── noise03_sparse25_seed2026/
    ├── contrast05_sparse25_seed2026/
    ├── combined_sparse25_seed2026/
    └── ...
```

### 4.2 正式训练输出

建议固定为：

```text
runs/formal/<condition>/<method>/seed_<seed>/
```

示例：

```text
runs/formal/sparse25_clean/baseline/seed_2026/
runs/formal/sparse25_clean/proposed/seed_2026/
runs/formal/sparse25_noise03/baseline/seed_2027/
runs/formal/sparse25_noise03/proposed/seed_2027/
```

### 4.3 指标输出

```text
results/formal/<condition>/
```

每个条件至少生成：

```text
metrics_summary.csv
metrics_summary.md
metrics_per_view.csv
```

总表建议汇总到：

```text
results/formal/all_conditions_summary.csv
results/formal/all_conditions_summary.md
```

### 4.4 实验命名规则

```text
<method>_<condition>_seed<seed>_r<resolution>_30k
```

例如：

```text
baseline_sparse25_noise03_seed2026_r1_30k
proposed_sparse25_noise03_seed2026_r1_30k
```

---

## 5. 分阶段训练计划

## Phase 0：正式训练前冻结与检查

### 目标

确认代码、数据、随机性和审计链路适合正式实验。

### 必做清单

- [ ] 当前代码提交到一个明确 Git commit；工作树保持干净。
- [ ] 记录上游 commit 和当前扩展 commit。
- [ ] 全部 68 个测试继续通过。
- [ ] dataset、split、degradation 清单 SHA256 校验通过。
- [ ] `run_manifest.json` 能记录完整命令和代码状态。
- [ ] 训练 seed 能真实控制并记录 Python、NumPy、PyTorch、CUDA 随机性。
- [ ] baseline 新增损失权重为零时不会进入新增损失计算路径。
- [ ] baseline/proposed 对相同训练图文件计算出的 SHA256 完全一致。
- [ ] 终端训练日志被保存到独立日志文件。

### 退出条件

所有检查通过后，才能进入容量校准。

---

## Phase 1：显存与稳定性容量校准

### 条件

使用具有代表性的困难条件：

```text
Sparse-25 + Gaussian noise sigma=0.03
seed=2026
```

### 顺序

1. Baseline：2,000 iterations，`resolution 1`；
2. Proposed：2,000 iterations，`resolution 1`；
3. 两者均稳定后，各继续验证到 7,000 iterations；
4. 记录峰值显存、平均迭代时间、Gaussian 数量变化和是否出现 NaN/OOM。

### 决策

- 两条路径在 7k 前均稳定：正式矩阵使用 `resolution 1`；
- 任一路径稳定 OOM：改为 `resolution 2`，重新执行本阶段；
- 禁止只给 Proposed 降分辨率或改变 densification 参数。

### 输出

```text
runs/calibration/...
results/calibration/...
docs/formal_training_log.md
```

容量校准结果不进入论文主表。

---

## Phase 2：Baseline 复现锚点

### 目标

在不加退化、完整训练视角下，获得一个正式 30k Thermal3D-GS baseline 结果，作为后续实验的复现锚点。

### 条件

```text
Full-view clean
Baseline
seed=2026
30,000 iterations
```

### 必须保存

- 7k、15k、30k Gaussian/ATF/TCM；
- 34-view validation 渲染；
- 训练日志；
- PSNR、SSIM、LPIPS、T-MAE、E-MAE、Gradient preservation；
- 训练总时间、渲染时间和峰值显存。

### 退出条件

- 训练完整结束；
- 模型和渲染文件无缺失；
- 指标工具严格配对全部 34 个 val 视角；
- 结果明显优于 100-iteration 容量 smoke，且无系统性伪影或数值异常。

这里不设置外部论文数值复现阈值，因为当前 README 没有提供与本地环境、相同数据划分完全一致的官方 checkpoint 和标准值。

---

## Phase 3：新增损失超参数筛选

### 目标

只在 validation 上选择 Proposed 的损失权重。禁止查看 test。

### 固定条件

```text
Sparse-25 + Gaussian noise sigma=0.03
seed=2026
noise_beta=5
edge_gamma=3
```

### 第一轮：7k 快速筛选

| ID | λthermal | λedge | λsmooth | 目的 |
|---|---:|---:|---:|---|
| H0 | 0 | 0 | 0 | Baseline |
| H1 | 0.05 | 0.01 | 0.001 | 较弱热强度约束 |
| H2 | 0.10 | 0.01 | 0.001 | 当前默认配置 |
| H3 | 0.20 | 0.01 | 0.001 | 较强热强度约束 |
| H4 | 0.10 | 0.005 | 0.001 | 较弱边缘约束 |
| H5 | 0.10 | 0.020 | 0.001 | 较强边缘约束 |
| H6 | 0.10 | 0.010 | 0 | 去掉平滑项 |
| H7 | 0.10 | 0.010 | 0.002 | 较强非边缘平滑 |

### 第二轮：30k 确认

从 H1–H7 中选 validation 表现最好的 2 组，完成 30k。与 30k baseline 比较后冻结最终权重。

### 预注册选择规则

依次判断：

1. PSNR 相对 baseline 不出现明显下降；
2. T-MAE 更低；
3. E-MAE 更低；
4. Gradient preservation 更高；
5. SSIM 作为同等条件下的辅助判据；
6. LPIPS 只作为次要感知指标，不单独决定配置。

如果不同指标冲突，优先选择在 T-MAE、E-MAE 和 Gradient preservation 上更均衡，而非只追求单一 PSNR 峰值的配置。

### 限制

- 不根据 test 结果修改权重；
- 不在多个场景或条件间反复试到出现有利结果；
- 只允许一次预先记录的第二阶段小范围搜索；
- 最终参数冻结后，正式主矩阵中不得再改。

---

## Phase 4：正式核心实验矩阵

超参数冻结后，执行以下 7 个条件。每个条件运行 baseline/proposed × 3 seeds。

| 条件 ID | 训练视角 | 训练退化 | 研究问题 | 运行数 |
|---|---:|---|---|---:|
| C1 | 100% | clean | 完整视角基准 | 6 |
| C2 | 50% | clean | 轻度稀疏 | 6 |
| C3 | 25% | clean | 中度稀疏 | 6 |
| C4 | 12.5% | clean | 极端稀疏 | 6 |
| C5 | 25% | Gaussian noise `σ=0.03` | 低信噪比 | 6 |
| C6 | 25% | contrast `α=0.50` | 弱纹理/低热对比度 | 6 |
| C7 | 25% | contrast `0.50` → blur `σ=1.0, k=3` → noise `0.03` | 组合退化 | 6 |

核心正式训练总数：

```text
7 conditions × 2 methods × 3 seeds = 42 runs
```

### 每个条件的配对规则

例如 C5、seed 2027：

```text
baseline_sparse25_noise03_seed2027
proposed_sparse25_noise03_seed2027
```

这两次训练必须共享同一个：

- split manifest；
- degradation manifest；
- train image SHA256；
- 模型初始化 seed；
- 优化预算；
- 分辨率；
- densification 参数。

### 执行顺序

为了尽早发现问题，建议按以下顺序跑：

```text
C3 clean
→ C5 noise03
→ C6 contrast05
→ C7 combined
→ C1 full
→ C2 sparse50
→ C4 sparse12.5
```

每个条件先完成 seed 2026 的 baseline/proposed 配对并检查结果，再提交 seed 2027 和 2028。不要先把全部 baseline 跑完、数天后再跑 proposed，以免中途代码或系统状态变化。

---

## Phase 5：损失消融实验

### 固定条件

```text
Sparse-25 + combined degradation
seeds = 2026, 2027, 2028
30,000 iterations
```

### 消融配置

| 配置 | Lthermal | Ledge | Lsmooth |
|---|---|---|---|
| A0 Baseline | 关闭 | 关闭 | 关闭 |
| A1 Thermal only | 开启 | 关闭 | 关闭 |
| A2 Edge only | 关闭 | 开启 | 关闭 |
| A3 Smooth only | 关闭 | 关闭 | 开启 |
| A4 Thermal + Edge | 开启 | 开启 | 关闭 |
| A5 Full Proposed | 开启 | 开启 | 开启 |

A0 和 A5 已包含在核心矩阵 C7 中，因此新增训练量为：

```text
4 extra configurations × 3 seeds = 12 runs
```

消融阶段结束后可以回答：

- 噪声可靠性加权是否真正有效；
- 边缘保持是否重复了 TCM 的功能；
- 非边缘平滑是否带来过平滑；
- 三项组合是否存在互补作用。

---

## Phase 6：扩展退化曲线（资源允许时）

核心矩阵完成后，再补完整鲁棒性曲线。

### 低信噪比曲线

固定 Sparse-25：

```text
σ = 0.01, 0.03, 0.05
```

其中 `0.03` 已在 C5 中完成，只需新增 `0.01` 和 `0.05`。

### 弱纹理曲线

固定 Sparse-25：

```text
contrast α = 0.75, 0.50, 0.25
```

其中 `0.50` 已在 C6 中完成，只需新增 `0.75` 和 `0.25`。

### 模糊曲线

固定 Sparse-25：

```text
blur σ = 0.5, 1.0, 1.5
kernel size = 3 or 5，按预先配置固定
```

扩展矩阵应继续使用 baseline/proposed × 3 seeds。若时间有限，优先顺序为：

```text
noise severity > contrast severity > blur severity
```

---

## Phase 7：最终 Test 与统计报告

只有满足以下条件后才允许渲染 test：

- Proposed 权重已冻结；
- 正式分辨率已冻结；
- 正式 checkpoint 规则已冻结；
- 核心条件和 seed 已冻结；
- 不再计划根据 test 调整方法。

### Test 渲染

每个正式 run 使用：

```text
iteration = 30000
evaluation_partition = test
39 test views
```

### 必报指标

| 指标 | 方向 | 论文解释 |
|---|---|---|
| PSNR | ↑ | 整体像素重建质量 |
| SSIM | ↑ | 结构相似性 |
| LPIPS | ↓ | 感知差异，次要指标 |
| T-MAE | ↓ | 固定编码范围下的表观热强度误差 |
| E-MAE | ↓ | GT 热边缘加权的梯度误差 |
| Gradient preservation | ↑ | 热边缘与梯度保持能力 |
| ROI-MAE | ↓ | 仅在固定 ROI mask 可用时报告 |

### 统计方式

每个条件至少报告：

```text
mean ± standard deviation across 3 seeds
```

同时报告：

- Proposed − Baseline 的逐 seed 差值；
- 39 个 test 视角的逐视角配对差值；
- 推荐增加 95% paired bootstrap confidence interval；
- 若做显著性检验，使用逐视角配对检验，并同时报告效应大小。

不能只报告最好的 seed，也不能删除负结果。

### 定性图选择

在查看 Proposed 与 Baseline 差异前，预先固定 4–6 个 test view ID，覆盖：

- 热边缘明显区域；
- 大面积平坦弱纹理区域；
- 小型热目标；
- 遮挡或视角变化较大的区域。

论文图统一展示：

```text
GT | Baseline | Proposed | Baseline Error | Proposed Error | Sobel Edge
```

---

## 6. 正式训练命令模板

以下以 `Sparse-25 + noise03 + seed2026` 为例。路径需替换为实际生成的 25% split 和 degradation manifest。

### 6.1 Baseline

```powershell
.\.venv\Scripts\python.exe train.py `
  -s data\TI-NSD\heated `
  -m runs\formal\sparse25_noise03\baseline\seed_2026 `
  --eval `
  --dataset_manifest data\TI-NSD\heated\dataset_manifest.v1.json `
  --split_manifest data\TI-NSD\heated\splits\seed2026\sparse_nested-random_25.json `
  --degradation_manifest data\TI-NSD\heated\derived\noise03_sparse25_seed2026\degradation_manifest.v1.json `
  --evaluation_partition val `
  --resolution 1 `
  --data_device cpu `
  --load2gpu_on_the_fly `
  --iterations 30000 `
  --test_iterations 7000 15000 30000 `
  --save_iterations 7000 15000 30000 `
  --lambda_thermal 0 `
  --lambda_edge 0 `
  --lambda_smooth 0
```

### 6.2 Proposed

```powershell
.\.venv\Scripts\python.exe train.py `
  -s data\TI-NSD\heated `
  -m runs\formal\sparse25_noise03\proposed\seed_2026 `
  --eval `
  --dataset_manifest data\TI-NSD\heated\dataset_manifest.v1.json `
  --split_manifest data\TI-NSD\heated\splits\seed2026\sparse_nested-random_25.json `
  --degradation_manifest data\TI-NSD\heated\derived\noise03_sparse25_seed2026\degradation_manifest.v1.json `
  --evaluation_partition val `
  --resolution 1 `
  --data_device cpu `
  --load2gpu_on_the_fly `
  --iterations 30000 `
  --test_iterations 7000 15000 30000 `
  --save_iterations 7000 15000 30000 `
  --lambda_thermal 0.1 `
  --lambda_edge 0.01 `
  --lambda_smooth 0.001 `
  --noise_beta 5 `
  --edge_gamma 3
```

> 以上权重只是初始示例。Phase 3 结束后，应替换为冻结的最终权重。若容量校准最终选择 `resolution 2`，所有正式命令统一替换，不能只改部分实验。

### 6.3 Test 渲染

```powershell
.\.venv\Scripts\python.exe render.py `
  -m runs\formal\sparse25_noise03\proposed\seed_2026 `
  --iteration 30000 `
  --skip_train `
  --evaluation_partition test `
  --mode render
```

### 6.4 配对指标收集

```powershell
.\.venv\Scripts\python.exe tools\collect_metrics.py `
  --experiment baseline=runs\formal\sparse25_noise03\baseline\seed_2026\test\ours_30000 `
  --experiment proposed=runs\formal\sparse25_noise03\proposed\seed_2026\test\ours_30000 `
  --output-dir results\formal\sparse25_noise03\seed_2026 `
  --device cuda
```

正式结果不要使用 `--skip-lpips`。若 ROI mask 尚未冻结，则 ROI-MAE 保持 `unavailable`，不要临时根据结果画 ROI。

---

## 7. 每次训练必须记录的内容

每个 run 至少保留：

- 完整命令；
- Git HEAD、diff 状态和源码快照哈希；
- dataset/split/degradation manifest SHA256；
- 训练 seed；
- 开始与结束时间；
- GPU 型号、分辨率和峰值显存；
- 7k、15k、30k 的训练损失；
- Gaussian 数量随迭代变化；
- ATF、TCM 与 point cloud 的同迭代快照；
- val/test 渲染目录；
- 全部指标 CSV；
- 失败状态与失败原因。

建议终端日志写入：

```text
runs/formal/<condition>/<method>/seed_<seed>/stdout.log
```

runner 当前不会自动捕获 stdout/stderr，应由 PowerShell、作业调度器或外层脚本重定向。

---

## 8. 异常处理规则

### 8.1 OOM

- 不要只对失败方法降低分辨率；
- 先确认是否发生在 densification 后；
- 若需要改 `resolution`，整个正式矩阵从头统一执行；
- 不通过单独减少 Proposed 的 Gaussian 数量获得不公平显存优势。

### 8.2 NaN 或训练发散

- 保留失败目录和日志；
- 标记为 `failed`，不得静默删除；
- 先在同一 seed、同一输入上复现；
- 修复代码后使用新实验版本和新输出根目录重新训练；
- 代码修复后，受影响的 baseline/proposed 配对都应重跑。

### 8.3 训练中断

当前保存的 Gaussian、ATF 和 TCM 不是完整优化器 checkpoint。除非后续明确加入可靠的 resume 状态，否则正式实验中断后应从头重跑，不能把模型快照当作无偏续训点。

### 8.4 中途修改代码

一旦 Phase 4 开始：

- 禁止在同一实验版本中修改训练逻辑；
- 必须修改时，创建 `v2` 输出根目录；
- 所有受影响的 baseline/proposed 配对重新执行；
- 不把不同 commit 的结果混在同一均值中。

### 8.5 负结果

- 不删除不利 seed；
- 不删除困难视角；
- 不临时更换退化强度；
- 不在 test 上重新调损失权重；
- 将失败条件作为鲁棒性边界如实报告。

---

## 9. 完成判据

### 最低可用于论文主结论

- [ ] Full/50%/25%/12.5% clean 的 baseline/proposed 三 seed 完成；
- [ ] Sparse-25 + noise03 三 seed 完成；
- [ ] Sparse-25 + contrast0.5 三 seed 完成；
- [ ] Sparse-25 + combined 三 seed 完成；
- [ ] 42 个核心正式 run 均有完整日志和 run manifest；
- [ ] 12 个新增消融 run 完成；
- [ ] 39-view test 在超参数冻结后完成；
- [ ] 所有指标均报告 mean ± std；
- [ ] 至少提供逐视角配对差值；
- [ ] 论文定性视角在看结果前预先固定；
- [ ] 没有将热强度误差表述成真实温度误差。

### 完整版本

在最低版本基础上，进一步完成：

- [ ] noise `0.01/0.03/0.05` 曲线；
- [ ] contrast `0.75/0.50/0.25` 曲线；
- [ ] blur 曲线；
- [ ] 95% 配对置信区间；
- [ ] 固定 ROI mask 后的 ROI-MAE；
- [ ] 训练/渲染时间与峰值显存对比；
- [ ] 五类论文图和 figure manifest。

---

## 10. 推荐的实际启动顺序

不要直接一次提交全部 42 个正式 run。按下面顺序执行：

```text
1. 冻结 commit、随机 seed 和正式分辨率
2. Sparse-25 + noise03：baseline/proposed 容量校准到 7k
3. Full-view clean baseline：seed2026，正式 30k
4. Sparse-25 + noise03：完成 validation 超参数筛选
5. 冻结 Proposed 权重
6. 核心矩阵先跑 seed2026 配对
7. 检查全部目录、指标与图像是否正常
8. 再跑 seed2027 和 seed2028
9. 完成消融
10. 冻结全部设置后统一渲染 test
11. 汇总统计、曲线和论文图
```

第一批正式任务应当是：

```text
Phase 1：Sparse-25 + noise03 的 baseline/proposed 7k 容量校准
```

而不是立即把 30k × 多条件全部启动。这样可以在最小代价下先确认 6 GB 显存、densification 和新增损失均能稳定工作。
