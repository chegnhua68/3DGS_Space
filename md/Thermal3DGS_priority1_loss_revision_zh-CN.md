# Thermal3D-GS 第一优先级修改任务：修正边缘监督，简化新增损失

> 用途：交给 Codex，在现有 `Thermal3DGS_sparse` 仓库中实施最小修改。
> 范围：仅执行上一份泛化诊断计划的“第一优先级”。
> 状态：本文是待实施的修改规格，不表示代码已修改、测试已通过或性能已经提升。
> 本轮目标：检查含噪梯度监督和多项辅助损失是否造成性能损失，不预设已证实过拟合。

## 1. 本轮只做什么

保留原始 Thermal3D-GS 的 Gaussian、ATF、TCM、原始损失和优化流程，只新增一个可切换的损失版本：

1. 将新增边缘损失中的原始梯度匹配，改为**预测图与含噪观测经过相同轻度滤波后的梯度匹配**。
2. 新版本暂时设置 `lambda_thermal=0`、`lambda_smooth=0`，只保留固定小权重的边缘辅助项。
3. 保留旧版完整损失与旧版边缘单项，便于对照和回退。
4. 完成参数接入、最小日志、回归测试和四组 7k 验证实验配置。

**本轮不做：**可靠边缘掩膜、噪声阈值估计、warm-up/ramp/decay 调度、增密梯度分离、Gaussian 数量限制、网络结构修改、学习率调整、训练干净参考诊断系统、完整 resume，以及 30k 多条件正式矩阵。

新方法暂称：**滤波域边缘一致性正则化（Filtered-Domain Edge Consistency Regularization）**。本轮不将它称为经过标定的物理噪声模型，也不声称它能保证避免过拟合。

## 2. 依据、现状与源码核对边界

### 2.1 来源

- **[S1]** 用户上传的 `README_zh-CN(1).md`：现有损失公式、项目结构、清单契约、评估器和模型快照限制。仓库中的实际文件名可能仍为 `README_zh-CN.md`。
- **[S2]** `formal_training_log.md`：Sparse-25 + noise03、seed 2026 的 2k/7k 校准设置、输入哈希、指标和日志状态。
- **[S3]** `Thermal3DGS_generalization_revision_plan_v2_zh-CN.md`：上一轮提出的最小滤波域损失与 B0/O0/E1/E2 对照方案。

现有功能与运行事实来自 [S1]、[S2]；新损失、参数名称、输出目录和验收任务属于本次拟定的实现要求。

当前提供的是文档，不是完整本地源码。Codex 必须先读本地实现，不得假设下文建议的新参数或函数已存在。若源码与 README 不一致，先记录具体差异，再实施修改，不能悄悄改动旧方法来迎合文档公式。

README 的“当前验证状态”仍有早期 smoke 记录；训练日志另外报告了后续 2k/7k 校准。保留这种时间区别：以实际 `run_manifest.json` 核实具体运行，不因为旧状态描述而重新生成或覆盖已有数据。[S1, S2]

### 2.2 现有结果只说明什么

相同 Sparse-25 + noise03 + resolution=1 条件下，Proposed 的验证 PSNR 从 2k 的 `28.91107187` 上升到 7k 的 `30.22725793`，但仍低于对应 Baseline。[S2]

因此，当前是“新增方法尚未带来综合优势”，不是已经证明“训练越久，验证性能越差”。本轮修订用于检验监督目标设计，不把可能原因当作已确认结论。

### 2.3 修改前必须核对

- `train.py` 中新增损失实际接收哪张预测图：TCM 前、TCM 后，是否经过 clamp。
- 新增损失中的 `I_gt` 是否确实来自当前 `degradation_manifest` 指向的训练观测。
- 旧 `thermal_physics_loss.py` 的通道、Sobel 核尺度、padding、归一化和 reduction。
- 参数由 `arguments/` 中哪个类解析，runner 如何转发并写入 `cfg_args`。
- 三项权重全零时，是否确实跳过新增损失路径。

将核对结果写入 `docs/priority1_loss_revision.md`。若发现影响数据隔离或公平比较的问题，先停止运行并记录，不在一个实验分支中单独修复后与未修复分支比较。

## 3. 需要解决的两个具体问题

### 3.1 旧边缘权重被平滑，但监督目标仍然含噪

README 给出的旧公式为：[S1]

```text
E = normalize(|Sobel(Gaussian(I_gt))|)

L_edge_legacy = mean(
    (1 + edge_gamma * E)
    * |Sobel(I_pred) - Sobel(I_gt)|
)
```

在当前退化训练中，这里的 `I_gt` 是有噪声观测，而不是未退化真值。`E` 来自平滑图，并不意味着 `Sobel(I_gt)` 已去除高频噪声响应。

本轮新增代码用 `I_obs` 表示实际训练观测，避免将其误认成干净标签。只调整新增边缘项，不修改原 Baseline 所使用的训练目标。

### 3.2 额外加权热损失并没有替代 Baseline 的原始像素监督

当前是：[S1]

```text
L_total = L_baseline
        + lambda_thermal * mean(W * |I_pred - I_obs|)
        + lambda_edge * L_edge_legacy
        + lambda_smooth * L_smooth
```

只看其中的 L1 部分，若 Baseline 的 L1 系数为 `a`，合并后的像素系数为：

```text
a + lambda_thermal * W[p]
```

因此，`W` 小表示少增加一点附加权重，不代表已经将原始 Baseline 的该像素约束降权。这个判断是对文档公式的展开，不是断言该项一定无效。

为隔离问题，本轮新版本关闭额外热强度项和非边缘平滑项；**不要删除它们的旧实现，也不要替换 Baseline 的原 L1 或 SSIM。**

## 4. 冻结的数据和训练条件

以下条件沿用已完成校准，不重新设计：[S1, S2]

| 项目 | 本轮固定设置 |
| --- | --- |
| 数据根目录 | `data/TI-NSD/heated/` |
| 位姿协议 | 固定 COLMAP 位姿；不重估相机 |
| 稀疏划分 | Sparse-25，split seed 2026 |
| 训练视角 | 59 |
| 训练退化 | 已生成的固定 Gaussian noise，归一化 `sigma=0.03` |
| 训练随机种子 | `--seed 2026` |
| validation | 同一 34 个原始、未加本次合成退化的视角 |
| test | 同一 39 个视角；本轮不渲染、不计算指标、不查看图像 |
| 分辨率 | `resolution=1` |
| 图像加载 | `data_device=cpu`，`load2gpu_on_the_fly=true` |
| 每组训练 | 从头运行至 7000 iterations |
| 保存与验证 | 2000、7000；主比较固定为 7000 |
| 主干与原损失 | Gaussian、ATF、TCM、原 L1/SSIM/角点项全部保留 |
| 其他设置 | 继承原校准的学习率、优化器、背景、ATF/TCM 时序、增密、剪枝和 opacity reset |

输入路径从本地 `configs/experiment_matrix.calibration_sparse25_noise03_7k.json` 及实际运行清单读取。**不得猜测退化目录，也不要为了名称整齐重新生成退化图。**

[S2] 记录的预期输入文件 SHA256 为：

```text
dataset_manifest.v1.json
  d4fe9c97ed110fa1ecdf0636a19f7a7062169d3cfbd719d5594158f9d615e919

sparse_nested-random_25.json
  c9a1e0aa65e54e1558779b557539cb2eaba4d1faab765b786a4437aa519ba79e

degradation_manifest.v1.json
  06d68ec82b7faa8d65168b9bd2b35f013207331f4683a2ec0ad883837be3f03b
```

若实际哈希不符，停止并说明差异，不能静默覆盖清单或跳过校验。仅迁移了目录时也应核实解析基准，不能改变清单内容来强行通过。

四个方法共享初始状态、输入字节和抽样协议；新增分支不得额外消耗训练 RNG。不同损失可能导致最终 Gaussian 数量不同，不要求将结果人为剪成相同点数。

## 5. 新增损失的精确定义

### 5.1 输入和强度域

新函数只接收当前训练迭代的：

```text
I_pred：与旧辅助损失相同位置的预测图，保留计算图
I_obs ：当前退化清单加载的训练观测，作为无梯度目标
```

使用原数据的固定强度映射，不做逐图 min-max。不读取未退化训练图作为损失目标，不读取 val/test 生成任何训练约束。

新辅助项统一在单通道强度域计算：单通道直接使用；三通道使用固定 BT.601 权重 `(0.299, 0.587, 0.114)`。这只是新辅助项的设计选择，不更改原 Baseline 的通道处理，也不把伪彩色亮度解释为真实温度。

同时支持训练器实际传入的 `[C,H,W]` 和 `[B,C,H,W]`，内部统一成 `[B,1,H,W]`。仅接受明确处理后的 1 或 3 通道；其他形状报出清晰错误。保留已有加载器对 alpha 的处理，不在新损失里暗中重新合成背景。

若预测图在现有位置允许超出 `[0,1]`，不要为了新损失单独增加硬 clamp；沿用现有位置和计算图。

### 5.2 相同滤波作用于两条分支

```text
P = luminance(I_pred)
Y = stop_gradient(luminance(I_obs))

P_s = Gaussian(P)
Y_s = Gaussian(Y)
```

固定第一轮算子：

```text
Gaussian kernel size = 5
Gaussian sigma       = 1.0 pixel
Gaussian padding     = reflect，宽度 2
Gaussian weights     = 固定非学习参数，核权重和为 1
```

建议直接使用归一化二维核：

```text
G[i,j] = exp(-(i*i + j*j) / (2*sigma*sigma)) / Z
其中 i,j ∈ {-2,-1,0,1,2}，Z 为所有核元素之和。
```

反射 padding 的尺寸不满足时抛出明确错误，不悄悄换成零填充。不得在预测分支套 `no_grad` 或 `detach`；滤波和后续梯度必须能够反传至预测图。

### 5.3 固定 Sobel 尺度

只为新辅助项定义固定算子：

```text
Kx = [[-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1]] / 8

Ky = transpose(Kx)

Sobel padding = reflect，宽度 1
```

不得修改旧损失、原始 TCM 或指标工具所使用的公共 Sobel 实现。新旧梯度尺度可能不同，必须在修订记录中说明，不能将相同 lambda 理解为相同梯度影响。

### 5.4 滤波域梯度一致性

本轮 `M=1`，不使用可靠性图、边缘掩膜或逐图归一化：

```text
DxP = Kx(P_s)    DyP = Ky(P_s)
DxY = Kx(Y_s)    DyY = Ky(Y_s)

L_edge_filtered = 0.5 * (
    mean(abs(DxP - DxY))
  + mean(abs(DyP - DyY))
)
```

`mean` 对 batch 和全部空间位置求平均。使用两个有符号方向的梯度差，而不是只匹配梯度幅值。

这等价于在 `[B,1,H,W]` 上，将两个方向的绝对误差总和除以 `2*B*H*W`。不按梯度最大值、热边缘数量或图像动态范围再次归一化。

### 5.5 总损失

```text
lambda_thermal = 0
lambda_smooth  = 0
lambda_edge    = 0.001  # 本轮固定开发起点，未验证为最优

L_total = L_baseline + lambda_edge * L_edge_filtered
```

`lambda_edge` 从开始到结束保持不变。本轮不实现辅助权重调度。

### 5.6 明确禁止的替代实现

以下改法不符合本任务：

```text
只滤波 I_obs，却用未滤波 I_pred 匹配梯度；
直接用 |I_pred - Gaussian(I_obs)| 替代原图像重建项；
将平滑后的 I_obs 传回 Baseline 的全部损失；
只平滑 E，残差继续沿用原始 noisy Sobel；
将 I_pred.detach() 后计算新损失；
推理时对最终图像新增 Gaussian 平滑；
使用干净原训练图生成梯度目标或权重。
```

本轮只改变训练辅助项。滤波观测仍不是干净标签，也可能削弱细小热结构；效果必须由验证结果检查，不能预先写成“实现去噪成功”。

## 6. 参数和兼容性要求

### 6.1 拟新增参数

下列字段需要先在真实参数解析、配置序列化和 runner 中实现，再用于命令行。

| 字段 | 类型与默认值 | 含义 |
| --- | --- | --- |
| `aux_loss_version` | `legacy`，可选 `legacy` / `filtered_edge` | 默认保留旧行为 |
| `edge_filter_kernel` | 整数 `5` | 新边缘项的 Gaussian 核尺寸 |
| `edge_filter_sigma` | 浮点数 `1.0` | 新边缘项的滤波尺度 |

继续复用现有 `lambda_thermal`、`lambda_edge`、`lambda_smooth`；不新增含义重复的边缘权重字段。旧 `noise_beta` 和 `edge_gamma` 继续为 legacy 服务。

### 6.2 模式行为

```text
legacy:
  原三项公式、通道处理、算子、权重和 reduction 不变。

filtered_edge:
  要求 lambda_thermal == 0 且 lambda_smooth == 0；
  只计算本文件定义的 L_edge_filtered；
  不计算 E、N、W，不使用 noise_beta 或 edge_gamma。
```

选 `filtered_edge` 却传入非零 `lambda_thermal` 或 `lambda_smooth` 时，在启动前报错，防止实际训练与实验标签不一致；不能偷偷置零。

旧配置不含新字段时默认 `legacy`，旧 `cfg_args` 仍可安全加载。显式记录 resolved config，让日志可看出哪些字段实际生效，哪些旧字段在新模式下未使用。

### 6.3 全零短路

当三个权重均为零时：

```text
L_total = 原来的 L_baseline
```

不执行新增滤波、边缘图、可靠性图或额外反传，不创建可学习参数，不改优化器。参数合法性检查可在训练前完成，但 Baseline 的每步数值路径应保持原样。

### 6.4 建议函数契约

可在现有模块中新增相当于下述职责的函数；名称可以随代码风格调整并记录：

```text
filtered_edge_consistency_loss(
    prediction,
    observation,
    kernel_size=5,
    sigma=1.0
) -> scalar tensor
```

函数应有类型标注和形状说明。固定核使用确定性构造或非学习 buffer，保持相同 device/dtype，不引入新依赖或随机初始化。不得让函数自行读取图像文件或清单。

训练中仅在已有总损失合成位置加入新标量，仍通过原有的一次正式 backward 更新；不要为新项再调用一次会累计模型梯度的 backward。

## 7. 需要修改的文件

| 位置 | 本轮具体任务 |
| --- | --- |
| `losses/thermal_physics_loss.py` | 保留 legacy；加入滤波域边缘项、模式选择和零权重短路 |
| `arguments/` 中实际负责辅助损失的文件 | 注册新模式与滤波参数；验证非法组合；旧配置默认兼容 |
| `train.py` | 在原辅助损失接入点切换版本；保留主干与原损失；写最小分项日志 |
| `scripts/run_experiments.py` | 沿用现有日志捕获；转发新参数并写运行来源，不重做整个 runner |
| `configs/experiment_matrix.priority1_loss_revision.json`（新增） | 创建 B0/O0/E1/E2 四组配置；继承原校准输入 |
| `tests/test_thermal_physics_loss.py` | 新公式数值、梯度和 legacy 回归测试 |
| `tests/test_arguments.py`、`tests/test_experiment_runner.py` | 模式解析、参数校验、dry-run 转发、防覆盖测试 |
| `docs/priority1_loss_revision.md`（新增） | 源码核对结果、改动、参数语义、测试和实验状态 |
| 实际中文 README 与 `docs/formal_training_log.md` | 补充新模式和运行结果；未运行项只标记“计划” |

若文件实际名称不同，先按职责定位。不得为符合表格而重排仓库结构。

`render.py` 原渲染算法和 `tools/collect_metrics.py` 的指标定义保持不变。仅当旧参数反序列化确有兼容问题时做必要适配，并补测试；不能在修订损失的同时修改评价口径。

### 最小日志

每 500 次迭代沿用现有摘要节奏，补充可审计字段：

```text
iteration, aux_loss_version
lambda_thermal, lambda_edge, lambda_smooth
L_baseline, L_thermal_raw, L_edge_raw, L_smooth_raw
weighted_L_thermal, weighted_L_edge, weighted_L_smooth
L_total, Gaussian_count
```

新模式的 thermal/smooth 标记为 disabled；可将其加权贡献记为 0，但不能假装已经计算原始损失。Baseline 全零时同样不为日志而强制运行新增模块。

只 detach 已有标量后记录，避免保存计算图。本轮不增加梯度探针、干净训练参考、动态调权或复杂诊断框架。

## 8. 第一批验证矩阵

所有实验使用同一冻结条件、同一代码版本，从头训练至 7k。[S3]

| ID | 名称 | 模式 | λthermal | λedge | λsmooth | 用途 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| B0 | Baseline | legacy | 0 | 0 | 0 | 新版本代码下的零新增项对照 |
| O0 | 旧完整 Proposed | legacy | 0.1 | 0.01 | 0.001 | 复查原默认方案 |
| E1 | 旧边缘单项 | legacy | 0 | 0.001 | 0 | 简化旧辅助项并减小权重的对照 |
| E2 | 滤波域边缘单项 | filtered_edge | 0 | 0.001 | 0 | 评价本次最小新方案 |

O0、E1 保留旧 `edge_gamma=3`；O0 保留旧 `noise_beta=5`。E2 的这两个参数不生效；其滤波固定为 `5×5, sigma=1`。

**解释限制：**E1 与 E2 的区别包括滤波域监督、去除旧边缘加权以及明确的新算子/归一化约定，可能还包括通道处理差异。它们比较的是完整辅助项方案，不足以单独证明收益来自“只滤波目标”。本轮不扩展到更多单因素消融，也不作超出证据的归因。

顺序：

```text
单元测试 → 短步数兼容检查 → dry-run
→ B0 → O0 → E1 → E2
→ 在相同 7000 检查点收集全部 34 个 val 视角指标
```

短步数检查只验证读图、前反传和保存，不用来选择优胜方案。四个完整验证运行总预算为 28,000 次迭代，不自动追加 30k 或更大矩阵。

保存 2k 和 7k 快照；主比较使用 7k。如果 runner 只自动渲染最终快照，保持原行为，并用现有 `render.py` 单独渲染 2k 验证结果，不把 `test_iterations` 的控制台旧标签误当作 test 分区。

现有模型快照不含完整优化器状态，不能将旧 7k 权重当作这轮从头训练的替代起点。[S1]

## 9. 数据、输出与执行入口

### 9.1 输出目录

使用新增根目录，禁止覆盖旧 calibration/formal 结果：

```text
runs/priority1_loss_revision/
├── B0_baseline_sparse25_noise03_seed2026_r1_7k/
├── O0_legacy_full_sparse25_noise03_seed2026_r1_7k/
├── E1_legacy_edge_sparse25_noise03_seed2026_r1_7k/
└── E2_filtered_edge_sparse25_noise03_seed2026_r1_7k/

results/priority1_loss_revision/
├── val_2000/                  # 完成对应渲染后生成
├── val_7000/
│   ├── metrics_summary.csv
│   ├── metrics_summary.md
│   └── metrics_per_view.csv
└── revision_report.md
```

每个 run 保留既有 `run_manifest.json`、`stdout.log`、`stderr.log`、`cfg_args`、`point_cloud/`、`ATF/`、`TCM/`、`val/ours_<N>/renders/` 和 `gt/` 结构。

建议新增 `loss_components.csv` 保存最小分项日志。未经执行，不创建填有示例数值的结果表冒充真实输出。

### 9.2 配置生成与 dry-run

Codex 应复制并解析已有 7k 校准配置，只更改方法字段、新参数、实验名称和输出根目录；数据路径按原配置解析，不手写一个猜测的路径。

参数与配置实现后，使用已存在的 runner CLI：

```powershell
.\.venv\Scripts\python.exe scripts\run_experiments.py `
  --matrix configs\experiment_matrix.priority1_loss_revision.json `
  --dry-run
```

dry-run 必须显示：相同输入、seed、分辨率、迭代数、val 分区；仅方法配置和输出目录不同。实际输入文件与哈希还需单独验证，不能将“能打印命令”当作数据检查通过。

检查通过后执行：

```powershell
.\.venv\Scripts\python.exe scripts\run_experiments.py `
  --matrix configs\experiment_matrix.priority1_loss_revision.json
```

不使用 `--allow-existing` 绕过防覆盖。此命令只包含本轮四个实验，不得借用会启动其他正式任务的通用矩阵。

### 9.3 指标

继续使用冻结的 `tools/collect_metrics.py`，严格配对同名 renders/gt：

```text
PSNR ↑
SSIM ↑
T-MAE ↓
E-MAE ↓
Gradient preservation ↑
```

本轮为与原校准一致，可统一 `--skip-lpips`；LPIPS 必须显示 unavailable，而不是 0。ROI-MAE 在没有冻结 mask 时同样不报告数值。[S1, S2]

不修改评估 Sobel、归一化、PNG 写出或亮度变换来迎合新损失。无辐射定标时仍使用表观热强度，不写真实温度误差。

## 10. 必须通过的测试

### 10.1 旧行为和接线回归

- [ ] 缺省新参数等价于 `legacy`。
- [ ] 固定输入下 legacy 的各原始损失、总损失和预测梯度与修改前参考一致。
- [ ] 三项权重全零时不调用新增辅助函数；Baseline 的原损失、随机状态和更新路径不变。
- [ ] `filtered_edge` 搭配非零 thermal/smooth 权重会在启动前报错。
- [ ] runner 正确转发模式与滤波参数，`cfg_args` 和运行清单能够追溯实际配置。
- [ ] 旧 `cfg_args` 可加载，现有模型仍按原渲染流程工作。
- [ ] 不允许复用已有输出目录，清单哈希不符时拒绝运行。

对固定 CPU 张量的 float32 数值回归，可预先采用 `atol=1e-7, rtol=1e-5`；若需要改变容差，记录原因，不为掩盖公式变化而放宽。GPU 小训练还需结合原有非确定性限制核对，不承诺跨平台字节级一致。

### 10.2 新损失数值和梯度

- [ ] 与显式参考公式计算一致，包含两个梯度方向和系数 0.5。
- [ ] 相同预测与观测得到有限的零损失或规定容差内零。
- [ ] 常量图不会因边界填充产生虚假梯度。
- [ ] 预测与观测相差常数时，纯梯度项可为零；亮度差由原 Baseline 损失处理，不将此误判为实现失败。
- [ ] 预测分支的滤波可反传；非退化输入下产生有限、非零的预测梯度。
- [ ] 观测分支不产生目标梯度；固定核不是可学习参数。
- [ ] 支持规定的 CHW/BCHW 与单通道/三通道输入，并保持 device/dtype；非法形状清晰报错。
- [ ] 可用有限差分核查梯度；避开绝对值残差恰为零的不可微测试点。
- [ ] 不做逐图 min-max，不调用旧 E/N/W，不受 `noise_beta`、`edge_gamma` 改动影响。
- [ ] 构造固定信号和高频扰动，检查两条分支确实使用同一滤波；测试不能仅因为新算子尺度较小就宣布去噪有效。

### 10.3 数据隔离和实际运行

- [ ] 新损失只接收当前退化训练观测，没有读取干净原训练图的代码路径。
- [ ] validation 不参与反向传播，开发过程没有渲染或评分 test。
- [ ] 四组使用完全相同的 dataset/split/degradation 哈希及训练视角。
- [ ] 运行现有测试与新测试；GPU 测试被跳过时明确记录，不把 skipped 写成已通过 CUDA 验证。
- [ ] 完整 7k 实验前，先通过短步数前反传、模型保存及 val 渲染检查。

## 11. 验收与结果解释

### 11.1 实现验收不等于性能成功

满足以下条件即说明代码修改完成：新旧版本可切换、全零回归通过、新损失可微且符合公式、输入保持不变、四组配置可审计运行、输出和指标配对完整。

性能是否提升必须另行判断，不写“修复了过拟合”作为默认结论。

### 11.2 7k 比较方式

在同一 7000 检查点比较四组 mean validation 指标，并保留逐视角数据。结果报告至少包含：

```text
方法 / 新增损失版本 / 实际权重 / checkpoint
PSNR / SSIM / T-MAE / E-MAE / Gradient preservation
相对本轮 B0 的差值
Gaussian 数量 / 实际训练时间 / 可用显存记录
```

优先观察 E2 的 PSNR、T-MAE、E-MAE 是否共同改善，再看 SSIM 和梯度保持。不能因为 Gradient preservation 单独上升就接受方案，也不能只挑最好看的视角。

若指标冲突，原样报告冲突；单场景、单 seed、7k 只能提供开发证据，不支持统计显著性或跨场景泛化结论。

E1 与 E2 都不如 B0 时，保留结果，不自动加入 mask、调度或新网络。在 `revision_report.md` 中列出下一步待判断问题后停止本轮。

### 11.3 Codex 最终交付

- [ ] 实际改动文件清单和必要 diff 说明。
- [ ] 源码与 README 差异核对记录。
- [ ] 新参数说明及 B0/O0/E1/E2 配置。
- [ ] 单元测试、回归检查、dry-run 的真实状态与日志。
- [ ] 已完成运行的目录、输入哈希、代码状态和指标；未运行项明确标记。
- [ ] 一份 `results/priority1_loss_revision/revision_report.md`，分开写“代码完成情况”和“性能观察”。

**执行到此为止：本轮只验证监督目标修正和辅助项简化，不启动第二优先级的权重调度，不启动 30k 正式矩阵。**
