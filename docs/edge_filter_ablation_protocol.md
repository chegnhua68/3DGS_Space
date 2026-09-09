# Gaussian / Identity 边缘滤波单因素消融协议

> 冻结日期：2026-09-09<br>
> 起点提交：`6af9f0d21b7c5580cc4e1baa794c5261fd8c007d`<br>
> 7k 执行提交：`86b9b78b6b464795a41c3baa98c4219bb359c12f`<br>
> 当前阶段：Step 2 三组 7k 与 2k/7k val 分析已完成；30k 配置只准备和 dry-run，不执行。<br>
> 结果报告：[`results/edge_filter_ablation_7k/ablation_report.md`](../results/edge_filter_ablation_7k/ablation_report.md)

## 1. 问题与边界

本轮只回答：在 `filtered_edge` 的亮度转换、目标 stop-gradient、有符号 Sobel、padding、归一化、权重和训练设置完全相同时，是否执行 Gaussian 滤波会怎样影响验证集重建表现。

本轮不加入 mask、权重调度、新网络、噪声模型或 densification 修改，不调节 Identity 的权重，不重新生成数据，不访问 39-view test，也不从 7k checkpoint 续训。历史 Priority-1 结果只作背景；同轮比较必须重新训练 B0、E2-no-filter 和 E2。

## 2. 单因素定义

令 `P` 为 TCM 后且未 clamp 的训练预测，`O` 为 degradation manifest 指向的当前含噪训练观测：

```text
P_y = BT.601_or_identity(P)
O_y = stop_gradient(BT.601_or_identity(O))

F_gaussian(X) = Gaussian_5x5_sigma1_reflect(X)
F_identity(X) = X

L_edge(F) = 0.5 * (
    mean(abs(Sobel_x(F(P_y)) - Sobel_x(F(O_y))))
  + mean(abs(Sobel_y(F(P_y)) - Sobel_y(F(O_y))))
)

L_total = L_baseline + 0.001 * L_edge(F)
```

Sobel x/y 均使用 reflect padding，核为标准 3 x 3 Sobel 除以 8，并比较有符号分量。两组不做逐图 min-max，不新增 clamp，不调用 legacy E/N/W。Identity 只跳过 Gaussian 及其 reflect padding，之后的像素集合、Sobel 和 reduction 完全共用。

新参数为：

```text
--edge_filter_mode {gaussian,identity}
```

默认值是 `gaussian`，因此缺少该字段的旧 E2 命令和旧 `cfg_args` 保持原行为。`edge_filter_kernel=5` 与 `edge_filter_sigma=1.0` 在两组配置中都显式保留；Identity 的日志记录 `filter_active=false`，说明这两个参数没有进入计算。`filter_active` 仅表示 Gaussian 是否实际执行，不表示边缘损失是否启用。

## 3. 冻结输入

| 项目 | 冻结值 |
| --- | --- |
| 场景 | `data/TI-NSD/heated` |
| 训练视角 | Sparse-25 的 59 张含噪图 |
| validation | 34 张未退化图 |
| test | 39 张；禁止访问 |
| 数据/退化 seed | 2026 |
| resolution | 1 |
| data device | CPU，当前视角按需加载到 GPU |
| evaluation partition | `val` |

输入文件哈希：

| 文件 | SHA256 |
| --- | --- |
| `dataset_manifest.v1.json` | `d4fe9c97ed110fa1ecdf0636a19f7a7062169d3cfbd719d5594158f9d615e919` |
| `sparse_nested-random_25.json` | `c9a1e0aa65e54e1558779b557539cb2eaba4d1faab765b786a4437aa519ba79e` |
| `degradation_manifest.v1.json` | `06d68ec82b7faa8d65168b9bd2b35f013207331f4683a2ec0ad883837be3f03b` |

runner 在创建每个输出目录前核对这三项，adapter 随后继续核对清单引用、相机映射和逐图哈希。哈希不符时停止，不能更新 expected 值绕过失败。

## 4. Step 2 三组 7k

配置：`configs/experiment_matrix.edge_filter_ablation_7k.json`

| 实验 | `aux_loss_version` | `edge_filter_mode` | thermal/edge/smooth | train seed |
| --- | --- | --- | --- | ---: |
| B0 | legacy | gaussian（不生效） | 0 / 0 / 0 | 2026 |
| E2_no_filter | filtered_edge | identity | 0 / 0.001 / 0 | 2026 |
| E2 | filtered_edge | gaussian | 0 / 0.001 / 0 | 2026 |

三组从头初始化并按表中顺序串行训练 7000 iterations。保存与内部 validation 为 2000、7000；runner 自动独立渲染最终 7000，训练完成后再用相同 `render.py` 补 2000。输出根为 `runs/edge_filter_ablation_7k/`，不得覆盖 Priority-1。

训练日程不随方法改变：位置、ATF、TCM 学习率的 max steps 均保持当前默认 30000；角点权重在前 5000 步按现有公式衰减；densification 从 500 开始、每 100 步执行、到 15000 前有效，opacity 每 3000 步 reset。7k 与未来 30k 是独立训练轨迹。

启动前：

```powershell
.\.venv\Scripts\python.exe scripts\run_experiments.py `
  --matrix configs\experiment_matrix.edge_filter_ablation_7k.json `
  --dry-run
```

确认代码提交、工作树、输入哈希、三组命令与短跑均通过后，才执行不带 `--dry-run` 的同一命令。

## 5. 30k 待执行配置

以下文件本轮只解析和 dry-run：

```text
configs/experiment_matrix.edge_filter_validation_30k_seed2026.json
configs/experiment_matrix.edge_filter_validation_30k_extra_seeds.json
```

第一份包含 train seed 2026 的 B0/E2；第二份包含 train seed 2027、2028 的四组 B0/E2。数据、split 和退化仍固定 seed 2026，只改变训练 RNG。每组从头训练 30000，统一保存/验证 2000、7000、15000、30000。必须得到用户新的明确确认后才能执行。

## 6. 日志与资源口径

`quiet=true` 关闭 tqdm。runner 控制台只打印每个 train/render 的 started、completed 或 failed；子进程输出进入各 run 的 `stdout.log`/`stderr.log`。

训练启动时只打印一次 `[AUX]`，记录版本、模式、`filter_active`、有效 kernel/sigma 和三项权重。之后每 500 步只打印一行 baseline、edge raw/weighted、total、Gaussian 数量、核心 `avg_ms` 与 CUDA allocated peak；仅在 legacy 其他项实际启用时才追加其分量。完整字段进入 `loss_components.csv`，disabled 项不为日志而计算。TensorBoard 记录同一批已有标量，并用一次文本事件记录有效配置。

runner 的 `phase_timings` 是训练/最终渲染子进程墙钟时间。每个独立渲染目录的 `render_timing.json` 另记录 render-set 循环墙钟时间；该范围包含视角设备传输和 PNG 写入，但不含模型加载。`forward_mean_ms` 是同步 ATF、Gaussian renderer、TCM 的前向时间，排除前 5 张 warm-up，不含 PNG I/O。训练 `avg_ms` 是 CUDA event 核心区间累计均值，不能当成完整迭代墙钟；CUDA allocated peak 也不是整卡显存占用。

## 7. 主评估与配对差值

主结果来自独立 `render.py` PNG 和冻结的 `tools/collect_metrics.py`。训练内部 validation 仅观察：它在 renderer 后先 clamp 再加 TCM，而独立 renderer 是先加 TCM 再由 PNG 保存处理，二者不能混表。

每个方法/检查点必须有 34 对同名 render/GT；三个方法的对应 GT 字节 SHA256 必须完全相同。指标为 PSNR、SSIM、T-MAE、E-MAE、Gradient preservation。LPIPS 和 ROI-MAE 保持 `unavailable`。评价 E-MAE 使用现有 GT 权重与公共指标 Sobel，并不等于训练的 filtered-edge 公式；本轮不修改评价定义。

每次指标收集额外生成 `metrics_manifest.json`，绑定逐图 render/GT SHA256、三个表格输出、
`metrics.py`、实际 SSIM 实现、Python/PyTorch 版本和运行参数。收集器在评估前后核对输入库存，
并默认拒绝覆盖四个受管输出。结果分析器会再次校验这些哈希以及正式 run 的冻结命令、完整
34-view 相机记录和图像尺寸，然后才生成差值与定性图。

固定比较顺序：

```text
E2 - E2_no_filter
E2 - B0
E2_no_filter - B0
```

delta 始终为左侧减右侧。PSNR、SSIM、Gradient preservation 的正值更好；T-MAE、E-MAE 的负值更好。逐视角胜/负/平按指标 CSV 保留 8 位小数后的有向差值判定，精确为 0 才算持平，不根据结果修改容差。

正式结果分析命令：

```powershell
.\.venv\Scripts\python.exe tools\analyze_edge_filter_ablation.py `
  --metrics-dir 2000=results\edge_filter_ablation_7k\val_2000 `
  --metrics-dir 7000=results\edge_filter_ablation_7k\val_7000 `
  --run-dir B0=runs\edge_filter_ablation_7k\B0_trainseed2026 `
  --run-dir E2_no_filter=runs\edge_filter_ablation_7k\E2_no_filter_trainseed2026 `
  --run-dir E2=runs\edge_filter_ablation_7k\E2_trainseed2026 `
  --output-dir results\edge_filter_ablation_7k
```

分析器默认拒绝覆盖自己的七个派生产物；`--overwrite` 仅用于明确重建同一批结果，不能绕过
输入、命令、相机、指标或哈希校验。

## 8. 预注册定性视角

在查看本轮方法差值前固定选择规则：将 34 个 val 视角按现有渲染 sequence index 排序，取 `floor(k*(N-1)/(K-1))`，`N=34`、`K=4`、`k=0..3`。因此固定文件为：

```text
00000.png -> 原始 val 001.jpg
00011.png -> 原始 val 101.jpg
00022.png -> 原始 val 202.jpg
00033.png -> 原始 val 302.jpg
```

局部放大固定为图像中心三分之一，Pillow 半开坐标为 `[floor(W/3), floor(H/3), ceil(2W/3), ceil(2H/3)]`。当前 715 x 479 图像对应 `[238,159,477,320]`。图中展示 GT、B0、E2_no_filter、E2；全图显示范围固定 `[0,1]`，绝对误差统一使用 `magma`、`vmin=0`、`vmax=0.25`，不做逐图归一化。

## 9. 停止条件

遇到输入哈希不符、非有限 loss、OOM、快照迭代不一致、缺图、GT 不一致或评估路径不一致时，停止受影响阶段并保留输出。指标变差属于有效负结果，不属于运行失败。

完成三组 7k、2k/7k 独立指标、逐视角差值、资源表、固定定性图和 `ablation_report.md` 后停止。不得自动运行任何 30k 配置，不得访问 test，也不得自动添加新模块或重新调权重。
