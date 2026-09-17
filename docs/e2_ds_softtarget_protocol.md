# E2 vs E2+DS 正式实验协议

## 目的

验证 BM3D 软目标是否改善 E2 在稀疏视角、退化红外训练集上的验证集表现。比较仅包含 E2 与 E2+DS，不引入 B0、Detail、GD、低学习率或其他模块。

## 固定数据与监督

- 数据：`data/TI-NSD/heated`；训练 59 views，验证 34 views；测试集不读取。
- 退化训练图像记为 `Y`。
- 对每个 RGB 通道独立使用 BM3D normal profile、`ALL_STAGES`、`sigma=0.03` 得到 `Z`。
- 监督目标固定为 `T = 0.25Y + 0.75Z`，float32、HWC、范围 `[0,1]`。
- 共享缓存：`data/TI-NSD/heated/supervision/ds_bm3d_rgb_s003_r075_v1`。
- E2 的原始边缘物理项仍以 `Y` 为目标；主 L1/SSIM/corner 项在 E2+DS 中使用 `T`。

## 实验设计

六个 run 按以下顺序执行，每个 30,000 iterations：

`e2_seed2026_30k`、`e2_ds_seed2026_30k`、`e2_seed2027_30k`、`e2_ds_seed2027_30k`、`e2_seed2028_30k`、`e2_ds_seed2028_30k`。

每个 run 保存并评估 `7,000 / 15,000 / 30,000`。同一 seed 的 E2/E2+DS 共享初始化包、相机序列、数据划分、退化清单和所有通用超参数；只切换监督模式。

## 启动与监控

```powershell
.venv\Scripts\python.exe scripts\run_experiments.py `
  --matrix configs\experiment_matrix.e2_ds_softtarget_30k.json

.venv\Scripts\python.exe -m tensorboard.main `
  --logdir runs\e2_ds_softtarget\v1 --host 127.0.0.1 --port 6009
```

训练日志写入 `runs/e2_ds_softtarget/v1/<experiment>/stdout.log` 和 `stderr.log`；控制台只保留阶段状态。创建 `runs/e2_ds_softtarget/v1/STOP_REQUESTED` 可请求在安全边界停止。

## 汇总

六个 run 和三处 checkpoint 完成后执行：

```powershell
.venv\Scripts\python.exe tools\summarize_e2_ds_softtarget_30k.py
```

结果位于 `results/e2_ds_softtarget/v1/summary`，包括均值/标准差、同 seed 配对差值、逐视图差值、7k→30k 晚期退化、资源和输入/代码哈希审计。报告只覆盖验证集，不把结果外推到测试集。
