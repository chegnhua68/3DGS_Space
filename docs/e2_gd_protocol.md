# E2 + Gaussian Dropout Protocol

本轮在纯 E2 分支上验证训练期低比例 Gaussian opacity dropout。GD 仅作用于训练渲染中、原 activated opacity 进入 rasterizer 之前的临时张量：

```text
p(t) = 0.05 * clip((t - 1000) / (3000 - 1000), 0, 1)
a_render = a * keep / (1 - p)
```

`keep` 由独立设备匹配的 `torch.Generator` 逐步采样，seed 为 `train_seed + 104729`。E2 对照的 `gd_max_rate=0` 不抽样、不改变默认 RNG。掩码不写入 `_opacity`、optimizer、checkpoint 或永久 pruning；增密统计仍使用原 `radii/visibility_filter`，即 `upstream_visibility`。

训练和评价路径显式分离：内部 val、独立 `render.py`、viewer 和导出不传 opacity override，因此 GD 为 0。TensorBoard 只写标量：核心 loss 在第 1 步及每 50 步，审计 CSV、GD 归约和资源值每 500 步；writer `flush_secs=5`，退出时 flush/close。每个 run 还保存 `live_loss.csv`、`status.json`、`gd_rng_audit.json` 和原有 stdout/stderr。

runner 按固定顺序串行运行六个独立进程：2026 E2、2026 E2＋GD、2027 E2＋GD、2027 E2、2028 E2、2028 E2＋GD。`STOP_REQUESTED` 是批次级粘性标志，训练/渲染在安全边界检查，退出码 130 记为 `interrupted`，后续任务记为 `not_started_due_to_user_stop`。

本协议借鉴 DropGaussian 的 inverted-opacity 机制，但 5% 上限、1k warm-up、3k ramp、三 seed 和本地监测/停止接口是本项目的待验证工程设计，不宣称复现论文完整协议或严格物理无偏。
