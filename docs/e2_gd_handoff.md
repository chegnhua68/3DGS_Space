# E2 / GD Handoff

更新时间：2026-09-11

## 当前阶段

- 当前分支：`e2-baseline`；正式训练尚未启动。
- 本轮只比较 E2 与 E2＋低比例 GD；不包含 Detail、B0、30k 或 test。
- 历史 E2/Detail 报告保留在 `sparse-ir-development` 与 `runs/e2_detail_pair/v1/`。

## 冻结输入

- 场景：`data/TI-NSD/heated/`，Sparse-25、noise03，训练 seed `2026/2027/2028`。
- 数据、split、degradation SHA256 见 `configs/experiment_matrix.e2_gd_lowrate.json`。
- 每个 run 从头训练 `7000` steps；独立 val 在 `2000/7000`，共 34 views。
- E2：filtered edge，Gaussian `5x5 sigma=1.0`，`lambda_edge=0.001`；其余辅助项和 `lambda_detail` 均为 0。

## 已实现但待验收

- `integrations/thermal3dgs/gaussian_dropout.py`：独立 Generator、5% 调度、inverted opacity。
- `integrations/thermal3dgs/run_control.py`：粘性 stop 文件、原子状态写入、退出码 130。
- `train.py`：TB 每 50 步、CSV/资源审计每 500 步、退出 flush、GD/RNG 审计。
- `render.py`：显式关闭 GD，并在相机边界检查 stop 文件。
- `scripts/run_experiments.py`：六 run 串行矩阵、子进程日志重定向、batch/run 状态。

## 下一步顺序

1. 运行 `--dry-run`，核对六个 run、路径与三个输入哈希。
2. 执行单元测试、日志事件测试、STOP_REQUESTED runner 测试与两组 2-step smoke。
3. 核对本地 CUDA rasterizer 接受补偿 opacity；失败则阻塞正式训练。
4. 启动仅监听 `127.0.0.1` 的 TensorBoard，返回实际 URL 和停止命令。
5. 冻结源码快照后运行正式六 run；出现 stop 文件立即停止当前 run 和剩余队列。

## 运行控制

- TensorBoard logdir：`runs/e2_gd_lowrate/v1/`。
- stop 文件：`runs/e2_gd_lowrate/v1/STOP_REQUESTED`。
- 创建停止请求：`New-Item -ItemType File .\runs\e2_gd_lowrate\v1\STOP_REQUESTED -Force | Out-Null`。
- 正式结果：`results/e2_gd_lowrate/v1/`；不覆盖已有输出，不自动删除 stop 文件。
