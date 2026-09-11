# B0 / E2 Handoff

更新时间：2026-09-11

## 当前阶段

- 当前分支：`b0-e2-confirm`，基于纯 E2 `48d67e1`，已复用实时 TensorBoard/停止接线提交 `22deede`。
- 本轮只比较 B0/E2 三 seed、7k；不包含 GD、Detail、30k 或 test。
- 当前尚未启动正式六组训练。

## 冻结输入

- 场景：`data/TI-NSD/heated/`，Sparse-25、noise03、val。
- 输入 SHA256 与六组顺序见 `configs/experiment_matrix.b0_e2_confirm_7k.json`。
- B0/E2 均为 filtered-edge 配置，分别 `lambda_edge=0/0.001`；GD=0、thermal/smooth/detail=0。

## 已接入

- TensorBoard：第 1 步及每 50 步标量，`flush_secs=5`。
- CSV/资源审计：每 500 步；stdout/stderr 由 runner 重定向。
- 粘性停止文件：`runs/b0_e2_confirm/v1/STOP_REQUESTED`。
- 正式配置、2-step smoke 与未执行 30k 候选已写入 `configs/`。

## 下一步

1. dry-run、完整测试、B0/E2 smoke 和 STOP/event 验收。
2. 启动 TensorBoard，返回真实 URL 和停止命令。
3. 冻结源码后串行运行六组 7k。
4. 独立渲染 2k/7k，汇总 `E2-B0` 方向统计并生成报告。
