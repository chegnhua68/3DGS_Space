# B0 / E2 三 seed 验证协议

本轮只比较原始基线 B0 与 filtered-edge E2。两者使用同一 TI-NSD heated、Sparse-25、noise03、val 分区和完整既有 COLMAP 先验；唯一算法差异是 `lambda_edge=0` 与 `lambda_edge=0.001`。`lambda_thermal=lambda_smooth=lambda_detail=0`，GD 显式关闭 (`gd_max_rate=0`)。

正式顺序固定为：`b0_seed2026_7k`、`e2_seed2026_7k`、`e2_seed2027_7k`、`b0_seed2027_7k`、`b0_seed2028_7k`、`e2_seed2028_7k`。每组从头 7000 iterations，保存并独立验证 2000/7000；只访问 34-view val，不访问 test。

日志协议：TensorBoard 与 `live_loss.csv` 在第 1 步及每 50 步写入，审计/资源 CSV 每 500 步写入，writer `flush_secs=5`，不写图像、直方图或 profiler。runner 终端只显示 START/DONE/FAILED/INTERRUPTED，完整 stdout/stderr 写入每个 run。停止文件为 `runs/b0_e2_confirm/v1/STOP_REQUESTED`，不会自动删除。

指标差值统一定义为 `E2 - B0`。PSNR、SSIM、Gradient preservation 的负/正方向按指标定义判断，T-MAE/E-MAE 使用负 delta 判断改善；报告同时保存 signed delta 和 `improved/degraded/tied` seed/view 计数。

30k 候选矩阵仅为 `configs/experiment_matrix.b0_e2_30k_candidate.json`，本轮不加载、不启动。结果路径为 `runs/b0_e2_confirm/v1/` 与 `results/b0_e2_confirm/v1/`。
