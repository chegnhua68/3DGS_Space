# B0 / E2 共同初始化 30k 交接

状态：实现已完成，正式训练尚未启动。

冻结路径：

- 矩阵：`configs/experiment_matrix.b0_e2_shared_init_30k.json`
- run 根：`runs/b0_e2_30k/v1/`
- results 根：`results/b0_e2_30k/v1/`
- 初始化根：`runs/b0_e2_30k/v1/initializations/`
- 停止文件：`runs/b0_e2_30k/v1/STOP_REQUESTED`

已完成：

- shared-init 30k 矩阵 dry-run 与输入 manifest 哈希核对接口
- step-0 保存/重载 smoke：Gaussian、ATF、TCM、优化器和 RNG 比较均通过
- `training_source_sha256` 实现，运行日志增长不会改变源码摘要
- 现有训练日志在 `--quiet` 下不刷逐 500 步 console；CSV/TensorBoard 保留
- 既有测试与本轮新增 runner 测试已运行

待执行：

- preflight、TensorBoard 实际端口和停止接口短跑
- 六组从头 30k 训练、18 组独立 val 渲染和指标收集
- 30k 主结果、7k/15k 趋势、资源与有限诊断报告

上下文只轮询小型 `status.json`，完整 stdout/stderr、event、CSV 和命令落盘后再汇总。若哈希、初始化比较、相机配对、OOM/NaN 或停止协议失败，停止队列并保留现场。
