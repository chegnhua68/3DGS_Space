# B0 / E2 共同初始化 30k 协议

本轮固定比较 B0 (`lambda_edge=0`) 与 E2 (`lambda_edge=0.001`)。两者共享同一 seed 的未经训练 step-0 包，`filtered_edge`、5x5 Gaussian、sigma=1、BT.601 和有符号 Sobel 保持现有实现；Detail、GD、thermal、smooth 全部关闭。

矩阵：`configs/experiment_matrix.b0_e2_shared_init_30k.json`

- train seeds：2026、2027、2028；每个 seed 串行执行 B0、E2
- checkpoints：7000、15000、30000；主比较点为 30000
- evaluation：只访问 34-view `val`，不读取或渲染 `test`
- 输入：Sparse-25/noise03 的既有 manifest，runner 强制校验 SHA256
- 初始化：`runs/b0_e2_30k/v1/initializations/seed_<seed>/init_step0.pt`
- 停止：`runs/b0_e2_30k/v1/STOP_REQUESTED`
- 实时曲线：本机 TensorBoard，默认端口需先探测；不杀现有服务

## 启动

```powershell
New-Item -ItemType Directory -Force .\results\b0_e2_30k\v1\checks | Out-Null
.\.venv\Scripts\python.exe scripts\run_experiments.py `
  --matrix configs\experiment_matrix.b0_e2_shared_init_30k.json `
  --dry-run *> .\results\b0_e2_30k\v1\checks\matrix_dry_run.log

.\.venv\Scripts\python.exe scripts\run_experiments.py `
  --matrix configs\experiment_matrix.b0_e2_shared_init_30k.json `
  *> .\results\b0_e2_30k\v1\runner.log
```

runner 会先为三个 seed 生成并原生重载初始化包，然后串行训练六组并在每组结束后渲染 7k/15k/30k。训练 stdout/stderr 写入各 run，初始化和 runner 输出写入 `results/b0_e2_30k/v1/checks/`。

每个 run 的 `initial_state_audit.json` 必须在第一次 forward/backward/optimizer step 前生成；它包含 Gaussian、ATF、TCM 的逐张量摘要、优化器配置/空状态、训练起始 RNG 摘要和比较结果。相机序列写入 `camera_sequence.json`，并提供 7k/15k/30k 前缀哈希。

停止请求保持不删除：

```powershell
New-Item -ItemType File .\runs\b0_e2_30k\v1\STOP_REQUESTED -Force | Out-Null
```

源码审计使用 run manifest 中的 `git.training_source_sha256`，只覆盖实际训练/渲染/指标源码，排除 `runs/`、`results/`、数据、环境和缓存；配置文件单独记录 `config_sha256`。
