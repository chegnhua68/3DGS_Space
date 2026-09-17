# E2+DS 交接清单

## 运行前

1. 确认 `data/TI-NSD/heated/supervision/ds_bm3d_rgb_s003_r075_v1/supervision_manifest.v1.json` 状态为 `completed`，manifest SHA256 为 `a27c1311828b3a289eb0ec4ae56f470e339c9410135c7cf43a1c71d0cdb084ca`。
2. 确认 `runs/e2_ds_softtarget/v1/STOP_REQUESTED` 不存在。
3. 运行矩阵 dry-run，然后启动正式矩阵；不要自动重试失败 run。

```powershell
.venv\Scripts\python.exe scripts\run_experiments.py `
  --matrix configs\experiment_matrix.e2_ds_softtarget_30k.json --dry-run
.venv\Scripts\python.exe scripts\run_experiments.py `
  --matrix configs\experiment_matrix.e2_ds_softtarget_30k.json
```

## 运行后

```powershell
.venv\Scripts\python.exe tools\collect_metrics.py `
  --experiment e2_seed2026_30k=runs\e2_ds_softtarget\v1\e2_seed2026_30k `
  --output-dir results\e2_ds_softtarget\v1\val_30000 `
  --skip-lpips --device cuda --overwrite
.venv\Scripts\python.exe tools\summarize_e2_ds_softtarget_30k.py
```

实际收集指标时将六个实验分别传给 `collect_metrics.py`，对 `val_7000`、`val_15000`、`val_30000` 各执行一次。最终入口是 `results/e2_ds_softtarget/v1/summary/report.md`；机器可读状态是 `summary/status.json`，完整文件哈希是 `summary/results_manifest.json`。

## Git 记录

训练产物和 BM3D 缓存不提交到 Git。提交代码、配置、测试和文档：

```powershell
git add arguments/__init__.py train.py scripts/run_experiments.py `
  integrations/thermal3dgs/soft_target.py tools/create_denoised_supervision.py `
  tools/summarize_e2_ds_softtarget_30k.py configs/experiment_matrix.e2_ds_softtarget_30k.json `
  tests/test_arguments.py tests/test_soft_target.py docs/e2_ds_softtarget_*.md
git commit -m "Add E2 BM3D soft-target validation workflow"
git status --short
```
