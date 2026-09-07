# Thermal3D-GS Sparse-IR 完整操作记录

[返回中文系统说明](README_zh-CN.md) | [English README](README.md)

本文记录本项目在 Windows 主机上的实际搭建、实现、验证和 GitHub 发布过程，并给出以后可以直接复用的命令。它回答两个问题：

1. 当前系统是怎样从上游 Thermal3D-GS 搭建并验证到可运行状态的；
2. 怎样安全地提交代码并推送到 `chegnhua68/3DGS_Space`。

系统原理、损失函数、数据契约和完整训练参数见 [中文系统说明](README_zh-CN.md)。正式实验约束见 [实验协议](docs/experimental_protocol.md)，按时间记录的实验结果见 [实验日志](docs/experiment_log.md)。

## 1. 最终状态

截至 `2026-08-27`，已确认：

| 项目 | 状态 |
| --- | --- |
| 工作目录 | `E:\1_Work\Graduate\Work\Thermal3DGS_sparse` |
| 本地开发分支 | `sparse-ir-development` |
| 上游基线 | `mzzcdf/Thermal3DGS@03366b2a350ac5db6690dfd7fca51a56ba9e89a7` |
| 自有仓库 | `https://github.com/chegnhua68/3DGS_Space.git` |
| 自有默认分支 | `main` |
| 自有开发分支 | `sparse-ir-development` |
| 首次发布系统提交 | `5dc6d92fb57b8d06a824355a5f6060d69c6c3c69` |
| Python | 3.11.9，解释器为 `.venv\Scripts\python.exe` |
| PyTorch | 2.5.1+cu121 |
| CUDA Toolkit | 12.1，`nvcc` 12.1.66 |
| MSVC | v142/14.29，`cl` 19.29.30159 |
| GPU | RTX 2060 Max-Q，6 GB，计算能力 7.5 |
| CUDA 扩展 | `simple_knn` 和 `diff_gaussian_rasterization` 均已构建并验证 |
| 自动测试 | 68/68 通过，包括 2 个真实 CUDA 测试 |

数据集、虚拟环境、训练输出和模型文件均由 `.gitignore` 排除，没有上传到 GitHub。

## 2. 命令执行约定

除特别说明外，所有 PowerShell 命令都从仓库根目录执行：

```powershell
Set-Location 'E:\1_Work\Graduate\Work\Thermal3DGS_sparse'
```

项目命令始终显式使用虚拟环境解释器：

```powershell
.\.venv\Scripts\python.exe
```

这样可以避免 Windows Store Python、全局 Python 3.14 或其他环境抢占命令。不要假定当前 Scoop Python 已注册到 Windows `py` launcher；本机执行 `py -3.11` 不可用。正式训练还必须显式传入 `--seed`，它会同时控制 Python、NumPy、PyTorch 和 CUDA 的随机状态。

## 3. 获取并固定上游源码

项目以官方 Thermal3D-GS 为来源，远程名使用 `upstream`，自己的 GitHub 仓库使用 `origin`。两个名字不要交换。

检查远程：

```powershell
git remote -v
```

当前正确结果应包含：

```text
origin    https://github.com/chegnhua68/3DGS_Space.git
upstream  https://github.com/mzzcdf/Thermal3DGS.git
```

若从上游仓库重新开始，可执行：

```powershell
git remote add upstream https://github.com/mzzcdf/Thermal3DGS.git
git fetch upstream
git switch -c sparse-ir-development 03366b2a350ac5db6690dfd7fca51a56ba9e89a7
```

本项目开发基线固定为 `03366b2...`，不能在没有记录的情况下自动漂移到上游最新提交。

## 4. 主机和 GPU 审计

开始安装前检查 Python、GPU、CUDA 和编译器：

```powershell
Get-Command python,python3,py,git,cmake,nvcc,nvidia-smi -ErrorAction SilentlyContinue
nvidia-smi
where.exe nvcc
where.exe cl
```

初次检查发现：默认 Python 3.14 环境不可用于本项目，`nvcc` 和当前 shell 中的 `cl` 也不可用。因此选择 Python 3.11、PyTorch cu121、CUDA Toolkit 12.1 和 MSVC v142/14.29。

## 5. 在 E 盘准备 CUDA 和 MSVC

### 5.1 CUDA Toolkit 12.1

仓库提供 Scoop 清单：

```powershell
scoop install .\configs\scoop\cuda-12.1.json
```

执行前需要先把 Scoop 根目录配置到 E 盘。当前实际 CUDA 位置是：

```text
E:\Software\Scoop\Apps\apps\cuda-12.1\current
```

Scoop 清单会设置用户级 `CUDA_PATH`，并把 CUDA `bin` 加入用户 PATH。安装完成后重新打开终端，再检查：

```powershell
where.exe nvcc
nvcc --version
```

PyTorch 的 `cu121` wheel 只提供运行时，不提供编译 CUDA 扩展所需的 `nvcc`，因此本地 Toolkit 仍然必要。

### 5.2 Visual Studio Build Tools

没有安装第二套 Visual Studio。现有 Build Tools 2026 位于：

```text
E:\Software\Visual Studio Build Tools\Tools
```

通过 Visual Studio Installer 修改该实例，确保具有：

- C++ Build Tools 工作负载；
- Windows 10 或 Windows 11 SDK；
- `MSVC v142 - VS 2019 C++ x64/x86 build tools (v14.29)`；
- 组件 ID `Microsoft.VisualStudio.ComponentGroup.VC.Tools.142.x86.x64`。

构建 shell 使用以下命令激活 v142：

```cmd
call "E:\Software\Visual Studio Build Tools\Tools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.29
```

普通 PowerShell 找不到 `cl` 不表示没有安装，因为 `cl` 不应永久加入全局 PATH。

## 6. 创建 Python 3.11 环境

### 6.1 当前主机复建方式

当前主机复用了 E 盘 Python 3.11 中已经安装的 CUDA PyTorch：

```powershell
$python311 = 'E:\Software\Scoop\Apps\apps\python311\current\python.exe'
& $python311 --version
& $python311 -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -e '.[figures]'
.\.venv\Scripts\python.exe -m pip install tqdm plyfile imageio imageio-ffmpeg wheel gdown
```

`--version` 必须输出 Python 3.11。`--system-site-packages` 只是这台主机复用已有 PyTorch 的选择。

### 6.2 全新机器的隔离安装方式

全新机器建议把 PyTorch 直接装进 `.venv`：

```powershell
$python311 = 'E:\Software\Scoop\Apps\apps\python311\current\python.exe'
& $python311 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe -m pip install -e '.[figures]'
.\.venv\Scripts\python.exe -m pip install tqdm plyfile imageio imageio-ffmpeg wheel gdown
```

检查 PyTorch 和 GPU：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 7. 构建 CUDA 扩展

运行仓库提供的构建助手：

```powershell
cmd /c scripts\build_cuda_extensions.cmd
```

脚本负责：

- 激活现有 VS Build Tools 2026 中的 v142/14.29；
- 设置 `CUDA_HOME` 和 `CUDA_PATH`；
- 使用 `TORCH_CUDA_ARCH_LIST=7.5`；
- 使用 `MAX_JOBS=1` 降低内存压力；
- 把临时目录放在 E 盘；
- 编译并安装两个 vendored CUDA 扩展。

验证扩展：

```powershell
.\.venv\Scripts\python.exe -c "import diff_gaussian_rasterization._C, simple_knn._C; print('extensions ok')"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_cuda_extensions.py -v
```

CUDA 专项测试必须实际运行 2 个测试，不能只看到退出码 0，因为缺少依赖时 unittest 可能报告 skipped。

构建过程可能修改被上游跟踪的 `.egg-info` 文件。它们属于生成元数据，不应混入功能提交：

```powershell
git restore -- submodules/depth-diff-gaussian-rasterization/diff_gaussian_rasterization.egg-info
git restore -- submodules/simple-knn/simple_knn.egg-info
```

## 8. 数据准备和清单生成

当前验证场景是 TI-NSD `heated`，本地目录为：

```text
data\TI-NSD\heated
```

该目录含 307 个已注册视角和 `sparse/0` COLMAP 数据。`data/` 被 Git 忽略，不会推送到 GitHub。

生成数据集清单和基础划分：

```powershell
.\.venv\Scripts\python.exe tools\create_colmap_manifest.py `
  --dataset-root data\TI-NSD\heated `
  --dataset-id ti-nsd-heated `
  --data-range 0 255 `
  --test-every 8 `
  --val-every 8
```

生成 seed 2026 的嵌套稀疏划分：

```powershell
.\.venv\Scripts\python.exe tools\create_sparse_split.py `
  --dataset-manifest data\TI-NSD\heated\dataset_manifest.v1.json `
  --base-split-manifest data\TI-NSD\heated\base_split.v1.json `
  --output-dir data\TI-NSD\heated\splits\seed2026 `
  --ratios 1 0.5 0.25 0.125 `
  --method nested-random `
  --seed 2026
```

实际划分是 234 个训练池视角、34 个验证视角和 39 个测试视角；四个训练子集分别含 234、117、59 和 29 个视角。

生成仅训练集 noise03 退化：

```powershell
.\.venv\Scripts\python.exe tools\degrade_ir_dataset.py `
  --dataset-manifest data\TI-NSD\heated\dataset_manifest.v1.json `
  --split-manifest data\TI-NSD\heated\splits\seed2026\sparse_nested-random_12p5.json `
  --output-dir data\TI-NSD\heated\derived\noise03_12p5_seed2026 `
  --degradation gaussian-noise `
  --noise-sigma 0.03 `
  --seed 2026
```

退化只作用于 29 张训练图，val/test 保持原始字节。

## 9. 实现的系统扩展

本次开发加入或修改了以下能力：

- 版本化 dataset、base split、sparse split 和 degradation JSON 契约；
- 确定性嵌套稀疏视角采样；
- 仅训练集的红外噪声、模糊和对比度退化；
- Thermal3D-GS manifest 适配器；
- 可选热强度、边缘和可靠性平滑损失；
- 严格分离 `val` 与 `test` 的训练和渲染路径；
- PSNR、SSIM、LPIPS、T-MAE、E-MAE、梯度保持和 ROI-MAE；
- 五类论文图和带哈希的图生成清单；
- 批量实验 runner、输出防覆盖和来源审计；
- Windows CUDA 扩展构建脚本；
- Python 3.11 回归测试和 CUDA 数值测试。

正式矩阵使用以下两项配套设置：

```text
data_device=cpu
load2gpu_on_the_fly=true
```

只设置 `data_device=cpu` 而不启用按需传 GPU，会让 CUDA rasterizer 收到 CPU 相机矩阵并发生设备不匹配。

## 10. 日志行为与验证操作

训练器默认每 1000 次迭代更新一次摘要；加入 `--quiet` 后关闭实时进度条，但仍保留检查点、验证和最终摘要。实验 runner 会将训练和渲染子进程分别保存到每个实验目录：

```text
runs/formal/<condition>/<method>/seed_<seed>/stdout.log
runs/formal/<condition>/<method>/seed_<seed>/stderr.log
```

主终端只显示实验开始、完成或失败状态。`run_manifest.json` 同时记录这两个日志文件名，便于审计和定位失败原因。

## 11. 验证操作

运行全部测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py' -v
```

当前结果是：

```text
Ran 68 tests
OK
```

静态编译和差异检查：

```powershell
.\.venv\Scripts\python.exe -m compileall -q arguments gaussian_renderer integrations losses scene scripts src tests tools train.py render.py
git diff --check
```

正式矩阵先做 dry-run：

```powershell
$env:THERMAL_DATASET_ROOT = 'E:\1_Work\Graduate\Work\Thermal3DGS_sparse\data\TI-NSD\heated'
$env:THERMAL_SPLIT_DIR = "$env:THERMAL_DATASET_ROOT\splits\seed2026"
$env:THERMAL_DEGRADATION_DIR = "$env:THERMAL_DATASET_ROOT\derived"

.\.venv\Scripts\python.exe scripts\run_experiments.py `
  --matrix configs\experiment_matrix.example.json `
  --dry-run
```

dry-run 必须显示训练命令同时包含：

```text
--data_device cpu --load2gpu_on_the_fly
```

已完成的 smoke 和容量校准只证明链路可运行，不是正式收敛结果。历史时间线见 [实验日志](docs/experiment_log.md)，
当前正式训练计划的阶段结果见 [正式训练记录](docs/formal_training_log.md)。

## 12. 首次提交到 GitHub 的实际过程

### 12.1 提交前检查

先确认当前分支和文件：

```powershell
git status --short --branch
git diff --stat
git diff --check
```

检查 Git 不会提交数据、虚拟环境、训练输出和模型：

```powershell
git status --ignored --short
```

再暂存所有正常源码和文档：

```powershell
git add -A
git status --short
git diff --cached --stat
git diff --cached --check
```

在执行 `git commit` 前必须认真阅读暂存列表。尤其注意：

- 不应出现 `data/`、`.venv/`、`runs/`、`output/` 或 `checkpoints/`；
- 不应出现 `.pyd`、`.so` 或构建缓存；
- 不应出现令牌、`.env`、私钥或凭据文件；
- 不应出现意外的 Git 子仓库模式 `160000`；
- GitHub 普通仓库中不应出现接近或超过 100 MB 的单文件。

### 12.2 清理误加入的嵌套仓库

首次提交时曾误把仓库内的另一个 Git 克隆记录成模式 `160000`。检查方法：

```powershell
git diff --cached --summary
git ls-files --stage
```

若看到不属于设计的 `160000` 路径，先从索引移除：

```powershell
git rm --cached -- 3DGS-space
```

本次还发现两个本地错误克隆：空的 `3DGS_Space/` 和指向错误连字符仓库的 `3DGS-space/`。在核对远程地址和 `git status --porcelain` 为空后，两者都已从当前工作目录删除。删除命令具有破坏性，其他机器不得照抄路径；必须先确认目标仓库没有未提交内容。

### 12.3 创建提交

首次系统提交使用：

```powershell
git commit -m "Add sparse-view infrared reconstruction system"
```

因为提交尚未推送，发现嵌套仓库和空白问题后使用 amend 修正：

```powershell
git commit --amend --no-edit
```

最终首次发布提交为：

```text
5dc6d92fb57b8d06a824355a5f6060d69c6c3c69
```

只应 amend 尚未共享的提交。已经被他人拉取的提交不要随意改写。

### 12.4 配置正确的 GitHub 远程

目标仓库使用下划线：

```text
chegnhua68/3DGS_Space
```

先读取远程分支，确认是否为空以及是否存在冲突历史：

```powershell
git ls-remote --symref https://github.com/chegnhua68/3DGS_Space.git HEAD
git ls-remote --heads --tags https://github.com/chegnhua68/3DGS_Space.git
```

本次查询没有返回分支或标签，说明目标仓库为空。随后配置 `origin`：

```powershell
git remote add origin https://github.com/chegnhua68/3DGS_Space.git
git remote -v
```

若 `origin` 已存在，先检查，不要直接覆盖：

```powershell
git remote get-url origin
```

只有确认地址错误时才修改：

```powershell
git remote set-url origin https://github.com/chegnhua68/3DGS_Space.git
```

不要把 GitHub token 写进远程 URL、脚本或 README。认证应交给浏览器或 Git Credential Manager。

### 12.5 首次推送空仓库

因为远程为空，本次一次创建 `main` 和开发分支：

```powershell
git push origin HEAD:refs/heads/main HEAD:refs/heads/sparse-ir-development
```

含义如下：

- `HEAD:refs/heads/main`：把当前已验证提交作为 GitHub 默认代码入口；
- `HEAD:refs/heads/sparse-ir-development`：建立同名开发分支；
- 不使用 `--force`，避免覆盖未知远程历史。

推送后，让本地开发分支跟踪自己的 GitHub 分支：

```powershell
git branch --set-upstream-to=origin/sparse-ir-development sparse-ir-development
```

验证远程：

```powershell
git ls-remote --symref origin HEAD
git ls-remote --heads origin main sparse-ir-development
git remote show origin
git status --short --branch
```

首次推送后的验证结果是：

```text
HEAD -> main
main                  -> 5dc6d92
sparse-ir-development -> 5dc6d92
```

## 13. 后续日常提交和推送

### 13.1 在开发分支提交

确认当前位于开发分支：

```powershell
git switch sparse-ir-development
git status --short --branch
```

修改完成后运行验证，然后提交：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py' -v
git diff --check
git add -A
git status --short
git diff --cached --stat
git diff --cached --check
git commit -m "Describe the change clearly"
git push
```

由于本地 `sparse-ir-development` 已跟踪 `origin/sparse-ir-development`，最后一条 `git push` 会更新开发分支，不会自动修改 `main`。

### 13.2 推荐：通过 Pull Request 更新 main

1. 先把开发分支推送到 GitHub；
2. 打开 `https://github.com/chegnhua68/3DGS_Space`；
3. 创建 `sparse-ir-development` 到 `main` 的 Pull Request；
4. 检查文件、测试和提交记录；
5. 合并后同步本地引用。

同步命令：

```powershell
git fetch origin
git log --oneline --decorate --graph --all -20
```

### 13.3 明确需要时直接快进 main

只有确认 `main` 没有其他人新增提交、当前开发分支已经完整验证时，才可以：

```powershell
git fetch origin
git push origin HEAD:main
```

若 Git 拒绝 non-fast-forward，不要使用 `--force`。先执行：

```powershell
git fetch origin
git log --oneline --decorate --graph --all -30
```

检查分叉原因，再决定 merge、rebase 或 Pull Request。

## 14. 同步上游 Thermal3D-GS

查看上游变化：

```powershell
git fetch upstream
git log --oneline --decorate HEAD..upstream/master
```

不要直接在有未提交修改时 merge/rebase。先确保：

```powershell
git status --short
```

没有输出。同步上游会影响训练核心和论文可复现基线，应建立单独分支、运行全部测试，并在 [上游来源审计](docs/upstream_provenance.md) 中记录新提交哈希。

## 15. Git 常见问题

### `remote origin already exists`

先检查：

```powershell
git remote -v
git remote get-url origin
```

不要重复添加；确认错误后使用 `git remote set-url origin ...`。

### `rejected non-fast-forward`

说明远程含本地没有的提交。执行 `git fetch origin` 并查看历史，不要直接 force push。

### GitHub 认证等待

首次 HTTPS push 可能等待浏览器或 Git Credential Manager 授权。完成 GitHub 登录后继续等待原 `git push` 命令，不要同时启动多个推送。

### `git status` 出现数据或模型文件

立即停止提交，检查 `.gitignore`。已经暂存但不应提交的文件可用：

```powershell
git restore --staged -- <path>
```

该命令只取消暂存，不删除工作文件。

### `git status` 出现嵌套仓库

先执行：

```powershell
git -C <nested-path> status --short --branch
git -C <nested-path> remote -v
```

确认它是否是独立项目。不要在未检查内容时递归删除目录。

## 16. 发布前最终检查表

每次准备把实验代码合并到 `main` 前，至少确认：

- 当前分支和目标远程正确；
- `git status` 中没有数据、权重、缓存和嵌套仓库；
- 训练和评估命令对应正确的 split、degradation 和 partition；
- baseline 与 proposed 使用相同输入字节和优化预算；
- 68 个测试全部通过，CUDA 测试没有 skipped；
- `compileall`、JSON 解析和 `git diff --check` 通过；
- smoke 结果没有被描述为正式收敛结果；
- 没有密钥、token、私钥或超过 GitHub 限制的大文件；
- 暂存差异已经人工检查；
- 提交信息能够说明实际改动；
- 优先通过开发分支和 Pull Request 更新 `main`；
- 不使用 `git push --force` 覆盖未知历史。

完成这些检查后，GitHub 上的代码、文档和实验记录才能与本地实际运行状态对应。
