# Thermal3D-GS 稀疏视角退化红外重建系统

[English README and upstream documentation](README.md) | [完整操作与 GitHub 提交记录](README_OPERATIONS_zh-CN.md)

本项目是在 [Thermal3D-GS](https://github.com/mzzcdf/Thermal3DGS) 上构建的可复现实验系统，面向稀疏视角、低信噪比和弱纹理条件下的红外新视角合成与三维表观红外强度场重建。系统保留上游 Gaussian、ATF 和 TCM 训练路径，同时增加确定性数据划分、仅训练集退化、噪声感知边缘保持损失、严格的验证/测试隔离、指标汇总、论文图生成和实验来源审计。

当前分支固定在上游提交 `03366b2a350ac5db6690dfd7fca51a56ba9e89a7` 之上。详细的上游兼容修复和来源说明见 [上游来源审计](docs/upstream_provenance.md)。

## 目录

- [系统目标](#系统目标)
- [核心能力](#核心能力)
- [系统数据流](#系统数据流)
- [方法说明](#方法说明)
- [项目结构](#项目结构)
- [环境与 CUDA 扩展](#环境与-cuda-扩展)
- [数据集与清单契约](#数据集与清单契约)
- [完整使用流程](#完整使用流程)
- [批量实验](#批量实验)
- [指标与论文图](#指标与论文图)
- [输出目录](#输出目录)
- [当前验证状态](#当前验证状态)
- [测试](#测试)
- [常见问题](#常见问题)
- [实验规范与限制](#实验规范与限制)
- [许可与引用](#许可与引用)

## 系统目标

系统研究的问题不是一般可见光 3D 重建，而是以下条件共同出现时的红外场景建模：

- 输入视角稀疏，训练图像仅占完整训练池的 100%、50%、25% 或 12.5%；
- 红外图像存在低信噪比、低对比度或模糊；
- 热边界需要保留，而平坦区域中的噪声响应需要抑制；
- 所有方法必须使用相同相机、相同训练图字节、相同验证/测试视角和相同优化预算；
- 每次实验必须能够追溯到数据清单、划分清单、退化清单、代码状态和命令行。

当前实现首先支持固定姿态协议，即复用完整 COLMAP 标定来研究外观和热强度重建。若要研究真正的稀疏端到端重建，还需要仅从稀疏输入重新估计相机位姿，并单独报告位姿失败率。

## 核心能力

| 模块 | 作用 | 主要入口 |
| --- | --- | --- |
| COLMAP 清单生成 | 固化视角 ID、相机引用、图像路径、强度范围和源文件哈希 | `tools/create_colmap_manifest.py` |
| 稀疏视角划分 | 生成确定性、互相嵌套的 100/50/25/12.5% 训练子集 | `tools/create_sparse_split.py` |
| 红外退化 | 对 `train_selected` 应用噪声、降对比度、模糊或组合退化 | `tools/degrade_ir_dataset.py` |
| Thermal3D-GS 适配 | 将清单选择映射到 COLMAP 相机，并逐文件校验 SHA256 | `integrations/thermal3dgs/manifest_adapter.py` |
| 改进损失 | 噪声可靠性加权、热边缘保持和非边缘平滑 | `losses/thermal_physics_loss.py` |
| 训练与渲染 | 训练 Gaussian、ATF、TCM，并按真实 `val`/`test` 分区输出 | `train.py`、`render.py` |
| 批量实验 | 从 JSON 矩阵构造命令并记录输入哈希和代码快照 | `scripts/run_experiments.py` |
| 指标汇总 | PSNR、SSIM、LPIPS、T-MAE、E-MAE、梯度保持和 ROI-MAE | `tools/collect_metrics.py` |
| 论文图生成 | 定性对比、误差图、边缘图、稀疏曲线和退化曲线 | `tools/make_figures.py` |

## 系统数据流

```text
COLMAP 场景
  |-- images/
  `-- sparse/0/
          |
          v
dataset_manifest.v1.json + base_split.v1.json
          |
          +--> 100/50/25/12.5% sparse split manifests
          |           |
          |           `--> 仅 train_selected 的 lossless 退化图和退化清单
          |                              |
          `------------------------------+
                                         v
                              manifest_adapter 严格校验
                                         |
                                         v
                       Thermal3D-GS + 可选热红外损失
                                         |
                                         v
                    Gaussian / ATF / TCM 模型快照
                                         |
                              +----------+----------+
                              |                     |
                           val 渲染              test 渲染
                              |                     |
                         调参与选择           超参数冻结后评估
                              +----------+----------+
                                         |
                                  指标 CSV/Markdown
                                         |
                                  定量曲线与论文图
```

这里的关键约束是：稀疏划分只从训练池取样，退化只作用于选中的训练图，验证和测试图始终从原始数据集读取。

## 方法说明

### Thermal3D-GS 基线

训练主干保留上游 Thermal3D-GS 的 Gaussian 表示、ATF、TCM、L1、SSIM 和角点损失。项目只修复了阻止当前代码运行或破坏可复现性的兼容问题，例如 CUDA rasterizer 返回值、设备按需加载、ATF/TCM 保存路径、模型快照迭代一致性和安全的 `cfg_args` 解析。

当以下三个参数均为零时，新增损失路径完全短路，训练行为保持为修复后的基线：

```text
lambda_thermal = 0
lambda_edge    = 0
lambda_smooth  = 0
```

### 噪声感知边缘保持热一致性损失

给定固定映射到 `[0,1]` 的预测图 `I_pred` 和目标图 `I_gt`，系统从目标图构造两个辅助图。

热边缘图：

```text
E = normalize(|Sobel(Gaussian(I_gt))|)
```

噪声代理和可靠性图：

```text
N = normalize(|I_gt - Gaussian(I_gt)|)
W = exp(-beta * N)
```

`W` 是图像高频残差得到的启发式可靠性，不是经过传感器标定的物理噪声模型。默认使用 5x5、`sigma=1` 的高斯平滑，边缘和噪声代理按图归一化。

三个新增损失为：

```text
L_thermal = mean(W * |I_pred - I_gt|)
L_edge    = mean((1 + gamma * E) * |Sobel(I_pred) - Sobel(I_gt)|)
L_smooth  = mean((1 - E) * |Sobel(I_pred)|)

L_total = L_baseline
        + lambda_thermal * L_thermal
        + lambda_edge    * L_edge
        + lambda_smooth  * L_smooth
```

建议的起始参数是：

```text
lambda_thermal = 0.1
lambda_edge    = 0.01
lambda_smooth  = 0.001
noise_beta     = 5
edge_gamma     = 3
```

这些数值只是预注册的起点，必须在验证集上选择，不能根据最终测试集结果反向调参。

### Priority-1：滤波域边缘一致性

`--aux_loss_version` 默认是 `legacy`，因此旧命令和旧配置仍使用上面的完整辅助损失。新增的
`filtered_edge` 模式只保留一个小权重边缘项，并要求
`lambda_thermal=0`、`lambda_smooth=0`；非法组合会在加载场景前报错。

新模式先将 TCM 后、未 clamp 的预测图和同一份含噪训练观测转换为固定 BT.601 单通道亮度，
再对两条分支施加相同的 5x5、`sigma=1.0`、reflect-padding Gaussian 滤波。随后以
reflect-padding、`/8` 尺度的有符号 Sobel 分量计算：

```text
L_edge_filtered = 0.5 * (
    mean(|Sobel_x(G(I_pred)) - Sobel_x(G(I_obs))|)
  + mean(|Sobel_y(G(I_pred)) - Sobel_y(G(I_obs))|)
)

L_total = L_baseline + 0.001 * L_edge_filtered
```

观测分支显式 stop-gradient，预测分支保留完整计算图；不做逐图 min-max，不读取干净训练图，
也不改变推理渲染。`0.001` 只是本轮固定开发起点，不表示已经找到最优参数。四组 B0/O0/E1/E2
验证配置见 [Priority-1 实验矩阵](configs/experiment_matrix.priority1_loss_revision.json)，实现与
实验状态见 [Priority-1 修订记录](docs/priority1_loss_revision.md)。

## 项目结构

```text
Thermal3DGS_sparse/
|-- arguments/                         # 训练、渲染和新增损失参数
|-- configs/
|   |-- experiment_matrix.example.json # 正式实验矩阵示例
|   |-- experiment_matrix.heated_smoke.json
|   |-- figure_spec.example.json       # 论文图规范
|   `-- scoop/cuda-12.1.json           # 本机 CUDA 12.1 Scoop 清单
|-- docs/
|   |-- experimental_protocol.md       # 冻结的数据、划分和评估协议
|   |-- experiment_log.md              # 实际环境、命令和运行记录
|   |-- python311_environment.md        # Python/CUDA/MSVC 环境
|   `-- upstream_provenance.md          # 上游版本和兼容修复
|-- integrations/thermal3dgs/           # 清单到原训练器的适配层
|-- losses/thermal_physics_loss.py      # 新增热红外损失
|-- scripts/
|   |-- build_cuda_extensions.cmd       # Windows CUDA 扩展构建
|   `-- run_experiments.py              # 可审计批量实验 runner
|-- src/thermal3dgs_sparse_ir/          # 清单、划分、退化、指标、绘图核心库
|-- tests/                              # CPU、契约和 CUDA 回归测试
|-- tools/                              # 数据准备、指标和绘图 CLI
|-- train.py
`-- render.py
```

`.venv/`、`data/`、`output/`、`runs/`、`checkpoints/` 以及 `results/` 下生成的指标和图片均被 Git 忽略，避免把环境、数据集和大体积模型产物误提交到仓库。放在其他目录的权重文件不会自动被忽略。

## 环境与 CUDA 扩展

### 已验证环境

当前 Windows 主机实际验证的组合为：

| 组件 | 版本或位置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 2060 Max-Q，6 GB，计算能力 7.5 |
| Python | 3.11.9，项目解释器为 `.venv\Scripts\python.exe` |
| PyTorch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| CUDA Toolkit | 12.1，`nvcc` 12.1.66 |
| CUDA 路径 | `E:\Software\Scoop\Apps\apps\cuda-12.1\current` |
| MSVC | v142/14.29，`cl` 19.29.30159 |
| VS Build Tools | 现有 VS Build Tools 2026，位于 `E:\Software\Visual Studio Build Tools\Tools` |

没有安装第二套 Visual Studio。CUDA 和 MSVC 主体均位于 `E:\Software`；构建脚本还把 `TEMP`/`TMP` 指向 E 盘并关闭 pip 构建缓存。Visual Studio Installer、Windows SDK 或系统组件仍可能在 C 盘保留少量共享元数据和日志，这不属于 CUDA/MSVC 主体安装目录。

上游原始环境是 Python 3.7.13、PyTorch 1.12.1、torchvision 0.13.1 和 CUDA 11.6。当前 Python 3.11 组合是经过测试的本地移植，论文或报告中应明确披露，不能称为字节级复现上游环境。

`torch==2.5.1+cu121` 中的 `cu121` 表示 wheel 携带的 CUDA 运行时，不包含编译自定义扩展所需的 `nvcc`。本机因此仍需单独安装 CUDA Toolkit 12.1。项目的 `pyproject.toml` 也不会自动安装 PyTorch 或 torchvision。

### 在新机器准备 E 盘工具链

当前仓库带有固定 CUDA 12.1 安装包的 Scoop 清单：

```powershell
scoop install .\configs\scoop\cuda-12.1.json
```

在仓库根目录执行上述命令。执行前应先把 Scoop 根目录配置到目标 E 盘位置；该 JSON 只固定 CUDA 包，不决定 Scoop 根目录，也不安装或更新 NVIDIA 显卡驱动。清单会设置用户级 `CUDA_PATH` 并把 CUDA `bin` 加入用户 PATH，安装后需要重新打开终端才能继承这些环境变量。

已有 Visual Studio Build Tools 2026 时，不需要再安装 VS2019 或 VS2022。在 Visual Studio Installer 中修改现有 Build Tools 2026，进入“单个组件”，添加：

```text
MSVC v142 - VS 2019 C++ x64/x86 build tools (v14.29)
Component ID: Microsoft.VisualStudio.ComponentGroup.VC.Tools.142.x86.x64
```

本机把该组件加入现有 `E:\Software\Visual Studio Build Tools\Tools` 实例。实例还需要 C++ Build Tools 工作负载以及 Windows 10 或 Windows 11 SDK；当前主机安装的是 Windows 11 SDK 26100。修改 Visual Studio 组件需要管理员权限；新机器的实例路径不同，不应照抄本机路径创建第二套 VS。

### 仅复建当前主机的 Python 3.11 环境

当前主机使用已有 CUDA PyTorch 的系统包：

```powershell
& 'E:\Software\Scoop\Apps\apps\python311\current\python.exe' -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -e '.[figures]'
.\.venv\Scripts\python.exe -m pip install tqdm plyfile imageio imageio-ffmpeg wheel gdown
```

`--system-site-packages` 只是当前主机复用已有包的选择，不适合作为全新机器的默认方案。

### 在全新机器创建 Python 3.11 环境

已经把 Python 3.11 安装到 E 盘后，可用其明确路径创建隔离环境并安装本项目验证过的 CUDA 12.1 wheel。下面是当前 Scoop 根目录对应的路径；若你的 E 盘 Scoop 根目录不同，只修改第一行：

```powershell
$python311 = 'E:\Software\Scoop\Apps\apps\python311\current\python.exe'
& $python311 --version
& $python311 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe -m pip install -e '.[figures]'
.\.venv\Scripts\python.exe -m pip install tqdm plyfile imageio imageio-ffmpeg wheel gdown
```

`--version` 必须显示 Python 3.11。只有通过 python.org 等方式安装并正确注册了 Windows Python Launcher 后，才可将前两条命令替换为 `py -3.11 -m venv .venv`；当前 Scoop 安装未向 launcher 注册，不能假定该命令可用。

PyTorch wheel 自带 CUDA 12.1 运行时，但仍要求兼容的 NVIDIA 驱动。只有运行 Python 模型时不需要本地 `nvcc`；编译本仓库的两个 CUDA 扩展时，仍必须另行安装 CUDA Toolkit 12.1 和对应的 MSVC 工具链。

检查运行时：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 构建 CUDA 扩展

Windows 上可以运行仓库自带的重编译助手：

```powershell
cmd /c scripts\build_cuda_extensions.cmd
```

脚本会编译并安装：

- `simple_knn`；
- `diff_gaussian_rasterization`。

该脚本不是环境安装器，不会安装 Python、PyTorch、驱动、CUDA、Visual Studio 或 v142。它固定使用仓库 `.venv`，并在环境变量未定义时采用本机 E 盘路径、`TORCH_CUDA_ARCH_LIST=7.5` 和 `MAX_JOBS=1`。如果当前 shell 已经定义了这些变量，脚本会沿用现有值。

非默认路径或其他 GPU 应在调用脚本前显式设置：

```powershell
$env:VS_BUILD_TOOLS = 'D:\path\to\BuildTools'
$env:CUDA_HOME = 'D:\path\to\CUDA\v12.1'
$env:TORCH_CUDA_ARCH_LIST = '8.6'
$env:MAX_JOBS = '1'
cmd /c scripts\build_cuda_extensions.cmd
```

验证扩展：

```powershell
.\.venv\Scripts\python.exe -c "import diff_gaussian_rasterization._C, simple_knn._C; print('extensions ok')"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_cuda_extensions.py -v
```

CUDA 专项测试必须显示实际运行 2 个测试且没有 `skipped`。依赖或 GPU 不可用时，unittest 可能跳过整个测试类但仍返回退出码 0，不能只看命令退出状态。

普通 PowerShell 中 `CUDA_HOME` 为空或找不到 `cl` 并不表示工具未安装。它只说明当前 shell 没有执行 `vcvars64.bat` 或设置 CUDA 路径。扩展已经安装后，普通训练只需要 NVIDIA 驱动和可用的 PyTorch CUDA runtime，不需要每次重新编译。构建脚本使用 `setlocal`，结束后也不会永久修改调用终端的环境。

更完整的环境说明见 [Python 3.11 环境文档](docs/python311_environment.md)。

## 数据集与清单契约

### COLMAP 场景结构

输入场景至少需要：

```text
<scene>/
|-- images/
|   |-- 000.jpg
|   |-- 001.jpg
|   `-- ...
`-- sparse/0/
    |-- cameras.bin
    |-- images.bin
    `-- points3D.bin
```

也支持 COLMAP 文本格式。相机模型应满足上游加载器要求，通常使用 `SIMPLE_PINHOLE` 或 `PINHOLE`。

### 可选的公开文件夹下载助手

仓库提供面向公开 Google Drive 文件夹的并发下载工具：

```powershell
.\.venv\Scripts\python.exe tools\download_gdrive_folder.py `
  --folder-id <PUBLIC_FOLDER_ID> `
  --output-dir data\downloads\<dataset> `
  --workers 8
```

上游 README 给出的 TI-NSD 根文件夹 ID 是 [`1scp7-dB0BVE84ra8KkgQ7z9zLk5yCPFH`](https://drive.google.com/drive/folders/1scp7-dB0BVE84ra8KkgQ7z9zLk5yCPFH)，当前已验证的 `heated` 子文件夹 ID 是 [`174B29yaR6bZ4QzvzdZOnX-hcs95opJJQ`](https://drive.google.com/drive/folders/174B29yaR6bZ4QzvzdZOnX-hcs95opJJQ)。外部文件夹的层级、内容和公开状态可能变化，使用前应从上游入口重新核对。

该工具依赖 `gdown` 和 `requests`。已有非空文件会被跳过，失败的完整文件可以在下次运行时重新下载；它不支持 HTTP Range 分块续传，也不会对已存在文件自动复核远端大小或哈希。实验完整性应以生成数据集清单后的源图 SHA256 为准。COLMAP 二进制文件若不在同一公开文件夹中，仍需从官方来源单独取得。

### 四类清单

系统不依赖文件夹枚举顺序，而是使用四类带版本和哈希的 JSON 清单：

1. `dataset_manifest.v1.json`：记录数据集 ID、统一强度范围、图像域、稳定视角 ID、相机引用、相对路径、序列索引和源 SHA256。
2. `base_split.v1.json`：冻结互不重叠的 `train_pool`、`val` 和 `test`。
3. `sparse_*.json`：从 `train_pool` 生成 `train_selected`，记录比例、方法、seed 和源清单哈希。
4. `degradation_manifest.v1.json`：记录每个训练图的输入/输出路径、SHA256、变换顺序、全局 seed 和稳定的逐视角 seed；验证和测试记录必须保持原始哈希。

清单内路径必须是相对于清单声明的对应根目录的规范相对路径，并由 `output_path_base` 等字段指定解析基准；不能使用绝对路径、`..` 或反斜杠。原图通常相对于数据集根目录，退化训练图则相对于退化清单的输出根目录。加载器在训练前验证数据集、划分、退化清单之间的 ID、内容哈希和文件哈希关系。

### 强度范围与图像格式

- `data_range` 是整个数据集共享的编码强度范围，例如 uint8 的 `[0,255]`。
- 禁止逐图 min-max 归一化，否则会破坏跨视角热强度一致性。
- 未标定数据使用 `thermal_intensity`；只有存在明确辐射定标时才使用 `temperature`。
- 伪彩色 RGB 只能作为普通图像输入，不能据此声称重建了真实温度。
- 离线退化接受 uint8 JPEG/PNG/TIFF 和 uint16 PNG/TIFF。
- JPEG 源图的退化结果统一写为 PNG；其他退化结果写为 PNG 或 TIFF，不进行第二次 JPEG 有损编码。
- 当前 Thermal3D-GS 训练适配器只接受 `data_range=[0,255]` 的 8 位 `L`、`LA`、`RGB` 或 `RGBA` 图像。uint16 退化可离线生成，但尚不能送入当前训练器。

## 完整使用流程

下面以已经验证的 `data\TI-NSD\heated` 为例。公共数据仍应从上游发布地址获取；下载权限不等于允许重新分发。

注意两套 CLI 命名风格不同：数据工具使用连字符参数，例如 `--split-manifest`；上游训练和渲染兼容参数使用下划线，例如 `--split_manifest`。

### 1. 生成数据集和基础划分清单

```powershell
.\.venv\Scripts\python.exe tools\create_colmap_manifest.py `
  --dataset-root data\TI-NSD\heated `
  --dataset-id ti-nsd-heated `
  --data-range 0 255 `
  --test-every 8 `
  --val-every 8
```

建议先加 `--dry-run` 检查视角数量和路径；已有输出默认不会覆盖，确实需要替换时才使用 `--overwrite`。

### 2. 生成嵌套稀疏训练集

```powershell
.\.venv\Scripts\python.exe tools\create_sparse_split.py `
  --dataset-manifest data\TI-NSD\heated\dataset_manifest.v1.json `
  --base-split-manifest data\TI-NSD\heated\base_split.v1.json `
  --output-dir data\TI-NSD\heated\splits\seed2026 `
  --ratios 1 0.5 0.25 0.125 `
  --method nested-random `
  --seed 2026
```

`nested-random` 保证：

```text
12.5% subset ⊂ 25% subset ⊂ 50% subset ⊂ 100%
```

也可以使用按轨迹均匀取样的 `--method trajectory-uniform`。在同一种采样协议下比较的 baseline 和 proposed 必须共享完全相同的 split 清单；不同采样协议应作为不同实验条件明确标注。

### 3. 生成仅训练集退化

高斯噪声示例：

```powershell
.\.venv\Scripts\python.exe tools\degrade_ir_dataset.py `
  --dataset-manifest data\TI-NSD\heated\dataset_manifest.v1.json `
  --split-manifest data\TI-NSD\heated\splits\seed2026\sparse_nested-random_12p5.json `
  --output-dir data\TI-NSD\heated\derived\noise03_12p5_seed2026 `
  --degradation gaussian-noise `
  --noise-sigma 0.03 `
  --seed 2026
```

`--noise-sigma 0.03` 表示统一强度范围映射到 `[0,1]` 后的标准差，不是 uint8 灰度级 0.03。

组合退化示例：

```powershell
.\.venv\Scripts\python.exe tools\degrade_ir_dataset.py `
  --dataset-manifest <scene>\dataset_manifest.v1.json `
  --split-manifest <split.json> `
  --output-dir <derived-output> `
  --degradation combined `
  --order contrast blur gaussian-noise `
  --contrast-alpha 0.5 `
  --blur-sigma 1.0 `
  --blur-kernel-size 3 `
  --noise-sigma 0.03 `
  --seed 2026
```

工具默认拒绝覆盖任何已有目标，并且从不修改源图。验证和测试图不会复制到退化目录，退化清单会把它们解析回原始数据集。

训练时的 `-s/--source_path` 始终指向原始完整 COLMAP 场景根目录。派生退化目录只是训练图覆盖层，不能直接作为场景根目录。

### 4. 训练严格配对的退化 baseline

baseline 也必须读取与 proposed 完全相同的退化清单，只把新增损失权重设为零：

```powershell
.\.venv\Scripts\python.exe train.py `
  -s data\TI-NSD\heated `
  -m runs\baseline_sparse12p5_noise03 `
  --eval `
  --dataset_manifest data\TI-NSD\heated\dataset_manifest.v1.json `
  --split_manifest data\TI-NSD\heated\splits\seed2026\sparse_nested-random_12p5.json `
  --degradation_manifest data\TI-NSD\heated\derived\noise03_12p5_seed2026\degradation_manifest.v1.json `
  --evaluation_partition val `
  --resolution 1 `
  --data_device cpu `
  --load2gpu_on_the_fly `
  --iterations 30000 `
  --test_iterations 7000 30000 `
  --save_iterations 7000 30000 `
  --lambda_thermal 0 `
  --lambda_edge 0 `
  --lambda_smooth 0
```

运行干净 baseline 时保留同一 split，但省略 `--degradation_manifest`。不要把“干净 baseline”与“噪声 proposed”作为方法对比，因为那会同时改变训练数据和损失。

### 5. 训练 proposed

```powershell
.\.venv\Scripts\python.exe train.py `
  -s data\TI-NSD\heated `
  -m runs\proposed_sparse12p5_noise03 `
  --eval `
  --dataset_manifest data\TI-NSD\heated\dataset_manifest.v1.json `
  --split_manifest data\TI-NSD\heated\splits\seed2026\sparse_nested-random_12p5.json `
  --degradation_manifest data\TI-NSD\heated\derived\noise03_12p5_seed2026\degradation_manifest.v1.json `
  --evaluation_partition val `
  --resolution 1 `
  --data_device cpu `
  --load2gpu_on_the_fly `
  --iterations 30000 `
  --test_iterations 7000 30000 `
  --save_iterations 7000 30000 `
  --lambda_thermal 0.1 `
  --lambda_edge 0.01 `
  --lambda_smooth 0.001 `
  --noise_beta 5 `
  --edge_gamma 3
```

`--data_device cpu --load2gpu_on_the_fly` 会把相机图像常驻 CPU，只在当前视角训练时传入 GPU，适合 6 GB 显存。若 densification 后仍显存不足，可先使用 `--resolution 2` 或 `--resolution 4` 做容量校准。

### 6. 渲染验证集

```powershell
.\.venv\Scripts\python.exe render.py `
  -m runs\proposed_sparse12p5_noise03 `
  --iteration 30000 `
  --skip_train `
  --evaluation_partition val `
  --mode render
```

manifest 模式会把结果写入：

```text
runs/proposed_sparse12p5_noise03/val/ours_30000/
```

验证集用于超参数、损失权重和模型迭代选择。训练过程中的上游控制台/TensorBoard 标签仍把当前 held-out 分区称为 `test`；当 `evaluation_partition=val` 时，该标签实际表示 val，磁盘输出目录仍会正确写为 `val/`。

### 7. 超参数冻结后渲染测试集

```powershell
.\.venv\Scripts\python.exe render.py `
  -m runs\proposed_sparse12p5_noise03 `
  --iteration 30000 `
  --skip_train `
  --evaluation_partition test `
  --mode render
```

测试结果写入 `test/ours_30000/`。没有 manifest 的上游兼容流程仍使用原始 `test/` 目录规则。

### 8. 收集指标

验证集配对指标示例：

```powershell
.\.venv\Scripts\python.exe tools\collect_metrics.py `
  --experiment baseline=runs\baseline_sparse12p5_noise03\val\ours_30000 `
  --experiment proposed=runs\proposed_sparse12p5_noise03\val\ours_30000 `
  --output-dir results\validation_sparse12p5_noise03 `
  --skip-lpips `
  --device cuda
```

`--skip-lpips` 不会初始化 LPIPS 网络，结果中会明确写 `unavailable`。需要 LPIPS 时移除该参数；若不希望模型缓存进入 C 盘，可先设置：

```powershell
$env:TORCH_HOME = 'E:\Software\TorchCache'
```

## 批量实验

### 实验矩阵

[正式矩阵示例](configs/experiment_matrix.example.json) 包含：

- baseline 和 proposed 的 100%、50%、25%、12.5% 稀疏实验；
- 完整视角 noise03 baseline/proposed；
- Sparse-25 + noise03 baseline/proposed。

正式示例统一使用 `data_device=cpu` 和 `load2gpu_on_the_fly=true`，确保图像及相机张量在当前视角训练前一起传入 GPU。两项必须配套；只把 `data_device` 设为 `cpu` 会导致 CUDA rasterizer 收到 CPU 相机矩阵。

该示例假定以下退化清单已经按对应 split 生成：

```text
${THERMAL_DEGRADATION_DIR}/noise03_full/degradation_manifest.v1.json
${THERMAL_DEGRADATION_DIR}/noise03_sparse25/degradation_manifest.v1.json
```

当前本地只有 12.5% 的 `noise03_12p5_seed2026` 退化集。运行正式矩阵前，应先用第 3 步的工具分别生成矩阵引用的完整视角和 Sparse-25 退化目录；`--dry-run` 只打印命令，不会替你生成这些输入。

先设置路径变量并 dry-run：

```powershell
$env:THERMAL_DATASET_ROOT = 'E:\1_Work\Graduate\Work\Thermal3DGS_sparse\data\TI-NSD\heated'
$env:THERMAL_SPLIT_DIR = "$env:THERMAL_DATASET_ROOT\splits\seed2026"
$env:THERMAL_DEGRADATION_DIR = "$env:THERMAL_DATASET_ROOT\derived"

.\.venv\Scripts\python.exe scripts\run_experiments.py `
  --matrix configs\experiment_matrix.example.json `
  --dry-run
```

确认命令后执行全部实验：

```powershell
.\.venv\Scripts\python.exe scripts\run_experiments.py `
  --matrix configs\experiment_matrix.example.json
```

或只运行指定实验：

```powershell
.\.venv\Scripts\python.exe scripts\run_experiments.py `
  --matrix configs\experiment_matrix.example.json `
  --only baseline_sparse25 ours_sparse25
```

runner 默认拒绝复用已有实验目录。每个实验在训练前写入 `run_manifest.json`，记录：

- 完整训练和渲染命令；
- dataset、split 和 degradation 清单文件 SHA256；
- Git HEAD、工作树状态、diff 哈希和包含未跟踪源码的快照哈希；
- Python、平台、开始/结束时间；
- `running`、`completed` 或 `failed` 状态。

矩阵还可以为每个实验提供 `expected_input_sha256`。runner 会在创建输出目录前比较 dataset、
split 和 degradation 清单哈希；任何不一致都会直接终止，避免在错误输入上生成实验结果。

runner 会将训练和渲染子进程分别写入实验目录下的 `stdout.log` 和 `stderr.log`，主终端只显示实验开始、完成或失败状态。训练器默认每 1000 次迭代更新一次摘要；传入 `--quiet` 时关闭实时进度条，但仍保留辅助损失 resolved 配置、低频损失分项、检查点、验证和最终摘要。每次低频摘要也写入 `loss_components.csv`，未启用的 raw 项标记为 `disabled`，不会为日志额外执行辅助算子。

安装 TensorBoard 后，训练器会每 5 秒刷新 event 文件。实时查看本轮四组实验：

```powershell
.\.venv\Scripts\python.exe -m tensorboard.main `
  --logdir runs\priority1_loss_revision `
  --host 127.0.0.1 `
  --port 6006 `
  --reload_interval 5
```

浏览器打开 `http://127.0.0.1:6006/`。主要曲线位于 `train_loss/*`、`iter_time` 和
`val/*`；disabled 的 raw 辅助项不会伪造数值，weighted 曲线会如实记录为 0。

`--allow-existing` 会允许复用已有目录，应仅在明确理解覆盖风险时使用。

Windows 也可以使用包装脚本：

```powershell
.\scripts\run_sparse_degradation_experiments.ps1 `
  -Matrix configs\experiment_matrix.example.json `
  --dry-run
```

[heated smoke 矩阵](configs/experiment_matrix.heated_smoke.json) 是当前单机链路验证的精确配置：

```powershell
.\.venv\Scripts\python.exe scripts\run_experiments.py `
  --matrix configs\experiment_matrix.heated_smoke.json `
  --dry-run
```

本机对应的 `runs\smoke_noise03_pair` 已存在，因此直接去掉 `--dry-run` 会触发预期的防覆盖错误。复现实验时应先修改配置中的 `output_root`，不要为图省事直接加 `--allow-existing`。

## 指标与论文图

### 指标定义

| 指标 | 含义 | 备注 |
| --- | --- | --- |
| PSNR | 峰值信噪比 | 使用固定 `[0,1]` 映射 |
| SSIM | 结构相似性 | 与 PSNR 使用同一图像配对 |
| LPIPS | 感知距离 | 亮度通道复制为三通道；默认 VGG，属于次要指标 |
| T-MAE | 热强度平均绝对误差 | 不进行逐图 min-max |
| E-MAE | 目标边缘加权梯度误差 | 按 GT 边缘权重总和归一化 |
| Gradient preservation | 梯度幅值保持 | 逐像素 Sobel 梯度幅值相似度的均值，范围 `[0,1]` |
| ROI-MAE | 指定目标区域热强度误差 | 没有固定 mask 或 mask 为空时记为 unavailable |

PSNR、SSIM 和 Gradient preservation 越高越好；LPIPS、T-MAE、E-MAE 和 ROI-MAE 越低越好。

图像读取采用固定编码范围：uint8 除以 255，uint16 除以 65535，不进行逐图 min-max；RGB 使用 BT.601 权重 `(0.299, 0.587, 0.114)` 转成亮度。SSIM 优先使用上游 3D-GS 实现，不可导入时才使用本地等价后备实现。LPIPS 默认使用仓库内 `lpipsPyTorch` 的 VGG 网络，并把亮度通道复制为三通道输入。

指标工具要求 `renders/` 和 `gt/` 中的文件名集合完全一致，并按文件名字典序配对。它不会静默忽略缺图，也不会用文件夹枚举顺序配对。评估图和绘图输入必须使用受支持的无损格式，例如 PNG、TIFF、BMP 或 PNM；JPEG、WebP 等有损结果会被拒绝。

### 生成论文图

先根据正式测试结果修改 [绘图规范示例](configs/figure_spec.example.json)，再运行：

```powershell
.\.venv\Scripts\python.exe tools\make_figures.py `
  --spec configs\figure_spec.example.json
```

一次生成五类 PNG：

- 定性对比网格；
- 绝对误差图；
- Sobel 热边缘对比；
- 稀疏视角性能曲线；
- 退化鲁棒性曲线。

绘图规范显式固定文件名、方法顺序、色标范围、曲线坐标和指标列。生成的 `figure_manifest.json` 记录规范与输出哈希，便于论文结果审计。

## 输出目录

runner 生成的单次实验通常具有以下结构：

```text
runs/<experiment>/
|-- run_manifest.json
|-- cfg_args
|-- cameras.json
|-- input.ply
|-- point_cloud/
|   `-- iteration_<N>/point_cloud.ply
|-- ATF/
|   `-- iteration_<N>/ATF.pth
|-- TCM/
|   `-- iteration_<N>/TCM.pth
`-- <partition>/
    `-- ours_<N>/
        |-- renders/
        |-- gt/
        `-- render_time.txt
```

`<partition>` 是本次运行唯一的 `evaluation_partition`。正式矩阵默认生成 `val/`；冻结超参数后，需要按第 7 步另行渲染 `test/`。runner 不会在一次实验中自动同时生成两个分区。

`cfg_args` 保存训练参数，渲染时会和命令行覆盖项安全合并。Gaussian、ATF 和 TCM 使用相同迭代号重载，避免组合来自不同迭代的模型。

渲染器当前还会创建 `depth/` 目录，但没有写入深度图。`point_cloud.ply`、`ATF.pth` 和 `TCM.pth` 是推理所需的模型快照，不包含优化器恢复状态，不能当作完整的可续训 checkpoint。

指标工具输出：

```text
metrics_summary.csv
metrics_summary.md
metrics_per_view.csv
```

## 当前验证状态

截至 `2026-08-27`，本机已完成以下验证：

- 本地只准备了 TI-NSD 的 `heated` 场景，共 307 个已注册视角；
- 基础划分为 234 个训练池视角、34 个验证视角和 39 个测试视角；
- 12.5% + noise03 生成 29 张无损 PNG，全部输出哈希通过验证；
- 两个 CUDA 扩展成功构建和导入；
- 68/68 个 unittest 通过，其中包含 KNN 数值对照和 rasterizer 前向/反向 CUDA 测试；
- Python 模块编译、配置 JSON 解析和 `git diff --check` 均通过。

seed 2026 的嵌套训练子集为：

| 标称比例 | 训练视角 | 实际比例 |
| --- | ---: | ---: |
| 100% | 234 | 1.0 |
| 50% | 117 | 0.5 |
| 25% | 59 | 0.2521367521 |
| 12.5% | 29 | 0.1239316239 |

同一 12.5% split、同一 noise03 退化字节下的 10-iteration 配对 smoke 结果如下。两组均使用 `--resolution 4 --data_device cpu --load2gpu_on_the_fly`，即图像常驻 CPU、当前视角按需传入 GPU；这些结果不能与下方原始分辨率容量校准直接比较。

| 方法 | 视角 | PSNR | SSIM | T-MAE | E-MAE | 梯度保持 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline，新增损失全零 | 34 val | 17.37711321 | 0.81107783 | 0.10965925 | 0.01910995 | 0.68279362 |
| Proposed，完整新增损失 | 34 val | 17.38011804 | 0.81109057 | 0.10956990 | 0.01911006 | 0.68301739 |

另有一次 234 训练视角、原始分辨率、100-iteration baseline 容量校准，得到 34 个 val 渲染，PSNR 为 20.88558902，SSIM 为 0.92215234。该运行在 densification 开始前结束，因此只能证明全分辨率基础链路可用，不能证明 7k/30k 峰值显存安全。

这些数值都是工程 smoke，不是收敛结果，也不支持方法优越性结论。当前尚未完成：

- 未发现与 TI-NSD `heated` 明确匹配的公开上游 checkpoint，当前未下载或使用任何预训练 checkpoint；
- 正式 7k/30k baseline；
- 三个或更多预注册 seed 的均值和标准差；
- 完整稀疏比例、噪声级别、弱纹理和消融矩阵；
- LPIPS、固定 ROI-MAE 和最终 39-view test 报告；
- 可用于论文结论的定量表和定性图。

当前 `sparse-ir-development` 工作树仍有未提交和未跟踪修改，因此还不存在能够代表本扩展现状的已发布 commit。现有 runner 通过 Git 状态、diff 哈希和源码快照哈希记录每次 smoke 的实际代码内容。

完整历史时间线见 [实验日志](docs/experiment_log.md)；正式训练计划的逐阶段执行记录见
[正式训练记录](docs/formal_training_log.md)。

## 测试

运行全部测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py' -v
```

按模块运行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_arguments.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_colmap_manifest.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_cuda_extensions.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_degradations.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_experiment_runner.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_figures.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_manifest_adapter.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_metrics.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_splits.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_thermal_physics_loss.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_upstream_loss_utils.py -v
```

静态检查：

```powershell
.\.venv\Scripts\python.exe -m compileall -q arguments gaussian_renderer integrations losses scene scripts src tests tools train.py render.py
git diff --check
```

## 常见问题

### `CUDA_HOME=None` 或普通终端找不到 `nvcc`/`cl`

Scoop 清单会设置用户级 `CUDA_PATH` 并把 CUDA `bin` 加入用户 PATH；安装前已打开的终端不会自动继承，重新打开终端后再检查 `where.exe nvcc`。它不会设置用户级 `CUDA_HOME`。MSVC `cl` 不加入全局 PATH，需要由 `vcvars64.bat` 激活。重编译时直接运行 `scripts\build_cuda_extensions.cmd`，脚本会在自身 `setlocal` 作用域内完成两者配置；不要仅因旧终端看不到命令就重复安装。

### `No module named simple_knn` 或 `diff_gaussian_rasterization`

确认正在使用 `.venv\Scripts\python.exe`，然后重新运行 CUDA 构建脚本并执行扩展导入测试。

### CUDA 扩展为错误显卡架构编译

构建前设置正确的 `TORCH_CUDA_ARCH_LIST`，再重新安装两个扩展。当前默认值 7.5 只针对 RTX 2060。

### 训练时显存不足

优先组合使用：

```text
--data_device cpu
--load2gpu_on_the_fly
--resolution 2
```

先运行超过第 500 次迭代的容量校准，因为 Gaussian densification 开始后显存需求会变化。不要仅根据 10 或 100 次迭代判断 30k 一定可运行。

### 退化清单出现 unknown field 或 SHA256 mismatch

不要手工删除清单字段。重新使用当前版本的 `degrade_ir_dataset.py` 生成清单，并确认 dataset、split、退化文件及源图没有在生成后被修改。

### runner 拒绝已有输出目录

这是防覆盖设计。优先使用新的实验名或新的 `output_root`。只有明确需要复用并理解覆盖风险时才使用 `--allow-existing`。

### val 图像为什么位于 `val/` 而不是 `test/`

manifest 模式按 `evaluation_partition` 使用真实目录名：验证集写入 `val/ours_<N>`，最终测试集写入 `test/ours_<N>`。旧版曾把两者都写到 `test/`，该问题已经修复。

### LPIPS 下载占用 C 盘

先将 `TORCH_HOME` 指到 E 盘，再运行不带 `--skip-lpips` 的指标命令。当前 smoke 为避免额外模型下载，LPIPS 明确记为 `unavailable`。

## 实验规范与限制

正式实验必须遵守 [实验协议](docs/experimental_protocol.md)：

- 先冻结 test，再从剩余视角划分独立 val；
- 只用 val 调整超参数和选择模型迭代；
- baseline 和 proposed 共享初始化、split、退化图字节、优化器、densification、迭代预算和评估器；
- 至少报告三个预注册 seed，条件允许时给出均值、标准差和逐视角配对结果；
- 同一表格不能混合固定姿态和端到端姿态协议；
- 负结果不能通过删场景、删 seed 或只选有利条件隐藏；
- 无辐射定标时使用“热强度”“红外辐射表示”或“表观热分布”，不能声称重建真实温度。

新增可靠性图只是图像派生的噪声启发式。除非后续加入经过标定的传感器噪声模型，否则不应将其描述为直接的热物理约束。

## 许可与引用

固定的上游仓库没有根目录级许可证文件。部分源码头声明仅用于非商业研究和评估，但这不是完整的仓库再分发授权。TI-NSD 数据和公开模型权重也没有在当前下载入口提供清晰、完整的再分发许可。因此：

- 本项目目前仅按内部非商业研究场景处理，这不构成授权判断；仓库不重新分发数据、权重或完整上游修改源码，使用者须自行确认官方条款；
- 不要把公开可下载等同于允许重新分发；
- 发布代码归档、数据镜像或模型前，应向原作者确认授权；
- vendored CUDA rasterizer 的独立许可证只覆盖该子模块，不自动覆盖整个 Thermal3D-GS 仓库。

引用上游 Thermal3D-GS：

```bibtex
@inproceedings{chen2024thermal3dgs,
  title={Thermal3D-GS: Physics-induced 3D Gaussians for Thermal Infrared Novel-view Synthesis},
  author={Chen, Qian and Shu, Shihao and Bai, Xiangzhi},
  booktitle={European Conference on Computer Vision},
  year={2024}
}
```

本扩展的实验结果还应同时记录：上游 commit、本分支 Git commit 或 diff 哈希、Python/PyTorch/CUDA/MSVC 版本、数据清单哈希、split seed、退化清单哈希和完整训练命令。
