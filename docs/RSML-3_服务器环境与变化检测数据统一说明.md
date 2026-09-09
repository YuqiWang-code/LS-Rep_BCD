# RSML-3 服务器环境、数据集与 Teacher Cache 统一说明

> **适用项目**：`LS-Rep_BCD_RSML_3`\
> **整理日期**：2026-09-06\
> **用途**：统一记录新服务器硬件与深度学习环境、项目目录约定、变化检测数据集，以及 Teacher Cache 的读取规范。\
> **来源**：由原《RSML-3 服务器与深度学习环境说明》和《遥感变化检测数据集与 Teacher Cache 统一说明》合并整理。

## 阅读导航

- **第一部分**：RSML-3 新服务器、CUDA/Conda 环境、GPU 使用、SFTP 工作流与环境恢复。
- **第二部分**：CDD、LEVIR、SYSU、WHU 数据集结构、DataLoader contract、split、label、Teacher Cache 与接入检查表。

## 当前项目迁移约定

| 项目 | 约定路径或方式 |
|----|----|
| Windows 本地项目 | `F:\Code_Repositories_2\CursorCode\LS-Rep_BCD_RSML_3` |
| 服务器项目目录 | `/home/yqwang/projects/LS-Rep_BCD_RSML_3` |
| 数据集根目录 | `/home/yqwang/datasets/CD` |
| Teacher Cache 根目录 | `/home/yqwang/datasets/CD_teacher_cache` |
| 预训练权重 | 本地位于 `pre-trained_weights/`；服务器建议放在项目同名目录或 `/home/yqwang/checkpoints/LS-Rep_BCD_RSML_3/` |
| Checkpoint 根目录 | `/home/yqwang/checkpoints/LS-Rep_BCD_RSML_3` |
| 日志与实验输出 | `/home/yqwang/outputs/LS-Rep_BCD_RSML_3` |
| 代码上传 | 使用 `.vscode/sftp.json` 手动上传，`uploadOnSave` 与 watcher 自动上传保持关闭 |

注意：`.vscode/sftp.json` 默认忽略 `pre-trained_weights/` 和常见权重文件，约 2.85 GB 的预训练权重不会随代码自动上传；需要时应单独传输，并在训练前检查权重路径。旧脚本中的 `/home/yqwang/project/LS-Rep_BCD` 与 `/storage/...` 不适用于当前 RSML-3 目录规划，项目内可执行脚本已经按上表更新。

------------------------------------------------------------------------

## 第一部分：RSML-3 服务器与深度学习环境

### RSML-3 服务器与深度学习环境说明

> **用途**：遥感变化检测、CNN / Transformer / Mamba 模型训练与推理\
> **服务器主机名**：`RSML-3`\
> **用户**：`yqwang`\
> **环境快照日期**：2026-09-06\
> **工作方式**：Windows 本地使用 Cursor 编写/维护代码，通过 SFTP 上传到服务器，在服务器端使用 GPU 进行训练和推理。\
> **说明**：本文档以 2026-09-06 的实际探查结果和环境导出文件为准。硬件实时占用、温度、GPU 空闲情况等会随时间变化。

------------------------------------------------------------------------

#### 1. 总览

| 项目                            | 当前配置                               |
|---------------------------------|----------------------------------------|
| 操作系统                        | Ubuntu 22.04.5 LTS (Jammy Jellyfish)   |
| Kernel                          | Linux 6.8.0-138-generic                |
| 架构                            | x86_64                                 |
| glibc                           | 2.35                                   |
| CPU                             | AMD Ryzen Threadripper 7960X           |
| CPU 核心/线程                   | 24 核 / 48 线程                        |
| 内存                            | 约 64 GB 物理内存，Linux 可见约 62 GiB |
| Swap                            | 2.0 GiB                                |
| 系统盘                          | ZHITAI TiPlus9100 4TB NVMe SSD         |
| 根分区                          | ext4，约 3.7 TB                        |
| GPU                             | 2 × NVIDIA GeForce RTX 5090            |
| GPU 架构                        | Blackwell                              |
| 单卡显存                        | `nvidia-smi` 报告 32607 MiB            |
| NVIDIA Driver                   | 595.80                                 |
| `nvidia-smi` CUDA compatibility | 13.2                                   |
| 系统 CUDA Toolkit               | CUDA 13.2 / nvcc 13.2.78               |
| Miniforge                       | `/home/yqwang/miniforge3`              |
| Conda                           | 26.5.3                                 |
| Mamba 包管理器                  | 2.5.0                                  |
| Conda solver                    | libmamba                               |
| 默认 channel                    | conda-forge                            |
| 主要环境                        | `tools`、`cd_base`、`mamba_base`       |

当前服务器已经具备面向 RTX 5090 / Blackwell（Compute Capability 12.0、`sm_120`）的 PyTorch CUDA 运行能力，`cd_base` 与 `mamba_base` 均已完成双 GPU 实际 forward/backward 验证。

------------------------------------------------------------------------

### 2. 操作系统与主机信息

#### 2.1 主机

``` text
Hostname: RSML-3
Hardware Vendor: ASUS
Architecture: x86-64
```

系统：

``` text
Ubuntu 22.04.5 LTS
Codename: jammy
Kernel: Linux 6.8.0-138-generic
glibc: 2.35
```

服务器采用标准 Ubuntu 22.04 LTS 环境，适合作为长期深度学习训练节点。

------------------------------------------------------------------------

### 3. CPU

#### 3.1 CPU 型号

``` text
AMD Ryzen Threadripper 7960X 24-Cores
```

| 参数         |        数值 |
|--------------|------------:|
| Socket       |           1 |
| 物理核心     |          24 |
| 每核心线程   |           2 |
| 逻辑 CPU     |          48 |
| CPU 最大频率 | 约 5362 MHz |
| CPU 最低频率 |  约 545 MHz |
| NUMA 节点    |           1 |
| NUMA node 0  |    CPU 0-47 |

CPU 缓存：

| 缓存 |                    容量 |
|------|------------------------:|
| L1d  | 768 KiB（24 instances） |
| L1i  | 768 KiB（24 instances） |
| L2   |  24 MiB（24 instances） |
| L3   |  128 MiB（4 instances） |

CPU 支持 AVX2、AVX-512 等指令集；PyTorch 当前构建也检测并使用 AVX-512 CPU capability。

------------------------------------------------------------------------

### 4. 内存

采集时：

``` text
Mem total:      约 62 GiB（Linux free -h）
Physical online memory: 64G（lsmem）
Swap:           2.0 GiB
```

采集快照：

``` text
MemTotal:       65325176 kB
MemAvailable:   61327120 kB
SwapTotal:       2097148 kB
SwapFree:        2097148 kB
HugePages_Total: 0
Hugepagesize:    2048 kB
```

因此可以将该机器理解为 **64 GB 物理内存配置**，操作系统实际可用内存约 62 GiB。

对于较大的遥感影像 DataLoader，需要注意这台服务器的主要资源优势在 GPU，而不是超大内存；不建议无节制提高 `num_workers`、预加载规模或内存缓存。

------------------------------------------------------------------------

### 5. 存储

#### 5.1 NVMe SSD

物理盘：

``` text
Device: /dev/nvme0n1
Model: ZHITAI TiPlus9100 4TB
Transport: NVMe
ROTA: 0
```

即：**4 TB NVMe 固态硬盘**。

分区：

``` text
/dev/nvme0n1p1    512M   vfat   /boot/efi
/dev/nvme0n1p2    3.7T   ext4   /
```

采集时根分区：

``` text
Size: 3.7T
Used: 80G
Avail: 3.5T
Use: 3%
```

当前没有独立 `/data` 数据盘；`/home/yqwang`、数据集、项目、checkpoint、输出等均位于同一块 NVMe 根文件系统上。

#### 5.2 科研目录

当前规划：

``` text
/home/yqwang/
├── projects/       # 项目代码
├── datasets/       # 数据集
├── checkpoints/    # 模型权重 / checkpoint
├── outputs/        # 实验结果、预测结果、日志等
├── cache/          # 下载/框架缓存
├── scripts/        # 通用服务器脚本
├── env_specs/      # Conda / pip / CUDA 环境档案
└── miniforge3/     # Miniforge 与 Conda 环境
```

采集时这些科研目录基本为空，根文件系统还有约 3.5 TB 可用空间。

------------------------------------------------------------------------

### 6. GPU

#### 6.1 GPU 配置

服务器拥有：

``` text
2 × NVIDIA GeForce RTX 5090
Architecture: Blackwell
```

两张卡的核心信息：

| 项目                    |        GPU 0 |        GPU 1 |
|-------------------------|-------------:|-------------:|
| 型号                    |     RTX 5090 |     RTX 5090 |
| CUDA Compute Capability |         12.0 |         12.0 |
| PyTorch 架构            |       sm_120 |       sm_120 |
| `nvidia-smi` 显存       |    32607 MiB |    32607 MiB |
| PyTorch 可见显存        | 约 31.35 GiB | 约 31.36 GiB |
| SM 数                   |          170 |          170 |
| Power Limit             |        575 W |        575 W |
| Persistence Mode        |      Enabled |      Enabled |
| Compute Mode            |      Default |      Default |
| MIG                     |          N/A |          N/A |

PCI Bus：

``` text
GPU 0: 00000000:21:00.0
GPU 1: 00000000:C1:00.0
```

#### 6.2 PCIe

采集时 GPU 处于空闲 P8 状态，因此 PCIe 会自动降速到 Gen1，这是节能行为，并不代表训练时只能运行 Gen1。

GPU 0：

``` text
PCIe Device Max: Gen5
Host Max: Gen5
Link width: x16
```

GPU 1：

``` text
PCIe Device Max: Gen5
Host Max: Gen4
Link width: x16
```

因此两张卡的主机 PCIe 路径并不完全相同：GPU 0 所在路径可达 PCIe Gen5，GPU 1 的 Host Max 探查结果为 Gen4。

#### 6.3 双卡拓扑

`nvidia-smi topo -m`：

``` text
        GPU0    GPU1    CPU Affinity    NUMA Affinity
GPU0     X      NODE    0-47            0
GPU1    NODE     X      0-47            0
```

含义：

- 两张 GPU 都属于 NUMA node 0；
- CPU affinity 均为 0-47；
- GPU0 ↔ GPU1 拓扑为 `NODE`；
- 没有显示 `NV#` 路径，因此当前拓扑中没有 NVIDIA NVLink 互联标识；
- 双卡通信主要经过 PCIe / 同 NUMA 节点的 PCIe Host Bridge 结构。

对于多 GPU 训练，优先使用 PyTorch DDP，而不是依赖 NVLink 的通信假设。

#### 6.4 采集时 GPU 状态

采集快照中两张卡基本空闲：

- GPU 0：约 35°C，P8；
- GPU 1：约 32°C，P8；
- GPU 利用率均为 0%。

GPU 0 当时有 Xorg、GNOME Shell 和远程桌面相关图形进程占少量显存；GPU 1 仅有很少的 Xorg 占用。实际训练前应重新执行 `nvidia-smi` / `nvitop`，不要依赖本文档中的实时占用数值。

------------------------------------------------------------------------

### 7. NVIDIA Driver 与 CUDA 分层

服务器上需要区分 **驱动、系统 Toolkit、PyTorch 自带 CUDA Runtime** 三个概念。

#### 7.1 NVIDIA Driver

``` text
NVIDIA Driver: 595.80
nvidia-smi CUDA Version: 13.2
```

这里 `nvidia-smi` 的 CUDA 13.2 表示当前驱动支持的 CUDA compatibility level。

#### 7.2 系统 CUDA Toolkit

``` text
nvcc: /usr/local/cuda-13.2/bin/nvcc
CUDA Toolkit: 13.2
nvcc version: V13.2.78
```

软链接：

``` text
/usr/local/cuda -> /usr/local/cuda-13.2/
```

Shell 路径：

``` text
PATH:
  /usr/local/cuda-13.2/bin

LD_LIBRARY_PATH:
  /usr/local/cuda-13.2/lib64
```

Shell 中：

``` text
CUDA_HOME=
```

即环境变量 `CUDA_HOME` 本身没有显式设置。

但是在 `mamba_base` 中：

``` python
from torch.utils.cpp_extension import CUDA_HOME
```

能够解析得到：

``` text
/usr/local/cuda-13.2
```

这对 CUDA Extension 编译是重要信息。

#### 7.3 各环境 CUDA Runtime

| 层级/环境                    | CUDA |
|------------------------------|------|
| NVIDIA Driver compatibility  | 13.2 |
| 系统 CUDA Toolkit / nvcc     | 13.2 |
| `cd_base` PyTorch Runtime    | 13.2 |
| `mamba_base` PyTorch Runtime | 13.0 |

因此：

- `cd_base` 与系统 CUDA 主版本/次版本一致；
- `mamba_base` 使用 PyTorch `cu130`，但 CUDA extension 编译工具链能够找到系统 CUDA 13.2；
- 目前该组合已经完成 RTX 5090 实际 forward/backward 验证，不应仅因为 13.0 / 13.2 的 minor version 差异而随意更换系统 CUDA 或驱动。

------------------------------------------------------------------------

### 8. 系统编译工具链

系统工具：

| 工具       | 版本   |
|------------|--------|
| GCC        | 11.4.0 |
| G++        | 11.4.0 |
| GNU Make   | 4.3    |
| CMake      | 3.22.1 |
| Git        | 2.34.1 |
| tmux       | 3.2a   |
| 系统 Ninja | 未安装 |

注意：

- “系统 GCC 11.4”与“PyTorch 构建时使用的 GCC 13.3”是不同概念；
- `cd_base` / `mamba_base` 的 PyTorch 官方构建信息显示其二进制构建时使用 GCC 13.3；
- `mamba_base` 环境内安装了 `ninja 1.13.2`，因此 Mamba/CUDA extension 本地编译时仍可使用 Ninja，无需系统级安装。

------------------------------------------------------------------------

### 9. Miniforge / Conda

#### 9.1 Miniforge

安装目录：

``` text
/home/yqwang/miniforge3
```

Conda：

``` text
conda 26.5.3
```

Mamba：

``` text
mamba 2.5.0
```

Base Python：

``` text
Python 3.14.6
```

Solver：

``` text
libmamba (default)
```

Channel：

``` yaml
channels:
  - conda-forge
```

主要环境：

``` text
base       /home/yqwang/miniforge3
tools      /home/yqwang/miniforge3/envs/tools
cd_base    /home/yqwang/miniforge3/envs/cd_base
mamba_base /home/yqwang/miniforge3/envs/mamba_base
```

`.bashrc` 当前存在：

``` bash
conda activate tools
```

因此新的 SSH shell 默认会进入 `(tools)`。

训练项目时必须主动切换：

``` bash
conda activate cd_base
```

或：

``` bash
conda activate mamba_base
```

------------------------------------------------------------------------

### 10. 开发与训练工作流

当前主要工作模式：

``` text
Windows
  │
  ├─ Cursor：代码编写、项目开发、Git 操作
  │
  └─ SFTP
       │
       ▼
RSML-3
/home/yqwang/projects/<project_name>
       │
       ├─ cd_base / 项目专属 cd 环境
       │       └─ CNN / Transformer / 常规遥感变化检测
       │
       └─ mamba_base / 项目专属 Mamba 环境
               └─ Mamba / SSM 类项目
       │
       ▼
GPU 训练 / 推理
       │
       ├─ /home/yqwang/checkpoints/
       └─ /home/yqwang/outputs/
       │
       ▼
SFTP 下载实验结果到本地
```

推荐保持这个模式：

- **本地 Cursor**：编辑代码、重构、查看 Git diff；
- **SFTP**：上传项目代码、配置文件、小型资源；
- **服务器**：只负责 Linux 环境、数据、训练、推理、日志；
- **大数据集**：尽量不要反复通过 SFTP 往返；
- **checkpoint / outputs**：按项目建立子目录。

示例：

``` text
~/projects/changeformer/
~/datasets/LEVIR-CD/
~/checkpoints/changeformer/
~/outputs/changeformer/
```

------------------------------------------------------------------------

### 11. `tools` 环境

#### 11.1 作用

`tools` 是轻量服务器工具环境，适合：

- SSH 登录后的默认环境；
- GPU 状态查看；
- 日常服务器操作；
- 不承载具体论文项目的深度学习依赖。

核心信息：

``` text
Environment: tools
Python: 3.11.16
Pip: 26.2.1
Path: /home/yqwang/miniforge3/envs/tools
```

`pip check`：

``` text
No broken requirements found.
```

#### 11.2 nvitop

当前：

``` text
nvitop 1.7.1
```

实际可执行文件：

``` text
/home/yqwang/.local/bin/nvitop
```

注意：它位于用户级 `~/.local/bin`，而不是 `tools/bin` 中。因此虽然通常在 `(tools)` 下使用，但它本质上是用户级命令。

日常使用：

``` bash
nvitop
```

或：

``` bash
watch -n 1 nvidia-smi
```

#### 11.3 环境快照

``` text
/home/yqwang/env_specs/tools/
├── conda-list.txt
├── environment.yml
└── requirements.txt
```

`tools` 保持轻量，不建议往其中安装论文模型依赖。

------------------------------------------------------------------------

### 12. `cd_base` 环境

#### 12.1 定位

`cd_base` 是当前的 **通用遥感变化检测 / 计算机视觉基础环境**。

适合：

- CNN；
- Transformer；
- 遥感变化检测；
- 图像分类、检测、分割；
- OpenCV / Albumentations；
- Rasterio / GeoPandas / Shapely；
- timm 模型；
- TensorBoard / TorchMetrics；
- 不依赖特殊 Mamba CUDA extension 的论文项目。

建议将其作为**模板环境**，真正复现论文时优先 clone 后再安装论文专属依赖。

#### 12.2 基础版本

``` text
Environment: cd_base
Path: /home/yqwang/miniforge3/envs/cd_base

Python: 3.10.21
Pip: 26.2.1

PyTorch: 2.14.0+cu132
Torchvision: 0.29.0+cu132
Torch CUDA Runtime: 13.2
cuDNN: 9.24.0
NCCL: 2.30.7
Triton: 3.8.0
```

环境大小：

``` text
6.3G
```

#### 12.3 RTX 5090 支持

PyTorch 支持：

``` text
sm_75
sm_80
sm_86
sm_90
sm_100
sm_120
```

因此已经包含 RTX 5090 所需：

``` text
sm_120
```

PyTorch 实际识别：

``` text
GPU 0: RTX 5090, capability 12.0, 170 SM
GPU 1: RTX 5090, capability 12.0, 170 SM
```

#### 12.4 主要科研库

| 包              |     版本 |
|-----------------|---------:|
| NumPy           |    2.2.6 |
| SciPy           |   1.15.3 |
| Pandas          |    2.3.3 |
| Pillow          |   12.3.0 |
| OpenCV headless | 5.0.0.93 |
| Albumentations  |    2.0.8 |
| timm            |   1.0.29 |
| einops          |    0.8.2 |
| scikit-learn    |    1.7.2 |
| Matplotlib      |   3.10.9 |
| TorchMetrics    |    1.9.0 |
| TensorBoard     |   2.21.0 |
| Rasterio        |    1.4.3 |
| GeoPandas       |    1.1.4 |
| Shapely         |    2.1.2 |

Geo/GDAL 相关环境中还包含：

- GDAL core 3.10.3；
- PROJ 9.6.2；
- GEOS 3.14.0；
- PyProj 3.7.1；
- Pyogrio 0.11.0。

#### 12.5 健康检查

``` text
pip check:
No broken requirements found.
```

GPU 实测：

``` text
GPU 0 forward/backward: OK
GPU 1 forward/backward: OK
```

因此 `cd_base` 当前是一个健康、可用的 RTX 5090 通用训练基础环境。

#### 12.6 使用方式

直接测试：

``` bash
conda activate cd_base
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0))"
```

指定 GPU：

``` bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

或：

``` bash
CUDA_VISIBLE_DEVICES=1 python train.py
```

#### 12.7 推荐：论文环境从它 clone

例如 ChangeFormer：

``` bash
conda create -n changeformer --clone cd_base
conda activate changeformer
```

后续只在 `changeformer` 中安装论文专属包。

不要为了某篇论文直接大幅修改 `cd_base`。

------------------------------------------------------------------------

### 13. `mamba_base` 环境

#### 13.1 定位

`mamba_base` 是针对 **Mamba / State Space Model / 遥感 Mamba 论文**维护的基础环境。

适合：

- Mamba；
- Mamba-based Change Detection；
- ChangeMamba；
- MambaBCD；
- 需要 `causal-conv1d` / `selective_scan_cuda` 的项目；
- 需要 RTX 5090 `sm_120` 支持的 Mamba 模型。

该环境经过较多兼容性排查，**应视为冻结模板**，不要随意升级 PyTorch、Mamba 或 CUDA extension。

#### 13.2 基础版本

``` text
Environment: mamba_base
Path: /home/yqwang/miniforge3/envs/mamba_base

Python: 3.10.21
Pip: 26.2.1

PyTorch: 2.10.0+cu130
Torchvision: 0.25.0+cu130
Torch CUDA Runtime: 13.0
cuDNN: 9.15.1
NCCL: 2.28.9
Triton: 3.6.0
```

Mamba 相关：

``` text
mamba-ssm: 2.3.2.post1
causal-conv1d: 1.7.0
einops: 0.8.2
ninja: 1.13.2
transformers: 5.16.1
huggingface-hub: 1.30.0
```

环境大小：

``` text
5.8G
```

#### 13.3 CUDA 关系

系统：

``` text
CUDA Toolkit / nvcc: 13.2
```

PyTorch：

``` text
torch 2.10.0+cu130
Torch CUDA runtime: 13.0
```

PyTorch extension 工具检测：

``` text
torch.utils.cpp_extension.CUDA_HOME
= /usr/local/cuda-13.2
```

当前组合已经完成实际 CUDA 运算和 Mamba forward/backward 测试，因此暂时不要仅因 Toolkit 13.2 与 PyTorch Runtime 13.0 不完全一致而重装驱动或系统 CUDA。

#### 13.4 RTX 5090 支持

PyTorch 支持架构：

``` text
sm_75
sm_80
sm_86
sm_90
sm_100
sm_120
compute_120
```

RTX 5090：

``` text
Compute Capability: 12.0
sm_120
```

双卡均正常识别。

#### 13.5 CUDA Extensions

关键扩展：

``` text
causal_conv1d_cuda:
  /home/yqwang/miniforge3/envs/mamba_base/lib/python3.10/site-packages/
  causal_conv1d_cuda.cpython-310-x86_64-linux-gnu.so

selective_scan_cuda:
  /home/yqwang/miniforge3/envs/mamba_base/lib/python3.10/site-packages/
  selective_scan_cuda.cpython-310-x86_64-linux-gnu.so
```

SHA256：

``` text
causal_conv1d_cuda:
7d0bee2a0a6b65a223c530561508f2026d2ec2f5717f79f706ea1a590e27f037

selective_scan_cuda:
5b7e0d901c825a21ce880103ae4ea82b95389d22b4c88e884ef60f13829be807
```

##### 关于 `cuda_extensions.txt` 中的 `libc10.so not found`

探查脚本中曾直接：

``` python
import causal_conv1d_cuda
```

得到：

``` text
ImportError: libc10.so: cannot open shared object file
```

同时直接执行 `ldd` 时，也能看到部分 PyTorch 动态库显示 `not found`。

这不能直接解释为 CUDA extension 已损坏，因为：

1.  正常 Mamba 导入路径会先加载 PyTorch；
2.  `causal-conv1d` 实际 forward/backward 已通过；
3.  `selective_scan` 实际 forward/backward 已通过；
4.  完整 Mamba slow path / fast path 已通过。

后续如果需要单独检查扩展，优先：

``` bash
python - <<'PY'
import torch
import causal_conv1d_cuda
import selective_scan_cuda

print(causal_conv1d_cuda.__file__)
print(selective_scan_cuda.__file__)
PY
```

即先 `import torch`，再直接导入 extension。

------------------------------------------------------------------------

### 14. Mamba RTX 5090 兼容性处理记录

这是当前服务器环境中最重要的特殊记录。

#### 14.1 最终成功状态

验证结果：

``` text
causal-conv1d forward/backward: OK

Mamba slow path / GPU 0:
Mamba slow-path forward/backward: OK

Mamba fast path / GPU 0:
Mamba fast-path forward/backward: OK

Mamba fast path / GPU 1:
Mamba GPU 1 fast-path forward/backward: OK
```

说明当前 `mamba_base` 已经具备实际训练所需的 CUDA 反向传播能力。

#### 14.2 `causal-conv1d` 问题

此前 cached wheel 与当前 Torch ABI 不匹配，出现过类似动态库 undefined symbol 问题。

最终解决方式为：在当前 `torch 2.10.0+cu130` 环境下重新本地编译 `causal-conv1d 1.7.0`：

``` bash
python -m pip uninstall -y causal-conv1d

MAX_JOBS=4 \
CAUSAL_CONV1D_FORCE_BUILD=TRUE \
python -m pip install causal-conv1d==1.7.0 \
  --no-build-isolation \
  --no-cache-dir \
  --no-deps
```

编译完成后的 `causal-conv1d`：

``` text
import: OK
CUDA forward: OK
CUDA backward: OK
```

**重要原则**：如果未来修改 PyTorch 版本，当前编译好的 CUDA extension 不能默认视为仍兼容，应重新验证，必要时重新编译。

#### 14.3 不使用 `mamba-ssm 2.2.5`

旧版本尝试编译时包含已被 CUDA 13.2 移除支持的旧 GPU architecture，例如：

``` text
compute_53
```

导致：

``` text
nvcc fatal: Unsupported gpu architecture 'compute_53'
```

因此当前基础环境保留：

``` text
mamba-ssm 2.3.2.post1
```

不要为了“版本更老可能更稳定”而再次退回 2.2.5。

#### 14.4 Mamba slow-path backward 的核心问题

最初完整 Mamba：

``` python
Mamba(
    d_model=64,
    d_state=16,
    d_conv=4,
    expand=2,
    use_fast_path=False
)
```

forward 正常，但 backward 报：

``` text
CUDA error:
CUBLAS_STATUS_NOT_INITIALIZED
when calling cublasSgemm(...)
```

排查过程确认：

``` text
Pure PyTorch forward/backward                  OK
causal-conv1d forward/backward                OK
selective_scan forward/backward               OK
selective_scan + z forward/backward           OK
out_proj 输入强制 contiguous 后               仍失败
```

使用：

``` python
torch.autograd.set_detect_anomaly(True)
```

最终定位到 `mamba_simple.py`：

``` python
dt = self.dt_proj.weight @ dt.t()
```

其 `MmBackward0` 在当前 RTX 5090 + Torch 2.10 环境中触发 cuBLAS 错误。

#### 14.5 当前 Mamba Patch

文件：

``` text
/home/yqwang/miniforge3/envs/mamba_base/lib/python3.10/site-packages/
mamba_ssm/modules/mamba_simple.py
```

原代码：

``` python
dt = self.dt_proj.weight @ dt.t()
dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
```

修改为：

``` python
dt = F.linear(dt, self.dt_proj.weight)
dt = rearrange(dt, "(b l) d -> b d l", l=seqlen)
```

补丁 diff：

``` diff
-            dt = self.dt_proj.weight @ dt.t()
-            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
+            dt = F.linear(dt, self.dt_proj.weight)
+            dt = rearrange(dt, "(b l) d -> b d l", l=seqlen)
```

这里没有给 `F.linear` 添加 bias；Mamba 原路径会在后面的 selective scan 中单独处理 `dt_proj.bias`。

该修改之后：

``` text
Mamba slow path backward: OK
Mamba fast path backward: OK
GPU 0: OK
GPU 1: OK
```

#### 14.6 Patch 备份

已保存：

``` text
~/env_specs/mamba_base/
├── mamba_simple_original.py
├── mamba_simple_patched.py
└── mamba_patch.diff
```

如果重新安装/升级 `mamba-ssm`：

``` bash
pip install --force-reinstall ...
```

当前 `site-packages` 中的补丁可能被覆盖。

升级 Mamba 前，应先保存/比较补丁并在升级后重新运行验证。

------------------------------------------------------------------------

### 15. `mamba_base` 的 pip check 状态

当前：

``` text
mamba-ssm 2.3.2.post1 requires apache-tvm-ffi, which is not installed.
mamba-ssm 2.3.2.post1 requires quack-kernels, which is not installed.
mamba-ssm 2.3.2.post1 requires tilelang, which is not installed.
```

因此严格从 Python package metadata 看，`pip check` 并不是全绿。

但是当前实际使用的 classic Mamba 路径已经通过：

- causal-conv1d；
- selective scan；
- slow path；
- fast path；
- GPU 0；
- GPU 1；
- forward；
- backward。

所以目前不应仅为了让 `pip check` 显示 “No broken requirements” 而直接向 `mamba_base` 塞入上述三个包。

如果某个具体项目明确需要 `tilelang`、`quack-kernels` 或 `apache-tvm-ffi`，推荐先 clone：

``` bash
conda create -n <paper_env> --clone mamba_base
```

再在论文专属环境中测试安装，避免破坏已经验证通过的 `mamba_base`。

------------------------------------------------------------------------

### 16. 三个环境如何选择

| 场景                                | 环境               |
|-------------------------------------|--------------------|
| SSH 登录、GPU 查看、服务器工具      | `tools`            |
| CNN / ResNet / U-Net                | `cd_base`          |
| Transformer 遥感变化检测            | `cd_base`          |
| ChangeFormer 类项目                 | clone `cd_base`    |
| OpenCV / Albumentations             | `cd_base`          |
| Rasterio / GeoPandas                | `cd_base`          |
| 普通 timm backbone                  | `cd_base`          |
| Mamba / SSM                         | `mamba_base`       |
| ChangeMamba                         | clone `mamba_base` |
| MambaBCD                            | clone `mamba_base` |
| 需要 causal-conv1d / selective_scan | `mamba_base`       |
| 论文依赖特殊/冲突较大               | 新建论文独立环境   |

推荐原则：

``` text
tools       = 工具环境
cd_base     = 普通深度学习/遥感模板
mamba_base  = Mamba 模板

论文项目 ≠ 直接长期污染 base 环境
论文项目 = 从对应 base clone
```

------------------------------------------------------------------------

### 17. 推荐的项目环境策略

#### 17.1 普通遥感项目

例如：

``` bash
conda create -n changeformer --clone cd_base
conda activate changeformer
```

然后：

``` bash
cd ~/projects/changeformer
python -m pip install <论文额外依赖>
```

#### 17.2 Mamba 项目

例如：

``` bash
conda create -n changemamba --clone mamba_base
conda activate changemamba
```

clone 的主要优点：

- 保留当前已经测试成功的 Torch/Mamba/CUDA extension；
- 避免从 pip cache 重新获得错误 ABI wheel；
- 保留当前环境中已修改的 Python source 文件；
- 论文依赖冲突不会影响 `mamba_base`。

> clone 完成后仍建议运行一次 Mamba forward/backward 验证，确认补丁和 extension 都被完整复制。

------------------------------------------------------------------------

### 18. GPU 使用方式

#### 18.1 查看 GPU

``` bash
nvidia-smi
```

或：

``` bash
nvitop
```

#### 18.2 指定 GPU 0

``` bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

#### 18.3 指定 GPU 1

``` bash
CUDA_VISIBLE_DEVICES=1 python train.py
```

注意：

``` bash
CUDA_VISIBLE_DEVICES=1
```

后，程序内部：

``` python
torch.cuda.get_device_name(0)
```

仍会显示逻辑 `cuda:0`，但它实际映射到物理 GPU 1。

#### 18.4 两张卡同时可见

``` bash
CUDA_VISIBLE_DEVICES=0,1 python train.py
```

如果项目支持 PyTorch DistributedDataParallel，优先按项目 README 使用 `torchrun` / DDP。

------------------------------------------------------------------------

### 19. 长时间训练：tmux

服务器系统已有：

``` text
tmux 3.2a
```

创建会话：

``` bash
tmux new -s exp1
```

训练：

``` bash
conda activate <project_env>
cd ~/projects/<project_name>
CUDA_VISIBLE_DEVICES=0 python train.py
```

离开但不终止训练：

``` text
Ctrl+B
D
```

重新进入：

``` bash
tmux attach -t exp1
```

查看：

``` bash
tmux ls
```

------------------------------------------------------------------------

### 20. Cursor + SFTP 推荐实践

由于代码主要在 Windows Cursor 编写，推荐：

#### 本地

``` text
project/
├── configs/
├── datasets/        # 仅保留配置/软链接说明，不放服务器大数据
├── models/
├── scripts/
├── train.py
├── infer.py
├── requirements.txt
└── README.md
```

#### 服务器

``` text
~/projects/project/
~/datasets/<dataset>/
~/checkpoints/project/
~/outputs/project/
```

建议让训练配置使用绝对路径或统一的配置项，例如：

``` yaml
data_root: /home/yqwang/datasets/LEVIR-CD
checkpoint_dir: /home/yqwang/checkpoints/project
output_dir: /home/yqwang/outputs/project
```

不要把 TB/GB 级数据集放在项目 Git 仓库中。

------------------------------------------------------------------------

### 21. 环境档案与备份

当前档案目录：

``` text
/home/yqwang/env_specs/
```

#### 21.1 `cd_base`

``` text
~/env_specs/cd_base/
├── overview.txt
├── validation.txt
├── conda-list.txt
├── requirements.txt
└── environment.yml
```

此外根目录中还保留过：

``` text
~/env_specs/cd_base.yml
~/env_specs/cd_base-pip.txt
```

根目录的旧快照可能与 `cd_base/` 中的当前快照存在少量版本差异；后续应优先以：

``` text
~/env_specs/cd_base/overview.txt
~/env_specs/cd_base/conda-list.txt
~/env_specs/cd_base/environment.yml
```

这一组 2026-09-06 最新探查文件为准。

#### 21.2 `mamba_base`

``` text
~/env_specs/mamba_base/
├── overview.txt
├── validation.txt
├── cuda_extensions.txt
├── environment.yml
├── requirements.txt
├── conda-list.txt
├── system_info.txt
├── mamba_patch.diff
├── mamba_simple_original.py
└── mamba_simple_patched.py
```

#### 21.3 `tools`

``` text
~/env_specs/tools/
├── conda-list.txt
├── environment.yml
└── requirements.txt
```

#### 21.4 总汇总

``` text
/home/yqwang/env_specs/env_specs_all.txt
```

这是环境信息的集中式文本快照。

------------------------------------------------------------------------

### 22. 环境恢复原则

#### 22.1 同一服务器上创建项目环境

最推荐：

``` bash
conda create -n <new_env> --clone cd_base
```

或：

``` bash
conda create -n <new_env> --clone mamba_base
```

这比“从头按照 requirements 重装”更适合当前 Mamba 环境，因为后者包含已经验证过的 CUDA binary extension 和手工 Python patch。

#### 22.2 `cd_base` 从快照恢复

参考：

``` text
~/env_specs/cd_base/environment.yml
~/env_specs/cd_base/requirements.txt
~/env_specs/cd_base/conda-list.txt
```

`cd_base` 的依赖关系目前健康，恢复难度较低。

#### 22.3 `mamba_base` 从零恢复

不能只看一个 `requirements.txt` 就认为完全恢复。

必须同时考虑：

1.  Python 3.10.21；
2.  Torch 2.10.0+cu130；
3.  Torchvision 0.25.0+cu130；
4.  mamba-ssm 2.3.2.post1；
5.  causal-conv1d 1.7.0；
6.  causal-conv1d 应在当前 Torch 上验证 ABI；
7.  必要时重新 source build；
8.  恢复 `mamba_simple.py` patch；
9.  重跑 causal-conv1d backward；
10. 重跑 Mamba slow path；
11. 重跑 Mamba fast path；
12. 至少分别验证 GPU 0 / GPU 1。

不要把“pip 安装成功”等同于“Mamba CUDA backward 可训练”。

------------------------------------------------------------------------

### 23. 当前已验证的关键状态

#### 服务器

``` text
Ubuntu                     OK
NVIDIA Driver              OK
System CUDA 13.2           OK
2 × RTX 5090               OK
NVMe SSD                   OK
Miniforge                  OK
```

#### `tools`

``` text
Python 3.11.16             OK
pip check                  OK
nvitop 1.7.1               OK
```

#### `cd_base`

``` text
Python 3.10.21             OK
Torch 2.14.0+cu132         OK
CUDA 13.2                  OK
RTX 5090 sm_120            OK
GPU 0 forward/backward     OK
GPU 1 forward/backward     OK
pip check                  OK
Geo/CV libraries           OK
```

#### `mamba_base`

``` text
Python 3.10.21                  OK
Torch 2.10.0+cu130              OK
CUDA runtime 13.0               OK
System nvcc 13.2                OK
RTX 5090 sm_120                 OK
causal-conv1d import            OK
causal-conv1d forward/backward  OK
selective_scan                  OK
Mamba slow-path GPU 0           OK
Mamba fast-path GPU 0           OK
Mamba fast-path GPU 1           OK
Mamba patch                     ACTIVE
pip check                       3 missing optional/current metadata deps
```

------------------------------------------------------------------------

### 24. 当前不建议做的操作

为了保护已经验证通过的基础环境：

##### 不建议直接升级系统 Driver/CUDA

当前 RTX 5090 已经可以正常训练，没有理由仅为了追新版本而重装。

##### 不建议在 `mamba_base` 里执行无约束升级

例如：

``` bash
pip install -U torch
pip install -U mamba-ssm
pip install -U causal-conv1d
```

这些操作可能改变 ABI，并覆盖补丁。

##### 不建议将所有论文依赖装进 `cd_base` / `mamba_base`

正确方式：

``` text
base 环境保持稳定
      ↓ clone
论文环境自由调整
```

##### 不建议为了 `pip check` 好看而直接补 Mamba 的三个依赖

如果论文确实需要，再在 clone 环境中处理。

------------------------------------------------------------------------

### 25. 快速命令表

#### 登录后查看当前环境

``` bash
echo $CONDA_DEFAULT_ENV
which python
python --version
```

#### 环境列表

``` bash
conda env list
```

#### 工具环境

``` bash
conda activate tools
nvitop
```

#### 普通遥感环境

``` bash
conda activate cd_base
```

#### Mamba 环境

``` bash
conda activate mamba_base
```

#### GPU

``` bash
nvidia-smi
nvitop
```

#### CUDA

``` bash
nvcc --version
which nvcc
```

#### PyTorch

``` bash
python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
PY
```

#### 磁盘

``` bash
df -hT
du -sh ~/datasets ~/projects ~/checkpoints ~/outputs
```

#### CPU/内存

``` bash
lscpu
free -h
```

------------------------------------------------------------------------

### 26. 数据源

本文档由以下实际服务器探查/导出资料整理：

``` text
nvidia-smi-full.txt
research-workspace.txt
env_specs_all.txt
```

以及此前的服务器硬件探查输出，包括：

``` text
hostnamectl
/etc/os-release
uname
lscpu
free
lsmem
lsblk
df
nvme
lspci
nvidia-smi
nvidia-smi topo
nvcc
gcc / g++
git
make
cmake
tmux
conda info
conda env list
```

环境快照根目录：

``` text
/home/yqwang/env_specs/
```

服务器科研工作区：

``` text
/home/yqwang/
```

------------------------------------------------------------------------

### 27. 最终使用原则

当前服务器环境可以概括为：

``` text
RSML-3
│
├── Ubuntu 22.04.5
├── Threadripper 7960X
├── 64 GB RAM
├── 4 TB NVMe SSD
├── RTX 5090 × 2
├── Driver 595.80
├── CUDA Toolkit 13.2
│
└── Miniforge
    │
    ├── tools
    │   └── 登录 / 监控 / 日常工具
    │
    ├── cd_base
    │   ├── Torch 2.14 + cu132
    │   └── CNN / Transformer / 遥感变化检测基础模板
    │
    └── mamba_base
        ├── Torch 2.10 + cu130
        ├── mamba-ssm 2.3.2.post1
        ├── causal-conv1d 1.7.0
        ├── RTX 5090 sm_120
        ├── Mamba patch
        └── slow/fast path 双卡验证通过
```

日常科研最重要的原则：

> **本地 Cursor 写代码，SFTP 同步；服务器只负责环境、数据、训练和推理。`cd_base` 与 `mamba_base` 作为稳定模板，具体论文从对应模板 clone 独立环境。**

这样可以最大限度减少“为了复现一篇论文，把已经能正常工作的基础环境改坏”的风险。

------------------------------------------------------------------------

## 第二部分：变化检测数据集与 Teacher Cache

### 遥感变化检测数据集与 Teacher Cache 统一说明

> **用途**：作为后续新项目编写 `Dataset / DataLoader`、配置数据路径、划分训练/验证/测试集、加载 Teacher Cache 的统一数据说明。\
> **服务器数据根目录**：`/home/yqwang/datasets/`\
> **本地审计日期**：2026-09-06\
> **整理依据**：`DATASET_INVENTORY.md` + `DATASET_DATALOADER_DEEP_AUDIT.md` + 公开数据集/Teacher 模型资料。\
> **适用项目**：CNN、Transformer、Mamba、普通监督变化检测、半监督变化检测、知识蒸馏/Teacher-guided 变化检测。
>
> **重要优先级**：写代码时，本文中的“服务器本地审计结果”优先级高于论文、官网或网络上对原始数据集的描述。公开资料描述的是**原始数据集**，服务器上使用的是已经裁剪、重命名并固定 split 的 **256×256 本地版本**。

------------------------------------------------------------------------

### 1. 一页速查

#### 1.1 当前数据根目录

``` text
/home/yqwang/datasets/
├── CD/
│   ├── CDD-CD-256/
│   ├── LEVIR-CD-256/
│   ├── SYSU-CD-256/
│   └── WHU-CD-256/
│
└── CD_teacher_cache/
    ├── OVCDistill/
    │   ├── dinov2_vitb14/
    │   │   ├── SYSU-CD-256/
    │   │   └── WHU-CD-256/
    │   └── preview/
    │
    └── SAMStruct/
        ├── sam2.1_hiera_large/
        │   ├── SYSU-CD-256/
        │   └── WHU-CD-256/
        └── preview/
```

#### 1.2 四个变化检测数据集

| 本地数据集 | 总样本 | Train | Val | Test | 图像 | Label | 主要特点 |
|----|---:|---:|---:|---:|----|----|----|
| `CDD-CD-256` | 15,998 | 10,000 | 2,998 | 3,000 | JPG, RGB, 256² | JPG, L | 季节/外观伪变化明显，多种变化目标 |
| `LEVIR-CD-256` | 10,192 | 7,120 | 1,024 | 2,048 | PNG, RGB, 256² | PNG, L | 高分辨率建筑变化，小目标、细边界 |
| `SYSU-CD-256` | 20,000 | 12,000 | 4,000 | 4,000 | PNG, RGB, 256² | PNG, L | 通用地表变化，变化类型丰富 |
| `WHU-CD-256` | 7,434 | 5,947 | 743 | 744 | PNG, RGB, 256² | PNG, L | 超高分辨率建筑变化，类别不平衡明显 |

四个数据集均已验证：

``` text
A 文件名集合 == B 文件名集合 == label 文件名集合
train ∩ val  = 0
train ∩ test = 0
val ∩ test   = 0
train ∪ val ∪ test = 全部有效样本
```

也就是说，**四个本地数据集的 canonical split 都是完整且互斥的**。

------------------------------------------------------------------------

### 2. 后续项目必须遵循的 DataLoader Contract

这是全文最重要的部分。

#### 2.1 标准路径约定

对于任意数据集：

``` text
/home/yqwang/datasets/CD/<dataset>/
├── A/
├── B/
├── label/
└── list/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

其中：

``` text
A/<filename>      = T1 / 时相 1
B/<filename>      = T2 / 时相 2
label/<filename>  = 二值变化标签
```

默认 split：

``` text
list/train.txt
list/val.txt
list/test.txt
```

**禁止把 `os.listdir(A)` 的自然顺序直接当 Dataset 顺序。**

Dataset 应当：

1.  先读取 split txt；
2.  以 txt 中的文件名作为 sample ID；
3.  用同一个文件名构造 A、B、label 路径；
4.  初始化时验证三个文件都存在。

推荐：

``` python
name = self.names[index]

path_a = root / "A" / name
path_b = root / "B" / name
path_y = root / "label" / name
```

------------------------------------------------------------------------

#### 2.2 推荐统一的 Label 读取规则

四个数据集都属于二值变化检测。

建议统一：

``` python
mask = Image.open(label_path).convert("L")
mask = np.asarray(mask)

mask = (mask >= 128).astype(np.int64)
```

原因：

- SYSU：审计值为标准 `0 / 255`；
- WHU：审计值为标准 `0 / 255`；
- LEVIR：绝大多数为 `0 / 255`，抽样还出现极少数 `156 / 254`；
- **CDD：Label 为 JPEG，存在 0–255 的 JPEG 压缩中间灰度值，必须阈值化。**

因此：

``` python
mask == 255
```

**不推荐**作为通用规则。

统一使用：

``` python
mask >= 128
```

最稳妥。

------------------------------------------------------------------------

#### 2.3 数据增强

A、B、label 必须共享同一个几何变换：

``` text
RandomCrop
HorizontalFlip
VerticalFlip
Rotate
Resize
...
```

即：

``` text
A ─┐
B ─┼─ 同步空间增强
Y ─┘
```

但：

``` text
Brightness
Contrast
ColorJitter
Normalize
```

这类图像颜色/归一化操作**不能直接作用于 label**。

------------------------------------------------------------------------

#### 2.4 归一化原则

本文后面记录了服务器本地抽样得到的 RGB mean/std。

但是：

> 如果 backbone 使用 ImageNet / DINOv2 / SAM 等预训练模型，优先使用该预训练模型官方要求的 normalization，而不是擅自换成本地统计值。

本地 mean/std 更适合：

- 从头训练；
- 做数据统计；
- 判断不同数据集域差异；
- 没有指定预训练 normalization 的项目。

------------------------------------------------------------------------

### 3. 服务器数据总体情况

#### 3.1 `/home/yqwang/datasets/CD`

``` text
总目录数：22
总文件数：160,927
总大小：10.82 GiB
```

扩展名：

``` text
.png  112,893
.jpg   47,994
.txt       37
.py         3
```

各数据集大小：

| 数据集       | 本地大小 |
|--------------|---------:|
| CDD-CD-256   | 1.31 GiB |
| LEVIR-CD-256 | 2.31 GiB |
| SYSU-CD-256  | 5.42 GiB |
| WHU-CD-256   | 1.78 GiB |

------------------------------------------------------------------------

#### 3.2 `/home/yqwang/datasets/CD_teacher_cache`

``` text
总目录数：19
总文件数：36,132
总大小：18.10 GiB
```

扩展名：

``` text
.pt    35,894
.jpg      232   # preview，可视化用，不参与训练读取
.json       6   # manifest/config 元数据
```

Teacher Cache 当前只覆盖：

``` text
SYSU-CD-256 train
WHU-CD-256  train
```

CDD 和 LEVIR 当前**没有 Teacher Cache**。

------------------------------------------------------------------------

### 4. CDD-CD-256

#### 4.1 本地路径

``` text
/home/yqwang/datasets/CD/CDD-CD-256/
├── A/      15,998 JPG
├── B/      15,998 JPG
├── label/  15,998 JPG
└── list/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

图像：

``` text
A: RGB, 256×256
B: RGB, 256×256
label: L/grayscale, 256×256
```

------------------------------------------------------------------------

#### 4.2 Split

| Split | 样本数 |
|-------|-------:|
| Train | 10,000 |
| Val   |  2,998 |
| Test  |  3,000 |
| Total | 15,998 |

完整性：

``` text
duplicates = 0
missing A = 0
missing B = 0
missing label = 0
train/val/test 无重叠
split union = 15,998
```

------------------------------------------------------------------------

#### 4.3 本地 RGB 抽样统计

> A+B 合并抽样统计；每个 split 抽样 256 对。

| Split | Mean RGB                         | Std RGB                          |
|-------|----------------------------------|----------------------------------|
| Train | `[0.417276, 0.448822, 0.410310]` | `[0.232498, 0.248929, 0.238022]` |
| Val   | `[0.412241, 0.442980, 0.405717]` | `[0.241070, 0.255401, 0.246631]` |
| Test  | `[0.428764, 0.461439, 0.421374]` | `[0.242136, 0.256226, 0.247326]` |

------------------------------------------------------------------------

#### 4.4 Label 特点

CDD 的 label 是 **JPEG**。

对每个 split 抽样 512 张 mask 后：

| Split | `mask >= 128` 变化像素比例 | 原始中间灰度 `(1..254)` |
|-------|---------------------------:|------------------------:|
| Train |                   11.9437% |                 1.0924% |
| Val   |                   11.2160% |                 1.0690% |
| Test  |                   13.1101% |                 1.1660% |

原始 mask 抽样出现：

``` text
0 ~ 255 共 256 个灰度值
```

这与 JPEG 压缩后的二值标签特征一致。

##### CDD 强制建议

``` python
mask = np.asarray(Image.open(path).convert("L"))
mask = (mask >= 128).astype(np.int64)
```

不要：

``` python
mask = (mask == 255)
```

否则 JPEG 边缘附近的变化像素可能被错误丢弃。

------------------------------------------------------------------------

#### 4.5 数据集公开特点

CDD 与 Lebedev 等人在 2018 年的 season-varying remote sensing change detection 工作相关。公开文献描述其包含来自 Google Earth 的跨季节双时相影像，重点挑战之一是：

``` text
真实变化
+
季节变化
+
光照/颜色变化
+
植被外观变化
=
大量 pseudo-change / 伪变化
```

常见变化对象并不只限于建筑，还包括：

- 建筑；
- 道路；
- 车辆；
- 植被等不同尺度目标。

因此 CDD 特别适合考察：

- 模型是否过度依赖 RGB 差异；
- 对季节/光照伪变化的鲁棒性；
- 多尺度变化目标检测能力；
- 跨外观变化的泛化能力。

##### 原始数据与本地版本区别

公开资料对 CDD 处理后样本数存在 16,000 与 15,998 两种常见描述。

**本服务器必须以本地审计为准：**

``` text
10,000 / 2,998 / 3,000
Total = 15,998
```

------------------------------------------------------------------------

### 5. LEVIR-CD-256

#### 5.1 本地路径

``` text
/home/yqwang/datasets/CD/LEVIR-CD-256/
├── A/      10,192 PNG
├── B/      10,192 PNG
├── label/  10,192 PNG
├── list/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
│
├── all_levir_list.txt
├── list_train.txt
├── list_val.txt
└── list_test.txt
```

A/B：

``` text
RGB PNG
256×256
```

Label：

``` text
grayscale PNG
256×256
```

------------------------------------------------------------------------

#### 5.2 Canonical Split

**默认只使用 `list/` 下这一套：**

| Split | 样本数 |
|-------|-------:|
| Train |  7,120 |
| Val   |  1,024 |
| Test  |  2,048 |
| Total | 10,192 |

完整性：

``` text
duplicates = 0
missing A/B/label = 0
train/val/test 无重叠
全部 10,192 样本均被 canonical split 覆盖
```

------------------------------------------------------------------------

#### 5.3 特别重要：LEVIR 有两套 split

根目录还存在：

``` text
list_train.txt
list_val.txt
list_test.txt
```

它们**不是** `list/train.txt`、`list/val.txt`、`list/test.txt` 的重复副本。

审计结果：

``` text
root list_test.txt vs canonical test:
overlap = 0 / 2048

root list_val.txt vs canonical val:
overlap = 0 / 1024

root list_train.txt vs canonical train:
overlap = 5072 / 7120
```

也就是说，这是一套**替代划分方案**。

##### 默认项目规则

除非论文代码明确要求根目录：

``` text
list_train.txt
list_val.txt
list_test.txt
```

否则一律使用：

``` text
list/train.txt
list/val.txt
list/test.txt
```

`all_levir_list.txt` 包含全部 10,192 个有效 sample。

------------------------------------------------------------------------

#### 5.4 RGB 抽样统计

| Split | Mean RGB                         | Std RGB                          |
|-------|----------------------------------|----------------------------------|
| Train | `[0.396484, 0.391759, 0.334584]` | `[0.198256, 0.190001, 0.175035]` |
| Val   | `[0.380532, 0.374350, 0.319038]` | `[0.190607, 0.178245, 0.161980]` |
| Test  | `[0.385375, 0.381398, 0.324319]` | `[0.198337, 0.187098, 0.170328]` |

------------------------------------------------------------------------

#### 5.5 Label 特点

抽样 mask 出现：

``` text
0, 156, 254, 255
```

其中中间值占比极低：

``` text
约 0.0000x
```

变化像素比例：

| Split | 变化像素比例 |
|-------|-------------:|
| Train |      4.1275% |
| Val   |      4.2118% |
| Test  |      4.8211% |

因此 LEVIR 具有明显前景/背景不平衡。

推荐统一：

``` python
mask = (mask >= 128)
```

------------------------------------------------------------------------

#### 5.6 数据集公开特点

LEVIR-CD 是经典的**建筑变化检测** benchmark。

公开项目说明：

``` text
637 对 VHR 双时相影像
原始单图：1024×1024
空间分辨率：0.5 m/pixel
时间跨度：5–14 年
区域：美国 Texas 多个区域
```

主要关注建筑：

- 新建建筑；
- 建筑消失/拆除；
- 不同类型建筑，包括住宅、公寓、车库、仓库等。

典型难点：

- 变化建筑面积占图像比例低；
- 小目标、密集目标；
- 建筑边界精细；
- 长时间跨度带来的外观变化；
- 光照差异；
- 轻微配准误差。

因此 LEVIR 很适合评价：

``` text
小目标变化
建筑边界质量
Attention / Transformer
长距离关系建模
Mamba 空间建模
```

------------------------------------------------------------------------

### 6. SYSU-CD-256

#### 6.1 本地路径

``` text
/home/yqwang/datasets/CD/SYSU-CD-256/
├── A/      20,000 PNG
├── B/      20,000 PNG
├── label/  20,000 PNG
└── list/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

全部：

``` text
A/B: RGB PNG 256×256
label: grayscale PNG 256×256
```

------------------------------------------------------------------------

#### 6.2 Split

| Split | 样本数 |
|-------|-------:|
| Train | 12,000 |
| Val   |  4,000 |
| Test  |  4,000 |
| Total | 20,000 |

完整性：

``` text
duplicates = 0
missing A/B/label = 0
split 无重叠
20,000 样本全部覆盖
```

------------------------------------------------------------------------

#### 6.3 RGB 抽样统计

| Split | Mean RGB                         | Std RGB                          |
|-------|----------------------------------|----------------------------------|
| Train | `[0.400212, 0.504991, 0.428774]` | `[0.230152, 0.183968, 0.182145]` |
| Val   | `[0.399955, 0.502252, 0.423679]` | `[0.229198, 0.184835, 0.184254]` |
| Test  | `[0.394009, 0.501927, 0.424776]` | `[0.228178, 0.181773, 0.180695]` |

------------------------------------------------------------------------

#### 6.4 Label

标准：

``` text
0
255
```

变化像素比例：

| Split | 变化像素比例 |
|-------|-------------:|
| Train |     21.1368% |
| Val   |     21.5150% |
| Test  |     22.7476% |

四个本地数据集中，SYSU 的变化像素比例明显更高。

------------------------------------------------------------------------

#### 6.5 数据集公开特点

SYSU-CD 官方仓库描述：

``` text
20,000 对
256×256
0.5 m aerial images
Hong Kong
2007–2014
```

主要变化类型包括：

1.  新建城市建筑；
2.  郊区扩张；
3.  建设施工前土地平整；
4.  植被变化；
5.  道路扩建；
6.  海岸/填海建设。

因此 SYSU 与 LEVIR/WHU 最大的实验意义区别是：

``` text
LEVIR / WHU → building-focused
SYSU        → general land-cover change
```

它特别适合验证模型是否：

- 只会识别建筑变化；
- 能处理更丰富的变化语义；
- 能处理道路、植被、施工、海岸等不同目标形态。

------------------------------------------------------------------------

#### 6.6 Teacher Cache

SYSU 的整个 canonical train：

``` text
12,000 samples
```

已经同时拥有：

``` text
12,000 OVCDistill caches
12,000 SAMStruct caches
```

覆盖率：

``` text
100%
```

因此 SYSU 是当前最适合直接进行双 Teacher / 知识蒸馏实验的数据集之一。

------------------------------------------------------------------------

### 7. WHU-CD-256

#### 7.1 本地路径

``` text
/home/yqwang/datasets/CD/WHU-CD-256/
├── A/      7,434 PNG
├── B/      7,434 PNG
├── label/  7,434 PNG
├── list/
│   ├── train.txt
│   ├── val.txt
│   ├── test.txt
│   ├── 5_train_supervised.txt
│   ├── 5_train_unsupervised.txt
│   ├── ...
│   ├── 100_train_supervised.txt
│   └── 100_train_unsupervised.txt
│
├── sample_*.png
└── WHU_CD/
    ├── sample_*.png
    └── metrics.txt
```

训练数据仍然只读取：

``` text
A/
B/
label/
list/
```

以下内容**不是训练输入目录**：

``` text
WHU_CD/
sample_*.png
metrics.txt
preview 类图片
```

------------------------------------------------------------------------

#### 7.2 Canonical Split

| Split | 样本数 |
|-------|-------:|
| Train |  5,947 |
| Val   |    743 |
| Test  |    744 |
| Total |  7,434 |

完整性：

``` text
duplicates = 0
missing A/B/label = 0
split 无重叠
全部 7,434 样本覆盖
```

------------------------------------------------------------------------

#### 7.3 半监督列表

WHU 已附带不同标注比例的 supervised / unsupervised 划分：

| 标注比例 | Supervised | Unsupervised |
|---------:|-----------:|-------------:|
|       5% |        297 |        5,650 |
|      10% |        594 |        5,353 |
|      20% |      1,189 |        4,758 |
|      30% |      1,784 |        4,163 |
|      40% |      2,378 |        3,569 |
|      50% |      2,973 |        2,974 |
|      60% |      3,568 |        2,379 |
|      70% |      4,162 |        1,785 |
|      80% |      4,757 |        1,190 |
|     100% |      5,947 |            0 |

审计已确认每个列表中的样本都属于 canonical train。

从数量上每组：

``` text
supervised + unsupervised = 5,947
```

因此它们显然是为半监督实验准备的配套 split。

> 当前审计报告没有单独输出每一组 supervised 与 unsupervised 的集合交集，因此若某个新项目要求严格证明“互斥 + 并集=train”，可以在项目开始前再做一次集合级 assert。普通使用时可按其设计意图使用。

------------------------------------------------------------------------

#### 7.4 RGB 抽样统计

| Split | Mean RGB                         | Std RGB                          |
|-------|----------------------------------|----------------------------------|
| Train | `[0.490911, 0.470216, 0.431042]` | `[0.198939, 0.193220, 0.205562]` |
| Val   | `[0.489612, 0.469175, 0.430599]` | `[0.196319, 0.190595, 0.203615]` |
| Test  | `[0.473949, 0.453885, 0.412760]` | `[0.190160, 0.182829, 0.195018]` |

------------------------------------------------------------------------

#### 7.5 Label

标准：

``` text
0
255
```

变化像素比例：

| Split | 变化像素比例 |
|-------|-------------:|
| Train |      3.4209% |
| Val   |      4.5804% |
| Test  |      4.1644% |

WHU 是四个本地数据集中**变化像素最稀疏**的之一。

因此训练时常需要注意：

``` text
background >> change
```

模型和 loss 很容易偏向“全部预测不变化”。

------------------------------------------------------------------------

#### 7.6 数据集公开特点

WHU Building Change Detection 来源于 Christchurch, New Zealand。

武汉大学官方资料说明：

- 原始航拍数据地面分辨率约 0.075 m；
- 区域经历 2011 年 Christchurch 地震及后续重建；
- change detection 数据对应 2012 与 2016 年影像；
- 2012 区域约 12,796 个建筑；
- 同区域 2016 年约 16,077 个建筑；
- 使用 GCP 做几何配准，官方报告约 1.6 pixel 精度。

##### 注意空间分辨率描述

不同论文中会看到 WHU：

``` text
0.075 m
0.2 m
0.3 m
```

等不同数字，因为有人使用原始航拍影像，有人使用官方/第三方重采样版本。

因此对当前服务器的 `WHU-CD-256`：

> **只确定它是 256×256 处理版本，不应仅根据名称武断声称本地 patch 的实际 GSD。**

如论文必须报告 GSD，应依据该项目采用的数据来源/预处理脚本进一步确认。

##### 典型实验难点

- 超高分辨率建筑边缘；
- 建筑小目标；
- 配准误差对边界影响大；
- 变化像素比例低；
- 明显类别不平衡。

------------------------------------------------------------------------

#### 7.7 Teacher Cache

WHU canonical train：

``` text
5,947
```

Teacher：

``` text
5,947 OVCDistill
5,947 SAMStruct
```

覆盖率：

``` text
100%
```

并且已经有半监督 split，因此 WHU 很适合：

``` text
Teacher-guided semi-supervised change detection
knowledge distillation
structure-aware change detection
```

------------------------------------------------------------------------

### 8. 四个数据集的实验角色对比

可以把四个 benchmark 理解为四种不同“考题”。

| 数据集 | 最主要考察点 | 变化类型 | 本地 Train 变化像素比例 |
|----|----|----|---:|
| CDD | 抗季节/外观伪变化、多尺度鲁棒性 | 通用、多种目标 | 11.94% |
| LEVIR-CD | 建筑小目标、细边界 | 建筑 | 4.13% |
| SYSU-CD | 通用变化语义、多目标类别 | 建筑/道路/植被/施工/海岸等 | 21.14% |
| WHU-CD | VHR 建筑边界、严重类别不平衡 | 建筑 | 3.42% |

对模型结果的一个常用解释框架：

``` text
CDD 高
→ 对季节/外观伪变化更鲁棒

LEVIR 高
→ 建筑细粒度和小目标变化能力强

SYSU 高
→ 通用变化类型泛化能力好

WHU 高
→ 高分辨率建筑边界与极不平衡场景处理较好
```

------------------------------------------------------------------------

### 9. Teacher Cache 总体结构

当前 Teacher Cache：

``` text
/home/yqwang/datasets/CD_teacher_cache/
```

存在两套 Teacher 信息：

``` text
OVCDistill
└── DINOv2 ViT-B/14

SAMStruct
└── SAM 2.1 Hiera Large
```

覆盖：

| 数据集       |  Train | OVCDistill | SAMStruct |
|--------------|-------:|-----------:|----------:|
| CDD-CD-256   | 10,000 |         ❌ |        ❌ |
| LEVIR-CD-256 |  7,120 |         ❌ |        ❌ |
| SYSU-CD-256  | 12,000 |     12,000 |    12,000 |
| WHU-CD-256   |  5,947 |      5,947 |     5,947 |

只缓存 **train**：

``` text
没有 val cache
没有 test cache
```

------------------------------------------------------------------------

### 10. Cache Sample ID 与文件名映射

这是实现 cache-aware Dataset 时最重要的规则。

每个 teacher dataset 目录都带：

``` text
manifest.json
```

Manifest 中有：

``` json
{
  "entries": {
    "00000.png": "a73b9e5aff2420064b797f69f7d3449623af8d0a.pt",
    "...": "..."
  }
}
```

##### 正确做法

``` python
with open(manifest_path, "r") as f:
    manifest = json.load(f)

cache_filename = manifest["entries"][sample_name]
cache_path = cache_root / "train" / cache_filename
```

**推荐始终使用 manifest lookup。**

------------------------------------------------------------------------

#### 10.1 SHA1 规律

审计对多个样本验证发现：

``` text
cache file stem = SHA1(sample_id)
```

例如：

``` text
sample_id = whucd_02438.png

SHA1(sample_id)
→ 00057b1c339b6364dc4f4f52f5dab347bdef954a

cache:
00057b1c339b6364dc4f4f52f5dab347bdef954a.pt
```

但项目代码中依然推荐：

``` text
manifest["entries"][sample_id]
```

而不是重新实现 hash。

原因：

- manifest 是生成 cache 时的权威索引；
- 将来 cache version 变化时映射规则可能改变；
- manifest 还能验证数据版本和 train-list hash。

------------------------------------------------------------------------

### 11. 非常重要：Cache 中保存的是旧绝对路径

Cache meta 中能看到类似：

``` text
/data/CD/SYSU-CD-256/A/07979.png
/data/CD/SYSU-CD-256/B/07979.png
```

或：

``` text
/data/CD/WHU-CD-256/A/whucd_02438.png
```

这些是 **cache 生成时服务器/容器中的历史路径**。

当前服务器真实路径是：

``` text
/home/yqwang/datasets/CD/...
```

因此：

> **不要直接用 cache 的 `meta.source_t1` / `meta.source_t2` 去打开文件。**

正确做法：

``` python
sample_name = cache["sample_id"]

A = current_root / "A" / sample_name
B = current_root / "B" / sample_name
```

或直接以 Dataset 当前 split 的 sample name 为主。

审计已经确认 sample cache 中 `source_t1/source_t2` 的 basename 与当前 A/B 文件匹配。

------------------------------------------------------------------------

### 12. OVCDistill / DINOv2 Teacher Cache

#### 12.1 路径

``` text
/home/yqwang/datasets/CD_teacher_cache/
└── OVCDistill/
    └── dinov2_vitb14/
        ├── SYSU-CD-256/
        │   ├── manifest.json
        │   └── train/*.pt
        └── WHU-CD-256/
            ├── manifest.json
            └── train/*.pt
```

------------------------------------------------------------------------

#### 12.2 数量与大小

| 数据集     | Cache 数 |    单文件 |     总大小 |
|------------|---------:|----------:|-----------:|
| SYSU train |   12,000 | 28.17 KiB | 330.14 MiB |
| WHU train  |    5,947 | 28.17 KiB | 163.61 MiB |

------------------------------------------------------------------------

#### 12.3 Teacher 配置

Teacher：

``` text
DINOv2 ViT-B/14
```

Manifest：

``` text
teacher layers: [8, 11]
teacher input size: 518
cache sizes: [32, 16]

mean:
[0.485, 0.456, 0.406]

std:
[0.229, 0.224, 0.225]

input interpolation:
bicubic

feature interpolation:
bilinear

photometric_confidence:
true
```

Checkpoint：

``` text
dinov2_vitb14_pretrain.pth
SHA256:
0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73
```

本地 manifest 显示 Teacher 原 repo 路径：

``` text
/home/yqwang/project/LS-Rep_BCD/models/third_party/dinov2
```

这是 cache 的来源记录，并不是当前新项目必须依赖的运行路径；只读取 `.pt` cache 时不需要重新运行 Teacher。

------------------------------------------------------------------------

#### 12.4 单个 OVCDistill cache 结构

典型：

``` python
cache = {
    "sample_id": str,

    "soft_change": {
        "l1": Tensor[1, 32, 32],  # float16
        "l2": Tensor[1, 16, 16],  # float16
    },

    "confidence": {
        "l1": Tensor[1, 32, 32],  # float16
        "l2": Tensor[1, 16, 16],  # float16
    },

    "relation": {
        "l1": Tensor[8, 32, 32],  # float16
        "l2": Tensor[8, 16, 16],  # float16
    },

    "meta": {
        "config_hash": str,
        "source_t1": str,
        "source_t2": str,
    }
}
```

##### 各字段可理解为

`soft_change`

``` text
Teacher 提供的软变化响应
```

适合：

- soft target；
- consistency；
- multi-scale distillation。

`confidence`

``` text
Teacher 对 soft change 的置信信息
```

适合：

- confidence weighting；
- 低置信区域降权；
- pseudo-label filtering。

`relation`

``` text
8-channel multi-scale relation feature
```

适合：

- relation distillation；
- feature-structure consistency；
- 多尺度关系约束。

> 上述“用途”是根据本地 cache 字段语义做的工程解释；具体 loss 定义必须以新项目代码/论文为准。

------------------------------------------------------------------------

#### 12.5 DINOv2 本身的特点

DINOv2 是 Meta/FAIR 的自监督视觉特征模型。

官方资料说明其模型学习：

``` text
robust visual features without labels
```

ViT-B/14：

``` text
Patch size: 14
Embedding dimension: 768
12 attention heads
约 86M 参数
```

因此当前 OVCDistill cache 更偏：

``` text
语义/表征层 Teacher
```

而不是直接保存实例分割 mask。

------------------------------------------------------------------------

#### 12.6 Manifest 可复现信息

SYSU：

``` text
source_train_list_sha256:
3fd039fa22f1d0022551176f3da577521813b8c3b5dfaa1034de2d145ee63c7a

config_hash:
f9c2cfa9b0bbeab56e6cd8d11c908ef9da64510ee6faf62807bc8fe822a1a09c
```

WHU：

``` text
source_train_list_sha256:
3c79956757c6b37b4cfa7af89b025a827f89ac102779ad6e35a99f15e8217c27

config_hash:
b556412f6dd59d6f53983a61e3ce5b659b0fd2b0e890535981bbd5a07756ad11
```

------------------------------------------------------------------------

### 13. SAMStruct / SAM 2.1 Teacher Cache

#### 13.1 路径

``` text
/home/yqwang/datasets/CD_teacher_cache/
└── SAMStruct/
    └── sam2.1_hiera_large/
        ├── SYSU-CD-256/
        │   ├── manifest.json
        │   └── train/*.pt
        └── WHU-CD-256/
            ├── manifest.json
            └── train/*.pt
```

------------------------------------------------------------------------

#### 13.2 数量与大小

| 数据集     | Cache 数 |   单文件 |    总大小 |
|------------|---------:|---------:|----------:|
| SYSU train |   12,000 | 1.00 MiB | 11.76 GiB |
| WHU train  |    5,947 | 1.00 MiB |  5.83 GiB |

这也是为什么整个 `CD_teacher_cache` 的主要空间占用来自 SAMStruct。

------------------------------------------------------------------------

#### 13.3 Teacher 配置

Teacher：

``` text
SAM 2.1 Hiera Large
```

本地 manifest：

``` text
teacher_type: sam2_struct_v2

points_per_side: 32
points_per_batch: 64
pred_iou_thresh: 0.8
stability_score_thresh: 0.9

crop_n_layers: 1
crop_n_points_downscale_factor: 2
min_mask_region_area: 9
max_area_ratio: 0.85

overlap_order:
large-first/small-last; low-quality-first-on-ties

boundary:
quality-independent-4-neighbor-gaussian-0.8
```

Checkpoint：

``` text
sam2.1_hiera_large.pt
SHA256:
2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318
```

Config：

``` text
configs/sam2.1/sam2.1_hiera_l.yaml
```

------------------------------------------------------------------------

#### 13.4 单个 SAMStruct cache 结构

典型：

``` python
cache = {
    "sample_id": str,

    "t1": {
        "instance_id": Tensor[1, 256, 256],  # int32
        "boundary":    Tensor[1, 256, 256],  # float16
        "quality":     Tensor[1, 256, 256],  # float16
    },

    "t2": {
        "instance_id": Tensor[1, 256, 256],  # int32
        "boundary":    Tensor[1, 256, 256],  # float16
        "quality":     Tensor[1, 256, 256],  # float16
    },

    "meta": {
        "teacher_type": "sam2_struct_v2",
        "config_hash": str,
        "source_t1": str,
        "source_t2": str,
    }
}
```

##### 字段特点

`instance_id`

``` text
SAM 对图像内部实例/区域的结构化编号结果
```

注意它不是 ground-truth change mask。

`boundary`

``` text
结构边界响应
```

适合：

- boundary-aware loss；
- structure consistency；
- 变化边缘增强。

`quality`

``` text
对应结构结果的质量/可靠性信息
```

可用于：

- confidence weighting；
- unreliable region suppression。

SAMStruct 同时保存 T1 和 T2，因此它更偏：

``` text
单时相结构 Teacher
+
双时相结构对比
```

------------------------------------------------------------------------

#### 13.5 SAM 2.1 Hiera Large 本身特点

Meta 官方 SAM 2.1 checkpoint 表中：

``` text
sam2.1_hiera_large
约 224.4M parameters
```

SAM 2 系列面向图像/视频可提示分割，擅长生成结构化 object masks。

因此与 DINOv2 cache 相比：

``` text
DINOv2 / OVCDistill
→ 更偏语义特征、软变化、关系蒸馏

SAM2.1 / SAMStruct
→ 更偏实例结构、边界、结构质量
```

两者提供的是**互补 Teacher 信息**。

------------------------------------------------------------------------

#### 13.6 Manifest 可复现信息

SYSU：

``` text
source_train_list_sha256:
3fd039fa22f1d0022551176f3da577521813b8c3b5dfaa1034de2d145ee63c7a

config_hash:
8da34cea04ddc6800c6fc8ff8a7d606580d53836b95a843b5c7441151e0c0ac1
```

WHU：

``` text
source_train_list_sha256:
3c79956757c6b37b4cfa7af89b025a827f89ac102779ad6e35a99f15e8217c27

config_hash:
6bc52561e569dbc0c2ec55470262acc25582487febcde75bf86323a0ec26004a
```

------------------------------------------------------------------------

### 14. 两套 Teacher Cache 的对齐关系

深度审计已经确认：

#### SYSU

``` text
OVCDistill cache stem set
==
SAMStruct cache stem set

intersection = 12,000
only OVCDistill = 0
only SAMStruct  = 0
```

#### WHU

``` text
OVCDistill cache stem set
==
SAMStruct cache stem set

intersection = 5,947
only OVCDistill = 0
only SAMStruct  = 0
```

因此同一个训练样本可以安全地同时索引：

``` python
sample
├── image A
├── image B
├── label
├── DINOv2 / OVCDistill cache
└── SAM2.1 / SAMStruct cache
```

这对双 Teacher 训练非常方便。

------------------------------------------------------------------------

### 15. 推荐的 Cache-aware Dataset 读取方式

``` python
import json
from pathlib import Path

class CacheIndex:
    def __init__(self, dataset_name):
        cache_root = Path(
            "/home/yqwang/datasets/CD_teacher_cache"
        )

        ov_root = (
            cache_root
            / "OVCDistill"
            / "dinov2_vitb14"
            / dataset_name
        )

        sam_root = (
            cache_root
            / "SAMStruct"
            / "sam2.1_hiera_large"
            / dataset_name
        )

        with open(ov_root / "manifest.json", "r") as f:
            self.ov_manifest = json.load(f)

        with open(sam_root / "manifest.json", "r") as f:
            self.sam_manifest = json.load(f)

        self.ov_train = ov_root / "train"
        self.sam_train = sam_root / "train"

    def paths(self, sample_name):
        ov_name = self.ov_manifest["entries"][sample_name]
        sam_name = self.sam_manifest["entries"][sample_name]

        return (
            self.ov_train / ov_name,
            self.sam_train / sam_name,
        )
```

读取：

``` python
ov = torch.load(ov_path, map_location="cpu", weights_only=True)
sam = torch.load(sam_path, map_location="cpu", weights_only=True)
```

建议：

- cache `.pt` 默认先在 CPU 读取；
- 需要的 tensor 再放 GPU；
- 不要在 Dataset 初始化时一次性把全部 cache load 到内存；
- SAMStruct 全量约 17.6 GiB，不能简单常驻 RAM；
- 可由 DataLoader 按 sample 逐个读取。

------------------------------------------------------------------------

### 16. Dataset 最小实现模板

``` python
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class BitemporalCDDataset(Dataset):
    def __init__(
        self,
        root,
        split="train",
        transform=None,
    ):
        self.root = Path(root)
        self.transform = transform

        list_path = self.root / "list" / f"{split}.txt"

        self.names = [
            x.strip()
            for x in list_path.read_text().splitlines()
            if x.strip()
        ]

        # 可在开发阶段开启完整 assert
        for name in self.names:
            assert (self.root / "A" / name).exists()
            assert (self.root / "B" / name).exists()
            assert (self.root / "label" / name).exists()

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]

        img_a = Image.open(
            self.root / "A" / name
        ).convert("RGB")

        img_b = Image.open(
            self.root / "B" / name
        ).convert("RGB")

        mask = Image.open(
            self.root / "label" / name
        ).convert("L")

        mask = np.asarray(mask)

        # 对四个本地数据集均安全
        mask = (mask >= 128).astype(np.int64)

        # A/B/mask 的几何增强必须同步
        # transform implementation 由具体项目决定

        mask = torch.from_numpy(mask)

        return {
            "name": name,
            "image_a": img_a,
            "image_b": img_b,
            "mask": mask,
        }
```

这个模板只定义**本地数据协议**。

具体论文项目还需要根据模型添加：

- ToTensor；
- normalization；
- crop；
- augmentation；
- cache；
- multi-scale target；
- patch embedding；
- batch collate 等。

------------------------------------------------------------------------

### 17. 推荐的数据集配置字典

以后新项目可以先按下面理解：

``` python
DATASET_SPECS = {
    "CDD": {
        "root": "/home/yqwang/datasets/CD/CDD-CD-256",
        "train": 10000,
        "val": 2998,
        "test": 3000,
        "ext": ".jpg",
        "mask_threshold": 128,
        "teacher_cache": False,
    },

    "LEVIR": {
        "root": "/home/yqwang/datasets/CD/LEVIR-CD-256",
        "train": 7120,
        "val": 1024,
        "test": 2048,
        "ext": ".png",
        "mask_threshold": 128,
        "teacher_cache": False,
        "warning": "Use list/*.txt by default; root list_*.txt is an alternative split.",
    },

    "SYSU": {
        "root": "/home/yqwang/datasets/CD/SYSU-CD-256",
        "train": 12000,
        "val": 4000,
        "test": 4000,
        "ext": ".png",
        "mask_threshold": 128,
        "teacher_cache": True,
    },

    "WHU": {
        "root": "/home/yqwang/datasets/CD/WHU-CD-256",
        "train": 5947,
        "val": 743,
        "test": 744,
        "ext": ".png",
        "mask_threshold": 128,
        "teacher_cache": True,
        "semi_supervised_lists": True,
    },
}
```

------------------------------------------------------------------------

### 18. 类别不平衡

本地抽样得到的 canonical train 变化像素比例：

``` text
CDD    11.94%
LEVIR   4.13%
SYSU   21.14%
WHU     3.42%
```

大致背景/变化比例：

``` text
CDD    ≈ 7.4 : 1
LEVIR  ≈ 23  : 1
SYSU   ≈ 3.7 : 1
WHU    ≈ 28  : 1
```

因此：

- LEVIR 和 WHU 的类别不平衡尤其明显；
- 仅看 OA/Accuracy 容易被大量 unchanged pixel 误导；
- F1、IoU、Precision、Recall 通常比单纯 OA 更重要。

Loss 是否采用：

``` text
BCE
Dice
Focal
BCE + Dice
class weighting
```

应由具体论文/项目决定，不能只根据本文档强制更改。

------------------------------------------------------------------------

### 19. 新项目接入 Checklist

每次把一个新变化检测项目接入服务器时：

#### Step 1：确定数据集

``` text
CDD
LEVIR
SYSU
WHU
```

#### Step 2：找到项目原 DataLoader

确认它原来期望：

``` text
A/B/label
还是
A/B/OUT
还是
T1/T2/label
```

然后适配到本文的实际路径。

#### Step 3：确定 split

默认：

``` text
list/train.txt
list/val.txt
list/test.txt
```

特别注意：

``` text
LEVIR 根目录还有另一套 list_*.txt
```

必须确认论文到底用哪套。

#### Step 4：Mask

统一：

``` python
mask >= 128
```

特别是 CDD。

#### Step 5：Normalization

如果有 pretrained backbone：

``` text
用 backbone 官方 normalization
```

否则才考虑本文的 train RGB stats。

#### Step 6：Teacher Cache

只有：

``` text
SYSU
WHU
```

有 cache。

普通监督项目：

``` text
完全不需要读取 CD_teacher_cache
```

Teacher/distillation 项目：

``` text
使用 manifest["entries"][sample_name]
```

#### Step 7：验证一个 batch

正式训练前：

``` text
打印 A shape
打印 B shape
打印 mask shape / dtype
打印 mask unique
打印 sample name
如果有 cache，再打印 cache keys / tensor shape
```

推荐期待：

``` text
A: [B, 3, 256, 256]
B: [B, 3, 256, 256]
Y: [B, 256, 256] 或 [B,1,256,256]
Y unique: {0,1}
```

------------------------------------------------------------------------

### 20. 哪些文件不应该误当作训练数据

#### LEVIR

默认不要误用：

``` text
root/list_train.txt
root/list_val.txt
root/list_test.txt
```

除非项目明确采用替代 split。

------------------------------------------------------------------------

#### WHU

不要把下面这些加入 Dataset：

``` text
WHU-CD-256/sample_*.png
WHU-CD-256/WHU_CD/sample_*.png
WHU-CD-256/WHU_CD/metrics.txt
WHU-CD-256/list/*.py
```

它们属于预览、结果或辅助脚本。

------------------------------------------------------------------------

#### Teacher Cache

不要把：

``` text
OVCDistill/preview/
SAMStruct/preview/
```

的 `.jpg` 当 cache。

真正训练 cache 是：

``` text
.../<teacher>/<dataset>/train/*.pt
```

------------------------------------------------------------------------

### 21. 数据与 Cache 的“权威来源”原则

以后项目遇到信息冲突时，按这个优先级：

``` text
1. 当前服务器实际文件
2. manifest.json
3. list/train.txt / val.txt / test.txt
4. 本统一文档
5. 项目代码/README
6. 原数据集官网或论文
7. 二手论文/博客
```

原因：

公开论文可能描述：

``` text
原图
不同裁剪方式
不同 split
不同重采样 GSD
```

而代码最终必须服从当前服务器真正存在的数据。

------------------------------------------------------------------------

### 22. Teacher Cache 的工程定位

当前两种 Teacher Cache 并不是 ground truth 的替代品。

可以理解为：

``` text
Ground Truth
└── label/
    └── 真正监督目标

OVCDistill
└── DINOv2 derived soft semantic/change information
    ├── soft_change
    ├── confidence
    └── relation

SAMStruct
└── SAM2 derived structural information
    ├── instance_id
    ├── boundary
    └── quality
```

它们可以作为：

- auxiliary supervision；
- distillation target；
- consistency constraint；
- structure prior；
- pseudo-supervision；

但不能默认等同于人工变化标签。

------------------------------------------------------------------------

### 23. Teacher Cache 模型公开背景

#### DINOv2

Meta/FAIR 的 DINOv2 是自监督视觉表示模型。

官方项目：

<https://github.com/facebookresearch/dinov2>

模型卡：

<https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md>

当前 cache 使用：

``` text
DINOv2 ViT-B/14
```

官方模型卡给出的 ViT-B/14 架构信息包括：

``` text
patch size 14
embedding dimension 768
12 heads
约 86M parameters
```

------------------------------------------------------------------------

#### SAM 2.1

Meta Segment Anything Model 2：

<https://github.com/facebookresearch/sam2>

当前 cache 使用：

``` text
SAM 2.1 Hiera Large
```

Meta 官方 checkpoint 表：

``` text
sam2.1_hiera_large
约 224.4M parameters
```

SAM 2 面向图像/视频分割，因此本地 SAMStruct 用它提取：

``` text
instance
boundary
quality
```

与 DINOv2 的语义/关系特征形成互补。

------------------------------------------------------------------------

### 24. 四个公开数据集资料来源

#### CDD / Season-Varying CD

Lebedev et al., 2018:

**Change Detection in Remote Sensing Images Using Conditional Adversarial Networks**

<https://doi.org/10.5194/isprs-archives-XLII-2-565-2018>

原始论文明确包含：

``` text
real season-varying remote sensing images
```

关于常用 15,998 patch 版本及 `10000/2998/3000` 的描述，可参见后续变化检测文献；本地 split 已由服务器文件直接审计确认，因此代码应以本地数字为准。

------------------------------------------------------------------------

#### LEVIR-CD

官方项目：

<https://github.com/justchenhao/LEVIR>

公开描述：

``` text
637 pairs
1024×1024
0.5m/pixel
time span 5–14 years
building-related change
```

------------------------------------------------------------------------

#### SYSU-CD

官方项目：

<https://github.com/liumency/SYSU-CD>

官方描述：

``` text
20,000 pairs
256×256
0.5m
Hong Kong
2007–2014
```

变化类型包括：

``` text
urban buildings
suburban dilation
groundwork
vegetation
road expansion
sea construction
```

------------------------------------------------------------------------

#### WHU Building Change Detection

武汉大学官方：

<https://gpcv.whu.edu.cn/data/building_dataset.html>

公开描述包括：

``` text
Christchurch, New Zealand
2012 / 2016
原始 aerial imagery 约 0.075m
2011 earthquake / reconstruction context
GCP registration accuracy 约 1.6 pixels
```

------------------------------------------------------------------------

### 25. 本地审计来源

本文合并整理了：

``` text
DATASET_INVENTORY.md
DATASET_DATALOADER_DEEP_AUDIT.md
```

原始探查目标：

``` text
/home/yqwang/datasets/CD/
/home/yqwang/datasets/CD_teacher_cache/
```

`DATASET_INVENTORY.md` 负责：

- 完整目录扫描；
- 文件数量；
- 扩展名；
- 图片尺寸；
- 通道；
- split list；
- cache 文件基本结构。

`DATASET_DATALOADER_DEEP_AUDIT.md` 负责：

- A/B/label 全量文件名一致性；
- split 完整性；
- split 重叠；
- RGB mean/std 抽样；
- label 像素统计；
- CDD JPEG mask 问题；
- LEVIR 两套 split 差异；
- WHU 半监督列表；
- Teacher manifest；
- cache tensor shape；
- cache sample ID；
- SHA1 映射规律；
- OVCDistill/SAMStruct 对齐；
- Teacher Cache 覆盖率。

------------------------------------------------------------------------

### 26. 最终结论

当前服务器的变化检测数据可以统一理解为：

``` text
/home/yqwang/datasets/
│
├── CD/
│   │
│   ├── CDD-CD-256
│   │   ├── 15998 pairs
│   │   ├── 10000 / 2998 / 3000
│   │   ├── JPG
│   │   ├── seasonal/general change
│   │   └── mask >= 128 必须注意
│   │
│   ├── LEVIR-CD-256
│   │   ├── 10192 pairs
│   │   ├── 7120 / 1024 / 2048
│   │   ├── building change
│   │   └── canonical list/ 与 root list_* 不同
│   │
│   ├── SYSU-CD-256
│   │   ├── 20000 pairs
│   │   ├── 12000 / 4000 / 4000
│   │   ├── general land-cover change
│   │   └── 100% train teacher-cache coverage
│   │
│   └── WHU-CD-256
│       ├── 7434 pairs
│       ├── 5947 / 743 / 744
│       ├── building change
│       ├── low change-pixel ratio
│       ├── semi-supervised split lists
│       └── 100% train teacher-cache coverage
│
└── CD_teacher_cache/
    │
    ├── OVCDistill / DINOv2 ViT-B14
    │   ├── soft_change
    │   ├── confidence
    │   └── relation
    │
    └── SAMStruct / SAM2.1 Hiera Large
        ├── instance_id
        ├── boundary
        └── quality
```

以后给新项目适配 DataLoader 时，最核心的默认约定只有：

``` text
root = /home/yqwang/datasets/CD/<dataset>

sample list:
root/list/{train,val,test}.txt

inputs:
root/A/<name>
root/B/<name>

target:
root/label/<name>

binary mask:
gray >= 128

teacher cache:
仅 SYSU / WHU train
通过 manifest.json entries 映射
```

**只要新项目先读取本文件，就应该能够在不重新探查服务器数据结构的情况下开始 DataLoader 适配。**
