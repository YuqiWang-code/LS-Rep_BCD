# AGENTS.md — LS-Rep_BCD_RSML_3 / SAM-HSD

> Last updated: 2026-09-07  
> 本文件是当前项目中 AI/Coding Agent 的工作约束。实验上下文仅覆盖 `train_scripts/SAM-HSD/` 中的 baseline、Run1、Run2(EIR-HSD) 与 Run3(Z2-SRD)，不记录或继承其他旧实验、旧服务器结果表或历史完成状态。

## GitHub 提交流程

完成 `models/` 或 `AGENTS.md` 的修改并通过必要检查后，按以下顺序提交到 GitHub：

```bash
git add models AGENTS.md
git status
git commit -m "Update code"
git push
```

执行 `git commit` 前必须先检查 `git status`，确认暂存区只包含本次准备提交的文件；不要把权重、缓存、数据集、日志或无关改动加入提交。

## 研究定位与优先级

本项目服务于 211 高校硕士研究生的学术论文研究与实验验证，不以通用软件工程建设为主要目标。Agent 应将具有论文价值的**模型创新和方法性改进**置于最高优先级，重点关注网络结构、特征表征与交互、训练期辅助机制及其可解释性，而不是把代码重构、工程封装、损失函数微调或超参数搜索当作主要贡献。

具体遵循以下原则：

1. 优先提出机制明确、区别于现有模块且能够形成论文贡献点的方法创新；每项方案都应说明创新动机、作用机制、与已有方法的差异、可证伪假设及最小消融实验。
2. 损失函数和超参数仅作为公平训练、稳定收敛与验证机制有效性的配套手段；除非存在明确的新理论或新机制，否则不得将调参结果包装成核心创新。
3. 工程修改只需服务于实验正确性、可复现性和必要的运行效率；不会阻碍实验或结论可信度的工程优化，不应挤占方法研究的优先级。
4. 评价方案时优先考虑学术新颖性、机制合理性、实验可验证性和论文叙事完整性，同时警惕无机理的模块堆叠与只追求单次指标的改动。

## 1. 作用域与权威来源

当前唯一活动实验族：

```text
train_scripts/SAM-HSD/
├── baseline/
├── Run1/
├── Run2(EIR-HSD)/
└── Run3/
```

按以下优先级判断事实：

1. 当前代码与 shell 参数；
2. `train_scripts/SAM-HSD/Run1/README.md`；
3. `train_scripts/SAM-HSD/Run2(EIR-HSD)/README.md`；
4. `train_scripts/SAM-HSD/Run3/README.md`；
5. `docs/RSML-3_服务器环境与变化检测数据统一说明.md`；
6. 本文件中的摘要。

不得从旧项目复制实验结果、GPU/存储配置、完成状态或结论。需要报告指标时，必须重新读取当前服务器上对应运行的正式 test block。

## 2. RSML-3 项目环境

```yaml
project:
  name: LS-Rep_BCD_RSML_3
  server: RSML-3
  user: yqwang
  path: /home/yqwang/projects/LS-Rep_BCD_RSML_3

environment:
  name: lsrep
  path: /home/yqwang/miniforge3/envs/lsrep
  base_template: cd_base
  creation_method: conda create -n lsrep --clone cd_base

runtime:
  python: 3.10.21
  pytorch: 2.14.0+cu132
  cuda_runtime: "13.2"
  cuda_available: true

gpu:
  count: 2
  currently_available_ids: [1]
  model: NVIDIA GeForce RTX 5090
  architecture: Blackwell
  compute_capability: "12.0"
  pytorch_arch: sm_120
```

关键依赖已验证可导入：

| 包 | 版本或状态 |
|---|---|
| `numpy` | `2.2.6` |
| `scipy` | `1.15.2` |
| `scikit-learn` | `1.7.2` |
| `pillow` | `12.3.0` |
| `pandas` | `2.3.3` |
| `rasterio` | `1.4.3` |
| `geopandas` | `1.1.4` |
| `shapely` | `2.1.2` |
| `tensorboardX` | `2.6.5` |
| `thop` | `0.1.1.post2209072238` |
| `torchvision`、`opencv`、`tqdm` | 已安装并验证导入 |

环境使用：

```bash
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate lsrep
cd /home/yqwang/projects/LS-Rep_BCD_RSML_3
```

环境规则：

- 训练、验证、推理和复杂度统计固定使用 `lsrep`，不要直接使用或污染 `cd_base`。
- 不要自行降级、替换或无约束升级 PyTorch/CUDA。
- 缺失的普通 Python 依赖只安装到 `lsrep`。
- CUDA Extension 或二进制算子必须先确认兼容 PyTorch 2.14.0、CUDA 13.2 与 `sm_120`。
- 环境有重要变化时同步更新本文件，并将快照保存到 `/home/yqwang/env_specs/lsrep/`。

推荐健康检查：

```bash
conda activate lsrep
python -m pip check
python -c "import torch, torchvision, cv2, numpy, scipy, sklearn, PIL, tqdm, tensorboardX, thop; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
```

## 3. 服务器目录约定

| 用途 | 路径 |
|---|---|
| 项目 | `/home/yqwang/projects/LS-Rep_BCD_RSML_3` |
| 数据集 | `/home/yqwang/datasets/CD` |
| Teacher Cache | `/home/yqwang/datasets/CD_teacher_cache` |
| 预训练权重 | `/home/yqwang/projects/LS-Rep_BCD_RSML_3/pre-trained_weights` |
| Baseline checkpoint | `/home/yqwang/checkpoints/LS-Rep_BCD_RSML_3/saved_models/SAM-HSD/baseline` |
| Baseline 训练日志 | `/home/yqwang/outputs/LS-Rep_BCD_RSML_3/saved_models/SAM-HSD/baseline` |
| Run1 checkpoint | `/home/yqwang/checkpoints/LS-Rep_BCD_RSML_3/SAM-HSD/Run1` |
| Run2 checkpoint | `/home/yqwang/checkpoints/LS-Rep_BCD_RSML_3/SAM-HSD/Run2` |
| Run1 launcher 日志 | `/home/yqwang/outputs/LS-Rep_BCD_RSML_3/logs/SAM-HSD/Run1` |
| Run2 launcher 日志 | `/home/yqwang/outputs/LS-Rep_BCD_RSML_3/logs/SAM-HSD/Run2` |
| Run3 checkpoint | `/home/yqwang/checkpoints/LS-Rep_BCD_RSML_3/saved_models/SAM-HSD/Run3` |
| Run3 训练/测试日志 | `/home/yqwang/outputs/LS-Rep_BCD_RSML_3/saved_models/SAM-HSD/Run3` |

`.vscode/sftp.json` 用于手动上传代码，自动上传保持关闭。该配置忽略 `pre-trained_weights/` 和常见权重文件，权重需要单独传输并在训练前验证。

## 4. 当前 SAM-HSD 实验批次

baseline 覆盖四个 canonical 数据集：

- `CDD-CD-256`；
- `LEVIR-CD-256`；
- `SYSU-CD-256`；
- `WHU-CD-256`。

Run1/Run2/Run3 实验数据集为：

- `SYSU-CD-256`；
- `WHU-CD-256`。

### 4.1 Baseline：A2Net-LWGANet-L0

baseline 使用 `B0/Baseline_A2Net_LWGANet_L0` 配方：只保留 A2Net 主网络与 LWGANet-L0 backbone，不加载 Teacher Cache，不创建 SAM-HSD/EIR-HSD 训练辅助。四个入口脚本为：

```text
train_scripts/SAM-HSD/baseline/train_cdd.sh
train_scripts/SAM-HSD/baseline/train_levir.sh
train_scripts/SAM-HSD/baseline/train_sysu.sh
train_scripts/SAM-HSD/baseline/train_whu.sh
train_scripts/SAM-HSD/baseline/smoke_test.sh
train_scripts/SAM-HSD/baseline/run_gpu0_cdd_sysu.sh
train_scripts/SAM-HSD/baseline/run_gpu1_levir_whu.sh
```

默认 GPU 按 CDD/SYSU→0、LEVIR/WHU→1 分配，脚本第一个位置参数可覆盖物理 GPU ID。checkpoint 与训练日志必须使用第 3 节列出的两个独立根目录；不得重新合并到同一目录。

正式训练前先运行 `smoke_test.sh <physical_gpu_id> <batch_size>`。双卡队列固定为 GPU 0 串行执行 CDD→SYSU、GPU 1 串行执行 LEVIR→WHU；队列脚本依赖单数据集脚本自身的完成跳过和精确恢复检查。

baseline 的预训练权重默认使用 `/home/yqwang/projects/LS-Rep_BCD_RSML_3/pre-trained_weights/lwganet_l0_e299.pth`。这是 backbone 初始化权重，不是 Teacher Cache 或训练期外挂。

### 4.2 Run1：SAM-HSD

| ID | 当前脚本语义 |
|---|---|
| H0 | Clean Anchor |
| H1 | Legacy SAMStruct Reference |
| H2 | Encoder Directional |
| H3 | Decoder SCGR |
| H4 | Full SAM-HSD |
| H5 | Boundary Scalar |
| H6 | Unsigned Evidence |
| H7 | No SCGR |
| H8 | No Robust Filter |

入口与说明：

```text
train_scripts/SAM-HSD/Run1/README.md
train_scripts/SAM-HSD/Run1/run_experiment.sh
train_scripts/SAM-HSD/Run1/run_gpu0_sysu.sh
train_scripts/SAM-HSD/Run1/run_gpu1_whu.sh
train_scripts/SAM-HSD/Run1/smoke_test.sh
train_scripts/SAM-HSD/Run1/dry_run_sysu.sh
```

### 4.3 Run2：EIR-HSD

| ID | 当前脚本语义 |
|---|---|
| R0 | Clean Anchor |
| R1 | Run1 Unsigned Reference |
| R2 | Exchange-Invariant Code |
| R3 | Spatial Residual Encoder |
| R4 | Residual Decoder Head |
| R5 | Full EIR-HSD |
| R6 | Fixed Fusion |
| R7 | No Residual Correction |
| R8 | Directional Restore |

入口与说明：

```text
train_scripts/SAM-HSD/Run2(EIR-HSD)/README.md
train_scripts/SAM-HSD/Run2(EIR-HSD)/run_experiment.sh
train_scripts/SAM-HSD/Run2(EIR-HSD)/run_priority_gpu0_sysu.sh
train_scripts/SAM-HSD/Run2(EIR-HSD)/run_priority_gpu1_whu.sh
train_scripts/SAM-HSD/Run2(EIR-HSD)/run_gpu0_sysu.sh
train_scripts/SAM-HSD/Run2(EIR-HSD)/run_gpu1_whu.sh
train_scripts/SAM-HSD/Run2(EIR-HSD)/smoke_test.sh
train_scripts/SAM-HSD/Run2(EIR-HSD)/dry_run_sysu.sh
```

`Run2_all_shell_scripts.txt` 是由 `collect_shell_scripts.py` 生成的汇总文件；修改任何 Run2 shell 后必须重新生成，不要直接编辑汇总文件。

### 4.4 Run3：Z2-SRD

Run3 首选方法为 Z2-SRD（Swap-Group Even/Odd Structural Residual Distillation）。它以 Run2 R2 为最小祖先，在训练期把 SAM2 二时相结构 teacher 严格分解为交换不变的 even 表示与交换反变的 odd 表示；两个分支分别监督，不使用 Run2 decoder correction，部署时完整删除。

| ID | 当前脚本语义 | 训练参数 | 部署参数 |
|---|---|---:|---:|
| N0 | Z2-SRD Full（even+odd separated） | 2,921,930 | 2,913,094 |
| N1 | Z2 Even Only | 2,921,882 | 2,913,094 |
| N2 | Mixed Signed Control | 2,921,370 | 2,913,094 |
| N3 | No Group Projection（R2-style invariant control） | 2,921,370 | 2,913,094 |
| N4 | Z2 Odd Only | 2,921,862 | 2,913,094 |

入口与说明：

```text
train_scripts/SAM-HSD/Run3/README.md
train_scripts/SAM-HSD/Run3/run_experiment.sh
train_scripts/SAM-HSD/Run3/run_stage1_gpu0_sysu.sh
train_scripts/SAM-HSD/Run3/run_stage1_gpu1_whu.sh
train_scripts/SAM-HSD/Run3/run_gpu0_sysu.sh
train_scripts/SAM-HSD/Run3/run_gpu1_whu.sh
train_scripts/SAM-HSD/Run3/smoke_test.sh
train_scripts/SAM-HSD/Run3/dry_run_sysu.sh
```

Run3 的 checkpoint 与 `train_log.txt` 必须使用第 3 节所列的两个独立根目录；相对目录包含 `steps_<max_steps>/seed_<seed>`，不得让 12k 筛选恢复到 40k 正式运行。当前 Run3 只使用物理 GPU 1：SYSU 与 WHU 队列内部各自串行，两条队列可以并发驻留同一张 32 GiB 卡，但启动后必须检查合计显存。N2 是故意不满足严格群约束的混合式负对照，不得表述为主方法。`Run3_all_shell_scripts.txt` 由 Run3 的 `collect_shell_scripts.py` 生成，修改 Run3 shell 后必须重新生成。

## 5. 模型与部署约束

- 主学生模型是 A2Net-LWGANet-L0；SAM2 Teacher Cache 和 HSD 辅助结构只允许参与训练。
- 推理参数量保持 `2,913,094`。256×256 输入以 `2.7475G` 为历史参考值；当前 RSML-3 `lsrep`/THOP 环境稳定实测为 `2.767634G`，代码使用 `±0.03G` 统计容差，不能仅因这一级别的统计差异阻断最终测试。
- 打开或关闭训练辅助不得改变主预测。
- `switch_to_deploy()` 必须移除训练辅助参数，部署前后最大误差要求 `< 1e-6`。
- 不得把 Teacher map 注入部署期主特征路径。
- A/B/label 与 Teacher Cache 必须重放完全一致的 crop、resize、flip 和时相交换。
- 实验语义集中在 `models/scripts/train.py`；wrapper 只负责选择实验、数据集、GPU 和运行目录。

核心代码：

```text
models/a2net.py
models/backbone/lwganet.py
models/decoder/a2net_decoder.py
models/datasets/
models/distill/teacher_cache.py
models/distill/sam_hsd/
models/losses/combined_loss.py
models/scripts/train.py
models/tools/sam/validate_teacher_cache.py
models/tools/smoke_sam_hsd.py
models/tools/smoke_eir_hsd.py
models/tools/smoke_z2_srd.py
models/tools/dry_run_z2_srd.py
```

## 6. Teacher Cache 规则

当前缓存：

```text
/home/yqwang/datasets/CD_teacher_cache/SAMStruct/sam2.1_hiera_large/SYSU-CD-256/
/home/yqwang/datasets/CD_teacher_cache/SAMStruct/sam2.1_hiera_large/WHU-CD-256/
```

- 先运行当前批次中的 `validate_sam_cache_*.sh`。
- 验证通过时不要重新生成缓存。
- 仅当验证失败、训练列表变化或 schema 明确变更时考虑重建。
- 不得删除或覆盖数据集、Teacher Cache、checkpoint，除非用户明确授权并已核对绝对路径。

## 7. 启动、恢复与结果纪律

推荐顺序：

1. 激活 `lsrep` 并进入项目目录；
2. 验证 SYSU/WHU Teacher Cache；
3. 执行对应 Run 的 smoke test；
4. 执行 SYSU dry run，确认真实 DataLoader、缓存、显存和梯度；
5. 按 README 的 staged launch 顺序启动正式矩阵；
6. 从当前运行日志收集正式 test 指标。

每个实验目录使用：

- `last_checkpoint.pth`：精确恢复；
- `best_model_F1=*.pth`：验证集选择出的最佳模型；
- `train_log.txt`：训练、恢复和最终测试记录。

只有 `train_log.txt` 中同时出现以下标记才算完成：

```text
=== TEST RESULTS ===
=== END TEST RESULTS ===
```

结果规则：

- 正式指标只能来自上述最终 test block，不能使用验证集最佳行或 checkpoint 文件名替代。
- 报告 Recall、Precision、OA、F1、IoU、Kappa，以及训练/部署参数量和 FLOPs。
- 不在本文件保存旧结果表、历史排名或“已完成”状态；每次均以当前服务器文件为准。
- 单数据集、单种子或阈值以下差异不能被表述为普适提升。

## 8. Agent 工作规则

1. 修改前读取当前代码、对应 Run README 与实际日志，不依赖旧项目记忆。
2. 方法创新优先，但每项主张必须对应明确机制、可证伪假设和最小消融。
3. 保持推理图、Teacher Cache replay 顺序和部署约束不变。
4. 修改 shell 后执行语法检查，并重新生成对应 Run2/Run3 汇总文件。
5. 修改模型后至少执行相关 smoke test；涉及数据或缓存时再执行 real-cache dry run。
6. 不恢复与当前 `train_scripts/SAM-HSD` 无关的旧实验模块、路径、指标或环境配置。
7. 不在 `cd_base` 中安装项目专属依赖，不擅自更换 PyTorch/CUDA 栈。
8. 任何删除数据、缓存或 checkpoint 的操作都需要用户明确授权和精确路径校验。
