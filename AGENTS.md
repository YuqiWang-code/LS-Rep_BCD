# AGENTS.md — LS-Rep_BCD_RSML_3 / SAM-HSD

> Last updated: 2026-09-10
> 本文件是当前项目中 AI/Coding Agent 的工作约束。实验上下文覆盖 `train_scripts/SAM-HSD/` 中的 baseline、Run1、Run2(EIR-HSD) 与 Run3(Z2-SRD)。Run4(CR-SRD) 已判定无效并回退，`models/` 已恢复为 Run3 版本；Run3 40k 正式矩阵已完成，主方法 N0 未胜过对照 N3，按 README 停止标准 Z2 主线应停止。**项目当前处于「方法有效性探寻阶段」，主线结论随时可能被新实验推翻；不得把本文档中的任何结果当作已定稿的论文结论。** 不记录或继承其他旧实验、旧服务器结果表或历史完成状态。

## GitHub 提交流程

完成 `models/` 或 `AGENTS.md` 的修改并通过必要检查后，按以下顺序提交到 GitHub：

```bash
git add models AGENTS.md docs/experiment_metrics.xlsx "docs/RSML-3_服务器环境与变化检测数据统一说明.md"
git status
git commit -m "Update code"
git push
```

执行 `git commit` 前必须先检查 `git status`，确认暂存区只包含本次准备提交的文件；不要把权重、缓存、数据集、日志或无关改动加入提交。`docs/experiment_metrics.xlsx` 与 `docs/RSML-3_服务器环境与变化检测数据统一说明.md` 随代码与 AGENTS.md 一起提交（指标表与服务器/数据环境说明是论文复现所需的一部分）。

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
  currently_available_ids: [0, 1]
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

本地下载与汇总产物：

| 用途 | 路径 |
|---|---|
| Run1/Run2 历史训练日志 | `saved_models/SAM-HSD/` |
| Run3 下载日志 | `outputs/SAM-HSD/Run3/` |
| 统一指标表 | `docs/experiment_metrics.xlsx` |
| Run3 代码与指标快照 | `docs/temporary/models_and_metrics_SAM-HSD_Run3.txt` |

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

Run3 的 checkpoint 与 `train_log.txt` 必须使用第 3 节所列的两个独立根目录；相对目录包含 `steps_<max_steps>/seed_<seed>`，不得让 12k 筛选恢复到 40k 正式运行。结果汇总时，`max_steps=12000` 必须标记为 `Stage1` 机制筛选，不能作为 40k 论文正式性能；只有与正式协议一致的 40k 运行才标记为 `Full`。当前 Run3 40k 正式矩阵用双卡：GPU 0 跑 SYSU 队列、GPU 1 跑 WHU 队列，两条队列内部各自按 `N3 → N1 → N0 → N2 → N4` 串行（`run_experiment.sh` 以 `unset CUDA_VISIBLE_DEVICES` + `--gpu_id <物理ID>` 直接映射物理 GPU）。N2 是故意不满足严格群约束的混合式负对照，不得表述为主方法。`Run3_all_shell_scripts.txt` 由 Run3 的 `collect_shell_scripts.py` 生成，修改 Run3 shell 后必须重新生成。

Run3 40k 正式矩阵（`steps_40000`，seed=2333，batch=64）已完成全部 10 run，正式 test 指标（百分比，来自 `docs/experiment_metrics.xlsx`）：

| ID | 数据集 | Recall | Precision | OA | F1 | IoU | Kappa |
|---|---|---:|---:|---:|---:|---:|---:|
| N0 | SYSU | 80.4740 | 85.4443 | 92.1622 | 82.8847 | 70.7719 | 77.8080 |
| N1 | SYSU | 78.1334 | 85.5290 | 91.7257 | 81.6641 | 69.0104 | 76.3355 |
| N2 | SYSU | 78.1646 | 85.7785 | 91.7945 | 81.7948 | 69.1973 | 76.5132 |
| N3 | SYSU | 81.0877 | 86.0014 | 92.4273 | **83.4723** | **71.6330** | **78.5663** |
| N4 | SYSU | 79.6303 | 85.0594 | 91.8977 | 82.2554 | 69.8591 | 77.0131 |
| N0 | WHU | 93.2352 | 94.0511 | 99.4976 | 93.6413 | 88.0430 | 93.3798 |
| N1 | WHU | 92.5720 | 95.3571 | 99.5265 | 93.9439 | 88.5795 | 93.6976 |
| N2 | WHU | 92.5984 | 95.2979 | 99.5251 | 93.9288 | 88.5525 | 93.6817 |
| N3 | WHU | 92.6049 | 95.5538 | 99.5356 | **94.0562** | **88.7794** | **93.8147** |
| N4 | WHU | 91.8238 | 94.6297 | 99.4689 | 93.2056 | 87.2758 | 92.9293 |

**40k 判定（Z2 even/odd 分解未获支持）**：数据集内排名 SYSU `N3 83.4723 > N0 82.8847 > N4 82.2554 > N2 81.7948 > N1 81.6641`，WHU `N3 94.0562 > N1 93.9439 > N2 93.9288 > N0 93.6413 > N4 93.2056`。主方法 N0 相对对照 N3 在**两个数据集均下降**（SYSU `ΔF1 −0.5876`，WHU `ΔF1 −0.4149`），IoU/Kappa 同向下降，未达到 README 的继续标准（平均 `ΔF1 ≥ +0.25` 或平均 `ΔIoU ≥ +0.35`，且单数据集 F1 不低于 `−0.15`）。N0 与 N2 也无一致差异（SYSU `+1.0899`、WHU `−0.2875`），不满足「分离优于混合」。按 README 第「继续与停止标准」，应**停止 Z2 主线**，不再继续堆 correction 或调权重。这是单 seed（2333）结果；README 要求的论文级主比较为 seed `2333/3407/5871`，因此当前结论是机制层面的否定证据，而非最终统计结论。12k Stage1（8 行）仍保留在 Excel 中，标记为 `Stage1`，仅作机制筛选参考。

#### SAM2 结构 teacher 的有效性（当前未获证实）

跨 Run 对照（全部取自 `docs/experiment_metrics.xlsx` 正式 test block）：

| 对比 | 协议 | SYSU ΔF1 | WHU ΔF1 | 可归因性 |
|---|---|---:|---:|---|
| Run1 H0 → Run2 R0（同配置重跑） | 64 / 40k | `+0.1132` | `−0.0752` | 单 seed 噪声基线 ≈ 0.1 |
| Clean anchor → Run3 N3（含 teacher） | 64 / 40k | `+1.4466` | `+0.1177` | **混杂 joint-BN，不可归因 teacher** |
| **Run4 J0 → J1（同协议，仅差 teacher）** | 128 / 同 steps，均含 joint-BN | **`+0.0061`** | **`−0.7159`** | 目前最干净对照 |

结论：

1. Run3 的 `z2_srd` 在训练态把 T1/T2 合并为 `2B` 一次前向（joint temporal BN），因此 Run3 相对 Run1/Run2 clean anchor 的差值 = `teacher 效应 + joint-BN 效应`，**不能把 SYSU 的 `+1.45` 归因于 teacher**。该混杂即 Run4 设计文档标记的 P0 归因问题。
2. 唯一同协议 teacher-vs-no-teacher 对照（Run4 `J0_JointBN_Clean` vs `J1_N2_Reproduction`，batch/steps/joint-BN 完全一致）显示 SYSU 差异 `+0.0061`（等价于零）、WHU `−0.7159`（teacher 反而有害）。
3. Run3 内 5 个 variant 共享同一 teacher，最简单的 N3（R2 式 abs/product 不变量）在两个数据集均最优，说明 teacher 的**结构化用法**同样未带来收益。

**因此：按现有正式结果，不得声称 SAM2 结构 teacher 有效。** 若要正面回答该问题，最小可证伪实验是同协议（同 batch/同 steps）的 `clean` vs `clean + joint-BN` vs `clean + joint-BN + teacher` 三臂对照；在完成该对照前，任何「teacher 提升」的表述都属未经验证。项目处于有效性探寻阶段，本节结论随时可能被新实验推翻。

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
- **有效性状态**：该 SAM2 结构 teacher 目前**未获证实有效**（见 4.4 节跨 Run 对照）。缓存文件本身已验证可用，但「teacher 能否提升主模型」这一问题仍需同协议三臂对照（`clean` / `clean + joint-BN` / `clean + joint-BN + teacher`）才能回答。在得到该证据前，不得在任何报告或论文中把 teacher 描述为有效组件。

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

本地统一汇总命令：

```bash
python -B analyse/extract_metrics.py
python -B analyse/models_to_txt.py --run3
```

`extract_metrics.py` 递归读取 `saved_models/SAM-HSD/**/train_log.txt` 与 `outputs/SAM-HSD/Run3/**/train_log.txt`，忽略 launcher tee 日志，只接收最后一个字段完整且数值可解析的正式 test block；任一候选日志不完整时必须报错，不得静默混入结果。Run3 的 `Stage` 判定：`max_steps=12000` 为 `Stage1`（机制筛选），`max_steps=40000` 为 `Full`（正式）。指标以百分数保存并保留 4 位小数。`models_to_txt.py --run3` 收录当前 `models/` 下全部 Python 源码与统一 Excel 全部工作表，输出 `docs/temporary/models_and_metrics_SAM-HSD_Run3.txt`。

## 8. Agent 工作规则

1. 修改前读取当前代码、对应 Run README 与实际日志，不依赖旧项目记忆。
2. 方法创新优先，但每项主张必须对应明确机制、可证伪假设和最小消融。
3. 保持推理图、Teacher Cache replay 顺序和部署约束不变。
4. 修改 shell 后执行语法检查，并重新生成对应 Run2/Run3 汇总文件。
5. 修改模型后至少执行相关 smoke test；涉及数据或缓存时再执行 real-cache dry run。
6. 不恢复与当前 `train_scripts/SAM-HSD` 无关的旧实验模块、路径、指标或环境配置。
7. 不在 `cd_base` 中安装项目专属依赖，不擅自更换 PyTorch/CUDA 栈。
8. 任何删除数据、缓存或 checkpoint 的操作都需要用户明确授权和精确路径校验。
