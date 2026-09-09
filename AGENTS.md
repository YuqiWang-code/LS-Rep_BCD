# AGENTS.md — LS-Rep_BCD_RSML_3 / SAM-HSD

> Last updated: 2026-09-08
> 本文件是当前项目中 AI/Coding Agent 的工作约束。实验上下文覆盖 `train_scripts/SAM-HSD/` 中的 baseline、Run1、Run2(EIR-HSD)、Run3(Z2-SRD) 与 Run4(CR-SRD)，不记录或继承其他旧实验、旧服务器结果表或历史完成状态。

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
├── Run3/
└── Run4(CR-SRD)/
```

按以下优先级判断事实：

1. 当前代码与 shell 参数；
2. `train_scripts/SAM-HSD/Run1/README.md`；
3. `train_scripts/SAM-HSD/Run2(EIR-HSD)/README.md`；
4. `train_scripts/SAM-HSD/Run3/README.md`；
5. Run4 的 shell 与 `docs/temporary/LS-Rep_BCD_RSML_3 - SAM-HSD：Run3 证据驱动复盘、Run4 方法选择与可实施实验设计.md`（Run4 无 README，以此设计文档为权威方法来源）；
6. `docs/RSML-3_服务器环境与变化检测数据统一说明.md`；
7. 本文件中的摘要。

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
| Run4 checkpoint | `/home/yqwang/checkpoints/LS-Rep_BCD_RSML_3/saved_models/SAM-HSD/Run4` |
| Run4 训练/测试日志 | `/home/yqwang/outputs/LS-Rep_BCD_RSML_3/saved_models/SAM-HSD/Run4` |
| Run4 launcher 日志 | `/home/yqwang/outputs/LS-Rep_BCD_RSML_3/saved_models/SAM-HSD/Run4/launcher` |

本地下载与汇总产物：

| 用途 | 路径 |
|---|---|
| Run1/Run2 历史训练日志 | `saved_models/SAM-HSD/` |
| Run3 下载日志 | `outputs/SAM-HSD/Run3/` |
| Run4 下载日志 | `outputs/SAM-HSD/Run4/` |
| 统一指标表 | `docs/experiment_metrics.xlsx` |
| Run3 代码与指标快照 | `docs/temporary/models_and_metrics_SAM-HSD_Run3.txt` |

`.vscode/sftp.json` 用于手动上传代码，自动上传保持关闭。该配置忽略 `pre-trained_weights/` 和常见权重文件，权重需要单独传输并在训练前验证。

## 4. 当前 SAM-HSD 实验批次

baseline 覆盖四个 canonical 数据集：

- `CDD-CD-256`；
- `LEVIR-CD-256`；
- `SYSU-CD-256`；
- `WHU-CD-256`。

Run1/Run2/Run3/Run4 实验数据集为：

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

Run3 的 checkpoint 与 `train_log.txt` 必须使用第 3 节所列的两个独立根目录；相对目录包含 `steps_<max_steps>/seed_<seed>`，不得让 12k 筛选恢复到 40k 正式运行。结果汇总时，`max_steps=12000` 必须标记为 `Stage1` 机制筛选，不能作为 40k 论文正式性能；只有与正式协议一致的 40k 运行才标记为 `Full`。当前 Run3 只使用物理 GPU 1：SYSU 与 WHU 队列内部各自串行，两条队列可以并发驻留同一张 32 GiB 卡，但启动后必须检查合计显存。N2 是故意不满足严格群约束的混合式负对照，不得表述为主方法。`Run3_all_shell_scripts.txt` 由 Run3 的 `collect_shell_scripts.py` 生成，修改 Run3 shell 后必须重新生成。

### 4.5 Run4：CR-SRD

Run4 唯一主方法为 **CR-SRD（Coherence-Routed Structural Residual Distillation，一致性路由结构残差蒸馏）**。它保留 Run3 最强 N2 的单一共享 mixed head，核心贡献是：teacher 的「变化强度可信」与「方向符号可信」被解耦，`direction_coherence = |signed| / (magnitude + ε)`（参数无关、交换不变）在像素/通道级路由监督——方向可靠时学 signed direction，方向模糊时退化为 sign-free magnitude supervision；stable channel 完全复用 N2。部署时该机制与 SAM2 cache 一起删除，部署图/参数/FLOPs 不变。

`direction_coherence` 由 `ExchangeInvariantStructuralEvidence` 在线派生（`models/distill/sam_hsd/temporal_evidence.py`），不写入 Teacher Cache schema。CR 损失在 `models/distill/sam_hsd/losses.py::coherence_routed_structural_loss`，student topology 与 N2 完全相同（`models/distill/sam_hsd/cr_srd.py::CRSRDAdapter`），`use_coherence_gate=False` 时数值回退到 N2 mixed signed loss。

| ID | 实验名 | 语义 | 唯一变量 |
|---|---|---|---|
| J0 | J0_JointBN_Clean | clean anchor + joint temporal BN on | P0 归因对照：解耦 Run3 joint-BN 混杂 |
| J1 | J1_N2_Reproduction | z2_srd mixed signed（Run3 N2 复现） | 强 anchor 可复现性 |
| J2 | J2_CR_SRD_MaskOnly | CR coherence gate on，magnitude fallback off | 只抑制低 coherence 的 signed 监督是否有效 |
| J3 | J3_CR_SRD_Full | CR gate on + fallback on | **主方法**：低 coherence 区 magnitude fallback 是否进一步有效 |
| J4 | J4_N4_Odd_Only_Closeout | z2_srd group odd-only（Run3 N4） | N0 失败来自 odd 本身还是分支冲突 |

Run4 固定协议（Stage1，全 200 epoch）：

- 数据集：`SYSU-CD-256`（`max_steps=18750`）、`WHU-CD-256`（`max_steps=9375`）；
- `batch_size=128`、`seed=2333`、`lr=5e-4`、`weight_decay=1e-4`、`backbone_lr_mult=1.0`、`dice_reduction=batch`、`hsd_lambda=0.06`、`hsd_max_ratio=0.12`、`diag_grad_interval=500`；
- 训练参数：J0 `2,913,094`；J1/J2/J3 `2,921,370`；J4 `2,921,862`；部署恒为 `2,913,094`。

Run4 目录为单层叶：`<experiment>/<dataset>/s<max_steps>_seed<seed>/`（例如 `J3_CR_SRD_Full/SYSU-CD-256/s18750_seed2333/`）。checkpoint 与 `train_log.txt` 必须使用第 3 节列出的两个独立根目录，日志与 checkpoint 使用完全相同的 relative leaf。launcher 由 `common.sh` 提供 `cr_prepare_job`（lock + `run_manifest.json` + `.complete` skip + `last_checkpoint.pth` exact resume，拒绝覆盖）与 `cr_mark_complete`（只有完整 `=== TEST RESULTS === ... === END TEST RESULTS ===` 才写 `.complete`）。`run_manifest.json` 记录 `project/run/experiment/dataset/seed/max_steps/batch_size/joint_temporal_bn/hsd_lambda/hsd_max_ratio`。

入口与说明：

```text
train_scripts/SAM-HSD/Run4/common.sh
train_scripts/SAM-HSD/Run4/run_experiment.sh
train_scripts/SAM-HSD/Run4/run_gpu0_sysu.sh     # J0..J4 串行，物理 GPU 0
train_scripts/SAM-HSD/Run4/run_gpu1_whu.sh      # J0..J4 串行，物理 GPU 1
train_scripts/SAM-HSD/Run4/smoke_test.sh
train_scripts/SAM-HSD/Run4/dry_run_sysu.sh
train_scripts/SAM-HSD/Run4/J0_JointBN_Clean/train_{sysu,whu}.sh
train_scripts/SAM-HSD/Run4/J1_N2_Reproduction/train_{sysu,whu}.sh
train_scripts/SAM-HSD/Run4/J2_CR_SRD_MaskOnly/train_{sysu,whu}.sh
train_scripts/SAM-HSD/Run4/J3_CR_SRD_Full/train_{sysu,whu}.sh
train_scripts/SAM-HSD/Run4/J4_N4_Odd_Only_Closeout/train_{sysu,whu}.sh
```

`Run4_all_shell_scripts.txt` 由 Run4 的 `collect_shell_scripts.py` 生成，修改 Run4 shell 后必须重新生成。Run4 用双卡：GPU 0 跑 SYSU 队列、GPU 1 跑 WHU 队列，脚本以 `CUDA_VISIBLE_DEVICES=<物理ID>` + `--gpu_id 0` 映射（物理 GPU 1 在进程内是逻辑 cuda:0）。

Run4 预注册成功门槛（Stage1 只做机制筛选，不得调 loss weight 网格「救方法」）：

- CR Full (J3) 对 N2 (J1)：两数据集分别 `ΔF1 ≥ +0.25`，或两数据集平均 `ΔF1 ≥ +0.40` 且任一 `ΔF1 ≥ 0`，同时 `mean ΔIoU > 0` 与 `mean ΔKappa > 0`；
- fallback 因子有效性：`J3 − J2` 平均 `F1 ≥ +0.15`；
- 若某数据集 Recall 或 Precision 下降 `>1.5 pp` 且 F1 增益不足，判定存在明显副作用。

通过 Stage1 gate 后才 fresh 跑 N2/CR 的 20k budget diagnostic（禁止 12k→20k resume），再决定正式 20k/40k 与 Stage2 3 seeds（`{2333, 3407, 4519}`）paired formal matrix。Run4 当前仅完成 Stage1 预算，尚无 40k 正式结果。

Run4 Stage1（200-epoch，seed=2333）已完成全部 10 run，正式 test 指标（百分比，来自 `docs/experiment_metrics.xlsx`）：

| ID | 数据集 | Recall | Precision | OA | F1 | IoU | Kappa |
|---|---|---:|---:|---:|---:|---:|---:|
| J0 | SYSU | 83.3601 | 80.8679 | 91.4249 | 82.0951 | 69.6283 | 76.4593 |
| J1 | SYSU | 79.0349 | 85.4150 | 91.8732 | 82.1012 | 69.6370 | 76.8547 |
| J2 | SYSU | 78.7178 | 85.2005 | 91.7565 | 81.8310 | 69.2491 | 76.5106 |
| J3 | SYSU | 78.5310 | 85.5023 | 91.7968 | 81.8685 | 69.3029 | 76.5802 |
| J4 | SYSU | 79.5394 | 85.0487 | 91.8773 | 82.2018 | 69.7819 | 76.9473 |
| J0 | WHU | 92.9783 | 95.4760 | 99.5466 | 94.2106 | 89.0548 | 93.9747 |
| J1 | WHU | 92.5707 | 94.4374 | 99.4889 | 93.4947 | 87.7841 | 93.2287 |
| J2 | WHU | 93.2005 | 94.8729 | 99.5304 | 94.0293 | 88.7314 | 93.7849 |
| J3 | WHU | 89.2575 | 95.4036 | 99.4032 | 92.2282 | 85.5774 | 91.9183 |
| J4 | WHU | 92.1876 | 95.1705 | 99.5044 | 93.6553 | 88.0677 | 93.3975 |

**Stage1 预注册判定（未通过主方法 gate）**：CR Full (J3) 相对 N2 (J1) 的 ΔF1 为 SYSU `-0.2327`、WHU `-1.2665`，两数据集均下降，远未达到 `≥ +0.25` 门槛；J2（MaskOnly）也只在 WHU 上略优于 J1（`+0.5346`），SYSU `-0.2702`。因此 CR-SRD 在 200-epoch Stage1 下**未显示相对 N2 的增益**。按预注册规则，不应通过调 loss weight 网格「救方法」，下一步应按设计文档第 11/12 节先诊断 coherence 分布、best-step 与 budget 效应，再决定是否进入 Stage1B/Stage2 或终止 CR 主方法。J4（N4 odd-only）两数据集均略高于 J1（SYSU `+0.1006`、WHU `+0.1606`），幅度在单 seed 噪声内。

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
models/tools/smoke_cr_srd.py
models/tools/dry_run_cr_srd.py
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

本地统一汇总命令：

```bash
python -B analyse/extract_metrics.py
python -B analyse/models_to_txt.py --run3
```

`extract_metrics.py` 递归读取 `saved_models/SAM-HSD/**/train_log.txt`、`outputs/SAM-HSD/Run3/**/train_log.txt` 与 `outputs/SAM-HSD/Run4/**/train_log.txt`，忽略 launcher tee 日志，只接收最后一个字段完整且数值可解析的正式 test block；任一候选日志不完整时必须报错，不得静默混入结果。Run4 的 `Stage` 判定为 `Stage1`（200-epoch 机制筛选，非 40k `Full`）。指标以百分数保存并保留 4 位小数。`models_to_txt.py --run4` 收录当前 `models/` 下全部 Python 源码与统一 Excel 全部工作表，输出 `docs/temporary/models_and_metrics_SAM-HSD_Run4.txt`（`--run3` 对应 Run3 快照）。

## 8. Agent 工作规则

1. 修改前读取当前代码、对应 Run README 与实际日志，不依赖旧项目记忆。
2. 方法创新优先，但每项主张必须对应明确机制、可证伪假设和最小消融。
3. 保持推理图、Teacher Cache replay 顺序和部署约束不变。
4. 修改 shell 后执行语法检查，并重新生成对应 Run2/Run3 汇总文件。
5. 修改模型后至少执行相关 smoke test；涉及数据或缓存时再执行 real-cache dry run。
6. 不恢复与当前 `train_scripts/SAM-HSD` 无关的旧实验模块、路径、指标或环境配置。
7. 不在 `cd_base` 中安装项目专属依赖，不擅自更换 PyTorch/CUDA 栈。
8. 任何删除数据、缓存或 checkpoint 的操作都需要用户明确授权和精确路径校验。
