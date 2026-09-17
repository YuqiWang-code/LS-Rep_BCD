# AGENTS.md — LS-Rep_BCD_RSML_3 / RDT-CD

> Last updated: 2026-09-17
> 本文件是当前项目中 AI/Coding Agent 的工作约束。当前唯一活动实验为 `train_scripts/RDT-CD/Run3/`（方向 C），主方法为 **BT-SAM-RDT（Bi-Temporal Structural SAM Reciprocal Dynamic Teacher）**；学生模型 A2Net-LWGANet-L0，部署参数 2,913,094 不变。Run1/RDT-CD 与 Run2/SCGR 均作为历史实验保留，不再作为当前活动实现继续迭代。项目长期处于「方法有效性探寻阶段」，主线结论随时可能被新实验推翻；不得把本文档中的任何结果当作已定稿的论文结论。旧实验（SAM-HSD、DART-R-TS 固定教师、旧 C0 梯度路由、RDT-CD Run1/Run2 等）只能作为历史证据，不得重新混入当前 Run3 代码路径。

## GitHub 提交流程

完成 `models/`、`AGENTS.md` 或当前 Run3 脚本/说明的修改并通过必要检查后，按以下顺序提交到 GitHub：

```bash
git add models AGENTS.md train_scripts/RDT-CD/Run3 docs/experiment_metrics.xlsx "docs/RSML-3_服务器环境与变化检测数据统一说明.md"
git status
git commit -m "Update code"
git push
```

执行 `git commit` 前必须先检查 `git status`，确认暂存区只包含本次准备提交的文件；不要把权重、缓存、数据集、日志、`__pycache__/`、`*.pyc` 或无关改动加入提交。

## 研究定位与优先级

本项目服务于 211 高校硕士研究生的学术论文研究与实验验证，不以通用软件工程建设为主要目标。Agent 应将具有论文价值的**模型创新和方法性改进**置于最高优先级，重点关注网络结构、特征表征与交互、训练期辅助机制及其可解释性。

1. 优先提出机制明确、区别于现有模块且能形成论文贡献点的方法创新；每项方案都要说明创新动机、作用机制、与已有方法的差异、可证伪假设及最小消融实验。
2. 损失函数和超参数仅作为公平训练、稳定收敛与验证机制有效性的配套手段；除非有明确的新理论或新机制，否则不得把调参结果包装成核心创新。
3. 工程修改只需服务于实验正确性、可复现性和必要的运行效率。
4. 评价方案优先考虑学术新颖性、机制合理性、实验可验证性和论文叙事完整性，警惕无机理的模块堆叠与只追求单次指标的改动。

## 1. 作用域与权威来源

当前唯一活动实验族：

```text
train_scripts/RDT-CD/Run3/      # 方向 C：BT-SAM-RDT（B0 / R3A / R3）
```

（`train_scripts/RDT-CD/Run1/`、`train_scripts/RDT-CD/Run2/`、`train_scripts/SAM-HSD/`、`train_scripts/DART-R/` 等均为历史实验；其中 Run2/SCGR、旧 Direction-C/gradient routing 等已停止，不再活动。）

按以下优先级判断事实：

1. 当前 `models/` 代码与当前 Run3 shell 参数；
2. `train_scripts/RDT-CD/Run3/README.md`（当前实验协议）；
3. `docs/RSML-3_服务器环境与变化检测数据统一说明.md`（环境/数据/路径）；
4. 本文件中的摘要。

不得从旧项目复制实验结果、GPU/存储配置、完成状态或结论。需要报告指标时，必须重新读取当前服务器上对应运行的正式 test block。

## 2. RSML-3 项目环境

```yaml
project:
  name: LS-Rep_BCD_RSML_3
  server: RSML-3
  user: yqwang
  path: /home/yqwang/projects/LS-Rep_BCD_RSML_3

environment:
  name: lsrep          # clone 自 cd_base
  path: /home/yqwang/miniforge3/envs/lsrep
  python: 3.10.21
  pytorch: 2.14.0+cu132
  cuda_runtime: "13.2"

gpu:
  count: 2
  model: NVIDIA GeForce RTX 5090   # Blackwell, sm_120, 每卡 32GB
```

环境使用：

```bash
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate lsrep
cd /home/yqwang/projects/LS-Rep_BCD_RSML_3
```

环境规则：

- 训练、验证、推理和复杂度统计固定使用 `lsrep`，不要直接使用或污染 `cd_base`。
- 不要自行降级、替换或无约束升级 PyTorch/CUDA。
- 缺失的普通 Python 依赖只安装到 `lsrep`；CUDA Extension 必须先确认兼容 PyTorch 2.14.0、CUDA 13.2 与 `sm_120`。

## 3. 服务器目录约定

| 用途 | 路径 |
|---|---|
| 项目代码 | `/home/yqwang/projects/LS-Rep_BCD_RSML_3` |
| 数据集 | `/share_datasets/CD` |
| Teacher Cache | `/share_datasets/CD_teacher_cache` |
| RDT-CD checkpoint | `/share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/RDT-CD`（当前 Run3 使用 `RDT-CD/Run3/<实验>/<数据集>/`） |
| RDT-CD 训练/测试日志 | `/home/yqwang/outputs/LS-Rep_BCD_RSML_3/RDT-CD`（当前 Run3 使用 `RDT-CD/Run3/<实验>/<数据集>/`） |
| 预训练权重 | `/home/yqwang/projects/LS-Rep_BCD_RSML_3/pre-trained_weights` |

（旧 Run1/Run2、SAM-HSD、DART-R-TS 的 checkpoint/log 仅作归档，不再活动。当前 RDT-CD 实验目录不使用 `steps_40000/seed_2333` 中间目录，这两项写进训练日志配置头。）

`.vscode/sftp.json` 用于手动上传代码，自动上传保持关闭；该配置忽略 `pre-trained_weights/` 和常见权重文件。

## 4. RDT-CD（方向 C）

方向 C 主方法经历过多次迭代：C0（旧梯度路由）、C1（DART-R-TS 固定教师）、Run1 RDT-CD 与 Run2 SCGR 均提供了历史诊断证据，但没有形成稳定、可归因的 Teacher 增益。当前活动实现为 **Run3 BT-SAM-RDT**。

学生模型始终为 A2Net-LWGANet-L0。Run3 的全部 Foundation Teacher Cache、Fast Teacher、EMA Target Teacher、difficulty diagnostics 与 GT audit 都是纯训练期机制，不进入部署期主预测路径。

`models/scripts/train.py` 当前只保留以下实验入口：

| ID | 机制 | 说明 |
| -- | -------------------------- | ---------------------------------------------------------------------------- |
| B0 | none | **统一 clean baseline**，无 Teacher、无 Cache |
| R3A | bt_sam_rdt + SAM-only | **最小机制消融**：BT-SAM structural prior + 单个 Fast/EMA Teacher，关闭 OV 信息 |
| R3 | bt_sam_rdt + SAM+OV | **当前主方法**：BT-SAM structural prior + OV semantic prior + reliability fusion + 单个 Fast/EMA Teacher |

当前实现版本：

```text
implementation_version=bt_sam_rdt_run3_v1
cache_replay=aligned
routing_signal=pixel_positive_brier_gain
reject_unit=pixel
```

### Run3 BT-SAM-RDT 机制

Run3 的核心实现在：

```text
models/distill/task_space.py
models/distill/dynamic_teacher.py
models/distill/diagnostics.py
models/distill/teacher_cache.py
```

Run1 维护 SAM 与 OV 两个独立动态教师并进行像素级竞争/路由；Run3 删除该设计，先在教师上游构造一个真正的双时相 Foundation prior，再使用一个统一的 residual teacher：

```text
SAMStruct T1/T2
        ↓
显式跨时相实例对应
        ↓
BT-SAM structural-change prior ───────┐
                                      ├─ reliability fusion
OVCDistill semantic soft-change prior ┘
        ↓
Fused Foundation Prior
        ↓
One Fast Residual Teacher
        ↓ EMA
One Target Teacher
        ↓
GT Positive-Brier-Gain Audit
        ↓
Student KD
```

其中：

* **BT-SAM structural prior**：仅由 T1/T2 SAM instance、quality、boundary 构造，不读取 Student prediction、Student feature 或 GT；
* **OV semantic prior**：由 OVCDistill 的 `soft_change` / `confidence` 提供互补语义变化证据；
* **Fast Teacher**：参与梯度优化，根据当前 Student 状态与训练监督学习 residual correction；
* **EMA Target Teacher**：由 Fast Teacher 通过 EMA 更新，`requires_grad=False`，用于产生 Student 实际接收的蒸馏 proposal；
* **Student KD loss (`total`)**：只允许更新 Student；
* **Teacher fitting loss (`teacher_total`)**：只允许更新 Fast Teacher，Student feature / prediction 在该路径必须 detach；
* **Foundation prior**：Student-independent、GT-independent；
* **GT**：只允许用于 positive-gain safety audit、Fast Teacher fitting 与 detached Student-difficulty weighting。

Run3 不再维护两个互相竞争的 SAM/OV residual experts，也不再使用 Run2 的 regional routing、gradient-concordance gate 或 SCGR。

### BT-SAM 双时相结构先验

当前 `models/distill/task_space.py` 显式匹配 T1/T2 SAM instances。

对于 T1 instance `i` 与 T2 instance `j`：

```text
I_ij   = |M_i^1 ∩ M_j^2|

IoU_ij = I_ij / (|M_i^1| + |M_j^2| - I_ij)

C12_ij = I_ij / |M_i^1|

C21_ij = I_ij / |M_j^2|
```

核心原则：

1. SAM `instance_id` 仅视为单时相内部的 opaque ID，**不得跨时相直接按 ID 相减或 XOR**；
2. counterpart 按最大 IoU 选择；
3. IoU / directional coverage 用于匹配与可靠性分析，**不直接当作 pixel-level change probability**；
4. 匹配实例的重叠稳定 core → structural change ≈ 0；
5. T1-only 区域 → disappearance / shrinkage ≈ 1；
6. T2-only 区域 → appearance / expansion ≈ 1；
7. 非对应实例的重叠 → structural disagreement ≈ 1；
8. SAM quality、boundary 与 correspondence confidence 主要调节 reliability，而不是直接改写结构标签。

这是 Run3 相对旧实现的重要修正：expansion/shrinkage 的变化必须空间局部化，不能因为 directional coverage 不对称而给稳定 overlap core 分配非零变化概率。

### Foundation Prior Fusion

Run3 将 SAM structural prior 与 OV semantic prior 在 task space 中融合：

```text
q_prior =
    (R_S * S + R_O * O)
    /
    (R_S + R_O + eps)
```

其中：

```text
S   = BT-SAM structural prior
O   = OV semantic soft-change prior
R_S = SAM structural reliability
R_O = OV semantic reliability
```

当两种 prior 同时可用时，`|S - O|` 作为 conflict，降低 fused reliability；只有单一来源可用时不人为施加 disagreement penalty。

R3A 与 R3 使用相同的 residual teacher 容量和训练链：

```text
R3A: use_ov=False
R3 : use_ov=True
```

因此 R3 vs R3A 用于检验 OV semantic prior 在相同 Teacher 容量下是否提供增量信息。

### GT Positive-Brier-Gain Audit

对于 Student prediction `p`、Target Teacher proposal `q` 与 GT `y`：

```text
Student Brier error:   E_s = (p - y)^2
Teacher Brier error:   E_t = (q - y)^2
Teacher gain:          gain = E_s - E_t
```

当前正式 policy 为 `advantage`：

```text
reliability > 0
AND
gain > numerical tolerance
    → accept

otherwise
    → reject
```

GT audit 只做训练安全门，不参与 Foundation prior 构造；不得把该 audit 描述为 Foundation Teacher 本身的变化知识。

### Run1 / Run2 历史证据与当前解释

Run1 单 seed 2333 的历史结果曾出现：

| Dataset | B0 F1 | D1 F1 | ΔF1 |
| --- | ---: | ---: | ---: |
| SYSU | 80.81 | 83.23 | +2.42 |
| WHU | 94.03 | 94.21 | +0.18 |
| CDD | 97.79 | 97.80 | +0.01 |
| LEVIR | 91.27 | 91.16 | -0.11 |

但后续统一 clean B0 在 SYSU 达到约 83.24，与 Run1 D1 的 83.23 几乎一致，因此 Run1 的 SYSU +2.42 正信号存在明显 baseline/protocol 混杂，不能继续作为 Teacher 有效性的独立证据。

Run2/SCGR 又引入 regional routing、error-mass normalization 与 gradient-concordance gate，但 SYSU/WHU 等结果没有形成可信正增益，因此已停止。当前代码中不再保留 Direction-C、SCGR router、D1NG、D1/D2、region gate、gradient gate 等活动实现。

Run3 的研究问题因此收缩为：

```text
先证明：
经过显式双时相实例对应与结构变化重构后，
Foundation knowledge 本身是否能稳定帮助固定的极轻量 Student。

再讨论：
是否值得继续更复杂的 Teacher Agent / routing。
```

当前不能默认宣称：

```text
BT-SAM 一定有效
SAM + OV 一定互补
Foundation Teacher 一定优于 clean baseline
提升一定来自 Foundation prior 而非 residual teacher 容量
```

这些都必须由同协议正式实验回答。

## 5. 模型与部署约束

- 主学生模型是 A2Net-LWGANet-L0；SAM2 Teacher Cache、OV Cache、Fast Teacher、EMA Target Teacher 与所有训练辅助只允许训练期使用。
- 部署参数量固定为 `2,913,094`。
- 当前 `train.py` 的 256×256 部署 FLOPs 契约为约 `2.7475G`，允许工程校验误差 `±0.03G`。
- 打开或关闭训练辅助不得改变主预测；`switch_to_deploy()` 必须移除训练辅助参数，部署前后主预测最大误差 `< 1e-6`。
- 不得把 Teacher map / Foundation prior 注入部署期主特征路径。
- A/B/label 与 Teacher Cache 必须重放完全一致的 crop、resize、flip 和时相交换。
- Student optimizer 不得包含 `training_auxiliary.*` 参数；Fast Teacher 使用独立 teacher optimizer；EMA Target Teacher 不进入任何 optimizer。
- 实验语义集中在 `models/scripts/train.py`；wrapper 只负责选择实验、数据集、GPU 和运行目录。
- 当前 checkpoint 仅支持 **Run3 format v3**；不得静默恢复旧 format-v2 / Direction-C checkpoint。

当前核心代码：

```text
models/__init__.py
models/a2net.py
models/backbone/lwganet.py
models/decoder/a2net_decoder.py
models/datasets/cd_dataset.py
models/datasets/cache_transforms.py
models/datasets/transforms.py
models/distill/__init__.py
models/distill/dynamic_teacher.py
models/distill/diagnostics.py
models/distill/task_space.py
models/distill/teacher_cache.py
models/losses/combined_loss.py
models/scripts/train.py
models/utils/checkpoint.py
models/utils/logger.py
models/utils/metrics.py
models/utils/scheduler.py
models/tools/smoke_dynamic_teacher.py
models/tools/validate_teacher_cache.py
models/tools/export_deploy.py
```

`models/distill/losses.py`、旧 routing/Direction-C/SCGR 源码及 `__pycache__/*.pyc` 已从当前 `models/` 清理，不得因历史引用重新恢复，除非用户明确改变研究方向。

## 6. Teacher Cache 规则

- 根目录为 `/share_datasets/CD_teacher_cache/`，当前使用 SAMStruct + OVCDistill 两套 paired cache。
- SAM cache manifest 必须满足当前 `TeacherCache` schema，且 `teacher_type=sam2_struct_v2`。
- `PairedTeacherCache` 要求 cache entries 与当前 `list/train.txt` **精确全覆盖**：不得缺样本、不得多样本、不得重复 sample ID。
- 每个 cache sample 的 `sample_id` 与 `config_hash` 必须与 manifest 一致。
- SAM cache：
  - `t1/t2.instance_id` 为非负 int32/int64 `[1,H,W]`；
  - `boundary/quality` finite 且在 `[0,1]`；
  - T1/T2 三类字段空间 shape 必须一致。
- OV cache：
  - `soft_change` / `confidence` 包含 `l1/l2`，单通道且 bounded；
  - `relation` 仍属于 cache schema，但当前 Run3 task-space 主机制只使用 semantic soft-change / confidence；
  - 同一 level 内空间 shape 必须一致。
- R3A 与 R3 当前都走相同 paired-cache loading path，因此即使 R3A `use_ov=False`，`train.py` 仍要求同时提供 `--sam_cache_root` 与 `--ov_cache_root`；R3A 只是在 BTSAMRDT 内不使用 OV prior。
- 正式训练前先跑 `models/tools/validate_teacher_cache.py`；该工具是只读验证器，不得重建、改写或覆盖 Cache。
- 不得删除或覆盖数据集、Teacher Cache、checkpoint，除非用户明确授权并已核对绝对路径。
- Cache 文件可用 ≠ Teacher 有效；「Teacher 是否提升 Student」必须由 B0/R3A/R3 同协议正式对照回答。

## 7. 启动、恢复与结果纪律

推荐顺序：

1. 激活 `lsrep` 并进入项目目录；
2. 跑 `models/tools/smoke_dynamic_teacher.py`，验证 BT-SAM expansion/shrinkage、fusion、GT audit、梯度隔离、EMA、auxiliary ON/OFF、deploy 删除与参数契约；
3. 对计划训练的数据集跑 `models/tools/validate_teacher_cache.py`；
4. 用真实数据 + 真实 Cache 对最复杂的 `R3/SYSU` 做短 dry run，检查数据/缓存同步、loss、Teacher update、checkpoint/resume 与吞吐；
5. 用 Run3 v3 checkpoint 跑一次 `models/tools/export_deploy.py`，确认只导出 Student、部署参数正确；
6. 优先启动 SYSU / WHU 的 B0 → R3A → R3；有效性通过后再扩展 CDD / LEVIR；
7. 从 `train_log.txt` 收集正式 test 指标。

当前正式默认协议由 `models/scripts/train.py` 定义：

```text
input              256×256
batch_size         64
max_steps          40000
seed               2333
student lr         5e-4
student wd         1e-4
backbone lr mult   1.0
dice reduction     batch
main loss weights  1,1,1,1
kd_lambda          0.06
```

R3A / R3 默认 Teacher 参数：

```text
teacher_lr            5e-4
teacher_hidden        24
teacher_ema           0.99
max_logit_delta       2.0
teacher_weight_decay  1e-4
teacher_grad_clip     5.0
boundary_radius       2
small_area            64
```

每个实验目录使用：

- `last_checkpoint.pth`：Run3 format-v3 精确恢复；
- `best_model_F1=*.pth`：验证集选择出的最佳模型；
- `train_log.txt`：训练、恢复和最终测试记录。

Run3 v3 checkpoint 至少包含：

```text
format_version=3
model
optimizer
teacher_optimizer   # R3A/R3；B0 为 None
epoch
global_step
best_val_f1
args
rng
```

R3A/R3 的 Fast Teacher 与 EMA Target Teacher 已注册在 `model.training_auxiliary`，因此随 `model.state_dict()` 一并保存。

只有 `train_log.txt` 中最后一个完整：

```text
=== TEST RESULTS ===
...
=== END TEST RESULTS ===
```

区块才是正式结果。

结果规则：

- 正式指标只能来自最后一个完整 test block，不能使用验证集最佳行或 checkpoint 文件名替代。
- 报告 Recall、Precision、OA、F1、IoU、Kappa，以及训练/部署参数量和 FLOPs。
- 比较前检查 dataset、batch、max_steps、seed、Student 初始化、参数量、数据增强与评估协议是否一致。
- 单数据集、单种子、微小差异不能被表述为普适提升。
- seed 2333 只用于首轮有效性；论文级主比较需要补多 seed（当前计划 `2333/3407/5871`）。
- 当前 Run3 首要比较：
  - `B0 vs R3A`：BT-SAM structural prior + residual teacher 是否有独立增量；
  - `R3A vs R3`：OV semantic prior 在相同 Teacher 容量下是否提供额外增量；
  - `B0 vs R3`：完整 BT-SAM-RDT 是否在同协议下改善固定 Student。

## 8. Agent 工作规则

1. 修改前读取当前代码、Run3 README 与实际日志，不依赖旧项目记忆；若 Run3 README 与代码冲突，以当前代码/shell 为准。
2. 方法创新优先，但每项主张必须对应明确机制、与已有工作的实质区别、可证伪假设、最小消融和失败判据。
3. 保持部署主图、Teacher Cache replay 顺序、Student/Teacher 梯度隔离和 `switch_to_deploy()` 约束不变。
4. 不得重新引入旧 Direction-C、SCGR、Run1 双教师竞争、旧 SAM transport 或历史 router，除非用户明确要求重新开启该方向。
5. 修改 shell 后执行 `bash -n` 语法检查；修改后重新检查 shell 中 dataset/cache/pretrained/checkpoint/log 路径。
6. 修改模型后至少跑 synthetic smoke；涉及真实数据/Cache 时再跑只读 cache validation 与 real-cache dry run。
7. 修改 checkpoint/resume 逻辑时保持 Run3 format-v3、Student optimizer、teacher optimizer、model state、训练进度和 RNG 的精确恢复能力。
8. `models/scripts/train.py` 中旧 best checkpoint 的主动删除/替换策略是当前有意保留的行为；除非用户明确要求，不要擅自移除。
9. 不恢复已删除的 `models/distill/losses.py`、历史 routing 文件或任何 `__pycache__/*.pyc`；这些不属于当前 Run3 源码。
10. 不在 `cd_base` 中安装项目专属依赖，不擅自更换 PyTorch/CUDA 栈。
11. 任何删除数据、Teacher Cache 或 checkpoint 的操作都需要用户明确授权和精确路径校验。
12. Git 提交前必须检查暂存区；不得提交权重、Cache、数据集、checkpoint、训练日志、临时 dry-run 产物或 Python 字节码。
13. 当前仍处于有效性探寻阶段：不得因为模块来自强论文、Foundation Model 或单次实验有提升，就默认该机制适用于本项目。
14. 若 Run3 不能在同协议下证明 Foundation prior / Teacher 的有效性，应优先停止并回到证据诊断，而不是继续堆叠 router、loss 或额外模块。
