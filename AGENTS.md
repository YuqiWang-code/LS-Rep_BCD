# AGENTS.md — LS-Rep_BCD_RSML_3 / RDT-CD

> Last updated: 2026-09-17
> 本文件是当前项目中 AI/Coding Agent 的工作约束。当前唯一活动实验为 `train_scripts/RDT-CD/Run1/`（方向 C），主方法为 **RDT-CD（Reciprocal Dynamic Teachers，互惠动态教师）**；学生模型 A2Net-LWGANet-L0，部署参数 2,913,094 不变。**Run2/SCGR 已根据现有实验结果判定无效并归档，不再继续迭代**。项目长期处于「方法有效性探寻阶段」，主线结论随时可能被新实验推翻；不得把本文档中的任何结果当作已定稿的论文结论。不记录或继承旧实验（SAM-HSD、DART-R-TS 固定教师、旧 C0 梯度路由、RDT-CD Run2/SCGR 等已归档）。

## GitHub 提交流程

完成 `models/` 或 `AGENTS.md` 的修改并通过必要检查后，按以下顺序提交到 GitHub：

```bash
git add models AGENTS.md docs/experiment_metrics.xlsx "docs/RSML-3_服务器环境与变化检测数据统一说明.md"
git status
git commit -m "Update code"
git push
```

执行 `git commit` 前必须先检查 `git status`，确认暂存区只包含本次准备提交的文件；不要把权重、缓存、数据集、日志或无关改动加入提交。

## 研究定位与优先级

本项目服务于 211 高校硕士研究生的学术论文研究与实验验证，不以通用软件工程建设为主要目标。Agent 应将具有论文价值的**模型创新和方法性改进**置于最高优先级，重点关注网络结构、特征表征与交互、训练期辅助机制及其可解释性。

1. 优先提出机制明确、区别于现有模块且能形成论文贡献点的方法创新；每项方案都要说明创新动机、作用机制、与已有方法的差异、可证伪假设及最小消融实验。
2. 损失函数和超参数仅作为公平训练、稳定收敛与验证机制有效性的配套手段；除非有明确的新理论或新机制，否则不得把调参结果包装成核心创新。
3. 工程修改只需服务于实验正确性、可复现性和必要的运行效率。
4. 评价方案优先考虑学术新颖性、机制合理性、实验可验证性和论文叙事完整性，警惕无机理的模块堆叠与只追求单次指标的改动。

## 1. 作用域与权威来源

当前唯一活动实验族：

```text
train_scripts/RDT-CD/Run1/      # 方向 C：RDT-CD（B0 / D1 / D2）
```

（`train_scripts/RDT-CD/Run2/`（失败的 SCGR 迭代）、`train_scripts/SAM-HSD/`、`train_scripts/DART-R/` 的固定教师与旧 C0 梯度路由均为历史归档，不再活动。）

按以下优先级判断事实：

1. 当前代码与 shell 参数；
2. `train_scripts/RDT-CD/Run1/README.md`（当前实验协议）；
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
| RDT-CD checkpoint | `/share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/RDT-CD`（当前 Run1 在 `RDT-CD/Run1/<实验>/<数据集>/`） |
| RDT-CD 训练/测试日志 | `/home/yqwang/outputs/LS-Rep_BCD_RSML_3/RDT-CD`（当前 Run1 在 `RDT-CD/Run1/<实验>/<数据集>/`） |
| 预训练权重 | `/home/yqwang/projects/LS-Rep_BCD_RSML_3/pre-trained_weights` |

（旧 SAM-HSD / DART-R-TS 的 checkpoint 也在 `/share_datasets/yqwang/checkpoints/` 下，仅作归档，不再活动。当前 RDT-CD 实验目录不再使用 `steps_40000/seed_2333` 中间目录，这两项只写进训练日志配置头。）

`.vscode/sftp.json` 用于手动上传代码，自动上传保持关闭；该配置忽略 `pre-trained_weights/` 和常见权重文件。

## 4. RDT-CD（方向 C）

方向 C 主方法经历过多次迭代：C0（旧梯度路由）、C1（DART-R-TS 固定教师）均被判无效；Run1 进一步提出 **RDT-CD（Reciprocal Dynamic Teachers，互惠动态教师）**。Run2 曾在 RDT-CD 基础上加入 **SCGR（Sparse-Change Gradient-Concordant Routing）**，但现有实验结果不支持 SCGR 的有效性，因此 **Run2/SCGR 已停止并归档，当前方法恢复为 Run1 RDT-CD**。

学生模型始终为 A2Net-LWGANet-L0。RDT-CD 是纯训练期辅助机制，不进入部署期主预测路径。

`models/scripts/train.py` 当前保留以下实验入口：

| ID | 机制 | 说明 |
| -- | -------------------------- | ---------------------------------------------------------------------------- |
| B0 | none | **统一固定 clean baseline**，无教师 |
| D1 | dynamic_teacher + cache | **当前主方法**：完整 RDT-CD，使用 SAMStruct + OVCDistill Cache 作为教师先验 |
| D2 | dynamic_teacher + no-cache | 与 D1 保持相同在线动态教师容量，但关闭 Cache conditioning，用于验证 Foundation Teacher Cache 的增量价值 |

### RDT-CD 机制

RDT-CD 的核心实现在 `models/distill/dynamic_teacher.py`。

SAMStruct / OVCDistill 的 Cache 不再被视为固定且不可改变的最终教师答案，而是作为 **teacher priors（教师先验）**。在此基础上，引入两个小型在线 `ResidualTeacherExpert`：

```text
SAMStruct / OVCDistill Cache
        ↓
Teacher Priors
        ↓
Fast Residual Teacher Experts
        ↓ EMA
Target Teacher Experts
        ↓
Dynamic Teacher Proposals
        ↓
Task-space GT Audit
        ↓
Student Distillation
```

其中：

* **Fast Teacher**：参与梯度优化，根据当前学生状态和监督信号不断学习；
* **EMA Target Teacher**：由 Fast Teacher 通过 EMA 更新，不参与梯度优化，用于生成学生实际接收的动态 teacher proposal；
* **Student KD loss (`total`)**：只允许更新学生；
* **Teacher fitting loss (`teacher_total`)**：只允许更新 Fast Teacher，学生特征与预测在该路径中必须 detach；
* **EMA Target Teacher**：`requires_grad=False`，只通过显式 EMA 更新。

因此 RDT-CD 构成一个训练期互惠过程：

```text
Teacher → Student:
动态教师 proposal 蒸馏学生

Student → Teacher:
当前学生状态与错误用于训练 Fast Teacher

Fast Teacher → EMA Target Teacher:
通过 EMA 形成更稳定的动态教师
```

RDT-CD 的目标不是强迫学生持续模仿 Foundation Teacher，而是让 Foundation Teacher Cache 作为先验参与构造可演化教师，并在教师知识确实优于当前学生时才提供监督。

### Run1 Task-Space Routing

当前 `models/distill/task_space.py` 已恢复为 Run1 的 **pixel-wise GT-audited positive-gain routing**，不再使用 Run2 的 SCGR 区域路由。

对于学生预测 p、teacher proposal q 和 GT y，定义：

```text
Student Brier error:   E_s = (p - y)^2
Teacher proposal error: E_t = (q - y)^2
Teacher gain:          gain = E_s - E_t
```

仅当 `gain > numerical tolerance` 时，该 teacher proposal 才允许在当前像素参与 KD。即教师提案比当前学生更接近 GT → admit；否则 → reject。

SAM 与 OV 的有效 proposal 会按照 proposal confidence 与相对 improvement 形成 pixel-wise soft mixture，再用于 Bernoulli task-space distillation。

当前 Run1 task-space **不包含** 16×16 regional routing、regional utility、error-mass KD normalization、classifier-gradient concordance、gradient gate、D1NG、SCGR region diagnostics 等——上述机制均属于已失败的 Run2/SCGR，不再作为当前活动方法的一部分。

### Run2 / SCGR 失败记录

Run2 曾针对 Run1 在稀疏变化场景下的 teacher rejection 问题，引入 16×16 regional routing、error-mass normalized KD、gradient-concordance gate，正式实验含 B0 / D1NG / D1 / D2 四组（其中 D1NG 关闭 gradient-concordance gate）。

当前已有结果：

| Experiment | Dataset | F1 |
| --- | --- | ---: |
| B0 | SYSU | 83.24 |
| D1NG | SYSU | 82.67 |
| D1 | SYSU | 82.64 |
| B0 | WHU | 93.69 |
| D1NG | WHU | 92.70 |
| B0 | CDD | 97.80 |
| D1NG | CDD | 97.75 |
| D1 | CDD | 97.76 |
| B0 | LEVIR | 90.98 |
| D1NG | LEVIR | 91.05 |
| D1 | LEVIR | 91.12 |

D1NG 已关闭 gradient-concordance gate，但 SYSU −0.57、WHU −0.99 仍明显低于 B0。因此当前证据不支持「Run2 失败只是因为 gradient gate 太严格」，现阶段判断为 Run2 的整套 SCGR 改造没有获得有效性支持。Run2/SCGR 已停止，不继续调 region size、调 gradient threshold、增加新的 routing gate 或继续完成 D2。Run2 结果仅作为失败实验与研究诊断证据保留。

### Run1 历史结果及当前解释

Run1 单 seed 2333 的历史结果为：

| Dataset | B0 F1 | D1 F1 | ΔF1 |
| --- | ---: | ---: | ---: |
| SYSU | 80.81 | 83.23 | +2.42 |
| WHU | 94.03 | 94.21 | +0.18 |
| CDD | 97.79 | 97.80 | +0.01 |
| LEVIR | 91.27 | 91.16 | -0.11 |

Run1 曾在 SYSU 上出现明显正信号，但其余数据集收益很小或为负，不能表述为普适提升。另外，后续统一 clean B0 在 SYSU 达到 83.24，与 Run1 D1 的 83.23 几乎一致，因此 **Run1 的 SYSU +2.42 F1 正信号与后续 B0 波动存在明显混杂，不能再直接作为 Teacher 有效性的独立证据**。

当前研究决定是：固定统一 B0，不再跨不同 Run 使用不同 baseline 解释方法收益；重新在同一训练协议下只比较 B0 与 Run1 RDT-CD D1，直接回答「Teacher 结构本身是否有效」。

因此当前方向 C 的状态应表述为：**Run2/SCGR 已无可信正信号并停止；Run1 RDT-CD 已恢复为当前活动教师机制，但 Teacher 是否真正有效仍需在统一固定 B0 下通过 D1 vs B0 重新验证。**

## 5. 模型与部署约束

- 主学生模型是 A2Net-LWGANet-L0；SAM2 Teacher Cache 及所有训练辅助只允许训练期使用。
- 推理参数量保持 `2,913,094`；256×256 FLOPs 实测约 `2.7676G`（`±0.03G` 容差）。
- 打开或关闭训练辅助不得改变主预测；`switch_to_deploy()` 必须移除训练辅助参数，部署前后最大误差 `< 1e-6`。
- 不得把 Teacher map 注入部署期主特征路径。
- A/B/label 与 Teacher Cache 必须重放完全一致的 crop、resize、flip 和时相交换。
- 实验语义集中在 `models/scripts/train.py`；wrapper 只负责选择实验、数据集、GPU 和运行目录。

核心代码：

```text
models/a2net.py
models/backbone/lwganet.py
models/decoder/a2net_decoder.py
models/datasets/cd_dataset.py
models/datasets/cache_transforms.py
models/distill/dynamic_teacher.py
models/distill/diagnostics.py
models/distill/task_space.py
models/distill/losses.py
models/distill/teacher_cache.py
models/losses/combined_loss.py
models/scripts/train.py
models/tools/smoke_dynamic_teacher.py
models/tools/validate_teacher_cache.py
models/tools/export_deploy.py
```

## 6. Teacher Cache 规则

- 根 `/share_datasets/CD_teacher_cache/`，两套 teacher：SAMStruct（`sam2_struct_v2`）与 OVCDistill（DINOv2 ViT-B/14），四数据集 train 全覆盖。
- 先跑 `validate_teacher_cache`；验证通过不重建缓存。
- 不得删除或覆盖数据集、Teacher Cache、checkpoint，除非用户明确授权并已核对绝对路径。
- **有效性状态**：缓存文件本身可用，但「teacher 能否提升主模型」仍需 D1 vs B0 同协议对照验证；在得到证据前，不得把 teacher 描述为有效组件。

## 7. 启动、恢复与结果纪律

推荐顺序：

1. 激活 `lsrep` 并进入项目目录；
2. 跑 `smoke_dynamic_teacher`（CUDA 契约测试，覆盖 D1 / D2 及部署删除、梯度隔离、EMA 更新等关键路径）；
3. 做一次 dry run（真实数据 + Cache 的短训练，验证全链路后删临时目录）；
4. 启动当前 Run1 RDT-CD 队列（双卡 tmux）；
5. 从 `train_log.txt` 收集正式 test 指标。

每个实验目录使用：

- `last_checkpoint.pth`：精确恢复（脚本检测到则自动加 `--resume`）；
- `best_model_F1=*.pth`：验证集选择出的最佳模型；
- `train_log.txt`：训练、恢复和最终测试记录。

只有 `train_log.txt` 同时出现 `=== TEST RESULTS ===` 与 `=== END TEST RESULTS ===` 才算完成。

结果规则：

- 正式指标只能来自最终 test block，不能使用验证集最佳行或 checkpoint 文件名替代。
- 报告 Recall、Precision、OA、F1、IoU、Kappa，以及训练/部署参数量和 FLOPs。
- 单数据集、单种子或阈值以下差异不能被表述为普适提升。

## 8. Agent 工作规则

1. 修改前读取当前代码、Run1 README 与实际日志，不依赖旧项目记忆。
2. 方法创新优先，但每项主张必须对应明确机制、可证伪假设和最小消融。
3. 保持推理图、Teacher Cache replay 顺序和部署约束不变。
4. 修改 shell 后执行 `bash -n` 语法检查。
5. 修改模型后至少跑 smoke test；涉及数据或缓存时再跑真实 Cache 校验。
6. 不恢复已归档的旧实验模块、路径、指标或环境配置。
7. 不在 `cd_base` 中安装项目专属依赖，不擅自更换 PyTorch/CUDA 栈。
8. 任何删除数据、缓存或 checkpoint 的操作都需要用户明确授权和精确路径校验。
