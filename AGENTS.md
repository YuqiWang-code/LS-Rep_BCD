# AGENTS.md — LS-Rep_BCD_RSML_3 / RDT-CD

> Last updated: 2026-09-15
> 本文件是当前项目中 AI/Coding Agent 的工作约束。当前唯一活动实验是 `train_scripts/RDT-CD/Run2/`（方向 C），主方法为 **RDT-CD + SCGR（Sparse-Change Gradient-Concordant Routing，稀疏变化梯度一致路由）**；学生模型 A2Net-LWGANet-L0，部署参数 2,913,094 不变。项目长期处于「方法有效性探寻阶段」，主线结论随时可能被新实验推翻；不得把本文档中的任何结果当作已定稿的论文结论。不记录或继承旧实验（SAM-HSD、DART-R-TS 的固定教师、旧 C0 梯度路由、RDT-CD Run1 等已归档）。

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
train_scripts/RDT-CD/Run2/      # 方向 C：RDT-CD + SCGR（B0 / D1NG / D1 / D2）
```

（`train_scripts/RDT-CD/Run1/`（旧 RDT-CD）、`train_scripts/SAM-HSD/`、`train_scripts/DART-R/` 的固定教师与旧 C0 梯度路由均为历史归档，不再活动。）

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
| RDT-CD checkpoint | `/share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/RDT-CD`（Run2 在 `RDT-CD/Run2/<实验>/<数据集>/`） |
| RDT-CD 训练/测试日志 | `/home/yqwang/outputs/LS-Rep_BCD_RSML_3/RDT-CD`（Run2 在 `RDT-CD/Run2/<实验>/<数据集>/`） |
| 预训练权重 | `/home/yqwang/projects/LS-Rep_BCD_RSML_3/pre-trained_weights` |

（旧 SAM-HSD / DART-R-TS 的 checkpoint 也在 `/share_datasets/yqwang/checkpoints/` 下，仅作归档，不再活动。Run2 起不再有 `steps_40000/seed_2333` 中间目录，这两项只写进训练日志配置头，后续实验沿用。）

`.vscode/sftp.json` 用于手动上传代码，自动上传保持关闭；该配置忽略 `pre-trained_weights/` 和常见权重文件。

## 4. RDT-CD（方向 C）

方向 C 主方法经历迭代：C0（梯度路由）、C1（DART-R-TS 固定教师）均被判无效（固定教师很快被学生超越、教师有效贡献趋近于零），Run1 迭代为 **RDT-CD（Reciprocal Dynamic Teachers，互惠动态教师）**，Run2 在 RDT-CD 之上加入 **SCGR（Sparse-Change Gradient-Concordant Routing，稀疏变化梯度一致路由）**。学生模型仍叫 A2Net-LWGANet-L0，RDT-CD/SCGR 都是训练期机制。`models/scripts/train.py` 注册以下入口：

| ID | 机制 | 说明 |
|---|---|---|
| B0 | none | clean baseline，无教师 |
| D1NG | dynamic_teacher + SCGR / 无梯度门 | RDT + Cache + 区域/误差质量路由，**关**梯度一致门 |
| D1 | dynamic_teacher + SCGR | **当前主方法**（RDT + Cache + 完整 SCGR） |
| D2 | dynamic_teacher + SCGR / 无 Cache | 同结构在线教师 + 完整 SCGR，去 Cache 先验的负对照 |

**RDT-CD 机制**（`models/distill/dynamic_teacher.py`）：SAMStruct / OVCDistill 的 Cache 不再是固定教师目标，而是**教师先验**；两个小型在线 `ResidualTeacherExpert`（fast，零初始化、输出有界 logit 修正）随训练优化，EMA 复制为 target teacher 生成学生看到的动态提案。学生与教师**互相更新**：教师→学生蒸馏（`total`，只更新学生）、学生误差→教师拟合（`teacher_total`，只更新 fast teacher）。

**SCGR（Run2 新增，`models/distill/task_space.py` 的 `task_space_route`）**：针对 Run1 在稀疏变化数据集上 image 级整图拒绝率过高、KD 梯度与 GT 反相的问题，把逐像素 Brier 增益路由升级为——(1) **16×16 区域路由**（避免稀疏有用变化被图像面积淹没）；(2) **误差质量归一化 KD**（按区域误差质量而非面积加权）；(3) **梯度一致门**（教师区域提案仅在分析分类器梯度与 GT 正 concordance 时准入）。部署图完全删除辅助，参数 2,913,094 不变，`implementation_version=rdt_scgr_v2`、`scgr_region_size=16`（固定，不暴露为 CLI 超参）。

**Run2 消融（当前活动）**（`train_scripts/RDT-CD/Run2/README.md`）：4 组（B0 / D1NG / D1 / D2）× 4 数据集 = 16 run，batch 64、40k、seed 2333；双卡并行——GPU 0 跑 SYSU+WHU、GPU 1 跑 CDD+LEVIR，各卡内 B0→D1NG→D1→D2 串行。判定：D1 vs B0 = 完整 SCGR 整体有效性；D1 vs D1NG = 梯度一致门的净贡献（唯一变量是门开关）；D1 vs D2 = Cache 先验的增量。关键健康信号看 `scgr_region_accept_ratio`/`scgr_gradient_conflict_ratio`（梯度门是否起作用）、`effective_per_error_mass`（不应过早坍缩）、`teacher_ema_update_norm`/`teacher_target_gap`（教师是否持续演化）。

**Run1 历史结论（已归档，单 seed 2333）**：D1 相对 B0 平均 ΔF1 ≈ +0.63，其中 SYSU +2.43 F1 / +3.49 IoU 是方向 C 首个可信正信号；WHU +0.19、CDD ≈0、LEVIR −0.11；Cache 先验有增量（D1 vs D2：SYSU +0.60、WHU +0.64，WHU 的 D2 纯在线教师反而 −0.45 被 Cache 拉回）。Run1 失效诊断（见 `docs/temporary/RDT-CD_教师演化与失效场景分析_2026-09-15.md`）：CDD 近天花板无 headroom、LEVIR 先验无信息、WHU 先验有信息但 change 稀疏致 image 级整图拒绝率 55%~75%——这正是 SCGR 要解决的三个点。**Run2（SCGR）当前训练中，正式结果待出**。

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
2. 跑 `smoke_dynamic_teacher`（CUDA 契约测试，覆盖 D1 / D1NG / D2 三条路径）；
3. 做一次 dry run（真实数据 + Cache 的短训练，验证全链路后删临时目录）；
4. 启动 Run2 队列（双卡 tmux）；
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
