# README.md — LS-Rep_BCD_RSML_3

> Last updated: 2026-10-09
> 当前主线：**CATA-CD v2 —— Capability-Validated Adaptive Teacher Agent for Lightweight Change Detection**。
> 完整方案见 `docs/temporary/CATA-CD_v2_扩展教师池_验证驱动Agent_完整方案_20261005.md`；
> 负结果复盘与下一轮可证伪方案见 `docs/temporary/CATA-CD_v2_负结果严格复盘_下一轮可证伪迭代方案_GitHub最新_20261008.md`。
> 项目长期处于「方法有效性探寻阶段」，主线结论随时可能被新实验推翻；不得把本文档中的任何结果当作已定稿的论文结论。
>
> **2026-10-08/09 复核后的状态**：教师选择主线经 J1 判决**未通过 Gate R1**（见
> 「Run2」一节），论文主轴已按方案 §11.1 转向**不依赖教师的 S-PCG 结构创新**。
> 教师相关的 Capability Registry / Cache / probe 保留为**诊断资产**，不再作为主创新。

## 任务与硬目标

- 任务：推理时**极轻量化**的遥感图像**全监督二值变化检测**（不涉及半监督/多类/语义分割）。
- 数据集（固定 4 个，256×256）：SYSU-CD、LEVIR-CD、WHU-CD、CDD-CD。
- **最终硬目标（必须同时全部达到）**：
  - 四数据集 F1：SYSU ≥ 85、LEVIR ≥ 92.5、WHU ≥ 95、CDD ≥ 98（IoU 与 F1 同方向）；
  - **有效推理参数 < 5M**。

## 主线（CATA-CD v2）

1. 先把每个候选基础模型做成独立 **Teacher Package**，在四数据集上用完全相同的学生、初始化、40K steps、seed=2333 逐一验证（`train_scripts/CATA-CD/teacher_adaptation/`）。
2. 只有被实验证明「至少在部分数据集产生正迁移」的教师包才进入 Teacher Capability Registry。
3. 用带 **None/拒绝动作**的离线上下文 Bandit 学习「数据集签名 + 教师 train-only probe → 教师包/None」的选择策略（LODO 防止变成 4 行查表）。
4. 把一个约 **0.37M 的对称 Deployable Change Adapter（DCA）**固化进学生部署图，使最终模型约 3.28M，推理完全不访问任何大模型。

**教师—数据集适配关系全部由实验决定，不预设结论。** 最终 Agent 候选动作只保留 3–5 个互补且被证明有效的 Teacher Package + None。

> **⚠️ 第 1–3 步的当前状态（2026-10-09）**：第 1 步完成（44 个 40K run），
> 第 2 步完成（Registry），**第 3 步经 J1 判决失败**——教师软目标在同等辅助容量下
> **无独立价值**（`J1-TASKKD − J1-GATE = −0.18pp`，见 Run2 一节）。
> 因此**第 3 步不再作为论文主创新**；第 4 步的 DCA 及其 S-PCG 扩展成为主轴。
> 教师包与 Registry 仍保留，用于解释"为什么强教师难以蒸馏"这一诊断性结论。

## 实验约定（固定）

- **永远不做多 seed**：每个机制只跑一组同协议（seed 2333、batch 64、**40000 steps**）消融表；不补 seed、不做显著性检验。
- **从头训练纪律**：主实验与全部消融一律从头训练（同一 ImageNet 预训练 `lwganet_l0_e299.pth` + 固定 seed），禁止用任何已有 checkpoint 微调/续训作为实验组。
- 唯一变量：C0（clean A2Net）/ C1（C0+DCA）对照 + TV-*（C1+教师包）/ M1 主实验。
- 教师效用 = `TV-* − C1`，逐数据集、逐指标报告；正式结果只读 `train_log.txt` 最后一个完整 `=== TEST RESULTS ===` 区块。

## Run1 结果（teacher_adaptation，44/44 完成）

教师能力矩阵 `U[D,T] = TV-* − C1`（F1 百分点，seed 2333 / 40K / batch 64）：

| 数据集 | C0 F1 | C1 F1 | SAM | D2 | D3N | D3S | RCLIP | MARS | ANYSAT | UNISAT | RADIO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SYSU | 82.13 | **82.57** | −0.28 | −0.12 | −0.24 | −0.67 | −0.81 | −0.03 | −0.66 | −0.27 | **+0.68** |
| WHU | 92.66 | **94.04** | −1.15 | −0.36 | −0.21 | +0.08 | +0.08 | **+0.56** | −0.13 | **+0.40** | −0.32 |
| CDD | 97.79 | **97.75** | −0.01 | −0.02 | −0.05 | −0.04 | −0.01 | −0.02 | −0.01 | +0.00 | −0.01 |
| LEVIR | 90.92 | **91.04** | +0.06 | −0.22 | +0.08 | +0.14 | −0.07 | −0.06 | +0.06 | −0.18 | −0.00 |

部署参数：C0 = 2,913,094（2.7676G FLOPs）；C1/TV-* = 3,280,274（3.2432G FLOPs）；swap/deploy 误差 = 0。

**结论（单 seed，不构成定稿结论）**

1. **DCA 是主增益来源**（C1−C0）：WHU +1.38 / SYSU +0.44 / LEVIR +0.12 / CDD −0.04（H3 初步成立）。
2. **H1 成立**：最优教师随数据集变化（SYSU→RADIO，WHU→MARS/UNISAT，LEVIR→D3S，CDD→无）。
3. **Strong Positive（ΔF1 ≥ +0.20 且 ΔIoU > 0）仅 3 组**：RADIO→SYSU (+0.68/+0.99)、MARS→WHU (+0.56/+1.01)、UNISAT→WHU (+0.40/+0.71)。
4. **教师效用强数据集依赖、多数未观察到强正**（36 组中 **26 组 ΔF1 ≤ 0、33 组未达强正阈值**；10 组 ΔF1 > 0）→ 实证支持「带 None/拒绝动作的选择 Agent」这一核心论点。（早期表述「30 组非正」为计数错误，已按 `registry_v3` 更正。）
5. CDD 已饱和（97.75），所有教师均无增益。
6. **硬目标未达**：SYSU 83.25/85、LEVIR 91.17/92.5、WHU 94.60/95、CDD 97.76/98。

## Stage 3/4（Registry + LODO Agent + M1）

- **Registry**（`outputs/CATA-CD/registry/`）：`research_report.json`（test 效用，仅论文证据）与 `agent_train_registry.json`（**仅 val 效用**，Agent 输入，无 test 泄漏）严格分离；另有 4 个 dataset signature + 36 个 teacher train-only probe。
- **Agent（A-LODO：tiny MLP + LODO，state = dataset signature + teacher probe）选择**：SYSU→None、WHU→None、CDD→dinov3_lvd、LEVIR→anysat。
- **H2 基本成立**：2/4 fold 与 oracle-val 精确一致，3/4 fold gap ≤0.20pp（WHU 0.221pp 略超）；明显优于基线 A-F（固定教师，gap 最高 +1.84pp）与 A-H（启发式，+0.86pp）。
- **M1 结果**（按 Agent 选择从头训练，2-way）：SYSU 82.76 / WHU 93.01 / CDD 97.74 / LEVIR 90.99 → `M1 − C1` = [+0.19, −1.03, −0.01, −0.05] → **H4 不成立**。

## ⚠️ 关键方法论发现：单 seed 结果不可复现（噪声 > 效应）

M1/SYSU 与 M1/WHU 的配置与 C1 **逐字段完全相同**（`dca_mode=moe128`、`teacher_package=none`、`seed=2333`、`batch=64`、`40K`、同一预训练权重），但结果分别为 **82.76 / 93.01**，而 C1 为 **82.57 / 94.04** → **同配置差异达 +0.19 / −1.03pp F1**。

- 训练曲线**从第 0 个 epoch 即分叉**（SYSU epoch0 loss 3.5051 vs 3.4897；WHU 5.2305 vs 5.2067）。
- 原始 C1 以 **3-way 并发**运行、M1 以 **2-way** 运行；并发度不同（`nn.SyncBatchNorm` backward 归约 + cuBLAS/cuDNN 算法选择 + best-val checkpoint 选择放大）是主要嫌疑。
- WHU 上"同配置"run 的 test F1 散布达 **1.9pp**（C0 92.66 … TV-MARS 94.60）。

**后果**：单 seed 的 run-to-run 波动与教师效应**同量级**，因此上面的能力矩阵、Agent 选择与 M1 差异**不能直接当作机制有效性的证据**，必须先测出每个数据集的 σ。

### 噪声底结果（已完成：C1 配置 × R1/R2/R3，12 runs，均 3-way，与 C1 同并发，test F1）

| 数据集 | C1 | R1 | R2 | R3 | **σ_3way** | 极差 |
|---|---:|---:|---:|---:|---:|---:|
| SYSU | 82.57 | 82.22 | 82.91 | 82.42 | **0.289** | 0.684 |
| WHU | 94.04 | 94.64 | 93.42 | 93.63 | **0.537** | 1.220 |
| CDD | 97.75 | 97.75 | 97.76 | 97.73 | **0.013** | 0.031 |
| LEVIR | 91.04 | 90.69 | 91.13 | 91.07 | **0.195** | 0.432 |

平均 σ = **0.259pp**，**随数据集强烈变化**（CDD 0.013 ↔ WHU 0.537）；并发度只在 WHU 上再加一点（σ_mixed 0.623）→ 主因是**内在 run-to-run 噪声**，不是并发。

**效应/σ 判读（⚠️ 仅为量级参考，不是显著性）**：σ 是**同一配置跨运行的 F1 标准差**，不是 `TV−C1` 差值的标准误；它没有覆盖教师臂自身方差、配对相关性、checkpoint 选择偏差与 9 教师×4 数据集的多重比较。因此只能把它当作粗噪声底：

| 观察 | ΔF1 | σ | ΔF1/σ（**仅为量级比，不是检验统计量**） |
|---|---:|---:|---|
| DCA→WHU（C1−C0） | +1.38 | 0.537 | 2.6 |
| RADIO→SYSU | +0.68 | 0.289 | 2.4 |
| MARS→WHU | +0.56 | 0.537 | 1.0 |
| D3S→LEVIR | +0.14 | 0.195 | 0.7 |
| CDD 全部教师 | ≈0 | 0.013 | ~0 |

→ **结论修正（措辞收紧）**：这些差值**都还不能称为已证实的机制效应**，只能表述为"**RADIO→SYSU 与 DCA→WHU 是值得优先复核的观察**"；WHU 上的教师效应处于噪声量级；CDD 为"**同协议单 run 未观察到可信正增益**"（不说"确认真无效应"）。正式判据见下一节的 Agent-only 决策实验。

## Stage 4（v2）：U-SafeAgent 与 Agent-only 消融

按 `docs/temporary/CATA-CD_v2_Agent结构改进_最新GitHub审查_完整科研方案_20261008.md` 实现：

- **P0 修复**：`build_capability_registry` 改取**最后一个完整 TEST block**；registry 改**严格枚举**且 Agent 侧读取遇 `test_*` 字段直接 `PermissionError`；`train.py` 分开上报 `Student/Aux/Train/Infer Params`（此前 TV-* 的 "Train Params" 漏掉了 KD 头 8,320 参数）；`dataset_signature` 新增 **schema v2**（20 维：修 `n_components` 零连通域、Sobel 用梯度幅值、面积分位改全局汇总，新增 empty-pair ratio / boundary density / change-area CV / hard pseudo-change / illumination-unchanged），**v1 的 18 维保留不覆盖**。
- **新 reward**：`ΔF1_val(pp)`（弃用 `ΔF1+0.5ΔIoU` 的复合奖励，避免同一检测效应重复计量）。
- **新 Agent**：`models/agent/safe_selector.py` —— 共享**加权 ridge**（按数据集簇均衡）+ **域级 jackknife** 模型不确定性代理 + **安全集** `S(D)={T: μ−κu>δ}`，`δ=+0.20pp / κ=1`，集合为空即 `None`。配套 `teacher_metadata.py`（5 个 test-free 元数据字段）与 `train_safe_teacher_agent.py`（LODO + 诊断 + 冻结动作）。
- **新 probe**：`capability_probe.py` 增加 **P1–P5**（balanced AUROC / hard-negative separation / boundary contrast / size-conditioned AUROC gap / unchanged-leakage ratio），`probe_teacher_capability.py` 改**分层采样**（按变化比率三分位）+ **CDD 路径 fallback** + 记录采样 SHA。

### Agent-only 消融结果（4 折 LODO，held-out **val** reward，pp）

| 臂 | 含义 | harm↓ | false act.↓ | coverage | pos-recall |
|---|---|---:|---:|---:|---:|
| AG-00 | legacy MLP，`max>0` | 0.00–0.50 | 0.25–0.75 | 0.25–1.00 | 0.5 |
| AG-01 | ridge，`μ>0` | 0.00–0.25 | 0.25–0.75 | 0.25–0.75 | 0.0 |
| AG-02 | ridge + `δ=0.20pp` | **0.00** | 0.00–0.25 | 0.00–0.25 | 0.0 |
| **AG-03** | **U-SafeAgent（主方法）** | **0.00** | **0.00** | **0.00** | 0.0 |
| AG-05 | AG-04 + metadata | 0.00 | 0.00 | 0.00 | 0.0 |
| AG-CF / AG-CH | 公平固定教师 / 启发式 + None | 0.00 | 0.00 | 0.00 | 0.0 |

（范围为 5 个特征阶梯 `sig_only → sig_probe → sig_probe_ext → sig_probe_ext_meta → legacy` 的取值区间。val σ：SYSU 0.411 / WHU 0.311 / CDD 0.013 / LEVIR 0.061 pp；oracle-val：SYSU −0.09 / WHU +0.12 / CDD −0.00 / LEVIR +0.15 pp。）

> **量纲更正（2026-10-08 复核）**：早期表述"所有 val 效用在 ±0.15pp 内"**不成立**。
> 正确的是：**正向 val 效用 ≤ +0.146pp，但负向可达 −1.077pp**（最差为 AnySat→SYSU）。
> 36 组 val 效用中 **6 正 / 30 负**。这不改变"无法学到选择策略"的结论（δ=+0.20pp 仍高于任何
> 正向标签），但"效应都很小"的说法只对**正向**一侧成立。

**判定：命中方案 §9.4 的 F0「机制不可识别」。** AG-03 在**全部 5 个特征阶梯、全部 4 折都选择 `None`**——val 噪声（0.013–0.41pp）不小于待判效应（最大 +0.15pp），`μ−κu>δ` 无法满足。按方案 §1.1 的可证伪边界，**不能为了"让 Agent 动起来"而降低 δ**。

→ 结论：**Teacher Selection Signal Insufficiency（拒答正确，但尚无可用的选择策略）**。据此当时**未启动 M2**（方案 §9.2 要求的 gate 未通过：无 ≥1 折选到 positive reward 的非 None 动作）。

> **后续（2026-10-08/09）**：按上述复盘方案执行了 **Run2（J0 + J1）**，
> 结论是教师软目标**无独立价值**（Gate R1 失败）→ 论文主轴转向 S-PCG。
> 详见「Run2」一节。此处"不启动 Run2"仅指当时（Stage 4 v2）的判断。

**剩余缺口（硬目标，未变）**：SYSU 82.57/85（−2.43）、LEVIR 91.04/92.5（−1.46）、WHU 94.04/95（−0.96）、CDD 97.75/98（−0.25）——即 44 个单教师 run 中最大的 test 差也只有 +0.68pp，现有证据并不显示仅靠"选一个教师"可达四硬目标。

## Run2（2026-10-08 复核后的下一轮：J0 可重复性 + J1 机制判决）

按 `docs/temporary/CATA-CD_v2_负结果严格复盘_下一轮可证伪迭代方案_GitHub最新_20261008.md` 执行。
代码与脚本见 `train_scripts/CATA-CD/Run2/`（README 顶部含预注册 Gate R0–R3）。

### P0 修复（均已验证）

| 项 | 问题 | 修复与验证 |
|---|---|---|
| P0-A | `AG-CH` 用留出集**真实 val 收益**决定是否拒答 | 选择与门控改为**只用训练域**；新增 `--verify_no_leak`（置换每折留出标签）→ **7 臂 × 5 阶梯动作逐个不变 = PASS** |
| P0-B | `_anchors.C0_test_pp` 仍是 0–1 量纲 | 统一走 `to_pp()`；`0.821332 → 82.1332`；**C1 锚点与 36 个教师 delta 逐字节不变**；新增防覆盖护栏与新 `registry_v3/` |
| P0-E | `u_noise` 读了留出数据集自己的 σ | 拆 `--noise_mode {inductive,transductive}`；inductive 下每折的 σ 来源恰为**另外三个域** |
| P2-A | `len(X) != len(y) != len(groups)` 链式比较 | 改为显式三方相等；拒绝返回"零 spread"假确定性；12 个单测 |
| P1 | `argsort(argsort())` 非 ties-aware Spearman | 改 `scipy.stats.spearmanr` + **置换零分布**；去掉无意义的 p 值 |

**P1 结论收紧**：置换零分布 95% 区间为 **±0.67**，**所有 probe 的跨域 macro 相关都落在其中**（含最大的 `size_conditioned_auroc_gap` = −0.34），跨教师 `ρ(meanAUROC, meanValΔ) = +0.117` 亦在 [−0.68, +0.67] 内。
→ 正确表述是"**在 9 教师/域这一样本量下检不出关系**"，**不是**"证明教师能力与可蒸馏性独立"。

### J0：同配置不可复现的机制已定位（`models/tools/audit_reproducibility.py`）

**初始化、数据流、RNG 与一次前向——全部逐位一致**；不一致的是**反向传播**：

| 配置 | step-0 `grad_norm` 相对差 | 187 步后 loss 差 |
|---|---|---|
| A TF32 开（生产协议） | 1.9e−4 | −1.04e−2 |
| B + `CUBLAS_WORKSPACE_CONFIG` | 4.6e−4（**无效**） | — |
| C + 关闭 TF32 | **1.6e−6（小 113×）** | **+4.64e−2（更大）** |
| D + `use_deterministic_algorithms(True)` | **运行失败** | — |

- D 失败原因：**`MaxUnpool2d` 在 PyTorch 中没有确定性实现**（LWGANet-L0 的 decoder 使用它）
  → **本架构无法获得逐位确定性**。
- **把 step-0 扰动压小 113 倍后，一个 epoch 后的分歧反而更大** → **训练是混沌的**，
  微小数值差异会被指数放大。

**对 Gate R0 的判定**：R0 字面要求的"可重复初始化 + 数据序列"**通过**；
"同配置 40K 可复现"**不通过且无可用开关可修**。
→ Run2 的 40K 结果只能按**单次抽样**解读，**不得**把 <约 1–2pp 的差值归因于方法差异。

### σ 必须上调：Run2 出现了更大的同配置散布

Run2 的 `S0-C1` 与 J1 的 `J1-C1` **逐字段同配置**（`moe128` / 无教师 / 无辅助头 / seed 2333 / 40K / 同预训练），
但在 SYSU 上跑出 **83.1391 vs 81.8087 → 相差 1.33pp**。

| 数据集 | 同配置 run（F1） | n | **σ_old（n=4）** | **σ_new** | τ=max(0.2, 2√2σ) |
|---|---|---:|---:|---:|---:|
| SYSU | 82.57 / 82.22 / 82.91 / 82.42 / **81.81** / **83.14** | 6 | 0.291 | **0.478** | 0.82 → **1.35** |
| WHU | 94.04 / 94.64 / 93.42 / 93.63 / **93.23** | 5 | 0.537 | 0.561 | 1.52 → 1.59 |

- SYSU σ 的 95% χ² 区间（n=6）为 **[0.298, 1.173]**——即**旧的 n=4 估计 0.291 几乎落在下界**，
  说明先前把 σ_SYSU 当成 0.29 是**低估值**。
- 直接影响：**Gate R1 在 SYSU 的正确阈值应是 ≈1.35pp 而非 0.82pp**。
  R1 的结论（−0.18pp）在两种阈值下都失败，**判决不变**；
  但**Gate R2 必须按 1.35pp 判读**。
- 教训：n=4 的 σ 估计不可靠（方案 §6.1 已警告过 CI 达 `[0.566, 3.729]×s`），
  **每新增一次同配置 run 都应重算 σ**。

### J1：教师软目标无独立价值（SYSU，四臂 40K，全部完整 TEST block）

| 臂 | Aux Task | F1 | IoU | Infer Params |
|---|---|---:|---:|---|
| J1-C1 | none | 0.818087 | 0.692171 | 3.2803M |
| J1-GT | gt（1ch GT 辅助头） | 0.822754 | 0.698881 | 3.2803M |
| J1-GATE | gate（+ 教师一致区域筛样） | **0.824470** | **0.701361** | 3.2803M |
| J1-TASKKD | taskkd（target 换成教师软响应 `q_T`） | 0.822657 | 0.698740 | 3.2803M |

| 唯一变量对照 | ΔF1 | ΔIoU |
|---|---:|---:|
| J1-GT − J1-C1（加 GT 辅助头） | +0.467 | +0.671 |
| J1-GATE − J1-GT（**只在 teacher∩GT 一致区域监督**） | +0.172 | +0.248 |
| **J1-TASKKD − J1-GATE（把 GT 目标换成教师软目标）** | **−0.181** | **−0.262** |

**Gate R1 失败**（要求 > +0.82pp 且 ΔIoU 同向）。关键对照是**同一张卡上的先后两次**，无跨卡混淆。
按方案的判决分支：**teacher 只贡献了"可靠区域筛样"，soft representation 无独立价值**
→ 不再把"知识蒸馏"当作主创新。四臂部署侧完全一致（3.2803M / 3.2432G / swap·deploy error 0）。

**train-only 校准诊断**（SYSU/dinov3_lvd，512 样本）与上述结论自洽：

| 量 | 值 |
|---|---|
| balanced AUROC(`q_T`, GT) | 0.852 |
| **unchanged leakage @`t_plus`** | **9.62%** |
| **AUROC boundary / interior** | **0.533 / 0.861** |
| AUROC by size（small/medium/large） | 0.748 / 0.810 / 0.848 |
| `t_plus` / `t_minus` | 0.6412 / 0.6462（**门槛几乎无分离**） |
| ECE raw → affine refit | 0.342 → 0.064 |

即：教师响应在**边界上≈随机**（强支持 H4），且其"可信变化"门槛并不比"可信未变化"更高——
这解释了为什么把它当软目标没有额外信息。

### S-PCG 阶梯（已完成：**Gate R2 失败**）

`moe128`（S0 legacy 全局 gate）保持**逐参数不变**（3,280,274），新增两个 mode：

| mode | 臂 | gate 输入 | 参数量 |
|---|---|---|---:|
| `moe128` | S0 | `GAP(c_s)` 全局 | 3,280,274 |
| `moe128_stats` | S1 | 逐尺度 `GAP(\|a_s−b_s\|)` | 3,280,466 |
| `moe128_sympcg` | S2 | S1 + `GAP((a_s+b_s)/2)` + `\|GAP(a_s)−GAP(b_s)\|` | 3,313,234 |

S1→S2 是干净的单变量对照（同架构、同流程，仅 gate 输入不同）；所有 gate 输入按构造**交换对称**（swap error < 1e−6，10 个单测覆盖）。

结果（同数据集三臂在同一张卡先后运行，均 40000 步、完整 TEST block、swap/deploy error 0）：

| 数据集 | S0-C1 | S1-STAT | S2-PCG | **S2−S1 ΔF1** | ΔIoU | τ | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| SYSU | 83.1391 | 82.3641 | **83.3315** | **+0.967** | +1.410 | 1.353 | **FAIL** |
| WHU | 93.2317 | 93.6414 | 93.6568 | **+0.015** | +0.027 | 1.587 | **FAIL** |

- **决定性证据是跨域不一致**：同一改动 SYSU +0.97pp、WHU +0.015pp；而该机制的
  主张（`m_s`/`v_s` 帮助区分伪变化）**预测跨域一致增益**。
- S1 臂自身符号相反（SYSU `S1−S0 = −0.775`，WHU `+0.410`），与噪声量级一致。
- 按预注册规则 → **拒绝"对称共性上下文带来独立增益"这一 gate 假设，不新增第四专家**。
- 正确表述：**"SYSU 上观察到值得优先复核的正向信号（+0.97pp，超 √2σ=0.676 但未达 τ=1.353），
  WHU 上未检出"**——不能写"S-PCG 有效"或"S-PCG 无效"。
- 即使按 SYSU 最大 +0.97pp 外推，83.33 + 0.97 = 84.30 **仍 < 85**，硬目标仍不可达。

完整数据、"允许/不允许宣称"清单与下一步选项见 `docs/temporary/CATA-CD_Run2_SPCG_结果.md`。

### 四数据集当前最好单次结果 vs 硬目标（**Gate R3 未达**）

| 数据集 | 最好单次 F1 | 来源 | 目标 | 缺口 |
|---|---:|---|---:|---:|
| SYSU | **83.3315** | Run2 S2-PCG | 85.0 | −1.67 |
| WHU | 94.04 | 历史 C1 | 95.0 | −0.96 |
| LEVIR | 91.17 | 历史 TV-D3S | 92.5 | −1.33 |
| CDD | 97.79 | 历史 C0 | 98.0 | −0.21 |

## 环境（RSML-3）

```yaml
project: LS-Rep_BCD_RSML_3
server: RSML-3
user: yqwang
path: /home/yqwang/projects/LS-Rep_BCD_RSML_3
env: lsrep            # PyTorch 2.14.0+cu132, CUDA 13.2, Python 3.10
gpu: RTX 5090 ×2      # 每卡 32GB, Blackwell/sm_120
```

```bash
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate lsrep
cd /home/yqwang/projects/LS-Rep_BCD_RSML_3
```

## 教师池

| ID | Teacher Package | 权重文件（pre-trained_weights/） | 波次 |
|---|---|---|---|
| T1 | SAM2.1 Hiera-Large | `sam2.1_hiera_large.pt` | Wave-A |
| T2 | DINOv2 ViT-B/14 | `dinov2_vitb14_pretrain.pth` | Wave-A |
| T3 | DINOv3 ViT-B/16 LVD-1689M（neutral） | `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` | Wave-A |
| T4 | DINOv3 ViT-L/16 SAT-493M | `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` | Wave-A |
| T5 | RemoteCLIP ViT-L/14 | `RemoteCLIP-ViT-L-14.pt` | Wave-A |
| T6 | MaRS-Base RGB | `mars_base_rgb_encoder_only.pth` | Wave-A |
| T7 | AnySat | `AnySat.pth` | Wave-B |
| T8 | UniverSat Base | `universat_base.safetensors` | Wave-B |
| T9 | C-RADIOv3-B | `c-radio_v3-b_half.pth` | Wave-B |

教师编码器的最小可加载实现位于 `models/thirdparty/<name>/`；`others/<name>-main/` 仅保留 LICENSE + README（参考）。

## 目录结构

```text
models/
  a2net.py                            A2Net-LWGANet-L0（+ 可选 DCA）
  decoder/deployable_change_adapter.py   DCA（3 专家 × 4 尺度，对称 gate，~0.37M）
  distill/                             cache_v2 / teacher_package / kd / capability_probe
  agent/                               dataset_signature / teacher_registry / offline_bandit
  thirdparty/                          8 个教师 backbone 最小加载
  tools/                               generate_teacher_cache_v2 / audit / smoke / agent 工具
  scripts/train.py                     CATA-CD 训练器（C0/C1/TV-*）
train_scripts/CATA-CD/
  teacher_adaptation/                  数据集适配性验证（C0/C1 + Wave-A 能力矩阵）
  Run1/                                Agent 选教师后从头重跑 M1
  Run2/                                复核后的下一轮：J0 可重复性 + J1 机制判决 + S-PCG
    README.md                          预注册 Gate R0–R3（先读这个）
    common.sh                          Run2 共享环境与 run_one（J1-*/S0/S1/S2 臂表）
    run_j0.sh / run_j0_switches.sh     J0 取证与开关隔离
    run_calibration.sh                 J1 的 train-only 冻结校准
    run_j1_sysu.sh / smoke_j1.sh       J1 四臂
    run_spcg.sh                        S-PCG S0/S1/S2 阶梯
    monitor.sh                         各臂进度 / F1 / GPU / tmux
tests/                                 无依赖单元测试（safe_selector / task_reliable_kd / dca_gate）
analyse/                               extract_metrics.py / models_to_txt.py
others/                                <name>-main/ 精简参考代码 + 一次性复核脚本
docs/temporary/CATA-CD_v2_...md       完整方案（20261005）
docs/temporary/CATA-CD_v2_负结果严格复盘_..._20261008.md   负结果复盘与下一轮方案
docs/temporary/CATA-CD_Run2_J1_结果.md J1 判决结果与"允许/不允许宣称"清单
```

## 部署约束

- 学生 = A2Net-LWGANet-L0；DCA 首版 width=128。
- 部署参数：C0 = 2,913,094；C1 = 3,280,274；**有效推理参数 < 5M**。
- 教师 / Cache / Translator / Agent 全部训练期使用，**不进部署图**。
- `switch_to_deploy()` 为 no-op；时间交换对称 `< 1e-6`；部署前后主预测误差 `< 1e-6`。
- checkpoint 仅支持 **format_version=8**（CATA-CD v2）。

## 训练与结果

```bash
# smoke
bash train_scripts/CATA-CD/teacher_adaptation/smoke_test.sh

# 生成教师 Cache（逐教师，四数据集）
bash train_scripts/CATA-CD/teacher_adaptation/prepare_teacher_cache.sh <teacher_id> <gpu_id>

# 训练（两卡排队；GPU 空闲后）
tmux new -s cata_gpu0 -d 'bash train_scripts/CATA-CD/teacher_adaptation/run_gpu0.sh 0'
tmux new -s cata_gpu1 -d 'bash train_scripts/CATA-CD/teacher_adaptation/run_gpu1.sh 1'

# 结果统计（实验完成后）
python -B analyse/extract_metrics.py          # 写入 docs/experiment_metrics.xlsx
python analyse/models_to_txt.py --cata        # 快照 → docs/temporary/models_and_metrics_CATA-CD_Run1.txt

# Run2（复核后的下一轮；先读 train_scripts/CATA-CD/Run2/README.md 的 Gate R0–R3）
python tests/test_safe_selector.py            # 无依赖单测
python tests/test_task_reliable_kd.py
python tests/test_dca_gate.py

bash train_scripts/CATA-CD/Run2/run_j0.sh 0 200 2                 # J0 可重复性取证
bash train_scripts/CATA-CD/Run2/run_j0_switches.sh 0 30 SYSU      # J0 开关隔离（诊断）
bash train_scripts/CATA-CD/Run2/run_calibration.sh 512 SYSU       # J1 train-only 校准
bash train_scripts/CATA-CD/Run2/run_j1_sysu.sh 0 J1-C1,J1-GT SYSU # J1 四臂
bash train_scripts/CATA-CD/Run2/run_spcg.sh 0 S0-C1,S1-STAT,S2-PCG SYSU,WHU
bash train_scripts/CATA-CD/Run2/monitor.sh SYSU                    # 进度监控
```

## GitHub 提交流程

```bash
git add models README.md train_scripts/CATA-CD docs analyse others
git status
git commit -m "CATA-CD v2: adaptive teacher agent + DCA"
git push
```

`others/` 只保留精简后的参考代码（供网页 GPT 阅读），不提交其中的权重、`__pycache__/`、`*.pyc`、`work_dirs/`、数据集或完整框架副本。执行 `git commit` 前必须先检查 `git status`；不得提交权重、缓存、数据集、日志或无关改动。
