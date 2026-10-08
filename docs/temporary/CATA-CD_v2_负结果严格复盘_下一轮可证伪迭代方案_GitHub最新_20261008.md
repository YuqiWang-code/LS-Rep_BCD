# CATA-CD v2：负结果严格复盘与下一轮可证伪方法迭代

> **研究对象**：LS-Rep_BCD（全监督二值遥感变化检测；SYSU / WHU / CDD / LEVIR，256×256）  
> **审查日期**：2026-10-08；**GitHub `main` 固定版本**：[`2b1c0870d42d01d4b05cc0e5589d65863fe9e970`](https://github.com/YuqiWang-code/LS-Rep_BCD/tree/2b1c0870d42d01d4b05cc0e5589d65863fe9e970)（2026-10-08 13:44:51，北京时间）；上一核心实现提交 [`0f8e412`](https://github.com/YuqiWang-code/LS-Rep_BCD/commit/0f8e412b44722be83e3ab05450aeeb61e19ce943)。  
> **来源界限**：通过已连接的 GitHub 仓库接口核对完整文件树（327 个被追踪文件）并逐文件审查与本轮结论有关的源码、Registry、LODO 报告、脚本和 README；**未能克隆服务器完整工作目录，未直接取得 RSML-3 上全部原始 `train_log.txt` / `probes_v2` JSON / GPU 实测记录**。仓库中的 `research_report.json` 是正式日志的*派生快照*，因此其数字属于“代码/归档结果直接事实”，**不等于我已独立重验每一条最后 TEST block**。本次没有修改远端仓库，没有启动训练。  
> **证据等级**：**[F]** 仓库代码/归档直接事实；**[I]** 基于事实的推断；**[H]** 待验证假设；**[M]** 信息或原始证据缺失。新实验的所有预期收益均为 **[H]**，不是结果。  
> **原则**：不回收已失败的 U-SafeAgent 原方案；不改 seed 2333、不做多 seed、不将 teacher/cache/agent 接入推理；训练预算固定 40K steps、batch 64、所有正式实验从相同 ImageNet 初始化独立从头训练。报告 F1/IoU/Recall/Precision/OA/Kappa、训练/辅助/部署参数、256² FLOPs。

---

## 目录

1. [执行结论与三选一判定](#1-执行结论与三选一判定)
2. [最新仓库审查、证据表及必要纠错](#2-最新仓库审查证据表及必要纠错)
3. [“教师能力—蒸馏效用”解耦的机制解释与诊断](#3-教师能力蒸馏效用解耦的机制解释与诊断)
4. [区分 a/b/c 的最小判决实验](#4-区分-abc-的最小判决实验)
5. [下一轮四种候选机制、排序及止损门槛](#5-下一轮四种候选机制排序及止损门槛)
6. [噪声、同 seed 重复、配对与统计边界](#6-噪声同-seed-重复配对与统计边界)
7. [首选方法数学定义、训练图与部署图](#7-首选方法数学定义训练图与部署图)
8. [逐文件修改、实验 ID、启动顺序、路径及验证](#8-逐文件修改实验-id启动顺序路径及验证)
9. [论文可发表性与不能再宣称的内容](#9-论文可发表性与不能再宣称的内容)
10. [2024–2026 年核验文献](#10-20242026-年核验文献)
11. [立即执行顺序、缺失证据与最终闸门](#11-立即执行顺序缺失证据与最终闸门)

---

## 1. 执行结论与三选一判定

### 1.1 必须三选一：**优先判定 (a) 测量/协议问题，但不能宣称 (b)/(c) 已被排除**

**结论**：此次已完成的 **U-SafeAgent 教师选择主张失败**。如果必须在三类原因中选一个作为**当前最优先处理、最有证据支持的瓶颈**，选 **(a) 现有测量/协议无法可靠辨认小幅教师效用**。

- **[F]** AG-03 在全部四折 × 五个特征阶梯均选择 `None`，覆盖率 0、正收益召回率 0。它实现了拒答，却**没有证明“学会了何时选教师”**。[LODO 报告](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/2b1c0870d42d01d4b05cc0e5589d65863fe9e970/docs/temporary/CATA-CD_registry/agent_safe/lodo_report.json)、[冻结动作](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/2b1c0870d42d01d4b05cc0e5589d65863fe9e970/docs/temporary/CATA-CD_registry/agent_safe/selected_actions.json)。
- **[F]** LODO 的 held-out **最佳 val** 教师效用上界依次为 SYSU −0.0945、WHU +0.1159、CDD −0.0010、LEVIR +0.1460 **pp**；预设强正阈值 δ=+0.20pp，**四数据集没有一个标签达到该阈值**。在这套标签与阈值下，所有策略均无合法正动作；即使完美 oracle 也选 `None`。因此“AG-03 有 0 harmful action”是**空动作的逻辑结果**，不能作为学习效力证据。
- **[F]** C1 同配置同 seed 的 val σ（四次运行）= 0.411/0.311/0.013/0.061pp；test σ=0.289/0.537/0.013/0.195pp（SYSU/WHU/CDD/LEVIR）。这揭示了不可忽略的运行波动，但**只测过 C1，不代表所有 TV-* 同方差**。
- **[F]** 使用当前冻结 Teacher Package 的 36 项 40K 实验中，test 强正（ΔF1≥0.20pp 且 ΔIoU>0）仅 3 项；best-val 上没有 ≥+0.20pp 项。用 val 选择教师必然倾向 None，test 事后“最优教师”又与 val 排序冲突；二者共同压缩了可辨认的学习信号。
- **[I]** `(b) 4 数据集无法学任何选择规律` **未证实**：LODO 仅四个独立域，无法识别丰富的跨域映射，但有可能学习**非常低自由度、预先定义的规则**。目前只能说“在这个样本/奖励/教师包组合下无可证实的部署策略”。
- **[I]** `(c) 教师知识对 3.28M 学生本质上没有价值` **未证实**：当前 36 项测试的不是“所有蒸馏”，而是**同一种固定随机投影 + 单尺度 cosine KD**，且 raw teacher probe 与 KD target 不一致。没有对教师任务对齐监督的干预对照，故不能从失败推导“教师根本不可蒸馏”。

**可证伪立场**：

> **主假设 H-A（当前首选）**：运行级随机性、best-val checkpoint 选择和有缺陷的效用估计，掩盖了小幅或不稳定机制差异；**但即便把测量做稳，如果 KD 仍不能优于 GT-only 辅助，旧教师选择主张也必须停止**。  
> **替代 H-B**：测量修正后教师蒸馏仍偶尔有明显正效应，但 4 个域不足以学习稳健策略。  
> **替代 H-C**：测量修正后，在**同等辅助容量、相同训练预算**下，不管是原始 128ch cosine KD 还是 GT-对齐的 teacher KD，均不能超过无教师 GT-only 辅助 → **当前教师 Cache 内容对这一路径无额外价值**（注意外推范围）。

### 1.2 现实的止损决定

**现在不继续投资源扩增 Agent、Teacher Pool，也不启动 U-SafeAgent→M2。** 先做不涉及训练的 P0 证据清洗/可复现性审计；再进行一项**机制剥离式对照**，识别“可用监督”究竟来自教师还是 GT。若没有明确超噪声增益，主论文改走**学生双时相表征结构**而非教师选择。

当前部署预算：C0=2,913,094 / 2.7676G；C1=3,280,274 / 3.2432G；距 5M 仍约 1.72M 参数空间，但**不是必须用满**。对应 C1 的四目标差距：SYSU **2.428pp**、LEVIR **1.463pp**、WHU **0.963pp**、CDD **0.245pp**。现有教师 40K 表最大单项增益仅 SYSU RADIO +0.677pp，因此**“完美选教师”也没有单凭现有实验证据同时解决四硬目标**。

---

## 2. 最新仓库审查、证据表及必要纠错

### 2.1 审查清单及可溯源位置

| 证据层 | 文件（固定 SHA 下可打开） | 审查要点 | 状态 |
|---|---|---|---|
| 总结 | [`README.md`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/2b1c0870d42d01d4b05cc0e5589d65863fe9e970/README.md) | 44 runs、12 次噪声底、U-SafeAgent F0 | [F] 更新，部分措辞仍需审慎 |
| 教师效用 | [`research_report.json`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/2b1c0870d42d01d4b05cc0e5589d65863fe9e970/docs/temporary/CATA-CD_registry/research_report.json) | test 逐数据集 F1/IoU delta | [F] 派生快照；[M] 缺完整原日志 |
| Agent 监督 | [`agent_train_registry.json`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/2b1c0870d42d01d4b05cc0e5589d65863fe9e970/docs/temporary/CATA-CD_registry/agent_train_registry.json) | best-val reward=ΔF1(pp) | [F] 无 `test_*` 字段 |
| Policy | [`safe_selector.py`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/2b1c0870d42d01d4b05cc0e5589d65863fe9e970/models/agent/safe_selector.py) | weighted ridge/group jackknife/safe-set | [F] 已实现；[I] 不是真正置信下界 |
| Policy 实验 | [`train_safe_teacher_agent.py`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/2b1c0870d42d01d4b05cc0e5589d65863fe9e970/models/tools/train_safe_teacher_agent.py) | 五特征阶梯、AG-00…AG-CH、LODO | [F] 存在 AG-CH held-out label 泄漏 |
| Cache 定义 | [`teacher_package.py`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/2b1c0870d42d01d4b05cc0e5589d65863fe9e970/models/distill/teacher_package.py) | teacher F.normalize → abs diff → 随机128ch | [F] 目标非原生 change logits |
| KD 实际行为 | [`kd.py`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/2b1c0870d42d01d4b05cc0e5589d65863fe9e970/models/distill/kd.py) | c4 64→128；cosine；confidence 权重；gradient cap | [F] 完全无 task-GT 一致筛选 |
| Probe | [`capability_probe.py`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/2b1c0870d42d01d4b05cc0e5589d65863fe9e970/models/distill/capability_probe.py) | AUROC 是 `confidence` 对 GT 的排序 | [F] 与 128ch KD target 不是同一对象 |
| Probe 统计 | [`_probe_utility_corr.py`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/2b1c0870d42d01d4b05cc0e5589d65863fe9e970/others/_probe_utility_corr.py) | Spearman、每域仅9教师 | [F] 用双 argsort，不正确处理 ties |
| 登记与日志 | [`build_capability_registry.py`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/2b1c0870d42d01d4b05cc0e5589d65863fe9e970/models/tools/build_capability_registry.py) | 最后完整 TEST block、单位 pp、日志哈希 | [F] 解析修复；C0 锚单位 bug |
| 学生/训练 | [`train.py`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/2b1c0870d42d01d4b05cc0e5589d65863fe9e970/models/scripts/train.py) 和 [`deployable_change_adapter.py`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/2b1c0870d42d01d4b05cc0e5589d65863fe9e970/models/decoder/deployable_change_adapter.py) | 40K、aux 8,320 train-only、DCA=deploy | [F] 当前单尺度蒸馏；[M] 全日志缺失 |
| Cache 几何重放 | [`cache_v2.py`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/2b1c0870d42d01d4b05cc0e5589d65863fe9e970/models/distill/cache_v2.py)、[`transforms.py`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/2b1c0870d42d01d4b05cc0e5589d65863fe9e970/models/datasets/transforms.py) | 裁切/翻转，同一 change label；时相交换 invariant | [F] 设计成立；[M] 实际对齐误差待 dry-run |

**注意**：GitHub 包含第三方引用代码、文献和历史实验文档。此次对 327 个 tracked files 完成文件清单核对；**实质逐行审阅范围是对本轮机制结论有因果关系的核心脚本**，不虚称已逐行重审所有第三方 backbone 实现。

### 2.2 当前结果矩阵：允许的解读

**test F1，单位百分点 pp；相对于 C1；来源为 `research_report.json`**：

| Dataset | C0 | C1 | SAM2 | D2 | D3-LVD | D3-SAT | RCLIP | MaRS | AnySat | UniverSat | RADIO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SYSU | 82.13 | 82.57 | −0.28 | −0.12 | −0.24 | −0.67 | −0.81 | −0.03 | −0.66 | −0.27 | **+0.68** |
| WHU | 92.66 | 94.04 | −1.15 | −0.36 | −0.21 | +0.08 | +0.08 | **+0.56** | −0.13 | +0.40 | −0.32 |
| CDD | 97.79 | 97.75 | −0.01 | −0.02 | −0.05 | −0.04 | −0.01 | −0.02 | −0.01 | ≈+0.00 | −0.01 |
| LEVIR | 90.92 | 91.04 | +0.06 | −0.22 | +0.08 | +0.14 | −0.07 | −0.06 | +0.06 | −0.18 | ≈0.00 |

仓库 **未舍入值**显示：36 个组合中 **10 个 ΔF1>0、26 个 ΔF1≤0、3 个达到“强正”阈值、33 个未达到强正**。附件“30/36 非正”以及历史“30/36 非强正”与当前归档**均不一致，需更正**。不能用这一错误计数支撑普适负迁移的概率主张。整体格局仍是“多数未观察到强正，但少量单次 test 点估计为正”。

**val 最佳值**：SYSU 最大 −0.0945、WHU +0.1159、CDD −0.0010、LEVIR +0.1460pp。附件“全部 val 效用在 ±0.15pp 内”不成立：SYSU AnySat 约 **−1.076pp**，WHU 最低约 **−0.427pp**。正确表述是“**全部正向 val 效用 ≤+0.146pp，绝大多数效用为负，部分负值较大**”。

### 2.3 必须修的 P0（影响结果可信度/公平性）

1. **P0-A：`AG-CH` 对照存在 held-out val reward 泄漏 [F]。** 代码在 `run_lodo()` 的 AG-CH 分支，先 `chosen = tids_te[idx]`，随后读取 `util = mean([y[i] for i in te if tids[i] == chosen])`，并以 `util > args.delta_pp` **决定是否选择该教师**。`te` 是留出数据集；等于“看过留出数据集 val 真实收益后再拒答”。AG-CH 零误选不能作为真实 LODO 基线。应改：选择与拒绝仅基于训练域拟合的 gate/固定 train probe 阈值，held-out y **只能用于事后打分**。此问题**不说明 AG-03 使用了 test**，但导致其对照实验不干净。
2. **P0-B：Registry `_anchors.C0_test_pp` 实际仍是 0–1 量纲 [F]。** `build_capability_registry.py` 只对 `C1_test_pp`/`C1_val_pp` 乘 100，`C0_test_pp` 直接保留 `last_test_block()` 的 0–1 比例值，违反 `_meta.unit=pp`。修复派生字段，**不要篡改原日志或重命名旧文件覆盖**；建立 `registry_v3`，版本化审计，再计算全部 C0 对照。当前论文 C0 数字可通过“明确单位换算”报告，不可直接由有问题的 `_anchors.C0_test_pp` 混算。
3. **P0-C：真实原始完整日志缺失 [M]。** GitHub 没有收齐 44+12+4 份 `train_log.txt` 最后完整 TEST 块；归档虽记录源日志 SHA，却不能等同第三方独立验证。导出每臂文本摘要（最后完整 block + 首末配置 + 数据 split SHA + checkpoint 选择 epoch）至**新生成审计产物**，不得用 best val 或 checkpoint 文件名代替 test。
4. **P0-D：对同 seed 运行的“同配置”不能只比较 CLI [F/I]。** 当前证据已观察从 epoch0 分叉，应先比对 seed 调用时机、模型初始权重 hash、首 10 batch sample+crop state 的 SHA、GPU/并发、TF32/cuDNN/cuBLAS、随机数状态、best checkpoint epoch。未知源不能武断归咎 SyncBatchNorm/3-way 并发。
5. **P0-E：`train_safe_teacher_agent` 的 `u_noise` 读取被留出数据集 C1 重复 **val** 的 σ [F]。** 这不含 test 标签，不等于 test leakage；但**不是纯粹“来自另外三个训练域的跨域预测”**。若目标是模拟“从未训练/验证过的新数据集，单纯靠签名和 train-only probe 决策”，将 held-out C1 σ 当先验违反该设定。必须明确其适用范围：只适用于**允许目标数据集已具备 4 次 C1 训练验证记录**的转导协议；严格 inductive LODO 应仅由训练三域估计 conservative noise floor。未来仅四公共数据集时，这个区分尤为重要。

### 2.4 P1（方法瓶颈）/P2（可靠性）

- **P1-A：Teacher “quality” probe 与 Teacher “KD target” 测量不同对象。** `extended_probes()` 的 AUROC 使用 `confidence = max_c |normalize(f_A)-normalize(f_B)|`；KD 则逼近 `R_0 |normalize(f_A)-normalize(f_B)|` 的 128 维**随机线性投影结果**（`R_0` fixed random），且只在 `change[2]`（c4，stride16）上对齐。给出的 AUROC 并非这个 128 维目标的可学习度。
- **P1-B：非任务一致 cosine 目标。** 单尺度 cosine imitation 只强制两种表征“方向相同”，没有显式要求像素分类边界、变化/未变化 margin、边缘结构或伪变化负例。强 teacher 响应甚至可能与监督梯度方向相反。
- **P1-C：5 个 probe 的相关性不能排除非线性或条件依赖。** 原相关性脚本用 `argsort(argsort(...))` 实现 Spearman，没有平均秩处理 ties；各域只有 9 教师，36 条记录不是 36 个独立域。pooled Spearman ≈0.03 **是描述统计，不是证明真实相关系数为零**。
- **P1-D：`delta=0.20pp` 大于所有正向 val 标签。** 无论哪种精巧学习器都无法通过预设门槛；降阈值“救活选择率”会构成后验调参，正确做法是更换可检测的价值主张或改善 teacher utility 的产生机制。
- **P1-E：梯度上限是局部不是全路径。** `gradient_budget_lambda` 在 `change[2]` 上按 `rho=0.25` 缩放，不能直接宣称“整个主干 teacher 影响 ≤25%”；需测每层梯度余弦、norm 和冲突率。
- **P2-A：`SafeSelector.fit` 的 `if len(X) != len(y) != len(groups)` 是链式比较，不等同“三者必须相等”，应为 `not (len(X)==len(y)==len(groups))`。应增加单元测试。
- **P2-B：`model_spread_pp` 的三次 leave-one-train-domain 预测 std 是**敏感性代理**，不是 95% CI、更不是 conformal guarantee；现有注释已认识到这一点，但论文绝不能把该 margin 写成带形式风险控制的 safe bound。
- **P2-C：`best_val` 使用跨完整训练过程的最大 epoch F1**（模型选择值），对不同臂带有不同程度的 winner's curse；作为选择 checkpoint 的规则可以保留，但当成 Agent 的极小效用回归标签，会放大随机排序差。
- **P2-D：Cache `confidence` 名称会误导。** 其实是归一化特征差的最大通道值，**不是概率或经过 GT 校准的 teacher posterior**；本轮只能称 change-response strength。

---

## 3. “教师能力—蒸馏效用”解耦的机制解释与诊断

### 3.1 首先厘清被相关分析比较的量

令预训练教师输出 `F_T^A, F_T^B ∈ R^{C×h×w}`，当前代码的三个中间量为：

\[
E_T(p)=\big|\operatorname{norm}_c F_T^A(p)-\operatorname{norm}_c F_T^B(p)\big|,
\qquad z_T(p)=\max_c E_{T,c}(p),
\qquad Y_T(p)=R_0 E_T(p)\in\mathbb R^{128}.
\]

- 你观测的 P1 AUROC 是 `AUC(z_T, GT)`；
- 训练期 student 逼近的是 `Y_T` 的**逐像素余弦方向**，而不是 `z_T` 的二值决策；
- 最终测的是四尺度 decoder 输出 `F1(student, GT)`；
- 因而“`AUC(z_T,GT)` 高 → `F1(student after KD)` 高”**根本不是当前损失所直接优化的关系**。

### 3.2 六条机制性解释，每条给反事实检验

| ID | [H] 机制假设 | 为什么符合已有现象 | **无需 40K 或极低成本的可证伪预测** | 失败判据 |
|---|---|---|---|---|
| H1 **目标失配** | 教师 z 有 CD 可分性，但 128ch `Y_T` 的 cosine imitation 不保持 changed/unchanged 任务 margin | D3-LVD 在 SYSU AUROC≈0.870，但 val 增益≈−0.095pp | Train-only 对固定 256 对图，比较 `AUC(z_T,GT)` 与 `AUC(a^T Y_T,GT)`（仅用 train 元子集拟合 ridge `a`、独立 train holdout 检验）；及按 GT 的 cosine 组间 margin | `Y_T` 不降可分性且 student feature 轻线性 probe 可稳定恢复 margin，则 H1 减弱 |
| H2 **表征/容量鸿沟** | 仅 c4=64ch 的学生与巨大 teacher 单尺度 128ch 目标不对齐，标签梯度与 KD 梯度冲突 | teacher AUROC 强却不能转移 | 取固定随机初始化 + 若干 **train-only** checkpoint，按 c2,c3,c4,c5 计算 `cos(∇_f L_main, ∇_f L_KD)`、有效 `λ`、蒸馏辅助可线性拟合度；比较 c2 与 c4 的 oracle linear probe AUC | 若梯度大多非冲突、且目标可拟合而无收益，纯容量解释不足 |
| H3 **教师响应与二值任务不一致** | `z_T` 只辨认差异，不知道 GT 所定义的真变化；伪变化高反应既进入权重也进入 target | 多种自然/遥感预训练 teacher 都无法超过无教师学生 | Train split 分四象限：GT+/z高、GT+/z低、GT−/z高、GT−/z低，统计错误强化质量、KD 梯度/GT 符号冲突率、未变化 FPR；**训练标签可用** | 若误导象限很少且梯度一致，则 H3 减弱 |
| H4 **空间分辨率/boundary 丢失** | 教师 14–64 栅格不同，统一归到 student c4；小目标/边缘易损伤 | 高 AUROC 未必提升像素 F1/IoU | Train-only 计算对象尺度分层 recall/AUC、boundary contrast，比较 `AUC(z_T upsampled)` 与 c2/c4 学生对应 GT margin | 若细粒度与边界都不受 stride 影响，则 H4 不足 |
| H5 **量测变量不稳定** | best-val+同 seed 非确定性给小效应噪声 ≥效应 | SYSU val σ≈0.411pp，WHU≈0.311；oracle 最大仅+0.146pp | 相同前100–1000 steps 两次运行 hash/曲线对齐；从现存各次 C1 的 val 曲线重算固定 step 的 F1 方差与 best-val 方差；无新增 40K | 若完全确定且固定 step 与 best-val 方差同大，需另找噪声源 |
| H6 **并非“per-image z-score 导致”**（反证重要） | 当前代码**没有 per-image z-score**；只有逐像素 channel L2-normalize 和全局描述 normalize | 排除附件中一个未经代码支持的猜想 | 对比 `E_T` 归一化前/后 `z_T` 的排序及边界 AUC；不再把“per-image z-score 擦除先验”归罪于现实现 | 若无逐图 z-score，相关解释不能成立；但仍可检验 channel normalization 信息损失 |

**最优先低成本检验**：H1 + H2 + H3；如果训练-only 的 task-aware student 线性 probe 能学到 teacher 的 GT 分离，而完整 40K F1 没涨，支持“优化/监督转译瓶颈”；如果 teacher 自身变化 AUROC 高但 false-positive 大，支持“不能直接把变化响应视为真变化”。

### 3.3 Probe–utility 相关结论的正确措辞

| 数据 | 已有数值 | 可以说 | 不能说 |
|---|---|---|---|
| balanced AUROC | 约 0.51–0.87（train Cache） | 教师响应对 train GT 的像素排序能力存在明显差异 | 教师就是精确的 change detector、概率已校准 |
| pooled Spearman | 约 +0.03（原脚本） | **按这套特征和效用定义**未观察到稳定单调关联 | 证明教师能力与可蒸馏性在总体上独立 |
| per-domain Spearman | SYSU +0.30、WHU −0.33、CDD +0.05、LEVIR +0.12 | 方向因域而异，样本仅 9 teacher/domain | 宣称这些微小相关具有统计稳定性 |
| teacher mean val gain | 9 教师均值均负 | 固定 translator+训练协议未产生平均正向 val 效用 | 永远不存在有价值的遥感教师 |

**复算要求**：用 `scipy.stats.spearmanr`（ties-aware）分别做**域内排名**，再做 descriptive macro-average；不要将 36 项当作 36 个独立数据集报告 p 值。描述 scatter 时标明四种颜色/域、9 教师点及相同横坐标近似重复。

---

## 4. 区分 (a)/(b)/(c) 的最小判决实验

### 4.1 为什么“三选一”不可能被一张四域表严格统计辨认

- **(a)** 是 measurement process；**(b)** 是 policy generalization；**(c)** 是 causal effectiveness of KD transfer。它们不是互斥机制，可同时成立；要求给“唯一根因”在统计上不可识别。
- 因而采用**预注册顺序决策树**：先排除不可复现实验，再区分“只是 GT 辅助监督有效”与“Teacher target 有独立效用”，只有后者成功才值得重新讨论跨域选择。

### 4.2 阶段 J0：测量可重复性（先免费再决定是否投 40K）

**预算**：不新增 40K run；在 SYSU 与 WHU，每个数据集重跑同一配置 **2 次 × 100~1000 steps**，记录重训初始权重/梯度/数据索引 SHA、batch augmentation hash、100/1000 step loss/grad hash。允许提前停止诊断 run，但**它们不计入正式 40K 消融**。

预设输出：

1. `J0/initial_hash`：backbone/pretrained 的 SHA、SWA/TFM/Decoder/DCA 初始化 tensor 哈希，验证同一 config 初值 **逐张量一致**；
2. `J0/data_order`：首 1000 step 所用 sample IDs 及 crop/flip/exchange 参数哈希一致；
3. `J0/loss_trace`：第0/1/10/100/1000 step loss、λ（教师臂）、逐层梯度 norm、实际 cudnn/TF32/bf16 flags；
4. `J0/decision`：**完全一致**：允许保留相同 seed 40K 单跑纪律；**明显不一致**：先定位 nondeterministic kernel / dataloader / RNG，冻结严格协议后**新 C1 anchor 必须重新从头跑**，旧表只能历史比较。

**不要**将 `torch.use_deterministic_algorithms(True)` 一键打开后拿“新 M1”与旧 C1 比——这会造成新旧优化协议混杂。不能改变 `cd_base` 或自换 PyTorch/CUDA。对 `CUBLAS_WORKSPACE_CONFIG` 等设置须在 Python/PyTorch 初始化前统一、记日志；不支持的 sm_120 CUDA 算子先 smoke。单独诊断 `nn.SyncBatchNorm` 与普通 BatchNorm 如要替换，属**新协议臂**，不能混入方法增益。

### 4.3 阶段 J1：机制判决（**SYSU 优先**；固定教师从 train-only probe 选）

教师预注册为 **neutral DINOv3-LVD**：在 SYSU **train-only** AUROC≈0.870，且 teacher 预训练与 cache 已有，是**辨认“强 train 变化响应能否变成学生有效监督”的诊断对象**。这不是按 test F1 选择它；本阶段不是重新选教师。若真实 Cache manifest/权重 SHA 不一致，J1 不启动。

**唯一变化层面**：所有臂均用同一 C1 学生+deployable DCA，主监督、40K、seed、batch、训练超参相同；增加**同一个 1-channel 训练期 auxiliary head**（比旧 128ch head更直接）、**相同空间抽样集合和 loss 权重策略**，通过“辅助目标”剥离教师提供的额外信息。不能用任意调大 λ 假冒新方法。

| ID | 学生（部署） | 训练期辅助 | 相对上一臂唯一变量 | SYSU 正式 40K | 作用 |
|---|---|---|---|---:|---|
| **J1-C1** | C1 3.280M | None | clean anchor | **1** | 无教师上限对照 |
| **J1-GT** | 同 C1 | task-aware 1ch auxiliary，GT-only，**固定像素样本/掩码** | 增加 GT auxiliary | **1** | 测 GT-only 训练辅助值 |
| **J1-GATE** | 同 C1 | 与 J1-GT 完全相同 head/loss，但只在 **teacher 与 GT 一致的区域**监督，目标仍 GT | 只改变教师用于**筛样** | **1** | teacher reliability gate 的单独价值 |
| **J1-TASKKD** | 同 C1 | **与 J1-GATE 同样的像素筛样**，但软目标改为 train-fitted teacher task response（不等同原 128ch cosine） | **只改变被监督的软目标** | **1** | teacher soft knowledge 独立贡献 |

严格分离说明：GT vs GATE 比较有*有效样本量差异*，因此要同时报告采样量/正负比例；J1-GATE vs J1-TASKKD 才是**最干净的“Teacher 软目标是否额外有价值”**的控制。训练代码让四臂的 auxiliary 参数相同；J1-C1 没有 aux 是机制总效应对照。

**阈值（非显著性）**：基于现有 test 噪声底 SYSU `σ≈0.289pp`，一对独立单次结果之差的经验尺度 ≈`√2 σ=0.409pp`。使用**探索性 strong evidence gate**：

\[
\Delta F1^{test}_{\mathrm{J1-TASKKD-J1-GATE}} >
\max(0.20, 2\sqrt2\widehat\sigma_{\rm SYSU})
\approx \boxed{0.82\text{ pp}},\quad \Delta IoU>0,
\]

且 train-val 方向一致、相同 F1 评估协议。**0.82pp 只是“两倍经验差值尺度”的保守工程阈值，不是 p<0.05；四次 σ 估计极不精确。** 如严格重现后 σ 大幅下降，该闸门应按**预先固定的新 σ 测量规则**计算，不允许看到结果后下调。

### 4.4 判决分支：识别“测量”还是“监督价值”

| J0 | J1-GT 相对 C1 | J1-GATE 相对 GT | J1-TASKKD 相对 GATE | 最合理解释 / 下一步 |
|---|---|---|---|---|
| 不可复现 | 任何 | 任何 | 任何 | **优先 (a)**，暂不接受机制结论；定位到可重现后重建 anchor |
| 可复现 | 正 | ≤0 | ≤0 | GT 训练辅助有效，教师附加知识不足：**停止 teacher 选择，转 GT-only/学生结构** |
| 可复现 | 正或无 | 正 | ≤0 | teacher 只提供了可靠区域*筛样*，soft representation 无独立价值；可写训练期筛选机制，不再称知识蒸馏主创新 |
| 可复现 | 正或无 | 任意 | 明显正，且 >噪声门槛 | 反驳“教师从不有价值”(c)；**继续验证新蒸馏机制**，暂不恢复 Agent |
| 可复现 | 无 | 无 | 无 | 只支持“在当前教师+当前 translator+SYSU+40K 下均无可辨认增益”；转结构线，**不外推全部教师** |

**区分 (b) 需要额外条件**：如果 J1-TASKKD 对至少两数据集有稳定显著于噪声尺度的不同教师效应，**且**训练域 leave-one-domain 预测失败，再支持“选择规律跨域样本不足”。若 J1 教师机制不奏效，研究 (b) 没有因果价值，直接止损。即：J1 一套实验不能逻辑上完全识别 a/b/c；它是最省算力、可依次排除的设计。

---

## 5. 下一轮四种候选机制、排序及止损门槛

> **效果预期的纪律**：所有数字为**为了值得投入而设定的最低可观察目标**，不是声称“理论会涨 +Xpp”。没有 2024–2026 文献可以证明某模块在本项目四数据集必涨。若实验未越过门槛则判失败；不因“小幅单点正值”续押。

| 优先级 | 候选 | 类别 | 与失败路线的实质区别 | 最小完整预算 | **预注册希望看到的幅度 / 理由** | 失败时止损 |
|---|---|---|---|---|---|---|
| **0（必做）** | **J0 可重复性/有效效应定义** | 改测量/协议 | 不是训练 Agent，不再争论 tiny MLP 的选择率 | `2域×2次×≤1000steps`（非正式） | 不追求 F1 增益；要证明 paired 初始化/增强/数值稳定 | 无法再现则暂停所有新机制 40K 比较 |
| **1（机制判决，首选）** | **R-TASK：GT 一致的教师任务信息转译** | 改蒸馏机制 | teacher 不再作为原始 128ch 模仿目标，而先让 GT 确认可用区域、再监督 1ch 任务分离；严格与 GT/GATE 拆分 | SYSU `4×40K`（含新 C1） | 需要 `TASKKD−GATE>0.82pp`（现噪声条件），使其大于 SYSU 差值噪声尺度；实际能否达到未知 [H] | 无明显独立效应→不再扩建 teacher pool/agent |
| **2（下一主论文首选）** | **S-PCG：对称时相上下文条件化 DCA Gate** | 改学生结构 | 原 DCA gate 只见差分 c2–c5；让 gate 见 `mean(F_A,F_B)` 与 `abs(F_A−F_B)`，在 3 专家结构**不增叠专家**情况下区分伪变化/真变化 | 4数据集 `2臂×40K=8×40K`（旧 C1 如可比可减少） | **SYSU ≥+0.9pp、WHU ≥+1.6pp** 才超过当前双独立-run 噪声阈；LEVIR≥+0.55、CDD≥+0.04，IoU 同向。门槛严苛，预期概率不保证；真实可重复后应重算 | 若双时相 context gate 没改善 unchanged FPR/boundary F1，停止 |
| 3 | **C-HN：GT 监督下的伪变化困难负样本一致性** | 改训练期监督 | 不用任何 teacher、只从 train GT 的 unchanged 中挖高 `|A-B|` 干扰像素，训练期辅助分类/对比；部署不加模块 | 首筛 SYSU+WHU `2臂×2域=4×40K` | 目标超出当前 SYSU~0.82/WHU~1.52pp 差值噪声；无实证收益承诺 [H] | 若只提升 OA 而 Recall/F1/IoU 不动，则无价值 |

**为什么先 J1 而不是直接四域 S-PCG？** J1 解决的是核心“教师强却难蒸馏”的科学归因，4 个 40K SYSU 比重新扩 9 教师经济得多；它的因果对照干净，可以**决定是否彻底停止教师选择**。但从“最终四数据集硬目标”出发，如果 J1 没有高于噪声尺度的 teacher 独立效应，**S-PCG 应取代 KD 成为论文主轴**，不能再往旧 Agent 换个包装。

### 5.1 S-PCG 的可证伪机制：不是简单“再加注意力”

问题：当前 DCA 三专家（detail/context/noise）用于变化特征 `c_s`，gate 仅见 `GAP(c_s)`，容易把“强烈外观变化”误认为“真 GT 变化”；因为前端双时相压缩为差分后，同一差值可能来自物体生灭也可能来自光照。**[H]** 增加一个不带 teacher 的 *对称共性上下文* 用于 gate（不是直接注入主特征路径）可以识别伪变化。

同尺度聚合特征 `a_s,b_s∈R^{64×h_s×w_s}`，设 `d_s=|a_s-b_s|`、`m_s=(a_s+b_s)/2`、`v_s=|GAP(a_s)-GAP(b_s)|`，gate 改为：

\[
\pi_s=\operatorname{softmax}\big(\mathrm{MLP}_s([GAP(d_s),GAP(m_s),v_s])\big),
\quad \hat c_s=\sum_{k\in\{detail,context,noise\}}\pi_{s,k} E_{s,k}(c_s).
\]

修改只作用于 DCA gate 的输入，**原有专家和 Decoder 完全不变**。`m_s`、`d_s`、`v_s` 对时相交换严格对称；推理不需 label/teacher；参数增量来自 gate 首层小矩阵，不应接近 1M。与直接 `concat(F_A,F_B)` 不同，它不会暴露顺序，不引入无意义的时相交换敏感性。**数学上可能存在“对称上下文不足以解耦伪变化”**，故必须对 train/test 不变区 FPR（评估只报告 test）与变化连通对象 IoU 作独立分析。

**最小消融**：`S0=C1 原 gate`、`S1=只增加 d_s 多尺度统计`、`S2=S-PCG 对称 mean+diff gate`；优先 SYSU/WHU，若 S2≤S1 则“配对共性信息”没有额外价值，不能把总收益归给它。所有臂固定相同专家数/width=128、40K/2333/64。

### 5.2 C-HN 的创新界限

若仅给困难负样本加权 BCE，不足以独立称论文创新。只有当它形成**明确的双时相伪变化机制**（例如仅在 GT=0 且亮度变化高、位移模拟高的区域生成训练期双分支判别约束），并证明相对“普通 class-balanced BCE”和“hard-negative random”有跨域差值，才能写成方法。C-HN 训练期辅助头删除，主学生结构不变，优先检验 SYSU/WHU unchanged FPR 而不是把 OA 上涨当主要证据。

---

## 6. 噪声、同 seed 重复、配对与统计边界

### 6.1 σ 的正式定义以及 n=4 的严重不足

对同数据集、同配置、同 seed、不同独立启动（不同运行环境状态）的 F1 值 `X_1,…,X_n`，经验标准差：

\[
\widehat\sigma=\sqrt{\frac1{n-1}\sum_{i=1}^n(X_i-\overline X)^2}.
\]

**在独立、同分布、近似正态且没有超参数/模型选择额外适应性筛选时**，方差的 χ² 区间为：

\[
CI_{95\%}(\sigma)=\left[\widehat\sigma\sqrt{\frac{n-1}{\chi^2_{0.975,n-1}}},\;
\widehat\sigma\sqrt{\frac{n-1}{\chi^2_{0.025,n-1}}}\right].
\]

| 重复次数 n | 95% σ 区间相对 `s` 倍数 | 为四个数据集从已有 4 次增到 n 的代价 |
|---:|---:|---|
| **4（已有）** | **[0.566, 3.729] ×s** | 0 |
| 6 | [0.624, 2.453] ×s | +8 个 C1 40K run |
| 8 | [0.661, 2.035] ×s | +16 个 C1 40K run |
| 12 | [0.708, 1.698] ×s | +32 个 C1 40K run |
| 20 | [0.760, 1.461] ×s | +64 个 C1 40K run |
| 30 | [0.796, 1.344] ×s | +104 个 C1 40K run |

这些 CI 依赖正态独立假设；同 seed 并发训练很可能不完全独立，故以上只是**条件化尺度诊断**，不是随意报告成严格统计置信区间。**不建议现在补到 n=8/12**：为更准估一个 σ 牺牲 16–32 个 40K，性价比太差，还未解决 teacher-target mismatch。

### 6.2 现有 σ 的经验保守阈值

若 C1/TV 两次独立运行且方差均近似 `σ_D²`：

\[
\sigma_{\Delta,D}=\sqrt{\sigma^2_{C1,D}+\sigma^2_{TV,D}}\approx\sqrt2\sigma_D,
\qquad \tau_D=\max(0.20,2\sqrt2\widehat\sigma_D).
\]

| 数据集 | σ_test(C1) pp | √2σ（单差）pp | `τ_D`=max(0.20,2√2σ) pp | 实验解读 |
|---|---:|---:|---:|---|
| SYSU | 0.289 | 0.409 | **0.82** | RADIO 的 +0.68 <阈值；可重点复核，不能定效 |
| WHU | 0.537 | 0.759 | **1.52** | MaRS +0.56 与噪声同量级 |
| CDD | 0.013 | 0.018 | **0.20** | 本表工程最小增益 0.20pp 主导；不是 CDD σ 不能检测小差 |
| LEVIR | 0.195 | 0.276 | **0.55** | D3-SAT +0.14 不可可靠判为正效应 |

**关键**：这里是工程阈值；并非“2σ 就 p<0.05”。**多重比较** 9 教师×4 数据集已先挑最大值，事后最大观察还需要考虑赢家偏差。若未来**有足量独立配对差值**（例如预先规定 `n_pair≥6` 只用于一次确认，不替代主单跑表），可在诊断性分析中采用配对 t 区间 `\bar d ±t_{1-α/2,n-1}s_d/√n` 并为预先设定的 K 项比较用 Holm 校正，只有显式说明前提才能称探索性显著。**项目正式论文遵守“不做多 seed、不做显著性检验”纪律**，因此目前应避免任何 p 值、2.6σ“接近显著”等字样。

### 6.3 是否用配对、能否扩大 val/test 来降 σ？

**建议配对**：统一 GPU 卡、并发方式、初始学生层权重、数据顺序、几何增强与 checkpoint 规则；建立 `pair_id`、配置 hash 及 `data_order_sha`，报告 `d_i=F1_{method,i}-F1_{C1,i}`。配对的作用是**减少两个臂共有的环境/数据抽样波动**，并非制造同配置相同 F1 的保证。只有在可证明具有正协方差时才会降低 `Var(d)`。

**不建议只重复推理**：同一 checkpoint、同一确定性测试集推理重复 100 次，F1 几乎是同一个数，不能消除**训练结果方差**。扩大评估样本仅可能降低有限测试样本下的抽样不确定性，不能降低由不同 40K 训练终点/val 选 checkpoint 导致的 run-to-run σ；**不能把新的测试图片无协议纳入四固定公开数据集，或重新划分公开 test 后混报基准**。可在 **train-only 内部分层诊断 split** 估差，或对固定 test 样本做**配对图像 bootstrap 作为模型预测差值的不确定性描述**（像素有空间相关、应按整图/场景聚类，不按像素独立采样；不得据此声称训练重复性）。

**固定 seed 重复与多 seed 的区别**：同 seed 多次重启检验**实现/硬件/调度残余随机性**；多 seed 测量对初始化/数据随机种子的敏感性。二者估计的随机变量不同，**都不能直接互相替代**。项目允许保留已有 C1 重复作为噪声底**诊断**，正式主实验仅报告一次从头完整 40K 的预登记 run，不能用重复均值包装主结果。

### 6.4 GPU 时间：只给真实可执行的核算式，不虚构每 run 小时数

令 `t_SYSU,t_WHU,t_LEVIR,t_CDD` 为对应现有 `train_log.txt` 末尾 `Total time` 解析出的**单个 40K 训练 GPU 小时**。则：

- J0 成本：每个域 2×(≤1000/40000)×`t_D`，两个域合计约 `0.05×(t_SYSU+t_WHU)` **GPUh**（实际不严格线性，初始化/验证固定开销需单独记录）；
- J1 四臂 SYSU：**4×`t_SYSU` GPUh**，若严格测量的新 C1 已完成，可少一次；
- J1 在 WHU 复核 **J1-C1/J1-GATE/J1-TASKKD**：**3×`t_WHU` GPUh**；
- S-PCG 四数据集 C1/S1/S2：`3×(t_SYSU+t_WHU+t_LEVIR+t_CDD)` GPUh；
- 若噪声底追加到 n=6（C1 两次/域）：`2×Σt_D` GPUh。

两块 RTX 5090 可同时跑，但 wall time 取决于 2-way/3-way 并发与显存，**不是 GPUh 直接除 2 就严格成立**。先在已有日志中提取时间，再精确填写实验预算表。

---

## 7. 首选方法数学定义、训练图与部署图

### 7.1 推荐 J1-TASKKD 是**判决机制**而非已成立新方法

**候选名称：GT-Conditioned Reliable Task Distillation（GRTD）**。其贡献假设不是“新 loss 提升一切”，而是：**大模型的任务相关软信息仅在 teacher 响应与 GT 一致的局部区域有效**；因此先用 train GT 将 teacher response 的可靠区域/不可靠区域分开，避免把背景伪变化和低分辨率边缘当有用监督。保留 teacher 的局部软排序，但不逼学生学随机128维表示方向。

变量：二时相学生主模型 `S_θ(A,B)`；经 DCA 后的 `c2`（64ch，stride4）；训练期 1ch aux `h_φ(c2)`；标签 `y∈{0,1}^{256×256}`；教师已缓存的非概率 change-response `z_T=confidence`；`q_T` 是 train-only 校准的连续目标。**教师编码器全部 frozen + offline Cache**。辅助预测不反馈部署 logits；通过辅助 loss 对学生主特征传梯度（不可 detach 学生）。

**步骤 A：全数据集固定 teacher response 映射（不做逐图 z-score）**。

\[
q_T(p)=\sigma\left(\frac{z_T(p)-m_T^{\rm train}}{s_T^{\rm train}+\epsilon}\right),
\quad m_T=Q_{0.5}(z_T\mid\text{train}),\quad s_T=Q_{0.9}-Q_{0.5}.
\]

同一 teacher+dataset 的 `m_T,s_T` 从**训练子集**估出且冻结；`q_T` 只是响应分数的单调压缩，不可谎称“教师校准概率”。是否采用 class-balanced affine/logistic calibration，须利用**train-meta 内部 heldout**评估 ECE/AUPRC，再冻结；绝不看 val/test 调整。若 q_T 在 unchanged 不可校准，则让 GATE 而非 TASKKD 成为研究结论。

**步骤 B：训练 GT 可靠区域 gate**。

\[
G^+(p)=y(p)\land[q_T(p)\ge t_+],\quad
G^-(p)=(1-y(p))\land[q_T(p)\le t_-],\quad
G=(G^+\cup G^-)\setminus B_r(y).
\]

其中 `B_r` 为 GT 边界 r=2px 的不确定带（在 **256×256 GT 后**定义）；`t_+,t_-` 只用 train 的 class-balanced quantiles 冻结。**拒绝在 G 为空时产生虚假 loss**；该 batch 跳过 auxiliary，而主 loss 照常训练。正负掩码分别归一化，避免 4.1%–21% 变化占比导致全背景占优。

**步骤 C：对齐辅助任务学习**。

\[
L_\mathrm{GATE}=\tfrac12\mathbb E_{p\sim G^+}\!BCEWithLogits(h_\phi(c2)(p),1)
+\tfrac12\mathbb E_{p\sim G^-}\!BCEWithLogits(h_\phi(c2)(p),0).
\]

\[
L_\mathrm{TASKKD}=\tfrac12\mathbb E_{p\sim G^+}\!BCEWithLogits(h_\phi(c2)(p),\operatorname{sg}(q_T(p)))
+\tfrac12\mathbb E_{p\sim G^-}\!BCEWithLogits(h_\phi(c2)(p),\operatorname{sg}(q_T(p))).
\]

`GATE` 与 `TASKKD` **同 head、同掩码 G、同主 loss/优化器/schedule、同样单时相交换不变的 Cache**，仅 target 不同。与当前 `dense_change_kd` 的本质区别是 task alignment、GT 冲突约束和同一教师软证据的独立因果对照。注意二值 change GT 下 `q_T` 在 gate 区域通常趋近 hard GT；其额外 soft 信息可能很少，因此 **[H] 很可能失败**，恰好能强有力地终止这条路线。

`L_total=L_main + λ·L_aux`。允许沿用同一个既有 gradient-budget 公式作为实验控制（但必须记录实际 λ 与梯度余弦、支持 c2 后模型梯度计算）；**不要为 TASKKD 单独调大 λ**。如果同一 loss 常被 λ 压到极低，先在 J0 报清，不把调 λ 称为创新。

### 7.2 训练图与部署图

```text
训练 A/B + GT (256²)                 OFFLINE Teacher Cache (train only)
       │                                          │
    Shared LWGANet-L0                           z_T, 128ch evidence
       │                                          │
  SWA(A), SWA(B)                         [only for J1: frozen z_T -> q_T]
       │                                          │
  symmetric TFM -> (c2,c3,c4,c5)                  ├─ GT-consistency region G
       │                                          │
    DCA (deployable)                              │
       ├── Decoder -> L_main                      │
       └── 1ch train-only aux on c2 ──────────────┘
                -> L_GATE or L_TASKKD

部署 A/B -> Shared LWGANet -> SWA -> symmetric TFM -> DCA -> Decoder -> change map
       （无 teacher、Cache、agent、translator、aux head、GT、G）
```

**容量/FLOPs**：J1 所有臂 inference **仍是 C1=3,280,274 参数，256²约3.2432G FLOPs**；1ch 训练期辅助 head 若为 `Conv2d(64,1,kernel_size=1)` 仅 65 参数，可忽略训练期容量差（但主表如实记 `Aux Params=65`），GATE/TASKKD 完全一致。`switch_to_deploy()` 必须删除/不注册 `aux`；部署主预测在 `aux_on/off` 与 deploy 前后相差 `<1e−6`，时间交换预测 max error `<1e−6`。**新 S-PCG** 若被启动，需要重新现场统计 params/FLOPs；不可直接延用 3.2432G。

### 7.3 最小消融与图表

- **机制矩阵**：C1、GT-only、GATE(GT label)、TASKKD(teacher soft target)。
- **teacher 信号核验**：`z_T` train GT-balanced AUROC、AUPRC（因正负不平衡更适合）、FPR@fixed recall、对象大小分层 AUC、boundary F1；不得将这些训练指标充当正式 test 结果。
- **Student 可学性**：train-only 线性 probe AUC/正负 margin；`L_main`/`L_aux` 每1000 step 梯度余弦；`λ` 中位数/10%分位。
- **最终图**：四数据集 test F1/IoU 相对 clean C1 的 waterfall（不能拿跨配置不同 seed 的旧结果填新臂），另附 Recall/Precision/OA/Kappa、params/FLOPs、主要 FPR/boundary/size 分析。

---

## 8. 逐文件修改、实验 ID、启动顺序、路径及验证

> 此节是**拟修改清单**，不是声称已改好。**严禁在未授权情况下覆盖原 Cache/旧 checkpoint/既有 registry**。所有修改单独新分支、独立目录、先审查暂存区。

### 8.1 P0/P1 分级逐文件表

| 优先级 | 文件 | 精确修改 | 测试/判定 |
|---|---|---|---|
| **P0** | `models/tools/train_safe_teacher_agent.py` | AG-CH 的阈值判定不读取 `y[te]`；heldout val **只后评**。严格模式 `u_noise` 不读取 heldout C1 val σ；保留现转导模式另名 `transductive_sigma` | 构造 heldout y 全置换，**动作应完全不变** |
| **P0** | `models/tools/build_capability_registry.py` | `C0_test_pp` 全字段 ×100；读取最后完整 TEST block 且检查完整性/哈希/配置; 建新 v3 registry 不覆盖 v2 | `C0_test_pp.SYSU.F1≈82.1332`，非 0.821332 |
| **P0** | 新 `models/tools/audit_reproducibility.py` | 初始模型 state_dict SHA、数据顺序 SHA、crop/flip/exchange SHA、每步 loss/gradient/RNG hash；收集 cuDNN/TF32/cuBLAS 设定 | 同 seed 2 次 J0 初始权重与首 batch 一致；记录数值分歧 |
| **P0** | `models/agent/safe_selector.py` | 修复 `len(X)!=len(y)!=len(groups)` 链式错误；若无训练域 group，不返回假确定性；标注 spread proxy 非 CI | mismatch raises；groups<2 fail |
| **P1** | `others/_probe_utility_corr.py` | 替换双 argsort → ties-aware scipy.spearmanr；报告各域有效点数、pearson/pool仅描述 | shuffled pairing check；随机标签相关分布 sanity |
| **P1** | `models/distill/capability_probe.py` | 新增 GT-balanced teacher soft response 质量与 student linear-probe 可蒸馏性对照，**只读 train** | 变换单调性、两类空样本、NaN 单测 |
| **P1** | 新 `models/distill/task_reliable_kd.py` | 实现 J1 的 train-only `q_T/G/L_GT/L_GATE/L_TASKKD`，所有参数/阈值 train 派生冻结 | 梯度流、G 正负均衡、G empty safely skip |
| **P1** | `models/scripts/train.py` | 新 CLI `--aux_task {none,gt,gate,taskkd}`；new aux 仅 train；主预测无条件独立，记录部署/训练参数 | C1 主路径 aux=on/off **before backward** logits max diff<1e−6 |
| **P1** | 新 `models/tools/audit_taskkd.py` | 统计 teacher Q 分布、分位阈值、joint confusion、分层 AUROC、teacher mask | 不能读 val/test labels |
| **P2** | `train_scripts/CATA-CD/Run2/`（新目录） | 独立脚本 `common.sh`、`run_j0.sh`、`run_j1_sysu.sh`、`README.md`，一臂一唯一目录和日志 | `bash -n` 全通过，无脚本自动删旧数据 |
| **P2** | `analyse/extract_metrics.py` | 新 J1 ID 逐数据集聚合 6 metrics + params FLOPs；正式仅最后完整 TEST block | 人工插入两个完整块及尾部不完整块的单测 |
| **未来可选** | `models/decoder/deployable_change_adapter.py`、`models/a2net.py` | 如果 J1 止损后选 S-PCG，仅改变 DCA gate 输入，从 TFM 前聚合特征读取对称 mean/diff 统计；不动 experts | swap/deploy <1e−6，新增参数和 FLOPs 现场实测 |

**这里不重做**：已完成的 20维 signature、5个 P1–P5 probe、metadata、ridge/LODO、Safe-set δ=0.20；它们是诊断性既有结果，不是下轮创新。

### 8.2 预注册实验 ID（所有**正式** run 同 ImageNet 预训练、seed=2333、batch=64、40K，从头训练）

| Stage | ID | 数据集 | 教师 | 模型与辅助（唯一变化） | 正式 40K | 主要检验 | 失败解释 |
|---|---|---|---|---|---:|---|---|
| Audit | J0-REP | SYSU/WHU | None | 同配置 2×100–1000 step 测试 | **0** | 数值/数据流一致 | 未稳定 → 阻断 J1 |
| 判决 | J1-C1 | SYSU | None | clean C1 DCA | 1 | strict anchor | 重现失败 → 先改协议 |
| 判决 | J1-GT | SYSU | None | C1 + 1ch GT auxiliary | 1 | GT 辅助是否有值 | ≈C1 → target-free aux 无用 |
| 判决 | J1-GATE | SYSU | D3-LVD (train-only) | J1-GT + teacher/GT 一致 region gate，GT target | 1 | 教师是否有筛样价值 | ≈J1-GT → gate 无用 |
| **判决主臂** | **J1-TASKKD** | **SYSU** | D3-LVD (train-only) | J1-GATE 的 GT target → teacher soft target | **1** | teacher 软信息独立效用是否超噪声 | ≤GATE 或 <0.82pp → KD 选师止损 |
| 可选复核 | J1-WHU-* | WHU | D3-LVD | 仅复制 3个最关键臂 C1/GATE/TASKKD | 3 | 机制跨域而非 SYSU 偶然 | 无可辨信号 → 停止 teacher |
| 独立结构 | S-PCG-C1 | 四数据集 | None | 现 C1（若新 strict protocol 需全重建） | 4 | 每域 clean anchor | anchor 混杂不可比 |
| 独立结构 | S-PCG-STAT | 四数据集 | None | gate 仅增加 pair-diff statistics | 4 | 容量/statistics control | 无收益则不进 S-PCG |
| **结构主臂** | **S-PCG-M1** | 四数据集 | None | gate 增加 symmetric pair mean/diff conditioning | **4** | 输入机制额外收益 | ≤STAT，缺乏共性上下文创新 |
| 备选 GT-only | C-HN-* | SYSU/WHU 首筛 | None | 训练期伪变化困难负样本辅监督，deploy同C1 | 每臂每域1 | 不靠教师是否提升 F1 | 不超噪声，不扩四域 |

**正式首轮最多优先执行 J0（无 40K）+ J1 SYSU 4个40K**，不同时启动四个方向。J1 如果不通过，不“换另一个教师重跑 36 宫格”；直接提交负结果分析，转 S-PCG。

### 8.3 训练/部署不可破坏的检查

1. **构图正确性**：同一 `models.a2net.A2Net_LWGANet_L0`，J1-C1/J1-GT/J1-GATE/J1-TASKKD 学生主图逐参数一致；1ch aux head 两个 teacher 控制臂完全等容量。
2. **时间交换**：对固定 `(A,B)` 和 `(B,A)` 预测四尺度 max abs diff `<1e−6`；Cache 已先对双时相做 abs change，无须按 swap 交换 map，但 crop/resize/flip **必须同样重放**。
3. **Aug replay**：缓存每个例子的 sample ID、crop/flip/exchange、teacher source 256、native cache H/W；在真 Cache dry run 上检查 GT 与 teacher spatial map 的 known translated grid，尤其 14x14/16x16/64x64。CDF label JPEG 使用统一≥128。
4. **Loss-only**：aux head 接 post-DCA student c2；teacher `detach/no_grad`；主预测在 aux 构造/开关时**同输入前向完全一致**；真实 backward `∇_{student}L_aux≠0`。测试 taskKD 的 teacher G 与 GT 在时相交换后不变。
5. **预算**：J1 训练 student=3,280,274 + aux=65；部署 <5,000,000，teacher/aux/cache 不能出现在部署 state_dict；256² GFLOPs C1 已归档约3.2432G，实际再测；deploy 前后 `<1e−6`。
6. **完整训练协议**：`max_steps=40000`、seed2333、batch64，预训练只加载 `pre-trained_weights/lwganet_l0_e299.pth`。`--resume` 只用于**同一实验 ID/完全同配置、真正中断恢复**；不准从旧 C1 checkpoint 继续训练新 arm。
7. **日志**：必须有最后一组完整 `=== TEST RESULTS ===…=== END TEST RESULTS ===`，含 Recall/Precision/OA/F1/IoU/Kappa/参数/FLOPs、fingerprint、完整 40K、训练耗时。若不完整则失败，不用验证最佳行顶替。

### 8.4 建议目录与命令骨架（**拟新建，不代表仓库当前已有**）

```text
train_scripts/CATA-CD/Run2/
   README.md
   common.sh
   run_j0.sh
   run_j1_sysu.sh
   run_j1_whu_optional.sh
models/distill/task_reliable_kd.py                 # new
models/tools/audit_reproducibility.py             # new
models/tools/audit_taskkd.py                     # new
outputs/CATA-CD/Run2/J1-C1/SYSU/train_log.txt     # proposed
outputs/CATA-CD/Run2/J1-GT/SYSU/train_log.txt
outputs/CATA-CD/Run2/J1-GATE/SYSU/train_log.txt
outputs/CATA-CD/Run2/J1-TASKKD/SYSU/train_log.txt
/share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/CATA-CD/Run2/<ID>/<DS>/
```

**示例：准备执行时的逐项检查（先不要直接运行训练命令；新 CLI 须按修改实现）**：

```bash
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate lsrep
cd /home/yqwang/projects/LS-Rep_BCD_RSML_3

git status --short
git rev-parse HEAD
python -m compileall -q models/agent models/distill models/tools
bash -n train_scripts/CATA-CD/Run2/common.sh
bash -n train_scripts/CATA-CD/Run2/run_j0.sh
bash -n train_scripts/CATA-CD/Run2/run_j1_sysu.sh

# 先 synthetic smoke，再 REAL teacher cache + exact augmentation dry run。
# 通过后由新 Run2 README 中唯一、明示的 run ID 启动正式 40K。
```

建议对 `strict_resume` 做实验 ID/seed/step/batch/teacher Cache manifest SHA/数据 split SHA/实现版本严格校验；两个臂即使只差 aux target，也**不能**共享同一个最佳 checkpoint。Git 提交前 `git diff --cached --name-only`、`git status --short`；禁止提交 pre-trained weights/Teacher Cache/data/log/checkpoint。

---

## 9. 论文可发表性与不能再宣称的内容

### 9.1 负结果单独作为论文主贡献，目前**不足以投稿高水平会议/期刊**

目前可写**技术报告、硕士论文中的证据章节或论文相关实验中的有价值的负结果**。但“Teacher Selection Signal Insufficiency in Lightweight Remote Sensing CD”作为主研究贡献，还缺少决定性的证据：

- **欠缺同等容量/同等计算的 teacher-target vs GT-only 对照**。目前只证明一套 128ch cosine KD 失败，没证明蒸馏迁移不可行；
- **Agent baseline 不完全干净**（AG-CH LODO held-out val 泄漏）；
- **实验独立单位为四域**、同 seed run-to-run 波动，LODO “zero action”可由 δ>所有 val正标签完全解释，缺少非平凡策略优化行为；
- **有选择性地挑最大 test 正收益后讨论教师排序会放大 winner's curse**；
- **缺 44+12+4 实验的可发布/可审计最后 TEST block 摘要和所有 train/probe 数据 SHA**；
- **关键科学链条**“teacher response quality→target learnability→student gradient alignment→CD pixel F1”尚未因果分解。

若 J1 发现“teacher soft target 无额外价值但 GT-only 明显提升”，可作为论文结构线的反例/机制诊断；若 J1 发现 taskKD 独立明显有效，才有资格围绕“任务可靠 teacher transfer”提出新的方法论文。负结果只能作为**经过限定范围的观察**，不得宣传普适不可蒸馏定理。

### 9.2 保留 / 冻结 / 放弃

| 决策 | 资产 | 原因 |
|---|---|---|
| **保留** | C0/C1 干净 anchor、DCA 与 swap/deploy invariant | 真实可部署，C1 在 WHU/SYSU 有值得复核的结构收益 |
| **保留作诊断资产** | 9 teacher×4 数据集 Capability Registry、Cache schema、train probe/P1–P5、40K 运行记录 | 可解释 teacher response 层差异、避免重复“教师越大越好”假设 |
| **封存非正式** | 旧 MLP、U-SafeAgent AG-03、AG-CH 消融 | 已失败/空动作/基线公平性待修；不再宣称策略成功 |
| **停止扩张** | 新 teacher 名单、LODO neural meta-agent、复杂 bandit/RL | 只有4独立域、0强正 val 标签，扩网络没有学习信号 |
| **下一篇论文首选** | S-PCG：利用对称共性上下文优化 DCA 门控 | 不依赖 teacher，针对二时相伪变化的机制可证伪；需真实四域 40K |
| **仅条件保留** | task-aware 教师知识转译 | J1-TASKKD 相对 GATE 超噪声才继续，失败即停 |

### 9.3 不能再宣称（投稿纪律）

1. ❌ “U-SafeAgent 学会了高效选教师，在四个数据集上避免负迁移”；实际 AG-03 **全选 None**，这是拒答而非策略选择能力。
2. ❌ “零负迁移证明安全 Agent 有 formal risk guarantee”；`u_model` 只有三组 leave-one-domain refit 的 std proxy，无 conformal coverage/CI。
3. ❌ “36 组中30组非正/非强正”；归档是**26个≤0 /33个非强正**。
4. ❌ “所有 val 效用在±0.15pp 内”；**正值**在+0.146pp以内，**负值**可达−1.076pp。
5. ❌ “DINOv3-LVD teacher AUROC 0.87 ⇒ 应该最适合 SYSU”；train AUC 测的是 change-response，不是 student learnability。
6. ❌ “pooled Spearman +0.03 ⇒ 教师能力和可蒸馏性客观独立”；只有四个相关数据集簇，rank 实现还有 tie 问题。
7. ❌ “RADIO→SYSU +0.68 已统计显著 / 2.4σ 接近显著”；σ 不是教师差值的 SE，且进行了九教师筛选。
8. ❌ “MaRS→WHU +0.56 是纯噪声/完全无效”；只能说**处于观测的单次运行噪声尺度**，不能证明真实效应=0。
9. ❌ “所有 foundation teacher 对轻学生无用/知识无法迁入”；只评估了一种冻结 teacher 证据生成与 cosine loss。
10. ❌ “C1 DCA 在四域普适提升”；CDD 是负差，LEVIR微小，原协议含 run-to-run 噪声。
11. ❌ “改 teacher selection 就能满足 F1 85/92.5/95/98”；现有效应远不够。
12. ❌ “每图 z-score 抹掉教师响应”；当前 `teacher_package.py` 没有做 per-image z-score。
13. ❌ “没有 test leakage，所有 LODO 基线完全公平”；AG-CH 的 held-out **val reward**参与动作拒答（与 test leakage 不同但依然严重）。
14. ❌ “论文正式结果独立来自最新 test 日志”；此审查尚缺原始完整日志，只核验过派生 JSON/源码。

---

## 10. 2024–2026 年核验文献

> 分类说明：CVPR/ICCV/ECCV 等列 **CCF-A 会议**；IEEE TGRS / ISPRS JPRS 列 **SCI 权威期刊**；ICML 2025 如引用需单独识别会议分类，不与 SCI 混称。引用的是**对机制或方法学的启发**，并不把它们在 COCO/ImageNet/语义分割的涨点直接转写为本项目涨点。只写已核到的论文主页/出版商页面和真实公开代码；无确定官方代码时注明“未核实”，不猜 URL。

### 10.1 CCF-A 会议

| 文献（题名、作者首位等） | 年份/venue/发表状态 | 核验论文主页 | GitHub / 与本项目关系 |
|---|---|---|---|
| **CrossKD: Cross-Head Knowledge Distillation for Object Detection**；Jiabao Wang et al. | **CVPR 2024，正式发表** | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection_CVPR_2024_paper.html) | [官方代码](https://github.com/jbwang1997/CrossKD)。解释 **feature imitation 与任务预测可能冲突**，但任务为目标检测，不是 RS-CD。 |
| **FreeKD: Knowledge Distillation via Semantic Frequency Prompt**；Yuan Zhang et al. | **CVPR 2024，正式发表** | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_FreeKD_Knowledge_Distillation_via_Semantic_Frequency_Prompt_CVPR_2024_paper.html) | 官方代码地址此轮**未完成核验**；说明逐尺度/高频细节里存在噪声与蒸馏关注区域问题。 |
| **Logit Standardization in Knowledge Distillation**；Shangquan Sun et al. | **CVPR 2024，正式发表** | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Sun_Logit_Standardization_in_Knowledge_Distillation_CVPR_2024_paper.html) | 代码地址此轮未核验。**文中 z-score 针对 logits KD**，不能误称本项目已经做逐图 z-score。 |
| **UNIC: Universal Classification Models via Multi-teacher Distillation**；Mert Bülent Sariyildiz et al. | **ECCV 2024，正式发表** | [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/581_ECCV_2024_paper.php) | [官方代码](https://github.com/naver/unic)。多 teacher 的收益依赖专门的蒸馏路径；其任务是分类，不保证本项目 teacher 筛选有效。 |
| **MoVE-KD: Knowledge Distillation for VLMs with Mixture of Visual Encoders**；Jiajun Cao et al. | **CVPR 2025，正式发表** | [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Cao_MoVE-KD_Knowledge_Distillation_for_VLMs_with_Mixture_of_Visual_Encoders_CVPR_2025_paper.html) | [官方代码](https://github.com/hey-cjj/MoVE-KD)。在目标任务差异下需要处理教师冲突和分歧；**不能直接借来多教师叠加**。 |
| **A Good Teacher Adapts Their Knowledge for Distillation**；Chengyao Qian et al. | **ICCV 2025，正式发表** | [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Qian_A_Good_Teacher_Adapts_Their_Knowledge_for_Distillation_ICCV_2025_paper.html) | 官方代码此轮**未核实**。直接支持“教师学生**能力与目标分布不匹配**可导致 KD 失效”的研究动机，未证明本项目是同一原因。 |
| **Perspective-Aware Teaching: Adapting Knowledge for Heterogeneous Distillation**；Jhe-Hao Lin et al. | **ICCV 2025，正式发表** | [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Lin_Perspective-Aware_Teaching_Adapting_Knowledge_for_Heterogeneous_Distillation_ICCV_2025_paper.html) | [官方代码](https://github.com/jimmylin0979/PAT)。处理异构教师/学生视角不一致；启发 adapter/区域可靠性，而非大模型越强越好。 |
| **What Makes a Good Dataset for Knowledge Distillation?**；Logan Frank, Jim Davis | **CVPR 2025，正式发表** | [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Frank_What_Makes_a_Good_Dataset_for_Knowledge_Distillation_CVPR_2025_paper.html) | [官方代码](https://github.com/osu-cvl/good-kd-dataset)。提醒 KD 效用依赖数据/协议；本项目为有标注 in-domain KD，**不可照搬 data-free 结果**。 |
| **Feature Spectrum Learning for Remote Sensing Change Detection**；Qi Zang et al. | **CVPR 2025，正式发表** | [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Zang_Feature_Spectrum_Learning_for_Remote_Sensing_Change_Detection_CVPR_2025_paper.html) | 官方代码此轮未核实。强调伪变化和频谱干扰，支撑 S-PCG/C-HN 研究动机，但结构与本项目方案不同。 |

### 10.2 SCI 权威期刊（不标 CCF-A）

| 文献 | 年份/期刊/状态 | 核验论文/DOI | GitHub / 与本项目关系 |
|---|---|---|---|
| **BiFA: Remote Sensing Image Change Detection With Bitemporal Feature Alignment**；Haotian Zhang et al. | **IEEE TGRS 2024，正式发表** | [IEEE Xplore，DOI 10.1109/TGRS.2024.3376673](https://ieeexplore.ieee.org/document/10471555) | [官方代码](https://github.com/zmoka-zht/BiFA)。双时相错位及光照干扰的相关先例；本项目若用 S-PCG，须明确其只改变 DCA symmetric gate 而非 BiFA 空间 flow alignment。 |
| **ChangeMamba: Remote Sensing Change Detection With Spatiotemporal State Space Model**；Hongruixuan Chen et al. | **IEEE TGRS 2024，正式发表** | [DOI 10.1109/TGRS.2024.3417253](https://doi.org/10.1109/TGRS.2024.3417253) | [官方代码](https://github.com/ChenHongruixuan/ChangeMamba)。时空互动的强先例；**不可将其复杂主干无代价迁入 <5M**。 |
| **Robust feature aggregation network for lightweight and effective remote sensing image change detection** | **ISPRS JPRS 2024，正式发表** | [出版社，DOI 10.1016/j.isprsjprs.2024.06.013](https://www.sciencedirect.com/science/article/pii/S092427162400251X) | 官方代码本轮未核实；轻量多尺度聚合的实质性相关工作，若发表 DCA gate 创新需对照区别。 |
| **Burden-Free Distillation From Foundation Model for Efficient Remote Sensing Change Detection**；Shuangru Wang et al. | **IEEE TGRS 2025，正式发表** | [DOI 10.1109/TGRS.2025.3584094](https://doi.org/10.1109/TGRS.2025.3584094) | [作者仓库](https://github.com/Younger-hua/Burden-Free-Distillation)。与“大模型训练期辅助、部署无额外负担”高度相关；不能由强论文假定本项目任意 Cache 有正迁移。 |

### 10.3 “负结果/不可复现性”文献适用性说明

2024–2026 年近似直接命中“**四域、单种子、同种子并行 GPU 非确定性、教师选择 F1 噪声底**”的**权威正式论文**本轮未找到可核验的一对一研究，**不虚构专用引用**。统计标准差 χ² 区间、配对差值设计、Holm 多重检验属于通用统计原理，**不以某个 2025 CVPR 新方法的名义包装成研究创新**。项目自身的 `C1+R1/R2/R3` 数据才是本论文相关方法学观察的直接证据；但它没有消除四域独立样本数量限制。

---

## 11. 立即执行顺序、缺失证据与最终闸门

### 11.1 立即执行顺序（严格逐项，不同时铺开）

1. **冻结仓库证据**：`git rev-parse HEAD` 确认为 `2b1c087…` 或记录新的差异；不要直接根据移动中的 `main` 混用旧日志与新源码；对所有 44+12+4 `train_log.txt` 生成最后完整 TEST block 摘要及 source SHA。（如当前 HEAD 已变，应重新 diff。）
2. **修正 P0**：AG-CH heldout val gating；C0 anchor pp 单位；安全 selector 长度检查；拆分 inductive vs transductive LODO；输出新的 `registry_v3/`、`agent_audit_v3/`，**不覆盖原有 Registry**。
3. **J0**：SYSU、WHU 各两次短运行排查第 0 epoch 分叉，输出 initial/augmentation/loss hashes；不投入正式 40K。
4. **Train-only 机制无训练探针**：固定 D3-LVD/SYSU，对比 `z_T` vs `Y_T` changed/unchanged margin、AUPRC/size/boundary、逐层梯度冲突、真实 KD λ。若 teacher response 经 fixed projection 后已没有足够可学信息，优先跳过旧 KD，不再做教师选择。
5. **J1 4×40K**：C1、GT-only、GATE、TASKKD，仅 SYSU 且一组同协议。全部从头初始化，无 checkpoint 续训当新实验。清晰报告四臂 test 最后 block，若结果不超噪声尺度即判蒸馏主张失败。
6. **转论文主轴**：如 J1 失败，开始 S-PCG `S0/S1/S2` 四域机制表；如果 S2 未独立优于 S1，拒绝该创新；不要改个模块名继续堆叠。
7. **最后才考虑恢复 Agent**：只有 teacher task transfer 在至少两个域明显可靠、且 train-only val 排序与 test 在关键方向相符，才允许对 4 域选择学习作探索；否则删除 Agent 主标题、保留分析证据。

### 11.2 仍需补充证据（不要用猜测替代）

| 缺口 | 为什么不可缺 | 获取方式 |
|---|---|---|
| 44+12+4 份真实训练日志最后完整 TEST block、完整 40K 与耗时 | 确认 Registry 对历史日志的追踪真实性，计算 GPUh | RSML-3 对每份 `train_log.txt` 使用**最后完整块**扫描，不覆盖文件 |
| 4×9 `probes_v2` 完整 JSON 和 sample-id hashes | 重算 Spearman ties-aware、teacher z 对 GT 的分层排序 | 将 train-only 探针摘要导出，不外传私有 Cache 权重数据 |
| 每个教师 Cache manifest weight SHA/native HW、抽样 replay 用例 | 排除差异由权重/输入尺寸/增强失配造成 | `audit_teacher_cache.py` + 新 mask replay sanity |
| C1 / R1/R2/R3 实际 val 完整曲线、best epoch | 区分 fixed-step 波动与 checkpoint selection 放大 | 从存量日志提取；不可新造随机数 |
| `nvidia-smi`/CUDA/cuDNN/TF32/PyTorch 运行设置、同 config 初始化 hash | 排查 epoch0 不一致的原因 | J0 paired-short forensic |
| 学生 c2/c3/c4/c5 线性 probe 能力及 teacher/GT 的梯度冲突 | 区分教师质量与可蒸馏性 | train-only，不用 test 选教师 |
| 当前各数据集准确 GPU 单次训练时长 | 给出小时而非推测 | 日志 `Total time`，按每个 run 统计 |

### 11.3 最终可证伪 Gate（写在新 Run2 README 顶部）

- **Gate R0（正确性）**：任何关键 P0 未修复、不能重复同 config 初始化/数据序列、缺最后完整 test block → **不得解释 40K 差值**。
- **Gate R1（teacher knowledge）**：J1-TASKKD 对 J1-GATE **同辅助容量、同训练预算、同 teacher gate** 的 SYSU F1 增益 **需超 0.82pp**（基于现有 σ 的工程门槛）、IoU 同方向，且 train-only teacher/task probe 机制证据一致；否则**停止 teacher selector 作为论文主线**。
- **Gate R2（structure）**：S-PCG-S2 同协议优于 S1 的四域 test F1/IoU；重点看 SYSU、LEVIR、WHU，CDD不下降；deploy<5M/轻量 FLOPs/swap/deploy<1e−6。未达则拒绝该 gate 假设，不新增第四专家。
- **Gate R3（硬目标）**：单次同协议公开测试四 F1 必须 SYSU≥85、LEVIR≥92.5、WHU≥95、CDD≥98，IoU 同向；未全部达到就仍是**方法探寻阶段**，不可宣称完成论文目标。

**最终建议**：这轮失败既不是“只需再给 Agent 加点特征”，也不能草率归因于“9 个大模型都没价值”。目前最有力的解释是：**四独立域、val 正向信号过小、运行噪声不小，加上一条非 task-aligned 的随机投影 cosine KD 链路，使“择师收益”无法被可信辨认和学习。** 先完成可重复性审计与**GT-only / teacher-gate / teacher-soft 三重拆分**；如果拆不出教师的独立贡献，就正式止损，转向**不依赖教师的对称双时相 DCA 门控**，逐数据集 40K 验证真正有论文价值的结构创新。
