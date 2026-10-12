# CATA-CD v2 第三轮：保持九教师与适配性选师 Agent 不变的知识通道重建方案

> **证据锁定日**：2026-10-12　　**仓库**：<https://github.com/YuqiWang-code/LS-Rep_BCD>  
> **唯一代码基线**：`e6e9827c8f35bb93a4c85de028a6775f249addba`（下文简称 `e6e9827`）  
> **任务**：四个 256×256 公共数据集上的全监督二值遥感变化检测；学生 `A2Net-LWGANet-L0 + DCA(moe128)`；九教师与数据集适配性选师 Agent 作为**不可替换的论文主线**。  
> **证据标签**：**[代码事实]**＝固定 commit 中能逐行看到；**[报告事实]**＝该 commit 已提交的实验汇总（未拿到原始服务器日志独立复算）；**[推断]**＝现有证据支持但不是唯一解释；**[待验证假设]**＝必须经新对照实验验证。  
> **读取范围与限制**：通过已连接的 GitHub 接口逐文件核对固定 commit 的核心模型、Cache、KD、Agent、训练入口、脚本和 Run2 报告，并对照用户提供的能力矩阵、环境文档及 2024–2026 年论文主页；运行环境无法直接解析 github.com 域名，**没有在本机完成 `git clone`**，也没有连接 RSML-3 / 查验原始 `train_log.txt` 或执行 GPU 测试。下文不会把结果汇总伪称为原始日志审计。

---

## 目录

1. [结论与边界](#1-结论与边界)
2. [固定代码和历史证据审计](#2-固定代码和历史证据审计)
3. [根因排序与可证伪判别](#3-根因排序与可证伪判别)
4. [候选知识通道比较：只选一个主方案](#4-候选知识通道比较只选一个主方案)
5. [首选机制：CTP-Critic 数学定义、训练图与部署图](#5-首选机制ctp-critic-数学定义训练图与部署图)
6. [Agent 重建与基线](#6-agent-重建与基线)
7. [实验 ID、GPU 成本、启动顺序与精确恢复](#7-实验-idgpu-成本启动顺序与精确恢复)
8. [论文组织、负结果价值与禁用主张](#8-论文组织负结果价值与禁用主张)
9. [2024–2026 年可核验文献](#9-20242026-年可核验文献)
10. [立即执行顺序与仍需证据](#10-立即执行顺序与仍需证据)

---

## 1. 结论与边界

**核心判断**：当前最有解释力的首要瓶颈不是“Agent 不够聪明”、不是“学生必须在推理时看见教师”，而是**教师训练目标与二值变化检测之间的任务语义不匹配，被目前的低分辨率变化证据与单尺度方向约束进一步放大**。排序为：**①可迁移任务信息不足/监督目标不对齐 → ②原生空间分辨率与单尺度梯度注入位置不匹配 → ③梯度在主任务与蒸馏之间的有效利用不足 → ④`max-channel` 响应的校准和排序失真 → ⑤固定随机投影本身 → ⑥学生容量/任务饱和**。这是**优先调查顺序**，不是已证实的因果顺序。[推断]

**第一轮只选一个主机制**：**CTP-Critic（Counterfactual Teacher-Privileged Critic，反事实教师特权训练评论器）**。维持九教师及 Agent 池不变；离线保留教师的双时相对称多通道特征，训练期由“教师注入的特权评论器”帮助学生学习**GT 可判定的变化证据**；测试期只运行原 C1 学生。其关键不是把教师输入接到部署路径，而是用**同一个辅助结构、相同训练损失、仅把教师输入从对齐实值换成零/错配**，辨别教师提供的真实增量信息。**预计超过 +1pp 是设计目标，不是预报或承诺**。[待验证假设]

四数据集 F1 目标 CDD 98 / LEVIR 92.5 / SYSU 85 / WHU 95 是努力方向，**不是单轮机制的有效性门槛**。如果某机制在同期、同协议、唯一变量消融下有值得解释的提升，即使仍未达四库目标，也不据此否定其方向。全程坚持完整 40K、seed=2333、batch=64、ImageNet 预训练初始化、每臂从头训练；**不改变九教师+Agent 的科研定位**。

**发现的两项容易造成研究误判的表述**：

- `[代码事实]` `models/a2net.py::switch_to_deploy()` **当前是 no-op**；它本身不“删除”任何模块。之所以目前部署图干净，是因为训练器将 `aux` 构造在**模型外部**，没有挂在学生 `state_dict` 上。第三轮必须延续这一所有权边界，或显式实现部署剥离。
- `[代码/报告事实]` Run2 的部署计算量记录为 **3.2432G FLOPs**（S2 约 3.2433G），而非旧阶段大约 2.77G；每个新机制必须按 256×256 同一 profiler 重新测量，不能把历史 FLOPs 继承过来。

## 2. 固定代码和历史证据审计

### 2.1 已核对到的关键数据流

| 环节 | 固定代码证据 | 研究含义 / 风险 |
|---|---|---|
| 训练部署学生 | `models/a2net.py:89–120`：两时相共享 LWGANet → `swa` → `tfm` → `dca` → decoder；主输出是四个 sigmoid mask | C1 为 3,280,274 有效推理参数；教师不作为学生前向输入。[代码事实] |
| 教师 Evidence | `models/distill/teacher_package.py:96–115`：`F.normalize` → `abs(fa−fb)`；`Conv2d` 固定 Gaussian 投到 128 通道；`confidence = max_c(evidence)` | `local_change` 丢失原通道坐标、原始方向与无差异时相共性；`confidence` 不是概率。[代码事实] |
| v2 Cache | `models/distill/cache_v2.py:5–15,51–80`：`local_change[128,Ht,Wt]`、`confidence[1,Ht,Wt]`、`global_desc[128]` | 训练现有 KD 只消费前两项；`global_desc` 目前没有在 `train.py` 的损失中使用。[代码事实] |
| 原始 KD | `models/distill/kd.py:10–47`：`change[2]`（c4/stride16）经 `64→128` 1×1 Conv，再做 confidence 加权 cosine | 只对齐**方向**，几乎不监督相对幅值、空间结构、显式标签任务。[代码事实] |
| KD 强度 | `models/distill/kd.py:50–60`：`λ=min(λmax,ρ‖∂L_main/∂c4‖/‖∂L_kd/∂c4‖)`，默认 `ρ=0.25` | 仅限定 c4 局部梯度比例，**并非**保证共享所有层的真实梯度占比；需审计 λ 分布和梯度夹角。[代码事实] |
| J1 | `models/distill/task_reliable_kd.py:95–216`：c2/stride4 的 65 参数 1ch head；GT/GATE/TASKKD 相同头 | J1 否定的只是 **GT 一致区域上的单通道 soft-q**，不是 teacher feature 所有形式。[代码事实] |
| Agent | `models/agent/safe_selector.py:70–172`：域加权 ridge、留一训练域 jackknife 敏感性代理、`μ−κu>δ` | `u` **不是校准置信区间**；四域 LODO 训练时仅三域，不能把 36 条教师-域数据当 36 个独立域。[代码事实] |
| Agent 特征 | `dataset_signature.py`：v1 18 维，v2 20 维；`teacher_metadata.py`：训练域、目标、stride、pyramid、cache_hw；`capability_probe.py`：P1–P5 | 标签/签名来自**训练集**；必须严禁用留出域的 val 效用或 test 指标作选师输入。[代码事实] |
| 训练/测试 | `models/scripts/train.py:250–316,395–420,514–561`：40K、best-val 权重载入后 test，输出完整 TEST 区块 | `=== TEST RESULTS ===` 是权威结果入口；`best_model_F1=...` 名称为 **val F1**，不等于 test。[代码事实] |
| 数据重放 | `models/datasets/transforms.py:34–61` 对图片使用对称 crop；`cache_v2.py:30–44` 用低分辨率 rounded crop 重放 | 整数 rounding 可能让 16×16 教师 patch 的 subpatch 位移无从精确表示；需在新 v3 上检查对齐误差。[推断] |

**尤其注意**：`dense_change_kd` 的 `cap=2.0` 位于 cosine loss 之上，正常 cosine 损失本就在 `[0,2]`，所以它**不是当前蒸馏收益微弱的强证据**；真正要测的是每步 `λ`、有效辅助梯度和主损失梯度的关系，不能仅凭超参名“cap”判断有截断问题。

### 2.2 采用同期 anchor 的历史证据

原 36 个教师选择试验基于历史 C1（SYSU 82.57 / WHU 94.04 / CDD 97.75 / LEVIR 91.04）；**绝不**将这组与 Run2 新 CTRL（SYSU 83.1391 / WHU 93.2317 / CDD 97.7454 / LEVIR 91.1783）交叉相减。[报告事实]

| 数据集 | 历史 C1 F1 | 36 单教师的最佳 test 增量 | 对应教师 | 历史同协议最低增量 | Agent 最高可用 val 收益 |
|---|---:|---:|---|---:|---:|
| SYSU | 82.57 | +0.68pp | RADIO | −0.81 | −0.0945pp（全负） |
| WHU | 94.04 | +0.56pp | MaRS | −1.15 | +0.1159pp |
| LEVIR | 91.04 | +0.14pp | D3-SAT | −0.22 | +0.1460pp |
| CDD | 97.75 | 约 +0.00pp | UniverSat | −0.05 | −0.0010pp |

**[报告事实]** 36 组 test：10 正/26 非正，只有 3 组“强正”（+0.20pp 且 IoU 同向）；val：6 正/30 负，四域 oracle-val 最大均低于先前安全集 `δ=+0.20pp`。这证明的是**旧通道下不存在可接受的正向 Agent 训练标签**，不是“不同教师客观上没有差异”。

**J1，SYSU，F1/IoU（%）**：[报告事实，来自 `docs/temporary/CATA-CD_Run2_J1_结果.md`]

| J1-C1 | J1-GT | J1-GATE | J1-TASKKD | TASKKD−GATE |
|---|---|---|---|---|
| 81.8087 / 69.2171 | 82.2754 / 69.8881 | 82.4470 / 70.1361 | 82.2657 / 69.8740 | **−0.1813 / −0.2621pp** |

结论严格限于“在同一区域 G、同等辅助容量下，用当前单通道教师响应替换 GT，没有观察到独立正收益”；单次小差值**不足以证明教师软标签有害，也不足以证明 GT-only 已稳定有效**。

**S-PCG，四域 F1（%）**：[报告事实，来自 `docs/temporary/CATA-CD_Run2_SPCG_结果.md`]

| 数据集 | 同期 S0-C1 | S1-STAT | S2-PCG | S2−S1 F1 pp | S2−S1 IoU pp | 历史门槛 τ |
|---|---:|---:|---:|---:|---:|---:|
| SYSU | 83.1391 | 82.3641 | 83.3315 | +0.9674 | +1.4097 | 1.353 |
| WHU | 93.2317 | 93.6414 | 93.6568 | +0.0154 | +0.0271 | 1.587 |
| LEVIR | 91.1783 | 90.8484 | 91.2383 | +0.3899 | +0.6569 | 0.546 |
| CDD | 97.7454 | 97.7731 | 97.7532 | −0.0199 | −0.0381 | 0.200 |

**[推断]** 不能据 S2 的 SYSU +0.97 宣称“跨域正向”；同样不能因小于 τ 宣称机制“普适无效”。本轮 R2 规则未通过属于真实研究记录，保留为未检出。

**J0/noise 审计**：[报告事实] 同 seed 相同配置 SYSU 81.81/83.14 曾相差 1.33pp；重复同配置历史 σ 为 SYSU 0.478、WHU 0.561、LEVIR 0.193、CDD 0.011（pp）。其 `2√2σ` 分别约为 1.35/1.59/0.55/0.031pp，CDD 的 **0.20pp** 是当时人为最小变化判读栏，并非噪声算出来的 0.20。**本项目第三轮按固定指令不做多 seed，也不新做多 seed 显著性检验**；历史 σ 只作为解释尺度，单次机制因果主张仍需谨慎，不能把“同 seed”当作“严格相同数值轨迹”。

### 2.3 容易遗漏的 P0 / P1 / P2 清单

- **P0-A（必须验证）**：新 v3 Cache 的 `sample_id`、crop/resize/flip、时相交换同步重放；模型通道图对齐与 GT 256 网格对齐。旧 v2 的 symmetric evidence 无需交换，但**新若存 A/B 原特征，交换必须同时互换**。`transform` 原 crop 以像素为单位，低分辨率 `round` 无法精确恢复亚 patch 几何。[代码事实＋待验证]
- **P0-B（必须验证）**：teacher/critic 不挂到 deployable student 子模块；输出四个 sigmoid mask 对 `aux_mode=on/off` 完全一致；`switch_to_deploy` 实测预测误差 `<1e-6`，部署 `state_dict` 中不得出现 teacher/cache/critic/agent 字段。[代码事实＋待验证]
- **P0-C（必须验证）**：新 CTRL/EXP **同一 data iterator、同一 cache 读取、同 batch/40K**，禁止使用“无需 cache 的旧 C1”作新 EXP 同期唯一变量对照。新 CTRL 用 `teacher_input_mode=zero`，其他分支、loss、参数表结构保持相同。[推断]
- **P0-D（研究审计）**：当前仓库可查阅的实验报告引用“最后完整 TEST block”，但未直接提供相应原始服务器日志；只称**报告值**。完整批量核实只接受最后一个 `=== TEST RESULTS ===` 至 `=== END TEST RESULTS ===` 区块。[缺失信息]
- **P1-A**：`confidence=max_c(E)` 与概率校准不同；`P5 unchanged_leakage_ratio` 在 `capability_probe.py:165–167` 实际定义为**未变/变化区域的均值之比**，不要把这个 P5 直接当“阈值下假阳性率 9.62%”；后者来自独立校准诊断。[代码事实]
- **P1-B**：教师 native grid 并不全为 16²：元数据记录 AnySat 25² / UniverSat 64² / RADIO 14²。先做分辨率分组，否则会把“教师好坏”与“可用空间信息量”混淆。[代码事实]
- **P1-C**：J1 的教师 q_T 标注为“单调压缩响应，非校准概率”；报告中 `t_plus=0.6412` / `t_minus=0.6462` 指向重叠分布，**并不是直接证明两个类别不可分**（整体 AUROC 仍有 0.852）。[代码事实＋推断]
- **P2-A**：已有 `run_one` 会在无完整 test 区块时按 `last_checkpoint.pth` 自动 resume；全新 EXP 必须写新目录且不自动借旧 checkpoint，真正断点恢复才允许同 ID 同配置 resume。shell 修改后 `bash -n`；不可在原目录“清理重跑”。[代码事实]
- **P2-B**：四数据集 split 必须用 RSML-3 实际 `list/*.txt` 指纹。BiFA 论文中的 SYSU 切分（8k/4k/8k）和本项目服务器说明（12k/4k/4k）不一致；**禁止直接横比外部 F1**。[附件事实]

## 3. 根因排序与可证伪判别

### 3.1 各条怀疑的裁决

| 优先级 | 候选原因 | 现有证据 | 严格可证伪的判别及失败解释 |
|---|---|---|---|
| 1 | **训练目标/教师预训练域与二值变化任务不对齐** | 教师 `E=|Norm F_A−Norm F_B|` 不表达“真实变化 vs 光照、伪变化”；J1 的 q_T vs GT −0.181pp。[推断] | `K-CTP`：相同辅助结构/损失，教师实特征 vs 全零；若真实教师优于 zero，说明确有可迁移增量；若仅 GT auxiliary 增益，则 teacher 主张未证实。 |
| 2 | **原生空间信息与单尺度 c4 监督不匹配** | 16×16 teacher → 256×256 边界不足；teacher boundary AUROC 0.533 vs interior 0.861（SYSU/D3-LVD）。[推断] | 在同一任务通道下 `K-HI`：同一教师多分辨率可用特征（真正 native 高分辨率） vs 将低分辨率插值成同尺寸；额外差值若不存在，空间解释被削弱。 |
| 3 | **蒸馏梯度与主目标冲突/有效注入位置** | cosine 单 c4、局部 `ρ=0.25`；未记录全主干梯度方向和 teacher-specific 改善区域。[待验证] | `K-GRAD`：保留同样 teacher 和辅助 loss，仅切换 **梯度通路 c4 vs c2**，不能同时换任务目标；读取 `cos(g_main,g_aux)`/λ/梯度范数。优势不出现则该项降序。 |
| 4 | **max-channel 响应压缩、校准缺陷** | AUROC 0.852 但 ECE raw 0.342→校准 0.064；边界问题更可能来自空间来源。[推断] | `K-CONF`：同特征/尺度/头，唯一变量为 `max_c` vs task-fitted 1ch summary，**训练集拟合冻结**；若改进只为 +0.2，优先在 LEVIR/CDD 筛，不扩到高噪声域宣称。 |
| 5 | **固定随机 `R0` 破坏表征** | 原始 C 通道随机投到 128，线性投影有信息损失但不能无证据断言是主因。[待验证] | `K-PROJ`：保存完全相同的教师原始特征，固定正交随机投影 vs 训练期可学 1×1 translator，同形输出/损失；需防止新增自由参数成为纯辅助容量效应。 |
| 6 | **3.28M 学生容量不足** | DCA 本身历史 WHU +1.38pp，说明加结构曾有帮助，但不证明教师知识无法进入现模型。[推断] | 保持学生不变，比较 oracle task critic 的训练集可拟合性与其输出教师信息对主损失的梯度收益；仅当 privileged 监督可预测但 student 无法拟合时才将容量列主因。不得先增部署参数。 |
| 7 | **数据集任务饱和** | CDD 97.75 左右、剩余头部空间仅 2.25 F1；LEVIR/SYSU 仍分别约 91/82–83。[报告事实] | 若 CDD 没有 >1pp，不能说明通道失效；主检验放 LEVIR/SYSU/WHU，CDD 只作小效应与伤害哨兵。 |

**反驳两条过强推断**：第一，**“教师不做学生输入，因此无法提供新信息”是错误命题**：冻结教师的目标通过蒸馏损失反向传播改变学生参数，可引入其预训练知识；只是在给定 `(A,B)` 时，教师前向特征不会增加原始观测的 Shannon 信息，而可能改善有限学生的归纳偏置与优化。第二，**“128 维固定随机投影一定是主因”未获证明**：若低维语义结构保留，随机映射并不必然毁掉教师可用信息；应先证明可学 translator 在同尺度与同目标上有增量。

### 3.2 判别实验原则（不以 0.2pp 的噪声偶然值选方向）

所有正式 CTRL/EXP 需 **完整 40K / batch64 / seed2333 / 同预训练 / 同 GPU 类别 / 固定 train-val-test split**，单臂一次是本项目默认纪律；**不新做多 seed，不做显著性检验**。用户要求的“大于 1pp 预期效应”理解为机制的**预设价值门槛**而非可保证的提升：`K-CTP` 与 `K-HI` 试图恢复跨目标、空间分辨率层面的信息，目标 ≥1pp；`K-PROJ/K-CONF` 属局部改造，若没有可证实的大效应，只在 LEVIR/CDD 低 σ 域作完整 40K 筛选，不在高 σ 域过度归因。

**防混杂**：区别“teacher vs no teacher”的唯一变量应是**是否给同一训练期评论器输入正确配对 teacher evidence**；区别“通道 A vs B”的唯一变量应仅是某一张量表示或梯度连接。不同 teacher cache 生成公式、分辨率、loss、decoder 同时变动，不能叫单因素消融。

## 4. 候选知识通道比较：只选一个主方案

| 排名 | 候选（区别于旧通道） | 为什么**可能**出现 >1pp | Agent 是否可选 | 部署开销 | 代码量估计 | 何时否定 |
|---|---|---|---|---|---|---|
| **1** | **CTP-Critic：教师特权输入 + GT 任务评论器 + teacher-free 蒸馏视图**：多通道 `E/M` 经训练期 translator 映射，和学生变化特征交互；对相同辅助图以 zero/shuffle 做反事实 | 在“旧 q 已近似 GT”之外捕捉纹理/位置/上下文及教师诱导的学习方向，训练期输入而不是直接单通道回归；若教师原始特征确有增量才可能 >1pp。[待验证] | **是**：每教师统一接口，收益标签直接为同协议 `M1(T,D)−C1(D)` | **0 参数/0 FLOPs**（学生完全不变；训练更多） | 新模块约 250–400 行，Cache v3 约 150–250 行，接线/测试 150–250 行 | 实 teacher−zero ≤0 且 teacher−shuffle ≤0；或收益只来自无教师辅助；或主预测依赖 teacher |
| 2 | 真原生多尺度结构蒸馏（c2/c3/c4，teacher 有对应原生尺度才用） | 提供小目标与边界细节，不用 16×16 伪高分辨图充数；有空间来源才可能 >1pp | 是，但教师金字塔能力成为选择特征 | 0 | 180–300 行 + v3 Cache | 多尺度 vs 同语义单尺度无优势，或只有插值伪收益 |
| 3 | GT 拟合可学 translator：保持旧 KD 注入点，`R0` 改 trainable task adapter；GT 只在 train 拟合 | 缓解高维异构对齐；**单独预计更可能 <1pp**，先低 σ 筛 | 是，固定同构头可比 | 0 | 90–180 行 | 低 σ 域仍无教师特异增益、辅助对照等效 |
| 4 | 结构/关系型蒸馏：局部 Gram、邻域相似、频率/边界结构，不强求逐像素 cosine | 抑制不同网络通道坐标差、利用邻域组织；若边界教师本就无信息，则未必 >1pp | 需要关系可比性与统一 loss 尺度 | 0 | 120–240 行 | LEVIR/CDD 无收益或损失梯度与 GT 冲突 |
| 5 | 任务对齐的独立 teacher soft head（不同于 J1 `q_T`，由多通道 teacher feature 预测 GT，冻结交叉头） | 和 J1 的区别是**从多通道学出的任务预测**，不是换一个温度；但容易成为 GT 重复监督 | 是，只有公平 CTRL 才成立 | 0 | 120–250 行 | 对照 GT-only head 等价；存在训练 label 记忆捷径 |

**不建议“直接复现 BAN 的 feature injection”**：BAN（TGRS 2024）的冻结 foundation 特征通过 bridging 模块**进入推理时的 CD 主干**，虽然效果好，却违反本项目“教师必须训练期退出”约束。可以学习其**跨域选择/空间对齐思想**，但必须改写为训练期支路。类似地，FreeKD 的频率先验不是本项目自动成立的创新。

**单轮只执行排名 1**；排名 2–5 是“若特定判别触发再启动”的候选，不与主实验堆叠。

## 5. 首选机制：CTP-Critic 数学定义、训练图与部署图

### 5.1 问题定义与知识内容

设双时相图像 `(A,B)`、二值 GT `Y`，学生是部署函数 `f_θ(A,B)`，其 TFM/DCA 后多尺度变化特征 `C^s_l`。教师 `T_j` 冻结、离线提取配对特征：

\[
F^A_{j,l}=T_{j,l}(A),\quad F^B_{j,l}=T_{j,l}(B),\quad
E_{j,l}=|\operatorname{LN}(F^A_{j,l})-\operatorname{LN}(F^B_{j,l})|,\quad
M_{j,l}=\tfrac12(\operatorname{LN}(F^A_{j,l})+\operatorname{LN}(F^B_{j,l})).
\]

`E` 表示变化证据，`M` 表示对称场景共性；都满足 `E(A,B)=E(B,A)`、`M(A,B)=M(B,A)`。**不使用 `F^A−F^B` 的有符号量作为唯一教师输入**；缓存可按原始时相分别保存再对称计算，但时相交换必须联合重放。不存在原生多尺度的教师**不造假**：先只用其真实 native grid，`resize` 仅用于数值对齐，不作为真实细节来源。所有教师经过同一尺寸定义的 adapter，避免因头自由度不等而产生假路由效应。

统一教师表征：

\[
U_{j,l}=P_{j,l}\big[\operatorname{Resize}(E_{j,l}),\operatorname{Resize}(M_{j,l})\big],\quad
U_{j,l}\in\mathbb R^{B\times d\times H_l\times W_l}.
\]

`P` 为**训练期** 1×1 Conv + normalization + GELU，小维度如 `d=32`（只能作为起始设计，不声称该宽度最优）。输入是 `E/M` 的**多通道结构**而非 `max_c`。对 zero CTRL，仍完整实例化、读 Cache、前向计算对应子网络，只把 `U` 的使用位置替换为全零；辅助参数、batch 顺序和正则保持一致。

### 5.2 训练期反事实评论器

在 `c2(stride4)` 为主的**学生可见变化特征** `Z=ψ(C^s_2)` 上，设置同一个 train-only critic `h_φ`：

\[
\begin{aligned}
\ell_{\rm pri} &= h_\phi\big(Z+g_\phi(Z,U)\big),\\
\ell_{\rm nul} &= h_\phi(Z+g_\phi(Z,0)),\\
L_{\rm task} &= \operatorname{BCEWithLogits}(\ell_{\rm nul},Y_\downarrow),\\
L_{\rm pri} &= \operatorname{BCEWithLogits}(\ell_{\rm pri},Y_\downarrow),\\
L_{\rm transfer} &= \operatorname{KL}_{\rm Bernoulli}\left(\operatorname{sg}\sigma(\ell_{\rm pri}/\tau)\;\|\;\sigma(\ell_{\rm nul}/\tau)\right)\tau^2,\\
L &=L_{\rm main}(f_\theta(A,B),Y)+\lambda_a (L_{\rm task}+L_{\rm pri})+\lambda_t L_{\rm transfer}.
\end{aligned}
\]

`sg` 为 stop-gradient；所有辅助损失只经 `Z` 回传到学生 encoder/TFM/DCA，**主输出 `f_θ` 从不调用 `h_φ` 和 `U`**。`g_φ` 可先设简洁的门控残差 `Conv1x1([Z,U])`，**不用注意力/专家堆叠**。`L_task` 强制 critic 的 teacher-free view 自身就能完成 GT 任务，防止辅助分支完全依赖 privileged shortcut。`L_pri` 是**训练期特权信息分支**；`L_transfer` 把教师条件的任务导向传给 teacher-free view，但其软目标不是旧的 `max-channel` `q_T`。

**关键的反事实设计**：

- `CTP-ZERO`：`U≡0`；网络、两视图、loss 权重、optimizer、cache I/O **均不变**。
- `CTP-SHUFFLE`：从**同 batch 内别的样本**取 `U`，保持幅值分布但破坏与图像/GT 的配对；禁止仅改变标签或把 GT 注入 teacher input。这是额外**必要机制对照**，不作为每个正式比较同时变动的因素。
- `CTP-REAL(T)`：真实同图配对 `U_T`；对比 zero 与 shuffle，证明配对教师信息有无价值。只有 REAL 明显胜过两个反事实分支，才可写“教师特异知识贡献”。

**需要强调的逻辑缺口**：[待验证假设] teacher 的 `U` 对训练期 critic 足够有用，且共享学生 `Z` 能把这种帮助转化为部署后学生准确度。如果 `L_pri` 显著下降但部署 F1 不动，表示“教师会帮助 train-only critic，但无法从该梯度跨越到学生”，不应立即改 Agent。

### 5.3 训练图、部署图、可学习量与预计空间

```text
训练图（teacher 模块绝不在 student.forward 内）
 A,B ---> [学生 LWGANet/SWA/TFM/DCA/Decoder] ---> 主 4-scale 预测 ---> L_main(GT)
                   |
                   +-- c2(Z) -------> h_phi(Z + g_phi(Z,0)) ---------> L_task(GT)
                          |
 teacher v3 Cache(E,M) --> P_j --> U --> h_phi(Z + g_phi(Z,U)) ------> L_pri(GT)
                                                         |
                                            teacher-conditioned logits
                                                         |
                                      stopgrad -> L_transfer(teacher-free view)

部署图：A,B ---> 原封不动 C1 student ---> 主变化 mask
         无 P_j / g_phi / h_phi / teacher / Cache / Agent
```

**部署**：仍 `3,280,274` 参数，预计同 profiler约 `3.2432G FLOPs`（**与同一 C1 重新测**，不是新实测）；最大误差 `<1e−6`、交换误差 `<1e−6`。**训练参数**须在新日志分别输出：学生 3,280,274；`P_j/g_φ/h_φ` 等辅助参数实测；总训练参数求和；教师 frozen 权重数**另列**但不计可训练参数。教师 encode/cache 生成成本单列，不可说“完全没有成本”。

**Cache v3**：另建 `/share_datasets/CD_teacher_cache/CATA_CD_v3/<teacher>/<DS>/train/`，绝不覆盖 v2。最低需求是 `E/M`，元数据 `schema_version=3`、teacher/weights SHA256、sample_id、native_h/w、taps、normalize/preprocess、数据 split fingerprint、shape/dtype、augmentation contract；可选每级原生特征。粗算一次 `[768,16,16]` 的一张 FP16 map 约 0.375MiB，`E+M` 约 **0.75MiB/样本/教师**；四库训练样本约 35,067，对应 **约 25.7GiB/教师**（未计 meta、压缩、索引与高分辨金字塔）。九教师全部重建会放大存储成本，**先做 2–3 个教师、两个筛选域的增量 Cache**；无需整池预备后才训练。

**准确性不变量**：`A/B/Y/Cache` 全部共享 crop、resize、flip；教师缓存若已是对称的 E/M，时相交换可 no-op；若缓存原 F_A/F_B，交换时必须 `F_A↔F_B`。只许训练集 GT 帮助 critic 拟合，禁止用 val/test GT 生成 teacher target 或调译器；Agent 仅用 train-signature/probe 和训练域 val utility 标签，不以本域 test 选师。

### 5.4 与文献的实质区别、反例和最小消融

- 相对 **BFD（TGRS 2025）**：同样“训练期 foundation 教师、部署不增加负担”的大方向**不是首创**；CATA 的新增主张必须是**多异质教师上的数据集级适配性选择、反事实证据是否促进 teacher-free 学生**，不能把“burden-free”本身写成独创。
- 相对 **CrossKD（CVPR 2024）**：借鉴监督路径隔离/跨头任务一致性，但本项目**没有现成冻结的教师二值变化 head**；训练的是 train-only 对称 privileged critic，不是把 student 检测头喂到现成 teacher 检测 head 的同构技术。
- 相对 **Fuse Before Transfer（ICCV 2025）**：借鉴异构教师/学生的表征不匹配警告，但主实验不在部署期融合多个教师，Agent **只选择某个适配教师或 None**，不会偷偷改成 ensemble/fusion 创新。

**预注册成功条件**：部署资源与对称性全部合格；LEVIR 上 `CTP-REAL−CTP-ZERO` **优先观察是否 ≥+1.0pp F1 且 IoU 同向**；同时 `CTP-REAL−CTP-SHUFFLE>0`，并出现可解释的 hard-negative/小建筑面积段或边界收益。此条件属于 **机制定位的高价值阈值**，不是“不达便取消研究”的 gate。**失败判据**：REAL 和 SHUFFLE 无差异＝配对知识不必要；REAL 胜 ZERO 但不胜 SHUFFLE＝更像噪声正则；REAL 辅助 critic 变好但学生 test 不变＝教师分支没有形成有效学生梯度；任意参数/交换/泄漏违规＝P0，不进入正式结论。

## 6. Agent 重建与基线

### 6.1 为什么暂时不重新拟合 20 维 Ridge

旧的 Agent 4 折 LODO 全为 None **是旧通道 val 可用收益低于 δ 的理性拒答**，而非说明 `safe_selector.py` 算法必然出错。**四域 LODO 训练只有 3 个数据集**，即使凑够 36 个 `(teacher,dataset)`，域层的独立环境数量仍只有 4；20 维 signature + P1–P5 + metadata 的泛化结论非常脆弱。训练期若多跑同一教师同一域，独立域数**仍不增加**。[推断]

优先预测**相对干净 anchor 的教师效应**：

\[
r(T,D)=100\,[F1^{\rm val}_{40K}(M1(T,D))-F1^{\rm val}_{40K}(C_{\rm same\,batch}(D))],
\]

训练 / LODO 时只能用训练域的 `r` 拟合，测试域的 `r` 只作为折外评价；教师非 None 选择必须**通过 train-only 签名和 probe**。先用一条低维、可解释的**效应分解**：`r(T,D)=α_T+β^T z(D)+γ^T q(T,D)`，其中 `z` 限选至多 2–3 个已声明的域难度特征（例如小目标比/伪变化比），`q` 限选 1–2 个有训练证据的教师能力项（如硬负分离/分辨率适配度）；禁止在四个数据集上自由搜索最佳组合。部署前 Agent 的输出是一条**数据集级动作**，不是对每张图片动态访问教师。

**数据量与 run 数的区别**：

- 通道预筛：2–3 教师×2 域（LEVIR/CDD），再加同期 C1 和机制对照，够检验“新通道是否让**同一个域不同教师**收益拉开”，**还不足以训练泛化 Agent**。
- 最小完整 Registry：**9 教师×4 域＝36 个 M1 40K**＋4 个同批次 clean C1 40K（若四域 CTRL 已由同版本新训练完成，可共享并按批次记录）；约 **96 GPU·h**，两卡完全并行理论最短约 **48 h wall time**（不含 cache、I/O、val/test、失败重跑）。若只做 3–5 个预固定候选教师，至少 `3×4=12` 至 `5×4=20` 个教师训练，标签更少、外推范围更窄，必须将排除规则事先冻结。
- 对四域外未见数据集的真实泛化性能，现有资料**不能**给出可靠的 run 数保证，因为独立域数量永远只有 4。论文应将 LODO 称为**有限四域留一验证**，不宣传跨任意新域泛化。

### 6.2 δ、效用排序与 Agent 对照

`δ=+0.20pp` 是**行动效用门槛**，不是“统计显著阈值”，不可见到新正例就事后降低。初期仍保留 δ=0.20（可与 `δ=0`、`δ=0.5` 的**预先指定敏感性表**做 Agent-only 分析；不得从 test 挑 δ），将 `u_model` 写为“jackknife 敏感性代理”，非概率保证。若新通道候选真实分化达 1pp，`δ` 的作用从“永远 None”变为“避免低效/有害教师动作”；若依旧 `None`，不要强制改规则让 Agent 必选教师。

**不可缺的四类 CTRL/BASELINE**：

| Agent baseline | 计算方式 | 是否用留出效用 | 作用 |
|---|---|---|---|
| **ALL-NONE** | 所有域都不加教师，只用新同期 C1 | 否 | 验证“选师”较彻底不用教师是否有净收益 |
| **FIXED-BEST-SINGLE** | 仅在 LODO 训练的三个域确定全局平均 val 最佳单教师，之后固定选它，必要时按事先门槛拒答 | 否 | 排除“只需要一个普适强教师” |
| **RANDOM** | 在训练阶段预定义的合格教师池中按固定 seed 2333 随机选，**不以留出域 val/test 调整** | 否 | 排除 Agent 胜过随机的偶然性；对照统一成本 |
| **CATA-AGENT** | 低维 train-only 效应模型 + 明确拒答机制，LODO 全程留出域不参与拟合 | 否 | 研究核心动作 |
| ORACLE-VAL（**仅上界，不做模型**） | 假设事后知道留出域所有 val 效用选最大 | **是** | 仅定义信息上界，严禁作为 Agent 成绩 |

建议 Agent 有效性看：4 折动作表、被选师的 fold 外 val 效用、`coverage`、相对 ALL-NONE/FIXED/RANDOM 的 regrets/harm、按数据集的主表 test F1 / IoU。**分清标签是 val、正式论文性能是最后 test block**。若新通道虽有个别 `T` +1pp，但同一教师四域皆最好，则**知识通道成功，Agent 的适配性优势尚未证实**：要寻找可证实的教师优先级翻转，而不是把固定教师包装成 Agent。

## 7. 实验 ID、GPU 成本、启动顺序与精确恢复

### 7.1 新一轮实验 ID 表（GPU·h 以单个 40K run 2.4 估）

> **严格避免新多 seed**：以下每个实验实例**只跑 seed=2333 一次**；主表和消融全为完整 40K。旧的“σ 重复配对 n=2/3”在本第三轮仅作历史噪声解释，不再扩成新增多 seed 设计。不同教师/数据集的多臂不叫多 seed。实验中的“配对”是**同数据集、同一预训练、同 cache I/O 与同 GPU 型号的 CTRL/EXP**，不是多次独立随机 seed。

| 阶段 | ID / 对照 | 数据集与教师 | 40K run 数 | GPU·h | 唯一变量与判据 | 不理想则下一步 |
|---|---|---|---:|---:|---|---|
| 实现核验 | `SMOKE-CTP` | SYSU/LEVIR 数据与真实 cache 各小批 | **不计正式 run** | 小量 | 梯度/数值/对称/删除/shape/manifest；**smoke 不代替 40K** | 先修 P0，严禁错误上服务器 |
| 低 σ 筛 | `C3L-Z` vs `C3L-R-D3S` | LEVIR，D3-SAT | **2** | **4.8** | 唯一变量 U=0 vs U=REAL；目标 ≥+1pp 且 ΔIoU 同向 | 没提升检查 critic 的 task AUROC / λ / 梯度流；不要立即改 Agent |
| 教师特异性 | `C3L-S-D3S` | LEVIR，D3-SAT | **1** | **2.4** | REAL vs SHUFFLE（只破坏样本对应）；REAL 优于 SHUFFLE | 若等价，教师作用可能是数据正则，不写知识创新 |
| 反向教师 | `C3L-R-D2` | LEVIR，DINOv2 | **1**（与已有同批 Z 比） | **2.4** | 与 D3-SAT 同机制，仅教师 ID 不同；检查同域教师收益排序差 | 若差距难辨，检查任务 head/空间失配，优先判别不用堆叠 |
| CDD 守护 | `C3C-Z`, `C3C-R-D3S`, `C3C-R-D2` | CDD，D3-SAT / DINOv2 | **3** | **7.2** | 同域 clean anchor + 两师；特别检查 IoU 是否退化 | CDD 无 +1pp 不算失败；显著负向方向需解释 |
| 高 σ 确认 | `C3S-Z`, `C3S-R-RADIO`, `C3S-R-D3S` | SYSU，RADIO/D3-SAT | **3** | **7.2** | 同批比较；观察 >1.35pp 量级正差、IoU 同向和教师排序 | 差值小则仅“未检出”，另看错误分桶 |
| 高 σ 确认 | `C3W-Z`, `C3W-R-MARS`, `C3W-R-SAM2` | WHU，MaRS/SAM2 | **3** | **7.2** | 优先观察 >1.59pp 量级；不用测试集选教师 | 若教师方向不分化，回到通道任务信息 |
| 四域主表 | `C3-MAIN-4D` | 冻结通道；每域按 Agent 选择 Teacher 或 None；四域 clean C1 同期匹配 | **8** | **19.2** | 4×M1 vs 4×C1，同协议完整结果；不是新调阈值 | 若 M1 没相对提升仍完整归档负结果 |
| 完整 Agent 学习 | `C3-REGISTRY` | 9教师×4域 + 4个 C1（可能复用当轮 clean） | **至多 40** | **至多 96.0** | 通道固定后建 36 个 val 效用，LODO + 三 CTRL | 信号不足则 Agent 继续 None，如实写不可识别 |

上表前六个筛/确认阶段共 **13 个 40K run ≈31.2 GPU·h**（2+1+1+3+3+3），两 GPU 理想平衡约 **15.6 h wall time**，而并行两个实验进程时应是**每卡单进程**，不因为两卡而把 batch64 变成全局 batch128。四域主表 `8×2.4=19.2 GPU·h`；完整 registry 的 40 次训练**包含**已完成可复用的同期同协议配置，正式总账按唯一 run ID 去重，不能机械累加造成重复计算。

**重要纠偏**：项目历史有“76 个 40K run”的归纳数字，但依据此轮已直接查看的仓库汇总，能逐项核对的组件是 **44（C0/C1/36 TV）+12（历史噪声 R1–R3×四域）+4（J1）+12（S-PCG）＝72** 个条目；其他 4 个是否来自 Run1 等尚未查到全部完整日志，**“76”待提交全局 run manifest 和服务器最后 TEST blocks 后方可对外宣称**。若有重复实验 ID/不同协议，应单独列账。

### 7.2 两卡排期与文件级改造（不声称已执行）

**建议新建的文件 / 改动范围**：

| 文件 | 修改职责 | 代码量预估 | 审核点 |
|---|---|---:|---|
| `models/distill/cache_v3.py`（新） | v3 manifest、E/M 或双时相 taps、几何重放、版本与 SHA 校验 | 150–250 行 | A/B/label/cache 形变一致；不读 val/test cache |
| `models/distill/privileged_critic.py`（新） | `P_j`、`g_φ`、`h_φ`、zero/real/shuffle 模式及 BCE/KL | 250–400 行 | stopgrad 方向、两视图共享结构、输入对称 |
| `models/distill/teacher_package.py` | 新增真实 native taps 的 train-only exporter，不修改 v2 旧生成器语义 | 50–100 行 | teacher frozen、原生分辨率真实、教师输入规范 |
| `models/tools/generate_teacher_cache_v3.py`（新） | 增量缓存/核验工具，写新 root，manifest 分批原子提交 | 100–200 行 | 校验 train list × teacher weight hash、不覆旧 Cache |
| `models/scripts/train.py` | `--channel_mode`、`--teacher_input_mode`、可选 Cache3、`aux_params` 分账、grad diagnostics、resume config hash | 100–180 行 | `C1` forward 与已有实现 bit-identical；40K/日志块统一 |
| `models/a2net.py` | **首轮原则上零改动**；只读 c2/change 输出 | 0 行 | 确保部署参数、FLOPs 不增长 |
| `models/agent/...` | 本轮不先重调 safe-set；补新 registry 构建和低维 effect model | 80–180 行 | train-only LODO、None/fixed/random 基线 |
| `tests/test_privileged_critic.py`, `tests/test_cache_v3.py`（新） | 梯度、shape、交换、parity、manifest、export | 120–200 行 | P0 优先 |
| `train_scripts/CATA-CD/Run3/{common.sh,run_screen.sh,run_confirm.sh,run_registry.sh,monitor.sh}`（新） | 保留旧 Run2；每个 ID 单独目录，失败和恢复日志分离 | 180–300 行 | `bash -n`、dry-run、无删旧文件 |

**命名与路径**：

```bash
PROJ=/home/yqwang/projects/LS-Rep_BCD_RSML_3
DATA=/share_datasets/CD
CACHE3=/share_datasets/CD_teacher_cache/CATA_CD_v3
CKPT=/share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/CATA-CD/Run3
LOG=/home/yqwang/outputs/LS-Rep_BCD_RSML_3/CATA-CD/Run3
PRETRAINED=$PROJ/pre-trained_weights/lwganet_l0_e299.pth
# 每臂：$CKPT/<EXP>/<DS>/last_checkpoint.pth
# 每臂：$LOG/<EXP>/<DS>/train_log.txt
```

建议两卡调度：GPU0 依次 LEVIR 筛选 → SYSU 确认；GPU1 依次 CDD 筛选 → WHU 确认；各自最大一条 40K。使用 tmux 单实例/卡，所有实验记录固定 commit SHA、cache manifest SHA、split SHA、命令、GPU ID、seed、是否从头，避免多实验共用输出目录。

**命令模板（新文件尚未实现，以下是实施目标而非现在可运行的命令）**：

```bash
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate lsrep
cd /home/yqwang/projects/LS-Rep_BCD_RSML_3
# 改 shell 后必须过
bash -n train_scripts/CATA-CD/Run3/common.sh
bash -n train_scripts/CATA-CD/Run3/run_screen.sh
# 核对固定代码与文件，确认没有覆盖历史
(git rev-parse HEAD; git status --short)
# smoke/真实 cache dry-run：需先实现对应脚本
python -m tests.test_cache_v3
python -m tests.test_privileged_critic
# 正式训练：具体传参按新增 Run3/common.sh 的审定接口生成
# 禁止将下面示意中的 --resume 加到第一次启动新实验里
```

### 7.3 必跑 smoke / 真实数据 dry-run / 部署判定

1. **数学 smoke**：随机 `B=2`、两时相交换、固定 GT，检查 `E/M` 完全对称；`L_pri` 对教师参数无梯度、对 student `c2` 非零梯度；`L_transfer` 不向特权 teacher view 回传，`zero/real/shuffle` 主预测**逐元素一致**；空变化标签时不 NaN。
2. **真实 Cache dry-run**：各取 SYSU、LEVIR、WHU、CDD 不同类型样本，按同一 `state` 重放 crop/flip/exchange，比对 GT 边界、teacher response 位置；manifest 中 `sample_id` 精确对应 `list/train.txt`，无测试/验证数据进入 Cache；检查原生 teacher hw 和 H×W，不盲信 meta。
3. **资源 smoke**：训练支路 tensor 形状、显存峰值、worker 读 Cache 的吞吐、DDP 单卡流程均正确。**smoke 只检正确性，不能当短跑机制结论**。
4. **部署测试**：导出学生 state dict 的键集合与同协议 C1 一致；teacher/critic 不存在；同一输入 `aux on/off` 及 `switch_to_deploy` 前后 `max|p1-p2|<1e−6`；交换输入 `max|f(A,B)-f(B,A)|<1e−6`；参数 `<5M`，256² FLOPs 同 C1 复测。
5. **40K 结果审计**：读取**同一个** `train_log.txt` 的最后一个完整 TEST block；输出 Recall / Precision / OA / F1 / IoU / Kappa、student / aux / total train / infer params、FLOPs、swap/deploy error，另外输出记录 `loss_main/loss_aux/λ/grad_cos` 的分布。只有 val 最优 checkpoint 但没有完整 TEST block 不入正式表。

### 7.4 故障恢复兼容性与安全

- 所有第三轮实验**从相同 ImageNet 权重初始化为全新训练**；`--resume` 只允许**同 experiment ID、同 Run3 code hash、同 Cache manifest、同 split hash、同辅助配置**的真中断恢复。禁止 `C3L-Z` checkpoint 变为 `C3L-R-D3S` 初始化，禁止拿 Run2 C1 微调。
- 仓库当前 `CHECKPOINT_FORMAT_VERSION=8`，第三轮若引入新的 `aux` 结构，应升到**新的格式（建议 9）**，保存 `critic/translator` 参数及 optimizer/RNG，并拒绝不匹配旧格式；必要时明确实现 `resume_epoch/loader` 校验，因为当前 `start_epoch=ck['epoch']+1` 不支持任意 batch 中断无损逐步续训。
- Cache 文件只读旧树，新生成用新目录与原子写入；原有 `checkpoint`、Run2 日志、数据集绝不覆盖/删除。Git 提交前 `git diff --cached --stat`、`git status --short` 审核，不提交 `.pt/.pth/.safetensors`、Cache、日志、大规模数据。

## 8. 论文组织、负结果价值与禁用主张

### 8.1 若 CTP 成功：论文主线

**建议论文暂定题**：*Adaptivity-Aware Multi-Teacher Privileged Distillation for Lightweight Remote Sensing Change Detection*（仅拟题，避免在未证实前用 “SOTA”）。

叙事按因果顺序：**9 位异质 foundation 教师拥有不同的预训练归纳偏置 → 旧统一随机投影/单尺度 cosine 在二值变化任务上显示可选择收益不足 → 我们设计 train-only 对称 CTP-Critic，使教师原生证据经任务监督形成可检验的学生增益 → Agent 只凭 train-only 数据集签名/能力探针估计教师效用并选择 Teacher 或 None → 部署仍为完全相同的 3.280M 学生**。关键贡献应是：①教师适配性效应**有可学的 domain×teacher 非平凡变化**；②用 REAL/ZERO/SHUFFLE 揭示教师特异增量；③无部署负担且四库受控 40K 证据。论文图里要画**训练与部署并排图**，并明确教师离线 Cache 在部署时完全消失。

必要主表：四库每域 Recall/Precision/OA/F1/IoU/Kappa、推理参数、FLOPs；机制消融（C1、GT/zero、shuffle、real、Agent）、教师收益矩阵、LODO 动作表、换师适配性交叉、边界/小目标/伪变化桶的可解释错误分布，以及上述证据口径。即使硬目标暂未全部达到，只讨论实际机制增益和剩余差距。

### 8.2 若 CTP 仍未成功：负结果论文可行性

**当前证据“有研究价值”，但直接写成高水平方法学负结果论文的证据还不够**。[推断] J1 只有一个域/一个教师的一次任务对齐检验；S-PCG 的幅度低于既有噪声尺度；J0 训练非确定性说明数值扰动传播，但不自动证明所有不同方法效果都属噪声。76 run 的完整数字尚需逐 run manifest 确认。较有发表潜力的方向是“**为什么遥感 foundation 教师在容量受限二值变化检测中蒸馏失败：空间失配、目标失配与可选效用不足的受控审计**”，要求补齐：

1. 正式 run manifest、最后 TEST blocks、可公开的 split checksum 与代码固定 SHA；展示正/负例、效用矩阵、val/test 差异；未提供日志时只能写 repo 汇总。
2. 至少**一个**可区分任务对齐 vs 空间/梯度失配的 40K 受控实验（CTP REAL/ZERO/SHUFFLE 是首选）；仅有多个负结果不能识别根因。
3. **教师强弱不能仅靠基础 AUROC 描述**：需比较教师对困难负样本/边界/小目标的可使用表征和相应学生增量。
4. **负结果的边界**：只适用于指定四 split、该学生、这些教师编码器及所检通道；不得推广成“foundation distillation 普遍失败”或“所有 Agent 不可能获益”。

由于用户坚持固定 seed、不做多 seed，论文中必须如实声明这个约束；不做 “p<0.05” 或“统计显著性”虚假包装，可将历史同配置 σ 用于**保守解释**，但不能声称这一做法等价于新方法跨随机种子的可靠统计检验。

### 8.3 禁止再使用的结论句式（包括反向过度断言）

| 禁用说法 | 更准确说法 |
|---|---|
| “九位大教师对小学生无用”“教师知识已被 GT 完全覆盖” | 在旧 128ch/stride16 cosine 与 J1 1ch q_T 的已测配置中，没有观察到足够稳定的增量；其他传递方式待验证 |
| “Agent 天然不可能选师 / U-SafeAgent 已失败” | 旧通道 val 正收益全部小于预设行动门槛，导致 Agent 全拒答；尚未有可选标签 |
| “只要把教师特征作为输入，学生就会学到新知识” | 教师特权输入可能产生捷径；必须 REAL/ZERO/SHUFFLE 和部署主分支验证 |
| “固定随机 R0 已证实是罪魁” | R0 是假设候选，需同尺度、同目标、同容量比较 |
| “teacher boundary AUROC 0.533 证明所有教师边界无用” | 只在 SYSU/D3-LVD 当前响应测到接近随机的边界排序 |
| “J1 TASKKD −0.18 证明蒸馏有害” | 同一次抽样的小幅负向观察，低于该域历史波动水平 |
| “S-PCG +0.967 为有效增益” | 单次 SYSU 观察，未超过历史判读门槛；其他域不一致 |
| “CDD 没上 98 说明方法失败” | 硬目标是方向，机制由受控实验判断；CDD 靠近饱和 |
| “实验 76 次已证明统计显著” | 当前可核对的汇总账至少 72 条；缺少全部原始日志与独立统计保证 |
| “部署 2.77G / switch_to_deploy 自动清理 teacher” | 当前 Run2 报告约 3.2432G；学生 `switch_to_deploy` 为 no-op，辅助靠不注册部署学生移除 |
| “我们首次提出无推理负担的 teacher distillation” | BFD 等已有该方向，必须强调适配性 Agent 与特权反事实迁移的具体新贡献 |
| “LODO 四折证明任意新域可泛化” | 仅四个固定域的留一检验，不是任意新域的普适外推 |

## 9. 2024–2026 年可核验文献

> 筛选标准：**CVPR/ICCV 属 CCF-A 会议**；**IEEE TGRS、IEEE TIP、ISPRS Journal of Photogrammetry and Remote Sensing 等属于 SCI 期刊，不称 CCF-A**。未查到作者官方仓库的论文直接注明“未核验到”，**绝不编造代码地址**。下面除 BFD DOI 等均附官方论文页或作者官方 GitHub；“可借鉴”**不等于**本项目可复现同样提升。

| 编号 | 题名 / 年份 / venue（层级） | 官方论文 / 可核验代码 | 发表状态 | 本项目能借 / 不能借 |
|---|---|---|---|---|
| [1] | **CrossKD: Cross-Head Knowledge Distillation for Object Detection**, 2024, **CVPR（CCF-A）** | [CVF 官方](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection_CVPR_2024_paper.html)；[代码](https://github.com/jbwang1997/CrossKD) | 正式会议论文 | 借双监督路径隔离、避免教师目标与 GT 冲突；**不能**直接等同于二值 CD 教师 head，也不能引用 COCO AP 作为 CD F1 证据。 |
| [2] | **FreeKD: Knowledge Distillation via Semantic Frequency Prompt**, 2024, **CVPR（CCF-A）** | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_FreeKD_Knowledge_Distillation_via_Semantic_Frequency_Prompt_CVPR_2024_paper.html)；[代码](https://github.com/Gumpest/FreeKD) | 正式会议论文 | 借空间降采样导致错误监督、关系/频率损失思想；**不能**把频率蒸馏的 COCO/Cityscapes 收益直接外推四库。 |
| [3] | **CLIP-KD: An Empirical Study of CLIP Model Distillation**, 2024, **CVPR（CCF-A）** | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_CLIP-KD_An_Empirical_Study_of_CLIP_Model_Distillation_CVPR_2024_paper.html)；[代码](https://github.com/winycg/CLIP-KD) | 正式会议论文 | 关系特征/对比蒸馏的经验依据；**不能**把 CLIP 语图对比目标视作 GT 二值变化语义。 |
| [4] | **Feature Spectrum Learning for Remote Sensing Change Detection**, 2025, **CVPR（CCF-A）** | [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Zang_Feature_Spectrum_Learning_for_Remote_Sensing_Change_Detection_CVPR_2025_paper.html)；**作者官方代码未核验到** | 正式会议论文 | 对伪变化/跨时相成像干扰给出机制参照；**不能**把其频率模块直接并入 student 部署图。 |
| [5] | **DUNE: Distilling a Universal Encoder from Heterogeneous 2D and 3D Teachers**, 2025, **CVPR（CCF-A）** | [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Sariyildiz_DUNE_Distilling_a_Universal_Encoder_from_Heterogeneous_2D_and_3D_CVPR_2025_paper.html)；**官方代码未核验到** | 正式会议论文 | 借“异构多教师知识未必直接可比”的认识；**不能**把融合全部教师替代按域选一教师的 Agent。 |
| [6] | **A Good Teacher Adapts Their Knowledge for Distillation**, 2025, **ICCV（CCF-A）** | [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Qian_A_Good_Teacher_Adapts_Their_Knowledge_for_Distillation_ICCV_2025_paper.html)；**官方代码未核验到** | 正式会议论文 | 借教师-学生输出分布不匹配 / 容量差距假说；**不能**把“大教师太强”当本项目已验证的主因。 |
| [7] | **Fuse Before Transfer: Knowledge Fusion for Heterogeneous Distillation**, 2025, **ICCV（CCF-A）** | [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Fuse_Before_Transfer_Knowledge_Fusion_for_Heterogeneous_Distillation_ICCV_2025_paper.html)；[代码](https://github.com/liguopeng0923/FBT) | 正式会议论文 | 借跨 CNN/ViT 不同通道/空间分布的 InfoNCE/转译思路；**不能**引入多教师融合取代选师。 |
| [8] | **Distilling Balanced Knowledge from a Biased Teacher**, 2026, **CVPR（CCF-A）** | [CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Kim_Distilling_Balanced_Knowledge_from_a_Biased_Teacher_CVPR_2026_paper.html)；**官方代码未核验到** | 正式会议论文 | 借教师偏差迁移警告和重平衡思路；其长尾多类分类不能当本项目二值 CD 结果。 |
| [9] | **A New Learning Paradigm for Foundation Model-Based Remote-Sensing Change Detection**, 2024, **IEEE TGRS（SCI 权威期刊）** | [DOI](https://doi.org/10.1109/TGRS.2024.3365825)；[代码](https://github.com/likyoo/BAN) | 正式期刊论文 | 借冻结 foundation 与 CD 表征桥接；**BAN 推理需 foundation 参与**，不符合本项目 teacher-free 部署图。 |
| [10] | **BiFA: Remote Sensing Image Change Detection With Bitemporal Feature Alignment**, 2024, **IEEE TGRS（SCI 权威期刊）** | [DOI](https://doi.org/10.1109/TGRS.2024.3376673)；[代码](https://github.com/zmoka-zht/BiFA) | 正式期刊论文 | 借空间/通道/多尺度配准为何影响边界；**不能**直接搬其较高 FLOPs 的网络到 `<5M` 学生。 |
| [11] | **Burden-Free Distillation From Foundation Model for Efficient Remote Sensing Change Detection**, 2025, **IEEE TGRS（SCI 权威期刊）** | [DOI](https://doi.org/10.1109/TGRS.2025.3584094)；**作者官方代码未核验到** | 正式期刊论文（DBLP 与 DOI 可核实） | 与本项目 train-only foundation teacher / 轻量 student **最直接先例**；贡献不能只写“部署不带教师”。具体模块、数据集与部署数字需从全文独立核对后才引用。 |
| [12] | **SATE: Efficient Knowledge Distillation with Implicit Student-Aware Teacher Ensembles**, 2026, **Pattern Recognition（SCI 期刊，补充）** | [出版社](https://www.sciencedirect.com/science/article/pii/S0031320325010167)；**官方代码未核验到** | 2026 正式期刊论文 | 借 capacity-aware 教师知识定制；**不能**以 ensemble 取代 Agent 的单教师/None 选择。 |
| [13] | **Reproducibility of Training Deep Learning Models for Medical Image Analysis**, 2024, **MIDL（非 CCF-A，方法学补充）** | [PMLR](https://proceedings.mlr.press/v227/bosma24a.html)；**官方代码未核验到** | 正式会议论文 | 借“重训方差足以误导方法比较”的方法学警示；非遥感、非 Blackwell，不能外推其统计数值。 |
| [14] | **Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference**, 2025, **NeurIPS（CCF-A）** | [NeurIPS 官方](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f80094a824ba5912d4a2de169c404a40-Abstract-Conference.html)；[代码](https://github.com/nanomaoli/llm_reproducibility) | 正式会议论文 | 可借浮点与硬件引入非确定性的审计观念；其对象是 **LLM 推理**，不是本项目 MaxUnpool2d 训练反传，不可当直接因果证明。 |

**关于检索范围**：本表在 2024–2026 年已核验文献中聚焦高度相关工作，未硬凑“2026 变化检测蒸馏 SOTA”或虚构 GitHub 地址。项目提供的论文合集中的 CrossKD、BAN、BiFA 已交叉对照；其他待核验文献不列为正式证据。

## 10. 立即执行顺序与仍需证据

### 10.1 立即执行顺序

1. **冻结基线**：在 RSML-3 确认 Git 完整 SHA 为 `e6e9827...`、`list/*.txt` 指纹、C1 原预训练 hash；**不改旧 Run2、不污染 cd_base**。
2. **先做 P0 读写审计，不做冗长“证明一定有效”诊断**：检查 teacher v2/v3 Cache 实际目录、样本名、native taps 与空间对齐；写 `cache_v3.py` 与最小 smoke。
3. **只实现 CTP-Critic 这一个机制**，先接通 `zero/real/shuffle` 三模式，保留 `a2net.py` 原 forward 和 DCA 完全不变；新增参数账、主预测不变式、loss/grad 日志。
4. **真实 Cache dry-run + smoke + export 测试全过后**，立刻用两卡各跑一个完整 40K：GPU0 `LEVIR / C3L-Z`；GPU1 `CDD / C3C-Z`；然后分别跑对应 REAL。接着再跑 LEVIR SHUFFLE 与第二教师（DINOv2），不等待所有 9 教师缓存齐全。
5. **先看同期逐数据集 F1/IoU 差值**与错误区域分桶，再启动 SYSU/WHU 完整 40K；不以“未达硬目标”终止研究。若 REAL 没有教师配对增量，先查critic/梯度/空间，**不先改 Agent**。
6. **通道冻结后重建 36×4 的效用矩阵与 Agent**，预注册效应模型/None 策略；用 LODO 对照固定最佳、随机、全 None，最后把选师结果与对应 clean anchor 做四库 40K 受控主表。
7. **统一日志审计并整理论文**：先从最后完整 TEST 区块取指标，再解释机制；任何负面实验也完整保留。

### 10.2 仍需补充证据（严格区分缺失与假设）

- **[缺失信息]** RSML-3 当前的原始 `train_log.txt`、所有 40K run manifest、checkpoint args、split SHA、cache manifest、完整附加 run 数；GitHub Run2 结果汇总只能作二手证据。
- **[缺失信息]** 九个教师在目标 256 输入下可缓存的**真实 stage taps**、输出分辨率、显存和生成成本；目前只由 `teacher_metadata.py` 得到标称 native grid。
- **[缺失信息]** 原始 KD 各阶段 `λ`、`‖g_aux‖/‖g_main‖`、梯度余弦、辅助损失 clip 次数、small/boundary/hard-negative 的**学生错误转移**，这些比扩大模型规模更能判别原因。
- **[待验证假设]** 教师 `E/M` 任务条件表示在现实四库有可利用、可蒸馏的额外语义；CTP-Critic 的 REAL 能胜 ZERO 与 SHUFFLE，且存在不同教师优先级随域翻转。
- **[待验证假设]** 如果 native 16×16 很低，带真实较高分辨率金字塔的教师会在边界/小目标有额外收益；这不等于给 16×16 图插值可以恢复细节。

---

### 固定 commit 可复核源码入口

- [固定 README](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/e6e9827/README.md)；[A2Net](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/e6e9827/models/a2net.py)；[DCA](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/e6e9827/models/decoder/deployable_change_adapter.py)
- [Teacher Package](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/e6e9827/models/distill/teacher_package.py)；[Cache v2](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/e6e9827/models/distill/cache_v2.py)；[原始 KD](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/e6e9827/models/distill/kd.py)；[J1 任务 KD](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/e6e9827/models/distill/task_reliable_kd.py)
- [Capability Probe](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/e6e9827/models/distill/capability_probe.py)；[Safe Selector](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/e6e9827/models/agent/safe_selector.py)；[训练入口](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/e6e9827/models/scripts/train.py)
- [Run2 J1 结果](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/e6e9827/docs/temporary/CATA-CD_Run2_J1_结果.md)；[S-PCG 结果](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/e6e9827/docs/temporary/CATA-CD_Run2_SPCG_结果.md)；[执行脚本](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/e6e9827/train_scripts/CATA-CD/Run2/common.sh)

**最后结论**：先使“教师真实配对知识”在同容量反事实实验中**对学生主预测产生可证伪的独立帮助**，再考虑 Agent 如何选师。Agent 不改成单教师、不取消 None，也不强迫产生动作。CTP 的任何正收益、负收益或无效观察都须通过完整 40K 同期对照记录；机制评估与四库硬目标相互关联，但不互为否决条件。
