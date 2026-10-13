# CATA-CD v2 第四轮研究方案：训练期教师条件化特征旁路与适配性选师信号重建

> **版本与证据基准**：2026-10-13；GitHub `YuqiWang-code/LS-Rep_BCD` 的 `main` 最新 HEAD **`7c236cf8eb95ccc4ebe209b9b10a6f9cd50d4264`**（提交时间 2026-10-13 00:49:44 UTC）。
>
> **阅读方法**：通过用户指定的 GitHub 连接核对 HEAD、仓库树、源码、Run3 结果文档、README、Run3 脚本及 `models_and_metrics_CATA-CD_Run1-3.txt`（71 个源码快照、169 条统一指标）。**当前执行容器无法解析 github.com，因此未能实际运行 `git clone`；使用 GitHub 连接读取的是上述固定 HEAD 的远端源码，而非本地可执行克隆。**未连接 RSML-3，未独立打开服务器原始 `train_log.txt`，也没有实际运行新的 GPU 消融。
>
> **证据等级**：**[代码事实]**＝逐行核实的 HEAD 实现；**[仓库报告]**＝已提交结果报告或机器可读指标快照；**[证据支持的推断]**＝有现象支持但尚未完成因果区分；**[待验证假设]**＝Run4 必须检验。下文绝不将建议中的运行命令或数值写成已执行结果。
>
> **研究定位**：仅 SYSU-CD-256 / LEVIR-CD-256 / WHU-CD-256 / CDD-CD-256；256×256、全监督二值变化检测；九个教师包与数据集适配 Agent **必须保留**；学生部署 `A2Net-LWGANet-L0 + DCA(moe128)`，**3,280,274 参数 / 3.2432G FLOPs（256²）保持完全不变**；所有 teacher/cache/Agent/旁路只在训练期。

## 目录

1. [先给结论](#1-先给结论)
2. [证据审计与纠错](#2-证据审计与纠错)
3. [D-1：传导低效的可证伪归因](#3-d-1传导低效的可证伪归因)
4. [D-2：四个训练期前向支路候选及唯一首选](#4-d-2四个训练期前向支路候选及唯一首选)
5. [D-3：首选 PSFB 的数学定义、张量及计算图](#5-d-3首选-psfb-的数学定义张量及计算图)
6. [D-4：Run4 最小消融与排期](#6-d-4run4-最小消融与排期)
7. [D-5：逐文件改造与测试](#7-d-5逐文件改造与测试)
8. [选师 Agent 的衔接与评价](#8-选师-agent-的衔接与评价)
9. [D-6：若 Run4 继续未检出提升](#9-d-6若-run4-继续未检出提升)
10. [2024–2026 年核验文献](#10-20242026-年核验文献)
11. [立即执行顺序](#11-立即执行顺序)
12. [仍需补充证据](#12-仍需补充证据)

---

## 1. 先给结论

**建议执行方案 D，但要先澄清一种无法绕过的部署逻辑**：如果把教师特征注入**唯一的主预测前向**，部署时又直接删除该输入，那么训练与部署函数就不相同，违反“辅助开关不改变主预测、部署图零增量”的约束。因此，正确设计不是“训练时教师直接修改主特征，测试时硬拔掉”，而是**同一次训练中保留完整无教师主预测，同时增加教师条件化的、仅用于训练的学生影子前向**。它可以共享学生特征与解码器权重，利用教师提供更密集的任务梯度，且测试仍是原封不动的 C1 模型。

**唯一首选**：**PSFB（Privileged Spatial Feature Bridge，教师特权空间特征桥）**。在 **DCA 后、Decoder 前的 `c3`（stride 8、64×32×32）**插入训练期残差旁路，真实/错配/零教师证据构造影子特征，并以**共享 Decoder 的任务损失 + 64 通道直接特征转移损失**指导学生；`model.forward()` 的主输出路径完全不接教师。相对 Run3 的 **1 通道 critic + logits KL**，这次直接作用于 **真实 Decoder 所消费的中间特征及其多尺度输出**。**“有潜力 >1pp”是机制假设，不是已证实的效果预测。**

**三条检验优先级**：

1. **正确配对是否贡献独立知识**：先看 **`REAL−SHUFFLE`**，再看 `REAL−ZERO`；只有 REAL 同时明显高于二者，才能讨论教师条件化信息。
2. **影子特征是否真正进入学生可部署表征**：量化 `∇c3 L_main`、`∇c3 L_shadow`、`∇c3 L_feat` 的夹角、相对范数、BN 状态污染和主测试 F1；辅助头自己变好不算学生获益。
3. **教师是否可区辨以支撑 Agent**：至少在同一数据集内完成 **两个以上教师的各自 ZERO/REAL 配对**，比较效应而非不同老师未经校正的绝对 F1；未分辨出教师效用之前不调整安全阈值来“逼 Agent 选师”。

四库目标 SYSU≥85 / LEVIR≥92.5 / WHU≥95 / CDD≥98 是**努力方向而非单轮淘汰门**。本轮须每个正式臂完整 **40K steps、seed 2333、batch 64、同 ImageNet 预训练、从头训练**；不做多 seed、不做统计显著性检验。部署参数/FLOPs/时间交换与部署前后误差是**不能放宽**的红线。

---

## 2. 证据审计与纠错

### 2.1 版本、来源与审计强度

| 来源 | 本次检验状态 | 结论的合法边界 |
|---|---|---|
| `README.md` | GitHub HEAD 读取；明确三轮截至 10/13 | 已掌握项目进度、纪律与概要结果；不等于原始日志 |
| `models/a2net.py`、`models/decoder/a2net_decoder.py`、`models/decoder/deployable_change_adapter.py` | 读取核心实现 | 确认 4 尺度特征、DCA 后接 Decoder；Decoder 含 BatchNorm |
| `models/distill/cache_v3.py` / `privileged_critic.py` | 完整读取 | 确认 E/M、teacher-conditioned critic 和三模式反事实 |
| `models/scripts/train.py` | 审查控制逻辑、40K 训练、保存恢复、test 日志 | 确认主模型在 `critic` 外，CTP 通过辅助梯度影响其参数 |
| `train_scripts/CATA-CD/Run3/README.md` / `common.sh` / `launch_2x.sh` | 完整读取主要脚本 | 读到历史并发控制与路径；其中部分旧说明未反映最终 2× 排队执行，应以最终结果文档优先 |
| `docs/temporary/CATA-CD_Run3_CTP_筛选结果.md` | 完整读取 | Run3 八臂 F1/IoU、消融、完成声明来自报告 |
| `docs/temporary/models_and_metrics_CATA-CD_Run1-3.txt` | GitHub 连接获取完整 578,947 字符，核对 169 条统一指标/源日志路径标记 | 是 xlsx 的**派生快照**，不是服务器原始日志的独立验证 |
| `docs/experiment_metrics.xlsx` | 仓库树确认存在（GitHub 连接未解析该二进制表本体） | 不应声称本轮独立重算了该表的所有单元格 |
| RSML-3 的原始 `train_log.txt`/manifest/checkpoint | **未访问** | 目前正式数值表述统一标记“仓库报告” |

**可核查源码链接**：

- [固定 HEAD](https://github.com/YuqiWang-code/LS-Rep_BCD/commit/7c236cf8eb95ccc4ebe209b9b10a6f9cd50d4264)
- [Run3 完整结果](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/7c236cf8eb95ccc4ebe209b9b10a6f9cd50d4264/docs/temporary/CATA-CD_Run3_CTP_%E7%AD%9B%E9%80%89%E7%BB%93%E6%9E%9C.md)
- [新 critic](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/7c236cf8eb95ccc4ebe209b9b10a6f9cd50d4264/models/distill/privileged_critic.py)
- [Cache v3](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/7c236cf8eb95ccc4ebe209b9b10a6f9cd50d4264/models/distill/cache_v3.py)
- [训练入口](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/7c236cf8eb95ccc4ebe209b9b10a6f9cd50d4264/models/scripts/train.py)
- [Decoder](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/7c236cf8eb95ccc4ebe209b9b10a6f9cd50d4264/models/decoder/a2net_decoder.py)

### 2.2 Run1 → Run3 的最小证据表

| 轮次 | 可问的问题 | 主要观测 | 允许的结论 |
|---|---|---|---|
| Run1：44 个教师能力验证（CATA Run1 共 60 个 40K，含后续噪声及 M1） | 当前固定投影/单尺度 cosine 能否提供教师效用 | 36 个单教师对照：test 10 正、26 非正；最大正 +0.68pp（SYSU/RADIO）；val 最大正 +0.146pp | 当前通道下 Agent 面临效用小、标签与噪声难区分的问题；不代表所有教师天然无价值 |
| Run1 Agent F0 | 安全集能否产生有效动作 | δ=+0.20pp 时各域 oracle-val 均未过门槛；LODO 全 None | **旧通道+旧效用标签**下安全集没有非 None 行动；不是 Agent 算法的一般否定 |
| Run2 J1：4 个 40K | 软目标是否额外优于同容量 GT | SYSU `TASKKD−GATE=−0.1813pp` | 本种 GT 一致区 1ch 软目标**未检出**增量 |
| Run2 S-PCG：12 个 40K | 对称共性上下文能否带来可靠收益 | 四域 `S2−S1` +0.9674/+0.0154/+0.3899/−0.0199pp，均未过各自 τ | 本轮预注册判据 0/4；不代表真实效应必为 0 |
| Run3 CTP：8 个 40K | 多通道 E/M 特权 critic 是否让配对教师起效 | LEVIR D3S `REAL−ZERO=+0.3022`，`REAL−SHUFFLE=+0.1349`；CDD 为轻微负差 | 训练期 critic **未观察到超过阈值的独立收益**，不能将 +0.302 归因于配对知识 |

**合计：Run1 60 + Run2 16 + Run3 8 = 84 个 CATA-CD 40K 结果**，这是仓库的报告/指标收录数；本文未重新执行这 84 次训练。

### 2.3 Run3 分数据集逐臂数值（%）

| 域 | ID | teacher / mode | F1 | IoU | 同域 D3S-ZERO 相对 F1(pp) |
|---|---|---|---:|---:|---:|
| LEVIR | C3L-Z | D3S / zero | 90.8685 | 83.2651 | 0 |
| LEVIR | C3L-R-D3S | D3S / real | **91.1707** | **83.7740** | +0.3022 |
| LEVIR | C3L-S-D3S | D3S / shuffle | 91.0358 | 83.5465 | +0.1673 |
| LEVIR | C3L-R-D2 | DINOv2 / real | 90.8700 | 83.2677 | +0.0015* |
| LEVIR | C3L-R-RADIO | RADIO / real | 90.9163 | 83.3454 | +0.0478* |
| CDD | C3C-Z | D3S / zero | 97.5553 | 95.2274 | 0 |
| CDD | C3C-R-D3S | D3S / real | 97.5467 | 95.2109 | −0.0086 |
| CDD | C3C-R-D2 | DINOv2 / real | 97.5486 | 95.2145 | −0.0067* |

*带星号的“相对 ZERO”是**跨教师且辅助参数量不同**的表观差值，不能作为“该教师特异效用”使用；D3S 的 critic `76,033` 参数，D2/RADIO 为 `59,649` 左右（仓库报告显示 0.0596M），不同教师需要各自同容量同 Cache 的 ZERO 对照。相关实现：`privileged_critic.py:69–77` 中 `Conv2d(2*teach_ch,d,1)` 的参数量依赖教师通道数；`train.py:550–558` 从 cache 元数据获取 `teach_ch`。**[代码事实]**

**关键纠错（必须避免重复误判）**：`SHUFFLE−ZERO=0.1673` 是 `REAL−ZERO=0.3022` 的 **55.36%**；**44.64%** 才是 `REAL−SHUFFLE=0.1349` 占总增量的比重。换句话说，“打乱配对仍保留约 55% 表观收益”才是正确的算术表述。该单次差值仍低于 LEVIR 的 0.546pp 噪声读数，**不能进一步量化为严格的因果归属**。

**新旧通道绝对值**：LEVIR/D3S 旧 v2 = `91.1732`，新 CTP = `91.1707`，差 **−0.0025pp**；历史 C1 = `91.0400`，新 zero CTRL = `90.8685`，相差 **−0.1715pp**。说明没有观察到超过噪声的新绝对提升；**不能将跨轮次的 −0.0025 或 CTRL 下移单独当成严格因果解释**，因为不同训练批次、不同辅助结构与不同并发背景并非同期唯一变量。[仓库报告+推断]

### 2.4 重要的源码 P0/P1 审计点

| 分级 | 具体事实 | Run4 风险及处理 |
|---|---|---|
| **P0** | `models/decoder/a2net_decoder.py:225–266`：Decoder 有多处 `BatchNorm2d` | **禁止直接调用训练模式的共享 Decoder 做第二次影子前向并写回 running stats**；必须采用克隆 buffer 的 `functional_call`，或证明采用的其他隔离方法不改变主路径 buffer |
| **P0** | `models/a2net.py:118–120` 中 `switch_to_deploy()` 是 no-op | 旁路**绝不能注册为 student 子模块**；独立建模与独立训练 checkpoint 保存；仅将 student `state_dict` 用作部署 |
| **P0** | v3 E/M 对时相交换构造不变；`cache_v3.py:114–142` 重放 crop/flip | 严查样本 ID、显式 manifest 指纹、几何一致性；不允许未经对齐的原始 A/B 特征绕开 swap 重放 |
| **P1** | `privileged_critic.py:113–131`：学生 c2 → `psi`，教师 E/M → `P`，1ch logits KL | 任务损失和特征级信息在 critic 头内汇聚；到主 decoder 只剩上游参数梯度（不是教师信息“传不了”，而是**可能弱或错向**） |
| **P1** | `train.py:383–395`：CTP 固定 `lambda_a=lambda_t=1`，未采用旧 KD 梯度范数控制 | 必须在 Run4 记录真实主/旁路梯度大小与夹角；不可凭 0.302pp 断言因果瓶颈是“KL 太弱” |
| **P1** | v3 教师 native grid 不同（D3S 16²、D2 16²、RADIO 14² 等） | stride8 不可凭插值恢复教师缺失的精细边界；需要按对象尺寸/边界评测而非只看总体 F1 |
| **P2** | Run3 文档 README 的“每卡单进程”旧版说明与最终 2× 队列记录并存 | 新实验务必逐 run 记实际同时进程、GPU ID、进程 PID、cache 背景负载，不能只从计划反推 |

---

## 3. D-1：传导低效的可证伪归因

> **总体判断**：Run3 的 +0.302pp 不足以“证明传导效率低的唯一原因”；下面是按优先排查价值排序的**候选因果机制**，每条都给区分证据。最可能的组合是 **任务增量信号弱 + 蒸馏的中间目标与主预测路径解耦 + 辅助梯度质量不明**，其中“主路径解耦”最适合这轮唯一变量方案 D 验证。

| 顺序 | 机制假设 | 当前支持/反证 | 最小可证伪判别（无需额外训练/必要时仅 40K） | 失败/排除条件 |
|---|---|---|---|---|
| H1 | **只有软任务头的梯度，缺少教师条件化的 decoder 内部前向** | J1/CTP 只有辅助 logits，真实 Decoder 见不到教师条件特征 | PSFB REAL vs 同结构 ZERO/SHUFFLE；另 REAL vs `no_feat`，均 40K | 若 REAL 仅优 ZERO、不优 SHUFFLE，说明不是正确配对空间知识带来的明显改善 |
| H2 | **旁路与主任务梯度相抵消** | 旧 128ch KD 有局部 cap；CTP 未限制梯度预算 | 在同一 batch 取 `g_main=∇_{c3}L_main`、`g_shadow`、`g_feat`，计算 cosine 和范数比，记录前/中/末训练阶段 | 若增益仍小、且梯度大部分同向，梯度冲突不再是主解释 |
| H3 | **辅助信息经过 1ch KL 压扁** | v3 保留全通道，但 Run3 最终 `KL` 是 1ch 对齐 | PSFB REAL（64ch feature transfer）vs `no_feat` REAL（相同结构仅关 feature loss） | 若 no_feat≈REAL 且直达特征梯度非零，增益不能归于特征直传 |
| H4 | **教师低原生网格导致边界/小目标缺失** | SYSU/D3L 旧诊断边界 AUROC 0.533；D3S 16² | 同时报告 source 16²→c3 32² 的边界 AUROC、small/medium/large recall；必要时只在已有教师 native 64² 的 UniverSat 上另作域内反证（Cache 成本高） | 若边界质量始终接近随机，即使 PSFB 不能承诺边界收益 |
| H5 | **教师与变化任务天然不对齐或配对伪变化强** | D3S LEVIR/shuffle 可复现表观收益一半以上；CDD 无正信号 | 在 train-only 样本固定对比 REAL/SHUFFLE/ZERO 的 `U`—GT 或误差相关；比较每教师配对 CTRL 差 | REAL≲SHUFFLE 说明尚无教师特异知识的有效证据；**不**证明教师语义无用 |
| H6 | **适配器初始随机值、zero 对照的数值路径和容量差异** | D3S/D2 critic 容量不同；zero 的 `P` 无有效梯度 | 同教师同构多臂先冻统一初始化，逐臂输出 feature/gradient SHA、zero 等价检验，运行前只读 | 若同教师零/真配对已逐参数/数据流一致，排除结构容量混杂 |
| H7 | **学生 3.28M 容量瓶颈/数据集饱和** | CDD 97.75 接近目标；SYSU/LEVIR 差距仍 1.2–1.7pp | Run4 不新增部署参数，只比较现有学生在 GT-only 旁路与 teacher 特权旁路的可分性 | 如全域连 GT-only 旁路的强训练监督也无反应，再考虑容量或优化上限；仍是待验证 |

**训练前诊断脚本应输出**：

\[
\rho_{m,a} = \frac{\langle g_{\mathrm{main}},g_{\mathrm{aux}}\rangle}{\|g_{\mathrm{main}}\|_2\|g_{\mathrm{aux}}\|_2+\epsilon},\qquad
r=\frac{\|g_{\mathrm{aux}}\|_2}{\|g_{\mathrm{main}}\|_2+\epsilon},\qquad
r_-=\frac{\|g_{\mathrm{aux}}\|_{\rho<0}}{\|g_{\mathrm{aux}}\|+\epsilon}.
\]

梯度必须在 **同一 `c3` 张量、同一次 forward、同一 loss 尺度**上测量，`torch.autograd.grad` 仅做检查且 `retain_graph=True`；禁止将取证代码的 `.backward()` 再计入真实更新。还应记录 `L_shadow`、`L_feat`、`\|U\|`、`\|\Delta\|`、REAL/SHUFFLE 的门控激活差以及 Decoder BN buffers 的 SHA。**如果没有这些，光看 F1 无法区分“信息弱”和“梯度没用好”。**

---

## 4. D-2：四个训练期前向支路候选及唯一首选

| 排序 | 机制 | 相对 v2 单尺度 cosine / v3 CTP-Critic 的实质区别 | 为什么可能大于 1pp（待验证） | 新 Cache? | 部署图 | Agent 能否得到教师差异? | 文件预估 |
|---|---|---|---|---|---|---|---|
| **1：PSFB** | `c3` 教师条件化残差 → **训练期真实共享 Decoder 影子前向**；64ch 特征传递 + GT | 不再只是 teacher 独立 critic 的 1ch logits KL；教师直接改变进入任务 Decoder 的空间特征，但**不改主预测图** | 教师若提供真实的 32² 中尺度空间证据，可同时影响 Decoder 的多级监督与 c3 表征；同级监督更贴近二值 CD | **无需**，沿用 v3 E/M | **0 参数、0 FLOPs 增量**；训练旁路独立 | **可以被检验，不保证**，要求 REAL−SHUFFLE 与各教师相对各自 ZERO 的效用均能区分 | 新 170–230 行，训练器约 120–180 行 |
| 2：Teacher-FiLM | 用 E/M 控制学生 c3 的 `γ(U)⊙S+β(U)` 影子视图 | 乘加调制允许每通道改变语义响应，但旁路仍有条件训练→测试缺口 | 适于大尺度域偏移；若教师只起条件归一化作用，收益可能更明显 | 无 | 同上，仅训练期 | 不同教师条件系数若有价值才有信号 | 新约 110–170 行 |
| 3：跨尺度 Teacher Skip | teacher E/M 训练期作为 c3→c2 decoder skip，旁路预测 GT | 边界信息可能直接注入高分辨率 decoder | 只有高原生分辨率教师可能受益；16² 教师插值无法造出真实边界 | 可能需要原生多层 Cache（**昂贵**） | 旁路全删 | 可能更偏向有金字塔教师，不具普适性 | 220–350 行 |
| 4：关系/注意相似度影子支路 | U 与 c3 形成局部相似图，辅助监督空间关系 | 传的是几何邻接/关系而非原始向量 cosine、输出 KL | 适于建筑轮廓与伪变化，但 32² token 二次复杂度易放大 | 无 | 训练期全部移除 | 不同教师关系结构若足够分化，可能 | 200–300 行 |

**只选 PSFB**。原因：①使用已建好且可信度相对高的 v3 Cache，不再堆 Cache 制式；②与 CTP 的本质差异可通过 “same branch REAL + **no_feat**” 消融直接证伪；③ `c3` 32² 是折中尺度，不在偏低的 c4 16² 上再次用 cosine，也避免 c2 64² 将教师 16² 硬插值后解释为细节；④严格保持 DCA/学生部署形状、参数和 FLOPs 不变。**不选多模块一起叠加**，后续只有在 PSFB 本身展现明确方向时才讨论更高分辨率或关系项。

**风险提醒**：PSFB 即使成功也不能宣称“只因来自 CVPR 机制所以必定 +1pp”；另一方面，未达 +1pp **只表明未达本轮优先判读门槛**，不是放弃多教师 Agent 的理由。

---

## 5. D-3：首选 PSFB 的数学定义、张量及计算图

### 5.1 数据流与张量表

令输入为两幅同为 `B×3×256×256` 的 A、B 图，学生原图：

\[
(C_2,C_3,C_4,C_5)=\mathrm{DCA}(\mathrm{TFM}(\mathrm{SWA}(\mathrm{Backbone}(A)),\mathrm{SWA}(\mathrm{Backbone}(B)))) .
\]

当前 `C_s` 每尺度 64 通道；`C_2∈R^{B×64×64×64}`、**`C_3∈R^{B×64×32×32}`**、`C_4∈R^{B×64×16×16}`、`C_5∈R^{B×64×8×8}`。其中交换对称由 `TFM` 的 `|A−B|` 与 DCA 对称输入保证。

| 对象 | 张量形状（例：D3-SAT） | 是否参与部署 | 梯度 |
|---|---|---|---|
| `A,B` | 各 `[B,3,256,256]` | 是 | 图片不学习 |
| `C3` | `[B,64,32,32]` | 是（原学生内部） | **主损失与旁路梯度共同回传** |
| `E=|norm(F_A)−norm(F_B)|`、`M=(norm(F_A)+norm(F_B))/2` | 各 `[B,C_T,16,16]`；D3-SAT `C_T≈1024`，以 manifest 实值为准 | **否** | `detach`/离线冻结 |
| `P([E,M])` | `[B,32,16,16]→[B,32,32,32]` | 否 | 对 P 参数训练，教师常量不回传 |
| `Δ=R([LN(C3),U])` | `[B,64,32,32]` | 否 | 对 R 与 C3 训练 |
| `C3^pri=C3+Δ` | `[B,64,32,32]` | 否 | 仅影子预测与特征 target 使用 |
| `C3^nul=C3+R([LN(C3),0])` | `[B,64,32,32]` | 否 | 作为 teacher-free feature learner |
| `Decoder` | **沿用现有学生 Decoder 权重** | 主 forward 原样部署 | 训练期一主一影子共享参数，但隔离 BN buffers |

### 5.2 训练期公式

对 v3 Cache（已双时相交换对称）首先做固定几何重放：

\[
E_T=|\widehat F_T^A-\widehat F_T^B|,\quad M_T=\frac12(\widehat F_T^A+\widehat F_T^B),\quad \widehat F = F/\|F\|_2 .
\]

只在教师 **原生网格**执行 `P` 的宽通道混合，再上采样窄通道特征，以节省训练显存：

\[
U_T=\operatorname{Resize}_{32^2}\big(P_T([E_T,M_T])\big),\quad U_T\in\mathbb R^{B\times32\times32\times32}.
\]

对单一样本定义反事实使用值：

\[
\widetilde U=\begin{cases}
U_T,&\mathrm{REAL},\\
0,&\mathrm{ZERO},\\
\mathrm{RollBatch}(U_T,1),&\mathrm{SHUFFLE}.
\end{cases}
\]

`RollBatch` 在完整 64 batch 下循环移位；各模式都执行相同 `P` 前向与 Cache 读取，**只有使用处的教师输入内容不同**，使唯一变量严格清晰。

\[
Z=\operatorname{GN}(C_3),\quad \Delta_T=R([Z,\widetilde U]),\quad \Delta_0=R([Z,0]),\\
H_T=C_3+\alpha\Delta_T,\quad H_0=C_3+\alpha\Delta_0.
\]

`R` 只包含最少 `1×1 Conv(96→64)`（或 `1×1 Conv`+GELU，二者择一、开实验前固定），`GN` 无 running stats，`α` 固定为 1（不把 α 调参当创新）。**ZERO 模式仍然执行 R 和 shadow decoder，只有 U=0**；不会引入 CTRL 缺少辅助容量的问题。

原部署主预测独立计算：

\[
\widehat Y_{\mathrm{main}}=\operatorname{Decoder}(C_2,C_3,C_4,C_5),\qquad
L_{\mathrm{main}}=\sum_{k=2}^{5}L_{\mathrm{BCEDice}}(\widehat Y_k,Y).
\]

训练期的教师条件化 **影子前向**（同一个 Decoder 权重、单独的临时 BN buffers）：

\[
\widehat Y_{\mathrm{shadow}}=\operatorname{Decoder}_{\theta_D,\,\mathrm{tempBN}}(C_2,H_T,C_4,C_5),\quad
L_{\mathrm{shadow}}=\sum_{k=2}^{5}L_{\mathrm{BCEDice}}(\widehat Y^{\mathrm{shadow}}_k,Y).
\]

将教师条件化的 64ch 特征用作**停止梯度的目标**，从 teacher-free 的 H0 回归它（避免只对齐 1ch logits）：

\[
L_{\mathrm{feat}}=\frac{1}{BHW}\sum_p\left[1-\cos\big(\mathrm{Norm}_c(H_0(p)),\mathrm{sg}(\mathrm{Norm}_c(H_T(p)))\big)\right].
\]

**但注意**：cosine 只是辅助特征桥的一个归一化距离，**不是复制“固定随机投影 + c4 128ch cosine”旧方案**：这里目标来自同一个学生 c3 经教师条件残差生成，原生多通道 E/M 未投到固定随机128通道，影子输出同时受 CD GT 任务损失约束；`L_feat` 的关键是 **直接给 teacher-free 中间表征梯度**。不能因为距离公式中有 cosine 就把它包装成全新相似度函数；论文创新只能来自“任务解码器内部的特权反事实旁路及直达表征传递”。可在实现时固定用 `SmoothL1(normalize(H0),stopgrad(normalize(HT)))` 代替 cosine 二选一，禁止两者结果择优后声称独立创新。

\[
\boxed{L_{\rm Run4}=L_{\rm main}+\lambda_s L_{\rm shadow}+\lambda_f L_{\rm feat}}.
\]

**协议建议**：`λ_s=1, λ_f=1` 作为预先冻结的初始权重；报告三项损失与梯度比例；如发现辅助梯度压倒主损失，应采用**所有臂统一、预先确定**的局部梯度预算（参考现有 `gradient_budget_lambda` 思路），且重做同预算 CTRL，不能仅对 EXP 调损失权重。不要在筛选时追逐多个 λ 组合。

### 5.3 梯度路径、冻结与无教师部署证明

```text
                    ┌──────────────────── 原学生 backbone→SWA→TFM→DCA ────────────────────┐
   A,B ─────────────┤                                                                   │
                    └── C2,C3,C4,C5 ── [原Decoder] ── 主四尺度pred ────── L_main(GT) ──┤
                               │                                                     │
                               │ C3                                                  │
   v3 E/M ── detach ── P_T ── U ── REAL/ZERO/SHUFFLE ── R(C3,U) ── H_T                │
                                                               │                    │
                                     [临时BN buffers + 共享Decoder权重] ─ L_shadow ────┤
                                                               │                    │
                                     stopgrad(H_T) ── feature匹配(H0) ─ L_feat ───────┘

   反向：L_main 按原图回传；L_shadow 经过真实 decoder→H_T→C3；
         L_feat 通过 H0→C3 回传，但 target H_T detach；教师 cache/encoder 全冻结。
   部署：删除独立的 P_T/R/旁路损失/影子前向；只导出原学生。
```

严格地说，**共享 Decoder 的梯度仍会改变学生主 decoder 训练后的权重**，这正是要检验的知识迁移；“辅助开关不改变主预测”指**固定同一组学生权重、同一 eval 输入、同一次主预测 forward**不依赖教师，不是说有/无辅助训练两组模型应训练到相同权重。

### 5.4 P0：共享 Decoder 的 BatchNorm 污染

**[代码事实]** Decoder 含多层 BN (`a2net_decoder.py:203–266`)，普通 `decoder(C2,HT,C4,C5)` 会在 `model.train()` 下第二次更新主路径 `running_mean/var/num_batches_tracked`。这可能改变**关闭教师旁路后的主预测**，破坏机制公平性与用户要求。**不能忽略。**

建议在训练期 shadow 前向使用官方 `torch.func.functional_call`，传递：

```python
from torch.func import functional_call

params = dict(model.decoder.named_parameters())      # 原共享权重，不复制/不 detach
buffers = {k: v.detach().clone() for k, v in model.decoder.named_buffers()}
shadow_raw = functional_call(model.decoder, (params, buffers), (c2, h_t, c4, c5))
# 影子 BN 更新的是临时 buffer，原 decoder BN 状态绝不写入。
```

这段是**实现草图，尚未在 RSML-3 测试**，必须配套断言：执行影子前向前后，主 Decoder 所有 `running_*` / `num_batches_tracked` **逐位不变**；影子监督到 `c3` 和 decoder 共享权重的梯度为非零；`model(pre,post)` 主预测在同一权重/同一 eval 模式下不因模式开关改变；若使用 DDP/AMP，须单独烟测 `functional_call` 行为。可采用功能等价的“显式无 running-stat shadow wrapper”，但**不允许**简单调 `model.decoder.eval()` 去影响主训练状态。

**额外训练计算**：影子 decoder 会增加**训练期**运算/激活，部署 **0 增量**。两卡每卡两训练进程能否稳定执行，必须先检查 `torch.cuda.max_memory_reserved`；超显存时回退每卡一进程，**不能把 batch64 分割改变协议**。

---

## 6. D-4：Run4 最小消融与排期

### 6.1 唯一变量对照规范

- `PSFB` 每一臂 **共享 C1 student、同一 v3 Cache、同路径 I/O、同数据顺序、同随机种子、同 batch64、相同预训练、完整40K**；只切 `teacher_input_mode`。
- `REAL`、`ZERO`、`SHUFFLE` 对于**同一教师**拥有同一 P/R 参数量与同一 shadow decoder、相同损失；**严禁拿 D3S 的 ZERO 当 D2 的严格 CTRL**，因为 `P` 参数量和初始化随机流不同。
- `NOFEAT` 对照是**相同 REAL、同影子路径、同参数和同训练预算**，只把 `λ_f=0`，检验特征层直传是否真有贡献。`NOFEAT` 与 `REAL` 两臂要一起从头训。
- 三模式不可依据 test 符号临时改阈值；选师阶段使用 **train/val** 的独立数据（不以 test 选教师）。
- 正式 F1、IoU 与其他指标从**单一日志最后完整 TEST block**读取；不能用 best-val 行当 test。

### 6.2 噪声阈值与判读规则

历史 σ（百分点评分）：`SYSU 0.478 / WHU 0.561 / LEVIR 0.193 / CDD 0.011`。沿用项目约定的启发式可分辨尺度：

\[
\tau_D(n)=\max\left(0.20,\frac{2\sqrt2\,\sigma_D}{\sqrt n}\right)\ \mathrm{pp}\quad\text{（n 为每臂同配置重复次数，非显著性检验）}.
\]

| 数据集 | 1 次/臂 σ 尺度 `2√2σ` | 项目判读下限 τ(1) | Run4 主判据 |
|---|---:|---:|---|
| **LEVIR** | 0.546pp | **0.546pp** | 机制优先目标 **`REAL−ZERO≥+1.00pp` 且 IoU 同向**；配对特异性还需 `REAL−SHUFFLE>+0.546pp`（高可信判读） |
| **CDD** | 0.031pp | **0.20pp**（约定地板，非真实σ） | 可在小效应筛选中观察 `REAL−ZERO≥+0.20pp` 且 IoU 同向；不要求 ≥1pp（接近饱和域） |
| **SYSU** | 1.352pp | **1.353pp** | 单次优先 `REAL−ZERO≥+1.35pp` 且 IoU 同向；小于此仅记观察 |
| **WHU** | 1.587pp | **1.587pp** | 单次优先 `REAL−ZERO≥+1.59pp` 且 IoU 同向；小于此仅记观察 |

**注意**：该 τ 是历史工程判读尺度，**不是**数学上的 95% 置信界，也不意味着差值一旦超 τ 就自动“证实”泛化因果。教师“可区辨”的额外要求是 **至少两名教师各自 `REAL−ZERO` 的效应差超过相应 τ 且 val 可选动作为正**。若只在 test 上分得开而 val 上无信号，Agent 不得偷看 test。

### 6.3 正式实验 ID 表：先低 σ，再确认，最后四库主表

所有行均为 **完整 40K、seed2333、batch64、从头训练，单行 1 run**。以下是**预注册拟议排期，尚未实际启动**。

| 阶段 | ID | 数据集 | 教师 | teacher input / 唯一变量 | 同期唯一对照 | 目标/失败解释 |
|---|---|---|---|---|---|---|
| **筛选** | D4-L-D3S-Z | LEVIR | D3-SAT | ZERO；同容量教师输入置零 | 同教师 REAL | clean anchor |
| 筛选 | D4-L-D3S-R | LEVIR | D3-SAT | REAL；真实配对 | D4-L-D3S-Z | `ΔF1≥1.0` 且 IoU 同向；否则未检出大效应 |
| 筛选 | D4-L-D3S-S | LEVIR | D3-SAT | SHUFFLE；batch roll 1 | D4-L-D3S-R | 若 `REAL≈SHUFFLE`，教师配对优势不可识别 |
| 筛选 | D4-L-D3S-NF | LEVIR | D3-SAT | REAL，无 feature transfer（λ_f=0） | D4-L-D3S-R | 若 `R≈NF`，无法把收益归因于 64ch 直传 |
| 筛选 | D4-L-D2-Z | LEVIR | DINOv2 | ZERO；D2 专属 CTRL | D4-L-D2-R | 允许与 D3S 的 P 不同，**必须配自己 ZERO** |
| 筛选 | D4-L-D2-R | LEVIR | DINOv2 | REAL | D4-L-D2-Z | 教师效应 `R−Z` |
| 筛选 | D4-L-RAD-Z | LEVIR | RADIO | ZERO；RADIO 专属 CTRL | D4-L-RAD-R | 检查异构教师对照 |
| 筛选 | D4-L-RAD-R | LEVIR | RADIO | REAL | D4-L-RAD-Z | 教师效应 `R−Z` |
| **筛选** | D4-C-D3S-Z | CDD | D3-SAT | ZERO | D4-C-D3S-R | 同期 anchor |
| 筛选 | D4-C-D3S-R | CDD | D3-SAT | REAL | D4-C-D3S-Z | 重点看能否超过 +0.20pp |
| 筛选 | D4-C-D3S-S | CDD | D3-SAT | SHUFFLE | D4-C-D3S-R | 防止把随机条件当配对知识 |
| **确认** | D4-S-RAD-Z/R/S | SYSU | RADIO | 三臂：ZERO/REAL/SHUFFLE | 同教师各臂 | Run1 在 SYSU 最佳 +0.68，但本轮必须重新做同期 CTRL；单次 τ≈1.35 |
| 确认 | D4-S-D3S-Z/R | SYSU | D3-SAT | ZERO/REAL | 同教师 | 测跨教师效应，不能选看 test 表现 |
| **确认** | D4-W-MARS-Z/R/S | WHU | MaRS | 三臂 | 同教师 | Run1 WHU +0.56 仅用于设计动机；单次 τ≈1.59 |
| 确认 | D4-W-D3S-Z/R | WHU | D3-SAT | ZERO/REAL | 同教师 | 至少两教师效应对照 |
| **四库论文主表** | D4-P-`<DS>`-CTRL / M1 | SYSU/WHU/LEVIR/CDD | Agent 事先按 train-only 信号选出的教师或 None | Agent action 由**留出域外训练域**确定；CTRL 全 None | 每库一对完整从头40K | 报六指标/参数/FLOPs；若 Agent 仍选 None，要如实标注该域无选师增量 |

统计：低 σ 筛选 **11 run**；SYSU/WHU 高 σ 确认 **10 run**；主表按四域 **8 run**，合计建议 **29 run**（其中高 σ 确认和主表为**有条件启动**，且有复用可严格证明同协议的结果时应在表中注明来源，不能直接跨轮混合）。*若已完成的 Run4 REAL/CTRL 训练也与论文主表完全一致且属于同一预注册实验，可复用该正式结果；禁止为了指标从不同批次挑最高值。*

### 6.4 GPU 成本与 4 槽排期

基于用户给定的经验数 **单臂 40K≈4.2“单进程折算 GPU·h”**，实际 Run4 因多一次 decoder 训练前向可能高于此；下面是**排期估计，不是实测 Run4 时长**。

| 阶段 | 40K run | 标称单进程等价时长合计 | 若 2 卡×每卡2进程稳定（4 槽）：理想墙钟 | 若每卡仅 1 进程：理想墙钟 |
|---|---:|---:|---:|---:|
| LEVIR+CDD 筛选 | 11 | 46.2 h | `ceil(11/4)×4.2≈12.6h` | `ceil(11/2)×4.2≈25.2h` |
| SYSU+WHU 确认 | 10 | 42.0 h | 12.6h | 21.0h |
| 四库论文主表 | 8 | 33.6 h | 8.4h | 16.8h |
| 总计（有条件全部启动） | **29** | **121.8 单进程折算小时** | **33.6h 理想** | **63.0h 理想** |

这里 **4 槽是“并行进程位”，不是四张物理 GPU**；121.8 是按“单独占用卡跑一次的基准工作量”折算，**不能同时称为真实物理 GPU 消耗**。两进程共享一张卡会争计算资源，所以理想墙钟只是粗排期下界，实际每 run 时长必须以 Run4 日志更新。Batch 始终是**每进程 batch64，不做 DDP 合并成 batch128**；显存/OOM 时每卡一进程，不缩水 40K。

**建议队列**：卡0 两槽优先 LEVIR D3S 的 ZERO/REAL/SHUFFLE/NOFEAT，卡1 两槽跑 D2/RADIO 与 CDD；每轮同域的对照混排、同步启动，避免所有 CTRL 都空闲独占、所有 EXP 都在 GPU 拥堵阶段的混杂。记录每一个 run 的同卡伴跑实验 ID，而非只写“并发 2×”。

### 6.5 若效应小于 1pp，需要多少次同 seed 同配置重复？

下面只提供**噪声量纲上的经验样本量换算**，不要求多 seed、不做显著性检验，也不把重复后结果当论文主结果替代单臂主表：

\[
 n_{\mathrm{rough}}\ge\left\lceil\left(\frac{2\sqrt2\sigma_D}{\delta}\right)^2\right\rceil.
\]

| 域 | 分辨 1.0pp | 0.5pp | Run3 `0.3022pp` | Run3 `0.1349pp` |
|---|---:|---:|---:|---:|
| SYSU | 每臂 2 | 8 | **21** | **101** |
| WHU | 3 | 11 | **28** | **139** |
| LEVIR | 1 | 2 | **4** | **17** |
| CDD（**仅物理 σ**） | 1 | 1 | 1 | 1 |

CDD 若额外执行项目设定的 0.20pp 地板，而非只按 σ，辨别 0.1349pp 大约需要每臂 **3** 次，0.3022pp 为 1 次。**实验决策**：在 LEVIR 若新机制仍只有 ~0.3pp，可以按纪律先记录为观察而不追加大规模重复；若确实要做同 seed 独立从头重复，优先 LEVIR ZERO/REAL 各 4 次总量（包含第一次）用于机制归因，且每次必须独立实验目录。**不建议在 SYSU/WHU 追一个只有 0.13pp 的信号**，所需重复不合理。重复实验并不能凭空使选择器获得数据集层面的新独立样本。

---

## 7. D-5：逐文件改造与测试

### 7.1 文件清单与修改边界

| 文件 | 操作 | 建议修改 | 预估行数 | 注意事项 |
|---|---|---|---:|---|
| **新** `models/distill/privileged_spatial_bridge.py` | 新建 | `P_T`、`R`、REAL/ZERO/SHUFFLE 模式、shadow 特征、stop-grad feature loss、梯度诊断接口 | 170–230 | **train-only `nn.Module`，不要赋给 `student.*`** |
| `models/scripts/train.py` | 修改 | 增加 `--channel_mode psfb`、`--teacher_input_mode`、`--psfb_feature_on`；构建独立 bridge；一次正式主预测 + 一次共享 Decoder 影子前向；分账/日志/格式验证 | 120–180 | `none/ctp` 的现有路径不可改语义；不要改原 lr/batch/steps |
| `models/a2net.py` | **原则上不动** | 利用已有 `return_change_features=True` 获得 DCA 后 4 尺度特征；原 `forward()` 不挂教师 | 0 | 若无法安全调共享 Decoder，才添加纯函数式辅助接口且不能注册参数 |
| `models/decoder/a2net_decoder.py` | **不动** | 训练器调用 `functional_call` 做影子前向，用克隆 BN buffers | 0 | 改 decoder 权重或结构将破坏 clean anchor，应直接否决 |
| `models/distill/cache_v3.py` | 优先不动 | 增加完整 manifest/source split hash 验证可放在 Reader 外的独立 audit 工具 | 0–30 | Cache v3 不重生成，沿用 E/M |
| **新** `tests/test_privileged_spatial_bridge.py` | 新建 | 形状、梯度、对称、零/错配、共享 BN 不污染、main on/off 不变、无 teacher deploy keys | 130–200 | CPU 优先、CUDA smoke 补测 |
| **新** `models/tools/audit_psfb_gradients.py` | 新建 | 一批 train-only 数据上取主/旁路/feature 的梯度方向、范数和 activations | 80–120 | **诊断非正式短跑**，不当作成功证据 |
| **新** `train_scripts/CATA-CD/Run4/{common.sh,smoke_psfb.sh,launch_2x.sh,monitor.sh}` | 新建 | 实验 ID、独立日志/checkpoint 根、4 槽队列与锁，不能主动删除既有目录 | 180–260 | 改 shell 后必须 `bash -n`；禁用不经明确确认的 `rm -rf` |
| `analyse/extract_metrics.py` / `analyse/models_to_txt.py` | 必要时修改 | 增加 `channel_mode=psfb`、teacher_input_mode、相对同期 CTRL 的四域汇总 | 30–60 | 正式指标严格最后完整 TEST block，不能读 val 名称 |
| `README.md` / 新 `docs/temporary/CATA-CD_Run4_结果.md` | 完成实验后修改 | 按真实日志填写数据、成功/失败说明 | 后续 | 不提前写“成功” |

预估 Python 主体 500–800 行，脚本另约 200 行；这只是工程量，不是论文贡献。本轮学术上的**唯一主张**是“真正的 Decoder 任务前向中，训练期特权空间特征的配对关系能够在保持部署模型不变时迁移到学生”。

### 7.2 P0/P1/P2 验收清单

**P0 正确性（跑 40K 前必须通过）**

1. `cache_v3` 元数据：只读 train，`weight_hash`、`split_fingerprint`、`native_hw`、`channels` 与当前 train manifest/样本一致；所有 sample_id 与 `list/train.txt` 对齐，CDD JPEG mask 统一 `>=128`。
2. `E/M` 交换完全不变；Replay 的 crop/resize/flip 与 A/B/label 同步；随机交换时因本 schema 仅存 E/M，不需交换 teacher channels；禁止原始 FA/FB 当作 E/M 误读。
3. 所有 REAL/ZERO/SHUFFLE `optimizer` 参数数量、学习率、batch 顺序、loss 项一致；同教师 arms 初始化一致性按 student、bridge 参数分别 hash。
4. **影子 Decoder 不污染 BN buffers**：前后状态逐位比较；同时检查每个共享 decoder 参数与 `C3` 的影子梯度非零。不同输入模式下**固定权重的主预测** max diff `<1e-6`。
5. `bridge` **不得**出现在 `model.state_dict()` 或部署导出；模型仅含原有 C1 3,280,274 参数；THOP 同协议 256² 为 **3.2432G 量级且测得完全一致**，不是手工填写。
6. `switch_to_deploy()` 前后固定 eval 输入主预测 max diff `<1e-6`；`A↔B` 时间交换 max diff `<1e-6`。
7. 每个实验目录唯一；**新臂必须 `global_step=0` 从同一个预训练权重开始**，不允许 `--resume` 旧臂 checkpoint。只允许同一实验 ID、同一配置真中断恢复；恢复还需冻结 cache 指纹、optimizer/aux/RNG、数据采样位置并验证继续训练与日志协议（见下）。

**P1 机制诊断（可边跑边记，不作为 40K 放行门）**：

- 每个 epoch 汇总 `Lmain/Lshadow/Lfeat`、`||g_main||/||g_shadow||/||g_feat||`、梯度 cos、`||U||`、`||Δ_REAL−Δ_ZERO||`、`||H_T−H_0||`、辅助 region 类别/小目标与边界性能；若 feature collapse（`||Δ_REAL−Δ_ZERO||≈0`），如实写失败原因。
- REAL/SHUFFLE/ZERO 都要输出 **相同量纲** 的特征与梯度统计；可在固定 train-only sample 上绘制教师条件空间响应差，不看 test 来设计门控。
- 对比 old CTP 时必须提醒结构、辅助损失与位置都变化，**不能把跨轮次“PSFB−CTP”当唯一变量消融**；主干结论来自 Run4 同期对照。

**P2 工程与日志**：`bash -n` 检查所有 `.sh`；训练文件/预训练哈希与 `git rev-parse HEAD` 必须入日志；跑后更新 `docs/experiment_metrics.xlsx` 与代码+指标快照；Git 暂存区检查，绝不提交权重、Cache、数据、日志；全部保留历史树。

### 7.3 smoke、真实 Cache dry run、部署与日志检查

```bash
# 预期在 RSML-3 执行；以下 Run4 文件与 --channel_mode psfb 需要按 §7.1 实现后才存在
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate lsrep
cd /home/yqwang/projects/LS-Rep_BCD_RSML_3

# 1. 代码静态/单测
python -m compileall -q models/distill models/scripts/train.py
python tests/test_cache_v3.py
python tests/test_privileged_critic.py
python tests/test_privileged_spatial_bridge.py
find train_scripts/CATA-CD/Run4 -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n

# 2. 真实 Cache dry-run：读 1 小 batch 的 v3 sample + crop/flip/swap，完成 1 次 forward/backward
# 只用来定位维度、显存和梯度；绝对不能用短跑结果代替 40K
bash train_scripts/CATA-CD/Run4/smoke_psfb.sh

# 3. 存储/资源审计（部署使用 student-only，不带 PSFB 模块）
# 在 smoke 脚本内部断言主预测 on/off/swap/deploy <1e-6、deploy=3,280,274 params
# 并记录 GPU peak memory，只有两进程能完整存活才允许 launch_2x

# 4. 正式 40K：使用 Run4 专属 ID 与目录，不覆盖任何 Run1/2/3 产物
# bash train_scripts/CATA-CD/Run4/launch_2x.sh   # 实现及 smoke 通过后执行

# 5. 从各 run 最后一个完整 TEST block 汇总全部六指标
python -B analyse/extract_metrics.py
python analyse/models_to_txt.py --cata
```

**结果 parser 的必要不变式**：从每份 `train_log.txt` 中找**最后一个** `=== TEST RESULTS ===` 与后续 `=== END TEST RESULTS ===` 的完整配对，再确认 `Metric Split: test`、seed2333、max_steps40000、batch64、完整 Recall/Precision/OA/F1/IoU/Kappa、Train/Aux/Infer Params、FLOPs、swap/deploy error，任何缺项判不完整。禁止用历史 D3S-C1 anchor 搭配新的 RUN4 REAL 做正式净增量。

### 7.4 checkpoint 与精确恢复

- 新格式建议 `CHECKPOINT_FORMAT_VERSION=10`；在 `{student, bridge, optimizer, scheduler/step, RNG, epoch, global_step, args, git_sha, pretrain_hash, train_split_fingerprint, cache_manifest_hash, sampler_state}` 中记录各项；与 Run3 v9 严格隔离。
- 输出新根：
  - 检查点：`/share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/CATA-CD/Run4/<EXP>/<DS>/`
  - 日志：`/home/yqwang/outputs/LS-Rep_BCD_RSML_3/CATA-CD/Run4/<EXP>/<DS>/train_log.txt`
  - 已有 Cache：`/share_datasets/CD_teacher_cache/CATA_CD_v3/<TEACHER>/<DS>/train/`
- 启动脚本默认 `fresh` 应**检测已存在并拒绝覆盖**，不得模仿旧脚本“fresh 即 `rm -rf`”的行为；需要新 run 时追加唯一 ID/目录。
- **重要恢复兼容性事实**：HEAD `train.py:605–620` 目前以 `epoch+1` 作为恢复位置，即它适于在完整 epoch 边界保存后恢复；若在一个 epoch 中间突然中断，当前保存路径并不能自动证明“精确重放到 batch 位置”。Run4 若声称精确中断恢复，需显式保存 DataLoader/sampler batch offset 与 worker RNG，或只承诺完整 epoch checkpoint 的恢复并按原协议验证。**不允许把从旧 M1 checkpoint 续训改称 Run4 从头训练**。

---

## 8. 选师 Agent 的衔接与评价

当前九教师和 Agent 的论文逻辑保留：

\[
X(T,D)=[z(D),q(T,D),m(T)],\quad \widehat\Delta_T(D)=f_\theta(X),\quad
a(D)=\arg\max_{T\in\mathcal S(D)}\bigl(\widehat\Delta_T-\kappa u_T\bigr),
\]

其中 `z` 是只用 train 的 18/20 维签名，`q` 是 P1–P5 train-only 探针，`m` 是教师元数据，`None` 是合法动作。**训练标签必须更新为 PSFB 同协议 `REAL(T,D)−ZERO(T,D)` 的 val F1 差**，绝不能拿旧 v2/Run3 的 val 效用直接训练“新通道 Agent”，更不能拿 test F1 排序作为训练标签。

**可学信号最低要求**：先在 LEVIR 上拥有 D3S/DINOv2/RADIO 各自 `REAL/ZERO` 的干净差值并比较 `IoU` 方向；CDD 补 D3S 反事实，后续 SYSU/WHU 继续至少两老师。若新通道下**不同教师差异**仍只有 0.3pp 而 σ 下限 0.55pp，所谓教师路由只是“不同标签上的噪声排序”，不应盲目训练新模型企图提高指标。即使收齐 **9×4=36** 对，也只有 **4 个独立数据集簇**，LODO 每次拟合只见 3 个域；不能将 36 条视作 36 个独立域，更不能宣称已学出可外推到全新域的高置信决策。

**δ 如何处理**：仍固定 `δ=+0.20pp` 作为与旧注册表衔接的起点；若目标是“有证据的可执行正收益”，需考虑 val 的训练波动及 teacher-specific CTRL 后，再预注册 `δ_eff(D)=max(0.20,工程噪声目标)` 的**诊断阈值**；切勿为了增加非 None 覆盖率直接把 δ 改成负数。旧 `μ−κu` 的 `u` 是三域 jackknife 敏感性代理，**不是**校准下置信界，亦非置信概率。

**Agent baseline 必须同时报告**：`全 None`、`固定最佳单教师（仅训练折 val 选）`、`随机教师（预先固定策略/seed 且不看留出域收益）`、`旧 U-SafeAgent`、`新 PSFB 效用模型`（必要时加简单 train-only heuristic）。比较 **LODO 四折 val action/reward/regret/harm/coverage** 和最终四库一次 40K 的 test 主表；固定老师和随机老师也需使用**同通道 PSFB**的 40K 效用，不能混入历史 KD。Agent 只能选一名教师或 None，不能推理期融合/teacher ensemble。

> **重要**：若 PSFB 在若干域产生强正向但教师间收益未分开，证明的是“训练期特权旁路可能有效”，**不是**“适配性选师已经有效”。论文中两贡献要分别做消融，不得用单一教师的效果替 Agent 背书。

---

## 9. D-6：若 Run4 继续未检出提升

**不建议以未达四库硬目标作为止损依据。** 真正有意义的判断是：D4 `REAL/ZERO/SHUFFLE` 是否在相同条件下出现配对教师可辨识的净收益；如果没有，也要严谨说明不支持哪个具体传输假设，不能从一个方案推断全部蒸馏无效。

### 9.1 “84 个 40K 结果+负结果归因”能否投稿？

**可形成一篇有研究价值的机制审计论文雏形，但当前**还不宜保证能以“全新 SOTA 算法论文”标准被 CVPR/ICCV/TGRS 接收。论文支点可能是：

1. **系统性负结果审计**：9 foundation 教师×4 数据集，两个知识通道、J1、S-PCG、CTP 等统一学生/预算对照；公开错误目标、训练期辅助容量、路由效用难辨识条件。
2. **反事实知识有效性**：teacher REAL/SHUFFLE/ZERO 及针对不同教师的专属 CTRL；以是否正确配对而非是否加了辅助损失定义教师知识特异性。
3. **非确定性问题的边界**：J0 指出 backward 低位差异与 epoch 后漂移；σ 是工程参考、不是显著性结论；主表诚实提供所有负结果而不只挑最好单次。
4. **部署等价性**：3.2803M、3.2432G 及无 teacher 推理图，解释什么训练期路径可撤销，什么不能直接撤销。

**仍需补充的证据**：所有原始日志与完整 test 区块的自动审计包；每教师各自 ZERO；同一 train-only 样本对齐与 teacher 条件梯度诊断；任务边界/小目标分层指标；固定评估协议下更多**机制对照种类**（不要求新数据集/多 seed）；清晰复现实验 ID/哈希/并发与中断恢复。**注意**论文科研叙事不能被“跑了 84 个实验”替代；数量只是审计力度，而非创新性本身。

### 9.2 禁止再宣称的表述（双向禁止）

| 不得声称 | 应改为 |
|---|---|
| “CTP-Critic 把效果翻倍” | 同期 `+0.302pp` 未过 τ；跨轮次的表观相对收益受 CTRL 漂移混杂 |
| “Shuffle 重现 45% 收益” | **55.36% 是 SHUFFLE/REAL 的表观增量比；44.64% 才是配对额外部分**，且均低于噪声尺度 |
| “教师知识没有信息”“大模型作为教师无用” | 当前指定通道下未检出可辨识净收益；其他方法未被证明无效 |
| “教师软目标有害” | J1 SYSU 本轮负差 −0.181pp，幅度小，不能定性普适有害 |
| “Agent 选 None 证明算法正确/最优” | 旧标签四域 val 均无超过 δ 的动作；None 是保守机制行为，未证明超出训练域泛化 |
| “PSFB 肯定能 >1pp” | 只是一条新的可证伪机制假设；需要四库 40K 唯一变量实验 |
| “PSFB teacher 输入接入主预测，部署却完全无影响” | 教师只接入**可丢弃的影子前向**，主预测始终 teacher-free；权重可因训练梯度改变 |
| “源码 `switch_to_deploy` 已显式删除 critic” | 该函数目前 no-op；critic 不在学生模块树中才是当前部署清洁的原因 |
| “有过 84 个 40K 结果，所以机制有效 / 论文可发” | 这是审计规模；论文价值需要明确方法主张、有效对照与可复核数据 |
| “没达到 SYSU85 就说明方法失败” | 硬目标只是努力方向；机制是否有效由同协议消融来评价 |
| “没检出 >τ 效应说明数学上效应为零” | 未检出不等于零效应，记录区分不了的噪声范围 |
| “相同 seed = bitwise 可复现 / 已做显著性检验” | J0 显示反向数值不确定性；本项目不做多 seed、不进行显著性检验 |
| “公开代码证明全部原始日志无异常” | 当前只核对 GitHub 汇总和指标快照，原日志仍待抽查/独立审计 |

---

## 10. 2024–2026 年核验文献

**来源原则**：只引用可核验的 CVF 论文页 / IEEE 或 DOI / arXiv 官方条目 / 作者 GitHub；会场层级分明，CVPR/ICCV 属于 **CCF-A 会议**，TGRS 等是 **SCI 期刊**，不能写成 CCF-A。**代码地址未确认的明确写“未核验到官方代码”**，不编造。以下均是外部论文，不表示作者已在本项目四库上验证有效性。

| 编号 | 题名 / 年份 | venue / 层级 | 已核验论文地址；官方代码 | 发表状态 | 对 PSFB 可以借什么 / 不能借什么 |
|---|---|---|---|---|---|
| [R1] | **CrossKD: Cross-Head Knowledge Distillation for Object Detection** (2024) | **CVPR 2024 / CCF-A** | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection_CVPR_2024_paper.html)；[代码](https://github.com/jbwang1997/CrossKD) | **已发表** | 能借“在学生特征进入实际任务头时分离教师/GT 监督路径”的动机；不能照搬检测头/框分配或宣称其 AP 改进会变成 CD F1 改进。 |
| [R2] | **KD-DETR: Knowledge Distillation for Detection Transformer with Consistent Distillation Points Sampling** (2024) | **CVPR 2024 / CCF-A** | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_KD-DETR_Knowledge_Distillation_for_Detection_Transformer_with_Consistent_Distillation_Points_CVPR_2024_paper.html)；[代码](https://github.com/wennyuhey/KD-DETR) | **已发表** | 能借“蒸馏对应位置一致”这一可检验思想，尤其 Cache native grid 与 c3 位置；不能复制 DETR object query 到本二值密集 CD。 |
| [R3] | **Active Object Detection with Knowledge Aggregation and Distillation from Large Models** (2024) | **CVPR 2024 / CCF-A** | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Active_Object_Detection_with_Knowledge_Aggregation_and_Distillation_from_Large_CVPR_2024_paper.html)；[代码](https://github.com/idejie/KAD) | **已发表** | 能借“训练阶段外部知识通过 oracle/teacher 解码路径辅助无额外知识输入学生”思想；不能将 ego-centric active object detection 直接视作遥感二时相变化检测。 |
| [R4] | **Class Incremental Learning with Multi-Teacher Distillation** (2024) | **CVPR 2024 / CCF-A** | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Wen_Class_Incremental_Learning_with_Multi-Teacher_Distillation_CVPR_2024_paper.html)；[代码](https://github.com/HaitaoWen/CLearning) | **已发表** | 能借“多个教师确有不同专长才谈互补”的论证结构；不能把它的 class-incremental/replay 机制移入四库二值 CD，亦不能改成推理融合。 |
| [R5] | **MoVE-KD: Knowledge Distillation for VLMs with Mixture of Visual Encoders** (2025) | **CVPR 2025 / CCF-A** | [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Cao_MoVE-KD_Knowledge_Distillation_for_VLMs_with_Mixture_of_Visual_Encoders_CVPR_2025_paper.html)；[代码](https://github.com/hey-cjj/MoVE-KD) | **已发表** | 能借“异构教师的特长/冲突必须分开测”，而非只比较 teacher checkpoint 强弱；不能引入 MoE 额外部署编码器或视觉语言任务。 |
| [R6] | **Cross-Architecture Distillation Made Simple with Redundancy Suppression** (2025) | **ICCV 2025 / CCF-A** | [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Cross-Architecture_Distillation_Made_Simple_with_Redundancy_Suppression_ICCV_2025_paper.html)；[代码](https://github.com/VISION-SJTU/RSD) | **已发表** | 能借“异构教师特征含冗余/结构差异，不能盲目逐通道复制”；不能未经实验证明当前瓶颈必为冗余。 |
| [R7] | **Feature Spectrum Learning for Remote Sensing Change Detection** (2025) | **CVPR 2025 / CCF-A** | [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Zang_Feature_Spectrum_Learning_for_Remote_Sensing_Change_Detection_CVPR_2025_paper.html)；**未核验到官方代码地址** | **已发表** | 能借“伪变化/风格混杂是重要错误源”的任务分析；本轮不叠加频谱模块，否则唯一变量失效。 |
| [R8] | **Burden-Free Distillation From Foundation Model for Efficient Remote Sensing Change Detection** (2025) | **IEEE TGRS 2025 / SCI 权威期刊（非 CCF-A）** | [DOI/出版社入口](https://doi.org/10.1109/TGRS.2025.3584094)；**未核验到官方代码地址** | **期刊已发表** | 能借“训练期利用 foundation model、推理不背负大模型”的定位；需仔细核查其训练图是否与 PSFB 实质重合；不可宣称本方法“首个 burden-free CD KD”。 |
| [R9] | **Remote Sensing Change Detection With Transformers Trained From Scratch** (2024; ScratchFormer) | **IEEE TGRS 2024 / SCI 权威期刊（非 CCF-A）** | [IEEE](https://ieeexplore.ieee.org/abstract/document/10489990)；[代码](https://github.com/mustansarfiaz/ScratchFormer) | **期刊已发表** | 能借如何分辨伪变化及检测中间特征结构的标准；其训练无需 ImageNet，不能直接当作当前学生同预训练公平比较。 |
| [R10] | **JL1-CD: A New Benchmark and a Robust Multiteacher Knowledge Distillation Framework for Remote Sensing Change Detection** (2026) | **IEEE Transactions on Instrumentation and Measurement 2026 / SCI 期刊（非 CCF-A、非 TGRS）** | [IEEE](https://ieeexplore.ieee.org/abstract/document/11498632)；[代码](https://github.com/circleLZY/MTKD-CD) | **期刊已发表（2026-04-28）** | 可借多教师互补需要明确任务/数据特性的证据链；**不能**把 JL1 新数据集加入本项目，不能将论文“融合多教师”改成本项目的单动作 Agent。 |

**专门说明“负结果与可复现性方法学”**：本轮检索没有找到足以确认题名/venue/代码的 2024–2026 专门“负结果+确定性”方法论文，因此**不虚构一个 NeurIPS/CVPR 引用**。该部分主要依据项目自己的 J0 数值取证、84-run 审计纪律以及上述实证研究的规范做法。需要正式投稿时可以另开一次专项检索，对确实核验到的论文再追加；不得把泛泛“应多 seed”当成本项目实施指令（本项目明确固定 seed 2333）。

**文献与本方案的实质差异**：CrossKD 的“跨头特权任务预测”很接近训练期支路的构型，但 PSFB 与之不同之处拟定位在**遥感二时相交换对称 E/M、多教师 train-only Cache、同 teacher REAL/SHUFFLE/ZERO 的配对反事实、共享变化检测 Decoder 且影子 BN 状态严格隔离、部署 3.28M/3.2432G 不变**。是否足以构成可发表的*方法新颖性*，仍需对照 CrossKD/BFD 的原文与完整消融，不能仅靠名称区分。[待验证假设]

---

## 11. 立即执行顺序

1. **锁定基线**：在 RSML-3 检查 `git rev-parse HEAD`（应是 `7c236cf...`，若服务器 SFTP 代码不是同一 HEAD，则记录源码 hash 并禁止跨协议混用）；确认 C1 3,280,274/3.2432G、`lsrep`、预训练权重与数据 split 指纹。**不要修改 cd_base/CUDA/PyTorch。**
2. **实施一个方案**：只新增 PSFB 训练期 bridge、影子 Decoder `functional_call` 与必要开关；不改 `a2net`/DCA/Decoder 部署行为、不堆额外频谱、FiLM、关系模块。
3. **P0 测试**：跑 Cache v3 真实数据 dry-run、时相交换/裁剪重放、主预测 on/off 等价、共享 decoder BN buffer 不污染、非零梯度、student-only 3.28M/3.2432G、checkpoint v10 全量字段；`bash -n` 审核 Run4 shell。
4. **直接启动低 σ 40K**：优先 LEVIR D3S-Z/R/S/NF，随后 D2 与 RADIO 各自 Z/R；CDD D3S-Z/R/S；有四个稳定资源槽才并发，不足则两槽。**Smoke 不取代正式 40K。**
5. **只从完整 TEST block 汇总**：逐域 F1、IoU、Recall、Precision、OA、Kappa、训练/辅助/部署参数、FLOPs、swap/deploy 误差；并计算同教师 `REAL−ZERO`、`REAL−SHUFFLE`、`REAL−NOFEAT`。如没有 >1pp 也按事实记录，不据此废弃主线。
6. **若低 σ 显现教师特异正效应，再走 SYSU/WHU**：至少两个教师各自 Z/R，并对最佳教师补 shuffle；按高 σ 的 τ 看是否“可区辨”，不从 test 选择 Agent。
7. **只有通道产生足够信号后重训 Agent**：新 PSFB train/val reward，严格 LODO；比较全 None、固定最佳单教师、随机、旧 Agent 和新 Agent。最终四库主表一律从头 40K。
8. **归档**：生成 Run4 README、Run4 结果文档、指标表及源码快照；Git 提交前检查暂存区，不提交 checkpoint/cache/数据/日志，不覆盖 Run1/2/3 文件。

---

## 12. 仍需补充证据

| 优先级 | 缺失项 | 会影响哪条结论 | 补充方式 |
|---|---|---|---|
| **P0** | Run3 八臂实际服务器 `train_log.txt` 最后完整 TEST 区块及 sha256、step 完整性 | 当前所有 Run3 数值只能称“仓库报告”，不是独立日志验证 | 只读拉取到审计目录，自动 parse 最后配对区块，保留 hash |
| **P0** | v3 cache manifest（各教师通道、native grid、weight hash、split_fingerprint、样本数） | 跨教师容量与 I/O 公平性、Cache 完整性 | 对真实 manifest 与 split 名单逐一审计 |
| **P0** | Run4 shadow `functional_call` 的 BN buffers 隔离及 main on/off 不变证据 | 部署不变式及合法唯一变量 | CPU+CUDA smoke，前后 state_dict/buffer sha |
| P1 | CTP 的 `L_task/L_pri/L_transfer`、`U_norm`、梯度方向/范数随 step 日志 | “通道传导低效”的具体原因 | 从已有日志和固定训练样本进行只读取证 |
| P1 | 同 teacher D2/RADIO 的 ZERO 反事实 | 目前跨教师绝对 F1/增量混有辅助容量差异 | Run4 按表各配独立 ZERO 40K |
| P1 | 新 PSFB 的 epoch 粒度边界、小目标及 hard-negative 错误切片 | >1pp 的作用区域和解释力 | 只读 train 分析+正式 test 固定评价口径；不能用 test 选择机制 |
| P2 | 四槽显存峰值/实际墙钟、真实同卡伴跑列表 | 预算估算及并发混杂 | 首次 smoke 和 run stdout/nvidia-smi 记录 |
| P2 | 更充分的 2024–2026 特权分支与 SCI/TGRS 论文核对 | 学术新颖性与 Related Work | 核验 publisher/CVF/作者原文，不能凭标题推断结构 |

**最终判断**：Run4 的价值不在于“又换一个蒸馏头”，而在于把命题收缩到一个可否证的问题：**当冻结教师的正确配对多通道信息真正进入学生任务 Decoder 的训练期空间前向，并且部署仍完全无教师时，学生是否获得超过同容量零输入与错配输入的可靠净收益；且不同教师的这一净收益是否足够分开，允许 Agent 学到适配性选择？** 若答案为否，仍保留九教师 + Agent 创新方向，但应将该具体桥接路径严格记为未检出，而不是继续包装相同机制。
