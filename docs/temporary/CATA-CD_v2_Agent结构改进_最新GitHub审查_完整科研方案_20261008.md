# CATA-CD v2：教师选择 Agent 结构改进与可证伪验证方案

**项目：** LS-Rep_BCD / LS-Rep_BCD_RSML_3  
**审查日期：** 2026-10-08（Asia/Singapore）  
**GitHub：** https://github.com/YuqiWang-code/LS-Rep_BCD  
**本报告固定源码版本：** [`1f2ee89b6a195d177b604215ba4a0af7d7d30a5f`](https://github.com/YuqiWang-code/LS-Rep_BCD/tree/1f2ee89b6a195d177b604215ba4a0af7d7d30a5f)（`main`，2026-10-08 04:57:37 UTC）  
**状态：** 研究设计与静态代码审查；**未**在 RSML-3 运行新训练/真实 Cache dry run；**未**修改仓库或服务器文件。  
**证据类型：** ① GitHub 源码/Registry/README 直接事实；② 根据事实作出的推断；③ **「待验证假设」**；④ 尚缺的源日志或配置。

> **审查范围声明。** 通过已连接的 GitHub 读取仓库的完整递归文件目录，并逐项读取本次主线的 README、Agent、Dataset、Cache、Trainer、评估、DCA、注册表、脚本和已有实验说明等关键源码。环境中对 GitHub 域名的直接 `git clone` 因 DNS 失败，故使用 GitHub 连接器完成读取。仓库约 295 个文件，包含历史第三方参考实现；本报告对当前主线关键链路做了细读，**不宣称逐行审完所有历史第三方文件**。原始 `train_log.txt`、服务器运行时和原始 Cache 未包含在 GitHub，因此数值使用提交的 Registry/README，正式论文数字仍需按日志末尾完整 TEST block 复核。

---

## 目录

1. [先给结论：是否失败、首选方案](#1-先给结论是否失败首选方案)
2. [不变主线与证据表](#2-不变主线与证据表)
3. [P0/P1/P2 代码与方法审查](#3-p0p1p2-代码与方法审查)
4. [为什么当前 Agent 学不稳](#4-为什么当前-agent-学不稳)
5. [五种候选 Agent 机制比较](#5-五种候选-agent-机制比较)
6. [首选：U-SafeAgent 的数学定义与数据流](#6-首选u-safeagent-的数学定义与数据流)
7. [奖励、标签、噪声与特征工程](#7-奖励标签噪声与特征工程)
8. [单 seed 下的可证伪验证协议](#8-单-seed-下的可证伪验证协议)
9. [实验矩阵、停止条件、硬目标](#9-实验矩阵停止条件硬目标)
10. [逐文件修改与可执行检查](#10-逐文件修改与可执行检查)
11. [2024–2026 可核验文献](#11-20242026-可核验文献)
12. [立即执行顺序与仍缺证据](#12-立即执行顺序与仍缺证据)

---

## 1. 先给结论：是否失败、首选方案

### 1.1 四条结论

**结论 A：当前 A-LODO/M1 的“可验证正收益”主假设失败。** Agent 对四数据集选择 `None / None / dinov3_lvd / anysat`；M1 相对 C1 的 test F1 为 `+0.19 / −1.03 / −0.01 / −0.05` 个百分点。没有四数据集一致正向，也没有接近硬目标。**不能将 Agent 的 2/4 LODO 精确命中解释为实际提升。**

**结论 B：现有 Agent 的确有结构与测量缺陷，但不能据此证明“教师选择永远不可学”。** 36 条教师-数据集记录不是 36 个独立环境；用于判断跨数据集泛化的独立单元只有 **4 个数据集**（LODO 每折训练 3 个）。MLP 的参数远多于独立任务数，无法可靠估计数据集级的教师收益函数。更重要的是，validation 与 test 的教师排序在 SYSU、WHU 直接反向。

**结论 C：不能从 `效应 / σ = 2.4` 得出“接近显著”。** 现有 σ 是同一配置跨运行的 F1 标准差，不是 `TV−C1` 的标准误。它未覆盖教师臂自身方差、对照/实验的相关性、模型选择偏差以及 9 教师×4 数据集多重比较。`RADIO→SYSU +0.68pp` 和 `DCA→WHU +1.38pp` 都是**值得优先复核的观察**，不是已证实的机制效应。

**结论 D：首选 **U-SafeAgent（Uncertainty-aware Safe-set Teacher Selector）**。** 保留 Teacher Package/Registry/DCA，**仅替换训练期 Agent**：

1. **先定义有效性**：以 C1 为零收益动作，采用不依赖 test 的 val 可靠性/负迁移判据；
2. **小容量估值**：数据集签名 + 经过审核的 teacher probe + 极少量 teacher metadata，使用带正则的线性/层级模型估计每教师收益；
3. **预测“安全候选集”而非盲目 argmax**：对估计不确定或优势过小的教师直接拒绝；
4. **唯一动作仍是单个 Teacher Package 或 None**：最终硬选择，绝不做同时蒸馏/权重融合；
5. **在 LODO 之后才允许 M1 从头训练**，且按固定协议报告是否真的跑赢 C1。

**重要可证伪边界**：用 3 个训练数据集无法校准严格的“95% 不负迁移保证”，本方案中 uncertainty bound 是**工程上的保守预测界/筛选门**，不是具有分布无关 coverage 的 conformal 证书。若 LODO 全部选 None，结论是**数据不足以支撑教师选择主张**，不能为了“让 Agent 动起来”降低阈值。

### 1.2 按用户要求分开裁定

| 问题 | 裁定 | 核心依据 | 不能推出什么 |
|---|---|---|---|
| (a) Agent 结构有缺陷吗？ | **是，P1** | 仅 4 环境；高维 MLP；教师 metadata 未实际入模；probe 无校准；固定教师/启发式基线不含公平 None | “改成 RL 就会涨” |
| (b) 选择问题本身不可学吗？ | **现协议下无法证明可泛化，非数学上不可学** | 有效域样本仅 4；36 条同域相关；LODO 只有 4 个留出决策 | “所有训练期条件选择都会失败” |
| (c) 噪声导致不可证伪吗？ | **现有单次点差结论识别不足，但可设计可证伪实验** | 同配置同 seed 运行不一致；σ 不同数据集相差约 40 倍；val/test 排名错位 | “所有非零效应都是噪声” |

---

## 2. 不变主线与证据表

### 2.1 保持不变的部署图与训练图

```text
训练期：
四数据集 TRAIN 的 A/B/label ──→ dataset signature s_D ─────┐
冻结大模型的双时相 Cache ──→ train-only probe c(D,T) ──────┤
离线能力注册表（仅 train/val） ──→ U-SafeAgent ──→ T* / None
                                                       │
A/B ──→ LWGANet-L0 ─→ NFA ─→ TFM ─→ DCA ─→ Decoder ─→ 主监督
                                       └─→ 训练期辅助 KD Head
                                               ↑
                              仅选择的 T* 的 Cache/Translator

部署期：
A/B ──→ 共享 LWGANet-L0 ─→ NFA ─→ 对称 TFM ─→ DCA ─→ Decoder ─→ 二值变化图
         × 无基础模型  × 无 Cache  × 无 Translator  × 无 Agent
```

- C0：clean A2Net-LWGANet-L0；**2,913,094** 参数 / **2.7676 G** FLOPs。
- C1：C0 + DCA `moe128`；**3,280,274** 参数 / **3.2432 G** FLOPs。
- DCA 增量：**367,180 参数**；有效部署剩余量 `5,000,000 − 3,280,274 = 1,719,726`，但本轮 **不允许继续增大 DCA**。
- swap 对称以及 `switch_to_deploy()` 误差：提交的 README 报告 **0**；必须通过实际 smoke 再确认。
- 固定四数据集、全监督二值检测、256×256；seed=2333、batch=64、40K steps；C0/C1/TV-*/M1 均用同一 ImageNet 初始化、从头训练。**Agent 离线学习不等同于重训学生，也不允许使用已有学生 checkpoint 做新的学生实验组。**

### 2.2 44 个 run：GitHub Registry/README 的逐数据集观察

**下表单位是 F1 百分点（pp），均为相对 C1 的单 run test 差；此表是“已提交的实验记录”，不是已验证的教师因果能力排序。**

| 数据集 | C0 F1 | C1 F1 | SAM2 | DINOv2 | D3-LVD | D3-SAT | RCLIP | MaRS | AnySat | UniverSat | RADIO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SYSU | 82.13 | 82.57 | -0.28 | -0.12 | -0.24 | -0.67 | -0.81 | -0.03 | -0.66 | -0.27 | **+0.68** |
| WHU | 92.66 | 94.04 | -1.15 | -0.36 | -0.21 | +0.08 | +0.08 | **+0.56** | -0.13 | +0.40 | -0.32 |
| CDD | 97.79 | 97.75 | -0.01 | -0.02 | -0.05 | -0.04 | -0.01 | -0.02 | -0.01 | +0.00 | -0.01 |
| LEVIR | 90.92 | 91.04 | +0.06 | -0.22 | +0.08 | +0.14 | -0.07 | -0.06 | +0.06 | -0.18 | ±0.00 |

**C1−C0：** SYSU +0.44、WHU +1.38、CDD −0.04、LEVIR +0.12pp。此差值受同配置运行波动影响，尤其 WHU/SYSU 不宜直接归因于 DCA。

### 2.3 Agent 唯一可以用于拟合的 validation 记录

| 数据集 | C1 best val F1 | SAM2 | DINOv2 | D3-LVD | D3-SAT | RCLIP | MaRS | AnySat | UniverSat | RADIO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SYSU | 81.55 | -0.31 | -0.39 | -0.09 | -0.30 | -0.57 | -0.12 | -1.08 | -0.44 | -0.24 |
| WHU | 94.82 | -0.34 | -0.42 | -0.31 | -0.20 | ±0.00 | -0.15 | **+0.12** | -0.43 | -0.30 |
| CDD | 97.73 | ±0.00 | -0.02 | -0.05 | -0.04 | -0.02 | -0.02 | -0.02 | ±0.00 | -0.01 |
| LEVIR | 91.50 | -0.09 | -0.09 | **+0.13** | -0.03 | +0.01 | +0.10 | **+0.15** | -0.18 | +0.14 |

**直接事实**：SYSU 上 RADIO 的 test +0.68，但 val −0.24；WHU 上 MaRS 的 test +0.56，但 val −0.15。**推断**：使用 test 排序构造教师动作标签会污染 Agent；坚持 val-only 可能错过 test 上偶然优秀的教师，但这是合法评估必须接受的代价。

### 2.4 LODO 与 M1 的记录

| 数据集 | LODO MLP | val oracle | M1 test F1 | C1 test F1 | M1−C1（pp） |
|---|---|---|---:|---:|---:|
| SYSU | None | None | 82.76 | 82.57 | +0.19 |
| WHU | None | AnySat | 93.01 | 94.04 | -1.03 |
| CDD | D3-LVD | None | 97.74 | 97.75 | -0.01 |
| LEVIR | AnySat | AnySat | 90.99 | 91.04 | -0.05 |

- **2/4 命中**不等于“显著优于基线”；只有 4 个留出动作，没有稳定统计精度。
- **“从不选有害教师”与数据矛盾**：CDD 选 D3-LVD，其 held-out validation reward 为 **−0.093pp（F1+0.5IoU 的复合效用）**，实际上是一次负迁移选择。
- SYSU、WHU 的 M1 选择为 None，按定义与 C1 是同配置；其 test 相对 C1 分别 +0.19、−1.03，说明**不应将 M1−C1 全部归因于 Agent**。

### 2.5 现有同配置噪声底

**同 seed=2333，同 C1 配置，提交的 3-way 并发测试 F1：**

| 数据集 | C1 | R1 | R2 | R3 | `s`（pp） | 极差（pp） |
|---|---:|---:|---:|---:|---:|---:|
| SYSU | 82.57 | 82.22 | 82.91 | 82.42 | 0.289 | 0.684 |
| WHU | 94.04 | 94.64 | 93.42 | 93.63 | 0.537 | 1.220 |
| CDD | 97.75 | 97.75 | 97.76 | 97.73 | 0.013 | 0.031 |
| LEVIR | 91.04 | 90.69 | 91.13 | 91.07 | 0.195 | 0.432 |

`s` 是 **n=4 个训练结果** 的样本标准差（不是单张图像评估误差，也不是 F1 差的标准误）。未经重新实验，不能区分 GPU 内核非确定性、worker/初始化、同设备争用或 checkpoint 选择的贡献。**“并发度不是主因”只可表述为“仅改变并发度不足以解释全部观测波动”**，而不是已经排除了并发的因果影响。

### 2.6 证据质量与缺口

| 层次 | 当前情况 | 是否支持正式论文结论 |
|---|---|---|
| ① 代码/配置 | `models/`、`train_scripts/CATA-CD/`、部分 JSON 可直接检视 | 支持“源码如此实现”，不等于通过运行测试 |
| ① 汇总数值 | README、`research_report.json`、`agent_train_registry.json` | 支持“仓库记录如此”，待原始日志末尾 test block 复核 |
| ② 推断 | val/test 排名冲突、有效域样本少、原有策略低信噪 | 支持设计动机，不支持因果归因 |
| ③ 假设 | 教师元数据+新 probe 能预测真实正迁移；安全集合降低误选 | **待验证假设** |
| ④ 缺失 | 44 个原始 train_log、全 checkpoint eval、设备确定性实测、各臂 val 噪声 | 无法给正式置信区间和独立重现结论 |

---

## 3. P0/P1/P2 代码与方法审查

### 3.1 P0：先改准确性/泄漏/可归因问题

| 优先级 | 文件与现状 | 风险 | 直接修复与验证 |
|---|---|---|---|
| **P0** | [`models/tools/build_capability_registry.py`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/1f2ee89b6a195d177b604215ba4a0af7d7d30a5f/models/tools/build_capability_registry.py) `TEST_RE.search(text)` | 找到**第一个**完整 TEST block，违反项目“最后一个完整 block”纪律；append/resume 情况可能读旧结果 | `list(TEST_RE.finditer(text))[-1]`，无闭合块拒绝；写单元测试含两个完整块、尾部残块、指标缺失 |
| **P0** | 当前 `best_val()` 在相同 val split 上选择 checkpoint，亦作为 Agent 的 reward | 双重适应：checkpoint 用 val 选、Agent 再用 best-val 指标排序，存在 winner's curse；原 36 条标签信号不独立 | 明确称为“checkpoint-selected validation utility”；有 checkpoint 时**不改训练**，离线在事先固定、不参与 checkpoint 选择的 `V_utility` 子集评估；若数据不足先做敏感性分析 |
| **P0** | 用户要求 test 不做策略输入；源码将 research/agent_train 分文件，但 `TeacherRegistry.load(which)` 非严格枚举 | 后续维护易发生载入 test；运行时缺少硬防线 | Agent 工具**只接收 agent_train_registry.json 路径**；解析前断言不存在 `test_*` 字段，输出记录输入 SHA256；research 不可通过 Agent API 打开 |
| **P0** | [`models/tools/train_teacher_agent.py`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/1f2ee89b6a195d177b604215ba4a0af7d7d30a5f/models/tools/train_teacher_agent.py) 的 A-F 固定教师与 A-H 启发式都被迫选教师 | 比较不公平：A-LODO 有 None 安全出口，A-F/A-H 没有，即便所有教师为负仍要动作 | 增加 `A-F+None`、`A-H+None`；保留原版作为 legacy 描述，主报告比较同样可拒绝的基线 |
| **P0** | 数据集签名与 probe 可能读取不到/错误匹配 CDD 命名路径；`dataset_signature.py` 直接 `A/sid`、`label/sid`，而主 Dataset 用 `_resolve_path()` | CDD 可能存在 `test_` 名称/`.jpg` vs `.png` fallback；与训练输入错位会污染策略 | 复用 `CDDataset._resolve_path()` 并断言 sample ID、split、mask 阈值、尺寸一致；对 CDD/其他数据各 16 样本交叉检查 |
| **P0** | `train.py` 无复验日志的完整原始 artifact 在仓库 | 现有 test 值不能按正式纪律再审查 | 只读取 RSML-3 原始 `train_log.txt` 最后**完整** test block，附 SHA256、run ID 和路径，绝不复写日志 |
| **P0** | `models/scripts/train.py` 在创建 `ChangeEvidenceHead` **之前**计算 `train_params = sum(model.parameters())` | TV-* 日志里的 `Train Params` 会遗漏 1×1 KD 头的 **8,320 个训练参数**（64×128+128）；混淆训练容量与部署容量 | 分开报告 `student_trainable=3,280,274`、`aux_trainable=8,320`、`training_total=3,288,594` 与 `deploy_effective=3,280,274`；C1 没有 aux，数值仍为 3,280,274 |

**特别纠正：`None` 判据不是当前运行脚本中所谓“反标准化遗漏”的 bug。** `train_teacher_agent.py` 先将 `policy` 预测的标准化值乘 `ys` 再加 `ym`，然后比较 **原始 utility > 0**；这一流程是正确的。真正的问题是 utility 标签的质量、阈值 0 的统计安全性与独立的 `offline_bandit.select_action()` 接口没有统一定义。不能把“文档诊断”当源码事实。

### 3.2 P1：影响 Agent 可识别性的设计瓶颈

| 位置 | 静态直接事实 | 问题 / 建议 |
|---|---|---|
| [`dataset_signature.py`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/1f2ee89b6a195d177b604215ba4a0af7d7d30a5f/models/agent/dataset_signature.py) | `GEOM_NAMES=10`、`APPEAR_NAMES=7`、`REG_NAMES=1` | **实际 18 维**，非附件写的 17 维；与 5 probe 合计 **23 维**，非 22；以 JSON `names` 长度为运行时真值，拒绝 silent fallback |
| 同文件 `n_components = max(len(comp_areas),1)` | 无变化图也被报告为 1 个连通分量 | 修正为真实 0；另外对“有变化图”和“全样本”分开统计 |
| 同文件 `cv2.Sobel(img,1,1)` | 计算的是 xy 混合导数，而不是常规梯度幅值 | 如需梯度差，改为 `sqrt(Sobel_x²+Sobel_y²)`；**签名语义改变须版本化** |
| 同文件对 sample 特征 `mat.mean(0)` | 对局部连通区域面积分位先逐图求再平均 | 不能代表全数据集的对象面积分布；应存样本级充分统计后全局汇总，并保留每图分布分位 |
| [`capability_probe.py`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/1f2ee89b6a195d177b604215ba4a0af7d7d30a5f/models/distill/capability_probe.py) | 5 个均值/比值来自随机投影 `local_change` 和 `confidence` | `confidence` 实为通道差异 max，并非经过校准的“正确概率”；不宜称 confidence calibration；新增排序性/AUC、边界和目标尺度 probe |
| [`probe_teacher_capability.py`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/1f2ee89b6a195d177b604215ba4a0af7d7d30a5f/models/tools/probe_teacher_capability.py) | 默认取 train list **前 256** 个样本 | 顺序偏倚/数据域覆盖差；用固定 seed 的 stratified train-only 采样或完整 train，记录索引哈希 |
| `train_teacher_agent.py` | `pv = ...get(..., 0)` | probe 丢失被编码为真实 0；应用硬报错或 missing indicator + 明确缺失处理 |
| `offline_bandit.py` / 主脚本 | docstring 写了 metadata，但主脚本实际只 `signature+probe` | 教师表征只靠 probe，可能无法识别架构/训练域；加 3–6 个**可审计离散/连续 metadata**，不直接加大深层 MLP |
| `train_teacher_agent.py` | MLP 23→96→32→1，800 epochs，按行 MSE | 模型复杂度远超独立域数；改强正则、小模型/基于风险的排序，统一 Agent seed=2333 |
| `train_teacher_agent.py` | 把 36 个 pair 当训练行，LODO 3 域×9 pair | 不能当 i.i.d. 的 27 个独立样本；缩放、Bootstrap、CV 都必须以**数据集簇**为单位 |
| `build_capability_registry.py` | reward 取 best-val F1 + IoU | 二值 CD 的 F1 与 IoU 通过 `IoU=F1/(2−F1)` 单调对应，同加权主要是在**重复计量**同一个检测效应 | 主 reward 用 **ΔF1_val（pp）**；IoU 用并行一致性约束，原复合 reward 仅做历史敏感性对照 |
| `train.py` | train loader 有 teacher 时通过返回状态的 Dataset 重构，而 none 时是另一 loader 实例 | 虽配置参数相同，但实际数据路径/随机流可能不同；损失差不能自动归因教师 | 固定训练样本 ID/增强 state 的前 100 批 SHA；用带教师但 KD λ=0 的数据流等价对照 |

### 3.3 P2：降低以后复现成本的工程修补

1. 在 Agent 模型 JSON 中固化 `git_sha, registry_sha, feature_schema_version, metric_unit='pp', threshold_rule, seed, folds`。
2. 输出每折真实训练域数/教师数/None 占比、预测上下界及未入模的缺失教师数量。
3. `offline_bandit.py` 和 `train_teacher_agent.py` 共享**一套** `score_raw_pp / None` 接口，不让任何预测值同时出现小数单位与 pp 单位。
4. `train.py` 的 resume 若整段后续训练都没有更好的 val，`best_path` 可能保持 None；需要从 checkpoint 保存/恢复 best 文件路径。此项不属于本轮 Agent 方法变化，**只做可靠性修复及测试，不用来制造新的实验组**。
5. README 的“CDD 已确认真正无效”“某效应接近显著”应改成“同协议单 run 未观察到可信正增益／待定”，避免在噪声估计过粗时下结论过度。

---

## 4. 为什么当前 Agent 学不稳

### 4.1 36 条监督的真实信息量

当前特征行 `x_{D,T}` 有 4 个 dataset signature，不同教师在同一数据集共享相同 `s_D`，各条 reward 都与同一 C1 锚定。因此 36 行高度相关，LODO 只有 3 个独立域可拟合 `dataset→teacher utility` 的外推。把这些行随机划成 train/test 会造成**域泄漏**：同一数据集的 signature 出现在训练和测试中。

**可证伪预测 H1（待验证假设）**：当改用更简单的收缩线性模型 + 显式拒绝后，LODO 的有害选择数量和 regret 降低；如改善只出现在原数据而 4 折并无改善，则结构假设失败。

### 4.2 要区分“预测排序”与“预判可靠可用”

我们真正需要的不是 `argmax_T E[r(D,T)]`，而是

`是否有证据支持某 T 在当前数据集优于 None（即 C1）？`

再在满足安全要求的教师间选择。现实现无法估计不确定度，最大预测效用只要略高于 0 就行动；CDD 上 `D3-LVD` 是直接反例。

### 4.3 val/test 反向不是“可以偷偷用 test 修正”的理由

SYSU 的 RADIO 和 WHU 的 MaRS 具有 test 正收益但 val 负收益。可能原因包括：评估域结构差、best checkpoint 选择噪声、学生对教师知识的收益随机、验证样本不足等；**未验证假设，不能做定论**。如果开发者用 test 上的 RADIO/MaRS 正差来写路由规则，之后的“独立 test”便失效，不能声称验证了 Agent 泛化。

### 4.4 “Agent”在当前设置更准确的表述

当前每个训练域上 **9 个教师全部有离线验证 reward**，这是一个 **full-information contextual action-selection / offline cost-sensitive policy learning** 问题，不是经典只观察被选择动作奖励的 online contextual bandit。若论文继续称 Agent 可以，但应避免声称具备在线试错反馈、RL 自主探索或真实终身学习。

---

## 5. 五种候选 Agent 机制比较

以下模型均仅用既有 36 条历史 reward 和 TRAIN signature/probe 做 **离线训练**，无额外学生 40K 成本；真正 M1 只有最终方案通过门槛后才从头训练。

| 候选 | 机制与创新焦点 | 对 6 个问题的作用 | 训练样本/额外算力 | LODO 主要风险 | 通过/失败判据 |
|---|---|---|---|---|---|
| **A1 两阶段层级（H-Gate）** | 第一关“该域有无可用教师”，第二关仅在通过者中排序 | 明确 None；避免持续选择弱教师；不确定性仍缺 | 4 域×9 教师，CPU 秒级 | 第一关只有 3 个训练域标签，二分类严重失衡 | 有害动作 ≤1/4、regret 优于 MLP；若全部 None 则无效覆盖 |
| **A2 双分支（SafeRank）** | `p(收益>δ)` + 条件收益 `μ` 两头，先准入后排序 | 让二值安全与收益幅度解耦；可用 pairwise ranking | 同上，CPU 秒–分钟 | 正例罕见导致 gate 塌缩；二头数据共享不能冒充 2 倍样本 | 相比单 MSE 降低负迁移且保留真正正动作 |
| **A3 保守预测集合（U-SafeAgent）★** | ridge/层级回归得 `μ,u`；`μ−κu>δ` 才进安全集；集合为空选 None | 同时处理 None、异方差、幅度不稳定、教师信息、argmax 脆弱 | 同上；若重算 probe 仅 CPU/GPU 短任务 | u 的跨域校准无法严密保证；过度保守 | 低有害率、低 regret、足够 coverage；任一未达即拒绝方法主张 |
| **A4 可解释规则+学习（Probe-Filter）** | 基于 train-only 判别力先淘汰质量差教师，再线性收益预测 | 强可解释、极低参数；可以测试 probe 机制价值 | 36 行，无学生训练 | probe 与蒸馏收益可能不相关，误过滤好教师 | 比 random+None、固定+None 好；否则去掉规则 |
| **A5 元学习/ProtoNet/MAML** | 数据集为 task，teacher 为 action，跨 task 学原型/初始化 | 理论可增广条件化，但不产生新独立任务信息 | 训练成本 CPU–GPU 小，但高元参数 | **只有 3 元训练任务**；极易伪泛化 | 默认否决；需有大量独立新遥感 CD 任务才有资格讨论 |

### 5.1 对每一候选的最小消融与可证伪性

- **A1**：与原 MLP 同 18+5 特征，唯一变量改为层级门；若在 LODO 中不减少负收益动作，失败。
- **A2**：与 A1 共用 train-only 特征，只把 `MSE` 换为 `binary+ranking`；若对弱正收益的 recall 明显降低而风险无改善，失败。
- **A3**：先用相同 ridge 均值预测，依次加入 `uncertainty bound`、`safe set`，每次只改一个组件；如果优于简单 `ridge+None` 的收益不足，不能称安全集合创新。
- **A4**：固定线性排序器，仅添加 probe 淘汰；与“等数量随机 teacher 淘汰”匹配；如果不能超过随机删候选，probe 没有机制价值。
- **A5**：仅做理论负对照，不建议加入本轮正式论文；不能用 3 个域训练 MAML 并据此宣称泛化。

### 5.2 为什么首选 A3 而不是 RL、深度集成或 conformal

RL 需要足够的交互/状态转移，当前是静态全信息表格；深度集成只能测模型拟合器不稳定，不代表跨域 epistemic uncertainty 已校准；严格 conformal coverage 至少要足够多的独立、可交换校准环境，只有 3 个训练域时不具备。**选可解释、小容量、可拒答的 A3 是当前最小结构赌注**，且易被 A1/A2/A4 证伪。

---
## 6. 首选：U-SafeAgent 的数学定义与数据流

**论文建议工作名（待验证）：** U-SafeAgent：*Uncertainty-aware Safe-set Teacher Selection for Capability-Validated Lightweight Remote Sensing Change Detection*。注意：安全并非数学上的绝对“无负迁移”，只表示“在事先定义的 val 证据门槛内没有发现高风险”。

### 6.1 条件决策对象

令 `D ∈ {SYSU, WHU, CDD, LEVIR}`，教师 `T∈𝒯`（9 个已完成 Teacher Package），基线 `T=∅` 代表 C1。定义

\[
\mathcal{A}=\{\varnothing,T_1,\ldots,T_9\},\qquad
U(D,T)=\operatorname{F1}_{val}(D,T)-\operatorname{F1}_{val}(D,C1),\quad U(D,\varnothing)=0.
\]

建议统一在代码与 JSON 中用 **pp**（百分点），即 `100 * (f1_fraction_T - f1_fraction_C1)`。`IoU` 为二值检测相关的伴随指标；当 `ΔF1` 和 `ΔIoU` 方向不一致时应检查代码/四舍五入/评估定义，而非把 IoU 权重当新创新。

训练特征必须在本折中独立合法得到：

\[
x_{D,T}=\left[s_D^{(q)},\ p_{D,T}^{(q')},\ e_T^{(q'')}\right],
\]

其中 `s` 仅来自 D 的 TRAIN A/B/label，`p` 仅来自 D 的 TRAIN Teacher Cache/label，`e` 为不随 val/test 改变的元数据；任何 teacher 的 TEST F1/IoU 都**不能作为 `e_T` 或 `p` 的替代标签**。

### 6.2 三层结构

**第一层：候选先验可行性（Train-only Feature Eligibility）。**

- A/B/label、Cache checksum、分辨率、split、NaN、常量特征、变化/未变化像素分布等全部通过才允许参与。
- 不用“teacher probe AUC<阈值即有害”的硬规则，除非阈值是外层训练折内确定；先仅把 probe 作为 covariates。
- 加入 `None` 为合法动作，始终保留。

**第二层：低容量效用与不确定性估计（Ridge+Group Jackknife）。**

使用共享岭回归，不按 `teacher_id` 单独训练 9 个拟合器：

\[
\hat{\boldsymbol\beta}
=\arg\min_{\boldsymbol\beta}
\sum_{D\in \mathcal{D}_{train}}w_D\sum_{T\in\mathcal{T}}
  \ell_{\delta}\big( U(D,T)-\phi(x_{D,T})^\top\boldsymbol\beta\big)
+\lambda\|\boldsymbol\beta\|_2^2.
\]

- `w_D=1/|𝒯_D|`，先按数据集簇均衡，避免教师更多的数据集“票数”更多。
- `ϕ` 优先为**8–12 维的固定预选解释性特征**或它们与 2–3 个教师大类的交互；**不使用 18+5+多 one-hot 全量维度的高阶交互**。
- `ℓδ` 可从平方误差开始，考虑 Huber 仅作为稳健性必要对照，不能包装成方法创新。
- `λ` 不可在 4 折 held-out outcome 上反复择优；先预定（如 `λ=1` 标准化尺度），或在训练域内极少折进行嵌套筛选并明确其不可靠性。

**模型不确定性代理：** 对某留出数据集 `D*`，在其余 3 个训练域内做 leave-one-*training-domain*-out 拟合，得到 `\hat\mu^{(-d)}`。令

\[
u_{model}(D^*,T)
=\operatorname{SD}_{d\in\mathcal D_{train}}
  \left[\hat\mu^{(-d)}(D^*,T)\right].
\]

它只有 3 个预测，因此**只是敏感性代理，不是合法 95% 置信标准差**。另引入只从训练折可用数据拟合的 val 运行波动代理 `u_noise`。如果只有 C1 重复，采用保守上界/敏感性情景，不能声称知道 Teacher-specific 方差。综合

\[
u(D^*,T)=\sqrt{u_{model}^2+u_{noise}^2+u_{shift}^2}.
\]

`u_shift` 可先设为 0 并独立做 ablation；若增加“距离训练域签名越远越保守”的距离惩罚，必须写成预先定义的 OOD 启发式，不可宣称泛化风险上界。

**第三层：安全集合 + None。**

\[
L(D,T)=\hat\mu(D,T)-\kappa\,u(D,T),\qquad
\mathcal S(D)=\{T:\ L(D,T)>\delta,\ \widehat{\Delta IoU}(D,T)>-\epsilon\}.
\]

\[
a(D)=
\begin{cases}
\varnothing,& \mathcal S(D)=\varnothing\\
\operatorname*{argmax}_{T\in\mathcal S(D)}L(D,T),&\text{其余情况}.
\end{cases}
\]

**首轮固定的门槛（属于工程筛查，不是统计显著性）：** `δ=+0.20pp`、`κ=1`、`ε=0pp`（IoU 仅用同一预测的可核验方向，若没有可信预测则不要伪造此 gate）。阈值来源是项目现有 Strong Positive 事前标准，并非根据 test 精调。为了不让 α/m 等概念被误读为正式统计检验，此处**不称 1σ 或 95% 保证**。

如果觉得 `κ=1` 过于严谨/松弛，**只预注册 κ=0/1 两个对照**，不要运行 10 个阈值再按 test 最好者选。

### 6.3 为什么能在小样本下提供相对清楚的机制解释

| 输入信号 | 被解释的知识需求 | 预测机制（均为待验证） | 可证伪手段 |
|---|---|---|---|
| 小目标比例、对象面积分位 | 空间分辨率/边界细节需求 | 细粒度教师的边界 probe 更可能产生正 utility | 移除相关数据签名，看动作/校准是否变差 |
| 未变化区的时相光度差 | 伪变化、域干扰 | teacher 若在 unchanged 区差异很高，应降低可信度 | 移除伪变化 probe，看误选是否增多 |
| Cache 变化/非变化区排名可分性 | 教师差分证据与真变化的相关性 | 高 separability 有可能更易蒸馏 | 控制教师 ID 后衡量 train 折 Spearman 与残差收益 |
| native stride、空间特征类型 | 密集几何 vs 语义上下文 | SAM/RADIO/ViT 互补性通过 metadata 表达 | 仅 metadata 与 metadata+probe 比较 |
| 模型对训练域的 distance | 外推风险 | 新域偏离已有 3 域时 None 更常被选 | 距离相关性与负迁移率的折间方向是否一致 |

**拒绝“偷换为多教师互选”：** `S(D)` 虽可包含多个“备选教师”，最终动作为单个 T 或 None，**训练器只加载一个教师 Cache**，无并行 teacher 特征融合、无基于老师互评的 teacher graph、无 teacher map 注入部署路径。

### 6.4 与当前 Agent 的实质区别

- 当前：MSE 分数 → `max>0` 就选；`None` 是一个数值阈值的结果。
- 建议：**特征有效性 → 低容量效用估计 → 风险余量 → 安全集合 → hard action**；None 是显式风险控制的默认动作。
- 当前：每行有效视为 i.i.d.，只输出选谁；建议：数据集簇训练，输出 `mu, model_spread, noise_proxy, margin, coverage, reason_for_rejection`。
- 论文新颖性只能归于“在多教师 Cache 的变化能力验证后，针对数据集级证据稀少的**可信决策与安全拒绝机制**”；不应仅将 ridge 或规则阈值包装成深度创新。

### 6.5 最小数学伪代码

```python
def decide(dataset_train_signature, teacher_train_probes, teacher_meta,
           fit_from_other_datasets, delta_pp=0.20, kappa=1.0):
    options = []
    for teacher in VERIFIED_TEACHERS:
        x = make_schema_checked_features(dataset_train_signature,
                                         teacher_train_probes[teacher],
                                         teacher_meta[teacher])
        mu = fit_from_other_datasets.predict_pp(x)
        u  = fit_from_other_datasets.uncertainty_proxy_pp(x)
        margin = mu - kappa * u
        if margin > delta_pp and fit_from_other_datasets.quality_gate(teacher, x):
            options.append((margin, teacher))
    if not options:
        return {"teacher": "none", "reason": "no_supported_positive_utility"}
    margin, chosen = max(options)
    return {"teacher": chosen, "margin_pp": margin,
            "reason": "accepted_by_predeclared_margin"}
```

---

## 7. 奖励、标签、噪声与特征工程

### 7.1 三种奖励定义（主次明确）

**R0（历史对照，不改变 44 run）：**

\[
r^{old}_{D,T}=100\,(\Delta F1_{val}+0.5\Delta IoU_{val})\ \text{pp}.
\]

**R1（首选）：**

\[
r^{new}_{D,T}=100\,[F1^{utility}_{val}(D,T)-F1^{utility}_{val}(D,C1)]\ \text{pp}.
\]

其中 `utility_val` 最理想应为预留的 **V_utility**（与 checkpoint selection 的 `V_select` 分离）；若当前手头仅有每个训练 run 的 best-val 标量，先重跑**离线 checkpoint evaluation**而不是重训学生；如没有可用 checkpoint，则保留历史 reward 并显著标注验证效用的筛选偏差，**不假装旧 val 已独立**。

**R2（稳健序数标签，可作为必要对照）：**

\[
y_{D,T}=
\begin{cases}
+1,&r^{new}_{D,T}>\delta_D\\
-1,&r^{new}_{D,T}<-\delta_D\\
0,&|r^{new}_{D,T}|\le\delta_D
\end{cases}
\]

`δ_D` 的校准**只能用训练折所允许的 val 可靠性证据**。对留出域 D* 不得用其 test σ（乃至其 held-out val outcome）反向决定阈值。R2 的 neutral 状态不是负例；应排除分类 loss 或赋较小权重，避免将近零随机符号硬编码为正确答案。

### 7.2 正收益分类、排序与回归的取舍

- **二分类 positive/negative**：解释清楚，但正标签极少、None 容易一边倒；必须引入 neutral/unresolved 第三类，不建议简单 0/1。
- **Pairwise Ranking：** 在相同数据集内对 `r(D,T_i)−r(D,T_j)` 建立顺序，能抵消共享 C1 的加性误差；然而教师臂本身仍波动、绝对 `>None` 无法由两两相对排序得到，因此**不能单独解决拒绝决策**。
- **Margin Ranking + None：** 把 None 当固定效用 0，只有 `r(D,T)>δ_D` 才与 None 建正排序 pair；`|r|≤δ_D` pair 不使用，减少微小差异过拟合。
- **Variance-weighted Loss：** 权重 `w∝1/(\hat v_{D,T}+v_0)` 只在 val 重复测量足够时有效；**不允许从 test σ 推入 Agent loss**，也不能把没有测到的教师臂方差填成 0。

### 7.3 val σ 的来源与成本

1. **优先从现有 C1/R1/R2/R3 的完整日志提取 `best val F1`，估计 val run-level σ**；不使用 test σ 标定 Agent。
2. 若每个 teacher run 有多个 epoch validation，**不能把同一 run 的 100 个 epoch 当作独立训练重复**；可以分析 peak 宽度或 `last-K` 稳定性，但不能以 epoch 方差当 run-to-run 方差。
3. **事先选定** checkpoint selection 规则（当前 best-val），固定后全组相同；可在历史 checkpoint 上离线评估固定 V_utility。若没保留 checkpoint，缺失就明确记录。
4. 仅为了估计 teacher-specific 运行方差而复制全部 36 臂非常昂贵且与“固定单次主消融”冲突。本轮**不推荐**，优先解决确定性和 reward 结构。

### 7.4 低成本新增 Probe（首批 5 个优先，第二批备选）

令 `M` 为 train 标签，`z(x)` 为教师 Cache 的非负变化响应图。首选用**原始 `confidence` 的响应值或归一化 `||local_change||_2`**；若使用跨教师统一绝对阈值必须先做尺度校准。用 TRAIN 内每图均衡抽样或分块统计，正负像素均衡、保留空间对象统计。

| 编号 | 公式/实现 | 意义 | 成本 | 风险与控制 |
|---|---|---|---|---|
| **P1：Balanced pixel AUROC** | `AUC(z[M=1], z[M=0])`；每图正负均衡采样，再跨图平均 | 判别真假变化而不依赖尺度阈值 | Cache-only CPU | 类别过少时方差大；零变化图单独统计 |
| **P2：Hard-negative separation** | `Q50(z|M=1)−Q90(z|M=0)`，再除 `MAD(z)+eps` | 真变化是否压过 unchanged 的“高伪响应” | Cache-only | 测试不同分位是调参，不当创新 |
| **P3：Boundary contrast** | `mean(z|d(M,∂M)≤r, M=1) − mean(z|外邻域窄带, M=0)`，`r=2` 像素 | 边界/定位证据 | Cache-only | Teacher stride 限制，记录上采样方式 |
| **P4：Size-conditioned AUROC** | 按 GT 连通分量面积 `<a0` 与 `≥a0` 分组，分别统计局部 AUROC，报告差值 | 小目标与大目标兼容性 | Cache-only | `a0` 必须相对 256×256 先固定，如 `<0.5%` 图面积 |
| **P5：Unchanged leakage ratio** | `E[z|M=0] / (E[z|M=1]+eps)` | 背景/伪变化干扰 | Cache-only | 像素级汇总易受大背景压制，按图统计 |
| P6：top-k localization precision | 在每图固定最高响应的 `k=max(16, ⌈0.01HW⌉)` 点中，计算 changed 占比 | 稀疏响应是否命中变化 | Cache-only | k 与真实变化比对，报告 sensitivity |
| P7：低/高频响应比 | `FFT(z)` 的高频能量占比 | 对边界纹理的响应 | CPU | 不等于真实有用；不优先 |
| P8：Multi-augmentation reliability | 同一 RGB 输入做指定 flip，Cache 特征等变误差 | 教师输出的几何一致性 | **需新增教师编码** | 成本增加；暂不纳入首批 |

首批只加入 P1–P5；**不要 18 signature + 5 原 probe + 8 新 probe 一股脑塞入 MLP**。训练时用预注册的强制缩维，如最多 10–12 个输入特征；新增 P1–P5 先做边际 Spearman（按域簇解释，只有 4 点无统计 power），再挑极少物理动机特征进入正式方案。

### 7.5 数据集 Signature 的补充与纠偏

当前 `dataset_signature.py` 是 18 维（10+7+1），建议保留旧 `schema_version=1` 以复现，再新增 `schema_version=2` 独立文件，不覆盖旧签名。建议优先改变/补充：

| 维度 | 明确定义 | 科学作用 |
|---|---|---|
| empty-pair ratio | `N(∑M=0)/N` | 直接描述稀疏变化，避免 n_components 的 `max(1)` 偏置 |
| object area 全局分位 | 汇总 TRAIN 所有 GT connected-component 面积占图面积比 `a_i/HW`，计算 P25/P50/P90 | 克服逐图分位取平均不能代表全域目标分布 |
| boundary density | `|∂M|/(|M|+eps)`，先按图再汇总中位 | 边界相对复杂度 |
| illumination on unchanged | `mean_{M=0}|A−B|`（需在明确亮度/归一化空间） | 不相关时相差异难度 |
| hard-pseudo-change fraction | `P(|A−B|>q_{90}(|A−B|\mid M=1)\mid M=0)`，即未变化像素差超过**训练集变化区**差异第 90 分位的概率 | 衡量伪变化是否接近或超过真实变化幅度；若变化像素缺失则记为不可计算；阈值仅用 TRAIN |
| change area CV | `SD(change_ratio_per_image)/(mean+eps)` | 变化密度跨样本异质性 |

**需核查的数据事实冲突：** 当前已提交 LODO 签名的 `change_ratio` 是 SYSU `0.157732`、WHU `0.046415`、CDD `0.120657`、LEVIR `0.067676`，与此前服务器说明中的大致占比不完全一致。差异可能来自统计范围、二值阈值、文件 split 或不同版本数据；**不能直接选其中一组为真**。先输出以 `list/train.txt` 清点的分母、正像素、图像数、文件 SHA，对每个源做同口径复算。

### 7.6 Teacher Metadata 的最小必要字段

推荐不含 Teacher ID 的以下 5 个字段/概念（实际编码可能 3–7 列，控制维数）：

- `training_domain`：自然图像 / 通用卫星或 EO / 混合（事先指定类别，查模型卡，不能使用实验结果分类）。
- `primary_objective`：segmentation / self-supervision / vision-language / agglomerated features。
- `native_stride`：编码器主输出空间 stride（数值编码为 `log2(stride)`）。
- `pyramid_available`：是否有原生多尺度输出（boolean）。
- `cache_resolution`：实际存盘 `Ht×Wt` 而非声称的原模型分辨率。

可选 `log(params_teacher)` 作为计算资源协变量，但本项目**不假定教师越大效果越好**。禁用由 SYSU/WHU/LEVIR/CDD **test 排名**构建的“teacher suitability embedding”。

---
## 8. 单 seed 下的可证伪验证协议

### 8.1 首要问题：必须先区分四类不确定性

1. **优化/实现不确定性**：同配置同 seed、同硬件/并发，因 CUDA 非确定性或 DataLoader 执行路径导致训练输出不同，现有 C1 重复主要测到这一类及 checkpoint 选择引起的放大效应。
2. **初始化/数据抽样不确定性**：真正多 seed 变化。**项目禁止多 seed**，本研究不估计这一类的分布。
3. **有限评估样本不确定性**：固定 checkpoint 在若从目标总体重新抽样验证/测试场景，其 F1 会怎样变化；可用图像/地块级 bootstrap 描述，不能替代训练波动。
4. **跨数据集外推不确定性**：只观察 4 个任务域，在未观察的新域能否判断教师效用；既不能靠 36 个同域 teacher pair，也不能靠同 seed 重复跑模型来补足独立任务数。

**方法学底线**：同 seed 重复可用于**确定性审计、噪声底估计及阈值灵敏度**，但不是“多 seed 取平均”的替代。主报告仍按固定 seed 单次正式 run 逐数据集报告；不能借同 seed 重复平均后的 test 来替换正式 F1。

### 8.2 题目 (i)：每数据集需要多少同 seed 重复才可估计 σ？

假设同配置运行结果在固定环境下近似独立、同分布、正态，观察 `n` 次 F1 的样本标准差为 `s`，则

\[
(n-1)s^2/\sigma^2\sim\chi^2_{n-1},\qquad
CI_{1-\alpha}(\sigma)=\left[
 s\sqrt{\frac{n-1}{\chi^2_{1-\alpha/2,n-1}}},
 s\sqrt{\frac{n-1}{\chi^2_{\alpha/2,n-1}}}
\right].
\]

**重要条件**：如果重复存在跨 GPU、操作系统、共享任务队列、训练日程变化，且条件没有固定，则“相互独立同分布/正态”不成立，以下区间仅是敏感性估计，不是可靠频率学覆盖。

按 `α=0.05`，95% σ 区间宽度因子如下：

| 同配置总次数 n（含原 C1） | σ 区间约为 `[下界,上界]`（乘 s） | 当前 4 数据集额外 40K run 数 | 额外总 steps | 科学判读 |
|---:|---:|---:|---:|---|
| **4（已完成）** | `[0.566s, 3.729s]` | **0** | 0 | 只能粗画噪声量级，远不能谈精确阈值 |
| 6 | `[0.624s, 2.453s]` | 8 | 320K | 仍非常宽 |
| 8 | `[0.661s, 2.035s]` | 16 | 640K | 可用作保守风险筛查，不适宜正式显著性断言 |
| 12 | `[0.708s, 1.698s]` | 32 | 1,280K | 仅约“上下 30%–70%”的粗估 |
| 20 | `[0.760s, 1.461s]` | 64 | 2,560K | 精度开始改善但成本高 |
| 30 | `[0.796s, 1.344s]` | 104 | 4,160K | 仍有 ±20%/34% 相对不确定性 |
| 50 | `[0.835s, 1.246s]` | 184 | 7,360K | 上下界才约为 −16%/+25% |

**直接回答“最少几个”：** 若“可用”指知道 σ 的大致量级、用于保守门槛敏感性分析，**至少 n≈8**（每数据集总 8 次）才勉强合适；若要求 95% σ CI 的上界至多约 `1.35×s`，**n≈30**。项目不接受这种训练开销，所以**本轮推荐沿用已完成 n=4 的粗 σ 作为风险提示，不额外跑 104 次主模型**，同时将“有统计把握的阈值”改为“未能估准”。

即使 n=50，也只表征固定 seed 在特定环境下的运行非确定性，不能替代不同初始化种子泛化。

### 8.3 题目 (ii)：同 seed 估方差与多 seed 有何不同？

应在方法节这样写：

> 本研究固定 student seed=2333，以评估机制在相同预训练初始化、训练预算、数据划分和训练流程下的逐数据集效果。另以完全相同的 seed/config 独立重复 C1，仅审计实现层面的 run-to-run 非确定性；这些重复不用于主结果取均值、不用于选择 test 最优 checkpoint，也不构成多 seed 的稳健性主张。正式指标均为指定 run 的最后完整 TEST RESULTS block。

要加**可复现环境锁定**：同 torch/cuda、same GPU ID、同 worker 数、训练样本次序、每 batch crop/flip/exchange 状态、初始 model weights checksum、AMP/TF32 设置、cuDNN/cublas 工作区与 deterministic 算子策略、同并发度。若启用 `torch.use_deterministic_algorithms(True)` 在某核报错，先保留 traceback，再找确定性替代；**不允许悄悄修改模型/损失使其“变稳定”后把旧结果直接对比**。

### 8.4 题目 (iii)：配对比较是否更合适？公式是什么？

**结论：比两个独立、异时运行的单点差更合适，但必须事先定义配对依据。** “seed 相同”不是自动形成配对的充分条件。真正配对实验必须相同 GPU/并发槽位、训练初始化、增强随机轨迹、数据顺序、epoch+step 规则，并在同一环境记录成对 `C1_k` 与 `TV_{T,k}`。

对 `k=1...n`，配对差

\[
 d_k = 100\,(F1_{TV,k} - F1_{C1,k})\ \text{pp},\qquad
 \bar d=\frac{1}{n}\sum_kd_k,\quad
 s_d^2=\frac{1}{n-1}\sum_k(d_k-\bar d)^2.
\]

**差分方差恒等式：**

\[
 \operatorname{Var}(TV-C1)=\sigma^2_{TV}+\sigma^2_{C1}-2\operatorname{Cov}(TV,C1).
\]

- 若两臂训练噪声高度正相关，配对可显著降低方差；若互不相关，配对并不降低。只测了 `σ_C1` 不能得到 `s_d`。
- **主实验目前项目纪律不做显著性检验**，所以主文件只用配对差**描述与定位**，不据此宣称 p 值证明。
- 如果用户未来**明确修改纪律，准许预注册的同 seed 重复显著性实验**（依然不换 seed），在独立配对、差值近似正态下，可用单侧检验

\[
H_0: \mathbb E[d]\le\delta \quad vs\quad H_1:\mathbb E[d]>\delta,
\qquad t=\frac{\bar d-\delta}{s_d/\sqrt n}.
\]

并以 `t_{1-\alpha/m,\,n-1}` 为 **Bonferroni 保守门**：

\[
\bar d- t_{1-\alpha/m,n-1}\,s_d/\sqrt n >\delta.
\]

`α=0.05`；如果事先正式检验 `9×4=36` 个教师×数据集组合，`m=36`（单侧 α/m≈0.00139）；如果只在锁定 M1 策略后**一次**测四数据集，`m=4`。更有力且没那么保守的方式是 Holm step-down：把 p 值从小到大排序，依序与 `0.05/(m−i+1)` 比较，首个不满足后停止。**绝不可在看见 test 结果后把 m=36 偷缩成 1，或把 test 用来反复选择 κ/δ。**

**必须写明两项边界**：上述检验只针对当前 seed、环境下的重复运行随机性，不是跨 seed/跨域总体显著性；对 `n=1` 根本没有 `s_d`，因此现有 TV 单次结果**不能套用该公式假造“显著”**。所有包含未来条件检验的部分，仅作可选的规范备选，不在当前固定“单次主消融、不做显著性检验”纪律下实施。

### 8.5 异方差加权与误选成本

如果未来训练折中有 val-only `\hat v_{D,T}`，可使用

\[
 w_{D,T} = \frac{1}{\max(\hat v_{D,T},\,v_{floor})},\qquad
\tilde w_{D,T}=\frac{w_{D,T}}{\sum_{T'}w_{D,T'}}.
\]

各数据集先平均各教师损失，再平均 3 个训练域，防止“CDD σ 很小”让整个模型几乎只学 CDD。**目前仅有 C1 test σ，不能直接生成 `\hat v_{D,T}^{val}`；缺失方差的权重不允许自动设为无穷大。**

更稳妥的决策评分在 LODO 中分拆：

\[
\mathrm{Regret}(D)=\max(0,\max_T r_{D,T})-r_{D,a(D)},
\]

其中 oracle 使用**held-out validation reward 只做评价**，不用于训练、阈值/超参数选择；同时报告

- `HarmRate = #(selected teacher with held-out r<−0.20pp)/4`；
- `FalseActionRate = #(selected teacher with held-out r≤0)/4`；
- `Coverage = #(selected non-None)/4`；
- `mean/median regret`；
- `AnyPositiveRecall = #(有效域选到 val 正教师)/(有效域数)`；
- `exact oracle match`（只作辅助，不作为首要优化指标）。

**不能只追 HarmRate=0**：恒选 None 得 0 harm，也必然不能创造任何 teacher 正收益；必须连同 Coverage/收益/regret 判断 Agent 是否有价值。

### 8.6 题目 (iv)：扩增评估图片或重复评估能否降低 WHU σ？

| 操作 | 能降低什么 | 不能降低什么 | 本项目建议 |
|---|---|---|---|
| 在**同一固定 checkpoint** 上重复 deterministic val/test | 通常什么也不会降低，应几乎完全相同 | 不能减训练期 run-to-run σ | 做一次相同 checkpoint×3 评估验证确定性；非科研结果 |
| 对既有 val/test 图像做**按地块/场景分组 bootstrap** | 可描述“有限评估样本”不确定性 | 不改变正式数据量，不抹平不同训练 run 的差异 | 仅做解释分析，F1 仍全 test 一次统一计算 |
| 增加同分布、完全独立且标签可靠的 **validation** 场景 | 有机会降低 val 估计的样本误差 | 不必然改善优化/内核波动 | 仅在事先固定的数据协议允许时；不能动 test split |
| 把 train/val/test 合并扩大 val | 看起来可能降低均值波动 | **会泄漏/改变基准** | **禁止** |
| 用相同样本多次增强推理后平均 | 可降低 TTA 波动 | 改变推理协议和 FLOPs；不解决训练噪声 | 不用于本轮主比较 |
| 同 seed 重跑 n 次独立训练取平均结果 | 降低该噪声机制下的估计均值方差 | 违反本项目“主结果固定单 run”；不等于多 seed 稳健性 | 仅限明确标记的 noise audit，不作为正式模型性能 |

WHU 官方 test 如果现协议只有 744 图，不能通过“再重复评估这 744 图”把样本数变成 `3×744`。应考虑**scene-level 相关性和小样本可靠性**，而不是伪增样。

### 8.7 留一数据集交叉验证 LODO 的严格嵌套顺序

对每个 holdout `D*`：

1. **排除** D* 的 `val reward/test reward`，但允许其 TRAIN signature 和 TRAIN Cache probe 作为**部署前上下文**。
2. 仅在其他 3 个训练数据集拟合 scaler、reward model、噪声 proxy、metadata 缺失编码；选择/预设阈值（尽量预先固定）。
3. 只使用训练域内部的 split/拟合来调 ridge λ，**禁止看 D* 的 oracle-val 再挑参数**。
4. 在 D* 的 train signature/probe 上输出一次 T*/None，冻结 decision JSON。
5. **仅评估** D* 已有的 held-out val reward 得到 regret/coverage/harm，不回写拟合器。
6. 四折汇总完成后，必须锁定首选结构与参数；不能在“反复看四折结果然后重新挑几十个参数”的条件下仍声称这是独立的 4 折泛化验证。至少保留完全预注册的单一主模型与极少必要消融。

**能力登记先于 Agent**：是否进入候选池的门槛也应 train/val-only；`research_report.json` 的 test 观察只能在研究报告里展示，不应通过“只保留 test 上正的 Teacher”形成隐蔽 test 泄漏。若既有 `VERIFIED_TEACHERS` 列表历史上受 test 表影响，需明确承认已有选择偏差，并重建 val-only 候选池。

---
## 9. 实验矩阵、停止条件、硬目标

### 9.1 最小 Agent-only 离线消融

| ID | 和参照臂的**唯一变量** | 输入及模型 | 数据集/验证 | 学生训练预算 | seed | 成功阈值（预注册） | 失败解释 |
|---|---|---|---|---|---|---|---|
| **AG-00** | 历史对照 | 原 23维 signature/probe + MLP + 0 门 | SYSU/WHU/CDD/LEVIR 四折 LODO | 不新增，复用 44 个 40K val 记录 | Agent 2333 | 仅作为对照 | 当前命中 2/4，非新成果 |
| **AG-01** | 只改拟合器 | 相同特征，ridge + 0 门 | 同上 | 0 | 2333 | harm 不比 AG-00 多，regret 不更大 | MLP 复杂度可能不是主因 |
| **AG-02** | 只改 None 门 | AG-01 + `δ=+0.20pp` | 同上 | 0 | 2333 | harm 下降且不能全部 None | 弱信号难以识别正教师 |
| **AG-03** | 只改可接受集合 | AG-02 + group-jackknife `μ−u>δ` | 同上 | 0 | 2333 | FalseAction≤AG-02；Coverage≥1/4 且 regret 改善 | 风险门未校准或任务不可识别 |
| **AG-04** | 只扩充 probe | AG-03 + P1–P5，旧维度按固定 schema 缩减 | 同上 | 0（Cache-only） | 2333 | 机制性解释增强、harm/regret 不恶化 | 新 probe 与真实收益无关 |
| **AG-05** | 只加 Teacher metadata | AG-04 + 预定义 metadata | 同上 | 0 | 2333 | 在某留出域动作/预测改善且无新负迁移 | 元数据不能跨域外推 |
| **AG-CF** | 固定教师基线改为公平拒绝 | 按训练域平均 val 最好教师，若 ≤δ 则 None | 同上 | 0 | 2333 | 主模型必须优于此基线 | 若无优势，Agent 复杂性没有价值 |
| **AG-CH** | 启发式基线改为公平拒绝 | train-only probe 选教师，设统一 δ/None | 同上 | 0 | 2333 | 主模型必须优于此基线 | 若无优势，学习器没必要 |

**严格防止过度模型选择：** 上表是结构分析菜单，不能在同样 4 个 LODO 折上将 AG-01~05、5 种模型、10 组阈值反复择优后声称“独立泛化”。正式对照限定 **AG-00（历史）、AG-02（必要门控对照）、AG-03（主创新）、AG-04（只为证明 probe 作用）和 AG-CF（公平固定教师）**；AG-05 仅在机制性证据明确时进入补充实验，其他作为附录或不跑。更强的独立泛化结论需要额外独立 CD 任务/公开数据集，但本项目固定四数据集，因此论文应将泛化措辞限定在“跨这四个基准的留一诊断”，不得称普适。

### 9.2 最终 M1 正式学生实验（仅条件成立后启动）

当且仅当 AG-03/04 在**完全 val-only 的 LODO** 满足 `regret≤AG-00`、`FalseAction≤AG-00`、不发生 `held-out val <−0.20pp` 的教师选择，且有 **≥1 折确实选择非 None 且 positive reward**，冻结 Agent 版本，再启动：

| ID | 唯一变量 / 与 C1 比较 | 数据集 | 预算 | seed / batch | 部署约束 | 成功条件 | 失败判据 |
|---|---|---|---|---|---|---|---|
| C0（已有） | 原干净学生 | 四个 | 40K/DS | 2333 / 64 | 2.913094M / 2.7676G | 对照 | 不参与 Agent 选择 |
| C1（已有） | C0+DCA | 四个 | 40K/DS | 2333 / 64 | 3.280274M / 3.2432G | 教师效用 anchor | 与 M1 运行条件不一致需重做准确性对照 |
| TV-*（已有） | C1+单 Teacher Package | 9T×4DS | 40K/arm | 2333 / 64 | 同 C1 | 仅 registry 证据 | 不得用 test 反选 Teacher |
| **M2-SYSU** | C1+AG-03/04 选择的单教师或 None | SYSU | **40K** | **2333 / 64** | **<5M**, no teacher deployed | `ΔF1>+0.20pp`、IoU 同向、无 P0 | 未过门或更差，判主张失败 |
| **M2-WHU** | 同上 | WHU | **40K** | **2333 / 64** | 同上 | 同上 | 同上 |
| **M2-CDD** | 同上 | CDD | **40K** | **2333 / 64** | 同上 | 同上 | 同上 |
| **M2-LEVIR** | 同上 | LEVIR | **40K** | **2333 / 64** | 同上 | 同上 | 同上 |

**说明：** 实验目录使用 `Run2`，而不是覆盖旧 `Run1/M1`。如果 Agent 返回 None，则该臂与 C1 完全同机制，不应把偶然差异当新方法收益；可保存执行轨迹作为可重复性审计，但 **不是教师策略涨点证据**。

### 9.3 两层成功定义：切不可把 LODO 选对与最终目标混淆

**Level A：Agent 方法门（必要但不充分）。**

- test 完全不参与任何训练/选择/阈值；
- 4/4 LODO 产出可追溯 action、margin 和拒绝理由；
- `FalseActionRate` 与 `MeanRegret` 不差于 AG-00/AG-CF，且至少 1 折有真实 positive val action（否则论文主张退化为 None 策略）；
- 新 probe/remove-metadata 消融产生机制一致的预期方向；
- `None` 安全动作必须在模拟无正收益域时被触发，例如所有 val reward≤0 的历史 SYSU/CDD。

**Level B：四数据集硬目标（论文最终门）。**

| 数据集 | C1 已报 F1 | 必须达到 F1 | 当前缺口（pp） |
|---|---:|---:|---:|
| SYSU | 82.57 | **85.00** | **+2.43** |
| LEVIR | 91.04 | **92.50** | **+1.46** |
| WHU | 94.04 | **95.00** | **+0.96** |
| CDD | 97.75 | **98.00** | **+0.25** |

同时 IoU 方向一致；有效部署参数 **严格 `< 5,000,000`**。在 44 个现有单教师 test 差仅最高 +0.68pp、绝大多数 <0.2pp 的条件下，**即使 Agent 完美选中了历史最大教师，当前证据也未显示可同时达到四硬目标**。这不能通过重命名 Agent 解决；若后续强行把 DCA 升级、额外 teacher 特征送进推理图，就违反本轮“只改 Agent”的任务边界。

### 9.4 定量失败判据

- **F0：机制不可识别**：AG-03/04 全四折选 None，且 val reward 没有可靠正信号；停止 M2，诚实报告“拒答正确但尚无可用选择策略”。
- **F1：有害教师未被拦截**：任一留出域选了验证效用明显为负（例如 `<−0.20pp`），风险建模假设失败。
- **F2：比极简规则更差**：regret/harm 不优于固定教师+None/仅 ridge+None，则 U-SafeAgent 复杂性不成立。
- **F3：指标假改善来自同配置漂移**：M2 None 与 C1 的差值超出历史 σ，优先审查重复性而不是当作 Agent 效应。
- **F4：越过部署边界**：任何 teacher/Cache/Agent 进入推理主路径、推理参数≥5M、swap/deploy 差≥1e-6，立即否决。
- **F5：泄漏**：任何输入 JSON/阈值/动作筛选源含 test 成绩，实验不可用于论文方法有效性证明。

---

## 10. 逐文件修改与可执行检查

### 10.1 推荐文件变更清单（仅设计，不代用户改仓库）

| 文件 | 要改什么 | 为何需要 / 唯一变量 | 验证 |
|---|---|---|---|
| `models/tools/build_capability_registry.py` | `TEST_RE.finditer` 选最后完整块；输出字段来源/指标单位校验；将 val 与 test registry 严格分离 | **P0 正确性**，不算方法创新 | 2 个 test block+尾部残块单测 |
| `models/agent/teacher_registry.py` | `load(which)` 改 enum 和只读安全 API，阻断 Agent 路径读取 research | P0 泄漏防御 | 人为注入 `test_f1` 后 Agent fail-fast |
| `models/agent/dataset_signature.py` | v1 不覆盖；v2 加 6 项、修零连通域/梯度、支持统一路径解析、严格 `names` | 训练域签名可信 | CDD 16 样本 ID/label 与 Dataset 一致性 |
| `models/distill/capability_probe.py` | 增加 P1–P5 的 train-only 统计；将 `confidence` 改称 `response` 以免误解 | 表征教师能力，不改变 KD | 正负/空图/尺寸/resize 合成单测 |
| `models/tools/probe_teacher_capability.py` | 修 ID 路径、train stratified ids、aggregate/per-image summary、样本索引 SHA | 不增加基础模型推理/Cache 成本 | real-cache dry run + manifest 审计 |
| `models/agent/teacher_metadata.py`（新） | 每个 T 的训练域、目标、stride、金字塔属性，与缓存实际尺寸对齐 | 教师类型元数据 | schema/重复 ID/缺失硬报错 |
| `models/agent/safe_selector.py`（新） | ridge + fold-only scaler + domain jackknife proxy + 安全集合 + None | **首选 Agent 主机制** | 预测分单位、None 极端输入、确定性 |
| `models/tools/train_teacher_agent.py` | 留下 legacy；新 CLI 指向 safe_selector；LODO 评价、诊断 JSON、固定 seed、动作冻结 | 可证伪与防过拟合 | 4 折/36 行/训练域3个 assertions |
| `models/agent/offline_bandit.py` | 统一 `score_pp` 和 `select_action()` 的 raw utility 语义 | 防标准化单位风险 | scaled/raw 检查 |
| `train_scripts/CATA-CD/Run2/`（新） | `run_m2.sh` 只读取冻结 `selected_actions.json`，不能写死 test 最优 T | 唯一变量：更新 Agent 决策 | `bash -n` + DRY_RUN 输出参数 |
| `README.md` + `docs/temporary/...` | 修注释/数据口径/阈值/状态与不足；汇总脚本相应更新 | 避免错误科研叙事 | 链接、结果 SHA 与文档一致 |

**明确不动**：`models/a2net.py`、`models/decoder/deployable_change_adapter.py`、`models/distill/kd.py` 的主预测结构；`models/scripts/train.py` 除 P0 数据流/日志准确性、安全 resume 修复外，不调整训练算法、训练 loss、student 参数量和部署逻辑。任何改训练器使 C1/TV 旧 run 不可比的变更，必须重建 clean anchor 和全组必要唯一变量对照。

### 10.2 必须先跑的无训练审计（现有 RSML-3 环境）

```bash
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate lsrep
cd /home/yqwang/projects/LS-Rep_BCD_RSML_3

git status --short
git rev-parse HEAD
# 建议与审查 SHA：1f2ee89b6a195d177b604215ba4a0af7d7d30a5f 对齐
python -m models.tools.train_teacher_agent \
  --signatures_dir /share_datasets/CD_teacher_cache/CATA_CD_v2/signatures \
  --probes_dir /share_datasets/CD_teacher_cache/CATA_CD_v2/probes \
  --registry_dir /share_datasets/CD_teacher_cache/CATA_CD_v2/registry \
  --output_dir /share_datasets/CD_teacher_cache/CATA_CD_v2/agent_legacy_audit
```

上述为**仓库已存在的 CLI**，用于冻结 legacy 结果。首选 `safe_selector` 的新 CLI 是建议实现，**现在不能冒充已经存在的可执行命令**；完成代码后应约定：

```text
python -m models.tools.train_safe_teacher_agent \
  --registry_dir .../registry \
  --signatures_dir .../signatures_v2 \
  --probes_dir .../probes_v2 \
  --teacher_meta_json .../teacher_metadata.json \
  --out_dir .../agent_run2 \
  --seed 2333 --delta_pp 0.20 --kappa 1.0
```

`train_safe_teacher_agent` 是**待创建模块**。不得把此示例当作当前仓库已经通过的命令。

### 10.3 Smoke 与 real-cache dry run

**Smoke（合成数据/无原始数据）**：

- `None` 准确：所有 `L≤δ` 则 None；至少 1 教师通过时返回唯一 ID；对于 NaN/missing probe 失败而非悄悄填 0。
- A/B swap：学生 C0、C1 和新 Agent 选定后的训练预测都必须按原协议 `<1e-6`（Agent 本身按数据集签名选教师，不能受 A/B 顺序改变）。
- 在有无 Agent/Cache 的推理接口上，**固定同一 student 权重，主预测绝对一致**，验证教师辅助没有误入部署路径。
- `switch_to_deploy()` 前后 `<1e-6`，参数数目精确 3,280,274，FLOPs 按原 THOP 流程约 3.2432G。
- LODO 稳定性：固定 seed，重复运行策略学习 2 次，预测 action JSON SHA 一致；若不一致不接受。

**Real-cache dry run（先各教师/数据集少量实际样本，不做完整 40K）：**

1. 随机采 16 个 train IDs（固定 seed，并覆盖无变化/小变化/大变化），核对 A/B/label ID 与 Cache manifest。  
2. 对 SAM2/DINOv3/RADIO 等至少三类 teacher 验证 `local_change/confidence` 的 shape、stride、dtype、有限值与幅度分布。  
3. 对 train label/Cached evidence 计算 P1–P5，验证几何变换与交换一致；除固定 `global_desc` 语义外不允许把教师数据送进推理模块。  
4. 对现有 44 run 文件生成 evidence audit：`run_id`, `split`, `steps`, `seed`, `train/val/test counts`, `last complete TEST block`, `weights/cache hash`；缺任一项标记待核验。

### 10.4 正式实验启动、路径与恢复纪律

约定保持现有路径结构：

```text
训练日志：/home/yqwang/outputs/LS-Rep_BCD_RSML_3/CATA-CD/Run2/<DS>/train_log.txt
Checkpoint：/share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/CATA-CD/Run2/<DS>/
Agent输入：/share_datasets/CD_teacher_cache/CATA_CD_v2/{registry,signatures_v2,probes_v2}/
Agent输出：/share_datasets/CD_teacher_cache/CATA_CD_v2/agent_run2/{lodo_report.json,selected_actions.json,feature_schema.json}
```

- **第一次实验必须创建全新 Run2 目录**，确保无已有 log/checkpoint；未经授权不清空、不覆盖任何 Run1 或 Cache。
- `selected_actions.json` 存完整折外决策和冻结的特征/Registry hash、参数规则与 `git_sha`。只允许写一次或版本化为只读。
- `M2` 的每一 run 必须从相同 ImageNet backbone 权重**全新初始化 student+DCA**，40K/2333/64；**禁止从 C1/TV-* checkpoint 微调**。`--resume` 只允许用于**同一实验同一优化轨迹遭中断后的精确恢复**，不是新的实验组。
- 若有恢复：checkpoint `format_version=8`，检查 run ID、cache hash、data split hash、optimizer、scheduler、global_step、RNG/worker 轨迹；缺信息则重新从头训练，而非无声续训。
- `bash -n` 检查修改后的 shell；改模型或数据路径必须 smoke/real-cache dry run；汇总脚本重生成后的**新结果**不能覆盖先前证据。
- Git 提交前 `git diff --cached --name-only` 审核：数据、权重、cache、checkpoint、训练日志、私有路径信息均不得提交。

---

## 11. 2024–2026 可核验文献

筛选原则：**CCF-A 会议**（CVPR、ICCV、ECCV、AAAI、NeurIPS、ICML）与 **SCI 权威期刊**（TGRS、JSTARS、Nature 等）分级列出；**JSTARS 是 SCI 期刊，绝非 CCF-A 会议**。论文相关性 ≠ 方法在本项目有效；本节只作为创新性定位、机制启发与证据边界。所有题名/年份/venue 均有会议、出版社页面或正式代码仓库可核验；若没有核验到官方代码，不臆造地址。

### 11.1 CCF-A 会议：多教师/动态选择/异质表征

| 编号 | 论文：作者；发表状态 | 官方论文 | 官方代码 | 对本项目的实质启示与区别 |
|---|---|---|---|---|
| **[1]** | **How to Trade Off the Quantity and Capacity of Teacher Ensemble: Learning Categorical Distribution to Stochastically Employ a Teacher for Distillation**；Zixiang Ding, Guoqing Jiang, Shuai Zhang, Lin Guo, Wei Lin；**AAAI 2024 正式发表** | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/29746) | 未核验到可归属作者的官方开源仓库 | **DynaKD** 学 categorical teacher 采样；不是数据集级、val-only 风险安全集；本项目不能再单纯宣传“会选教师”新颖 |
| **[2]** | **AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One**；Mike Ranzinger, Greg Heinrich, Jan Kautz, Pavlo Molchanov；**CVPR 2024 正式发表** | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Ranzinger_AM-RADIO_Agglomerative_Vision_Foundation_Model_Reduce_All_Domains_Into_One_CVPR_2024_paper.html) | [NVlabs/RADIO](https://github.com/NVlabs/RADIO) | 证明 SAM/CLIP/DINO 等教师的互补性可能共存，但**不能据此推断其 Cache 一定帮助 CD**；RADIO 是单个已训练 foundation teacher，而非本项目 Agent |
| **[3]** | **UNIC: Universal Classification Models via Multi-teacher Distillation**；Mert Bülent Sarıyıldız, Philippe Weinzaepfel, Thomas Lucas, Diane Larlus, Yannis Kalantidis；**ECCV 2024 正式发表** | [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/581_ECCV_2024_paper.php) | [naver/unic](https://github.com/naver/unic) | teacher dropping、可删投影器启发训练期专属 translator；本文针对多个教师合并为一模型，不包含本项目的 None/负迁移拒绝门 |
| **[4]** | **Reinforced Multi-teacher Knowledge Distillation for Efficient General Image Forgery Detection and Localization**；Zeqin Yu, Jiangqun Ni, Jian Zhang, Haoyi Deng, Yuzhen Lin；**AAAI 2025 正式发表** | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/32085) | [ZeqinYu/Re-MTKD](https://github.com/ZeqinYu/Re-MTKD) | Reinforced Dynamic Teacher Selection 用 RL 动态调权；与本项目“训练前 hard single teacher/None、4 个任务、离线全信息”不同；审稿时需显式对比 |
| **[5]** | **Multi-Teacher Knowledge Distillation with Reinforcement Learning for Visual Recognition**；Chuanguang Yang, Xinqiang Yu, Han Yang, Zhulin An, Chengqing Yu, Libo Huang, Yongjun Xu；**AAAI 2025 正式发表** | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/32990) | 未核验到论文作者的官方代码 | MTKD-RL 的 state 含 teacher performance 和 teacher-student gap，反馈优化多教师权重；本项目无足量 RL 交互，使用此文做**风险提醒**，不移植复杂策略 |
| **[6]** | **MoVE-KD: Knowledge Distillation for VLMs with Mixture of Visual Encoders**；Jiajun Cao 等（共 8 人）；**CVPR 2025 正式发表** | [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Cao_MoVE-KD_Knowledge_Distillation_for_VLMs_with_Mixture_of_Visual_Encoders_CVPR_2025_paper.html) | [hey-cjj/MoVE-KD](https://github.com/hey-cjj/MoVE-KD) | 训练期异质 encoder/LoRA/MoE 条件化，目标为 VLM；本轮禁止多教师推理交互/改变 DCA，不能照搬 MoE |
| **[7]** | **DUNE: Distilling a Universal Encoder from Heterogeneous 2D and 3D Teachers**；Mert Bülent Sarıyıldız, Philippe Weinzaepfel, Thomas Lucas, Pau de Jorge, Diane Larlus, Yannis Kalantidis；**CVPR 2025 正式发表** | [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Sariyildiz_DUNE_Distilling_a_Universal_Encoder_from_Heterogeneous_2D_and_3D_CVPR_2025_paper.html) | [naver/dune](https://github.com/naver/dune) | 显示异质教师的目标与表征差异很大，支持 teacher metadata 动机；并不说明“4 环境小样本任务选择”可以学习 |
| **[8]** | **Evidential Knowledge Distillation**；Liangyu Xiang, Junyu Gao, Changsheng Xu；**ICCV 2025 正式发表** | [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Xiang_Evidential_Knowledge_Distillation_ICCV_2025_paper.html) | [lyxiang-casia/EKD](https://github.com/lyxiang-casia/EKD) | 教师预测的 evidential uncertainty；**不同于**当前跨 run“teacher 是否有益”的方法不确定性，不能直接把证据学习输出当收益可信区间 |

### 11.2 CCF-A 会议：拒答/不确定性与训练期辅助

| 编号 | 论文：作者；发表状态 | 官方论文 | 官方代码 | 可用边界 |
|---|---|---|---|---|
| **[9]** | **Selective Generation for Controllable Language Models**；Minjae Lee, Kyungmin Kim, Taesoo Kim, Sangdon Park；**NeurIPS 2024 正式发表** | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5a6815122f533193a022cbc41786c1cc-Abstract-Conference.html) | [ml-postech/selective-generation](https://github.com/ml-postech/selective-generation) | 有可拒答、可控制错误发现率的机制启发；该文是文本生成，不可直接继承它的理论保证到 3 域 Agent |
| **[10]** | **Conformal Validity Guarantees Exist for Any Data Distribution (and How to Find Them)**；Drew Prinster, Samuel Don Stanton, Anqi Liu, Suchi Saria；**ICML 2024 正式发表** | [PMLR](https://proceedings.mlr.press/v235/prinster24a.html) | 官方论文页未提供经核验的实现仓库 | 说明 conformal 的理论假设/分布依赖必须明确；**本项目 3 个训练域不足以声称有分布无关路由保证** |
| **[11]** | **Online conformal prediction with decaying step sizes**；Anastasios Nikolas Angelopoulos, Rina Barber, Stephen Bates；**ICML 2024 正式发表** | [PMLR](https://proceedings.mlr.press/v235/angelopoulos24a.html) | 官方论文页未提供经核验的实现仓库 | online conformal 针对时间序列反馈，与当前静态离线教师表不同；不应把 LODO 偷称 online conformal |
| **[12]** | **U-Know-DiffPAN: An Uncertainty-aware Knowledge Distillation Diffusion Framework with Details Enhancement for PAN-Sharpening**；Sungpyo Kim, Jeonghyeok Do, Jaehyup Lee, Munchurl Kim；**CVPR 2025 正式发表** | [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Kim_U-Know-DiffPAN_An_Uncertainty-aware_Knowledge_Distillation_Diffusion_Framework_with_Details_Enhancement_CVPR_2025_paper.html) | [KAIST-VICLab/U-Know-DiffPAN](https://github.com/KAIST-VICLab/U-Know-DiffPAN) | 遥感 PAN-Sharpening 的 teacher uncertainty-map KD；**不是全监督二值 CD**，可借鉴“教师证据置信”概念，不能直接宣称任务可比 |

### 11.3 SCI 权威期刊（明确不是 CCF-A 会议）

| 编号 | 论文：作者；发表状态 | 出版社/代码 | 本项目价值与限制 |
|---|---|---|---|
| **[13]** | **Burden-Free Distillation From Foundation Model for Efficient Remote Sensing Change Detection**；Shuang Wang, Chonghua Lv, Dou Quan, Ning Huyan, Xianwei Cao, Jingxi Sun, Licheng Jiao；**IEEE TGRS 2025 正式发表** | [IEEE DOI](https://doi.org/10.1109/TGRS.2025.3584094)；[论文给出的 GitHub](https://github.com/Younger-hua/Burden-Free-Distillation)（本轮核验时仓库可能为空，不能声称可直接复现） | 教师训练期蒸馏、学生免重型推理，与 CATA-CD **部署范式重叠**；本项目的新意须落在经验证的教师能力决策，而非仅“教师可拆” |
| **[14]** | **PeftCD: Leveraging Vision Foundation Models with Parameter-Efficient Fine-Tuning for Remote Sensing Change Detection**；Sijun Dong, Yuxuan Hu, Libo Wang, Geng Chen, Xiaoliang Meng；**IEEE JSTARS 2026 正式发表，SCI 期刊** | [IEEE](https://ieeexplore.ieee.org/document/11458815/)；[dyzy41/PeftCD](https://github.com/dyzy41/PeftCD) | SAM2 与 DINOv3 在不同数据集各有优势；但其基础模型常在推理图，不能直接等同于本项目 Cache-only KD；引用具体数据须核对 split/评价协议 |
| **[15]** | **Accurate predictions on small data with a tabular foundation model**；Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, Frank Hutter；**Nature 2025 正式发表** | [Nature](https://www.nature.com/articles/s41586-024-08328-6)；[PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN) | 小样本表格预测的参照，但不能让 4 个高度相关任务变成大样本；不建议引入 TabPFN 并声称完成跨域元学习 |

### 11.4 与论文创新性直接有关的“不能再宣称”

- **不能宣称“首个教师选择/智能路由”**：DynaKD、MTKD-RL、Re-MTKD 已有类似决策范式。
- **不能宣称“首个多基础模型蒸馏为轻量单模型”**：AM-RADIO、UNIC、MoVE-KD、DUNE 已展示相关设定，遥感 CD 有 BFD。
- **可以作为待验证定位**：**“在 4 个异构全监督二值变化检测数据集上，通过预先验证 Teacher Package、估计 train-only 能力签名及风险边界，仅在预期收益足够时对轻量学生启用单一教师 Cache，并允许拒绝”**。此 claim 是否能成立要看 AG-03 与低成本公平对照的 LODO/M2 证据。
- **理论边界**：安全集 `μ−κu>δ` 是**风险感知工程判据**，不能自动继承 Selective Generation/Conformal paper 的有限样本风险保证。

---

## 12. 立即执行顺序与仍缺证据

### 12.1 立即执行顺序（按先后依赖，不做模块堆叠）

1. **锁定代码快照** `1f2ee89`，保存本轮审查的 `git_sha`；禁止在进行中主对照训练时混入最新代码改动。
2. **P0 正确性审计**：修最后完整 test block 解析；检查 `agent_train_registry` 无 test 字段；统一 CDD `A/B/label` 解析；验证 18+5=23 维和签名真实像素占比；记录数据 split SHA。
3. **解释当前 C1 重复**：检视所有同 config/seed run 的 init hash、train sample trace、并发/设备、首 epoch loss；抽取 **val** 的 run-level s 与 test 的 s 并排标注，不用 test 调策略。
4. **先做 AG-01/AG-02/AG-CF 的 4 折**，只复用 train/val Registry，确定“更简单的 ridge+None”是否已经足够；若足够，停止开发复杂 Agent。
5. **再做 AG-03**：域级预测敏感性、safe-set、可追溯 None。输出每折风险余量/动作/heldout val regret。若 4 折全 None，冻结“证据不足”结论，停止昂贵 M2。
6. **仅为解决明确失败模式增加 P1–P5 probe**：先验证与 train-only discrimination 的可解释关系，再做 AG-04；metadata AG-05 仅作为必要单变量对照。
7. **冻结 Agent 策略版本**；**绝不看 test 排名挑策略**。决定是否允许启动 Run2/M2。
8. **若通过 Gate，Run2 从头 40K**（四数据集同协议）；主指标只读最后完整 TEST block，报告 Recall/Precision/OA/F1/IoU/Kappa、部署/训练参数、256×256 FLOPs、swap/deploy 误差；计算 M2−C1 和相对硬目标缺口。
9. **失败时研究转向**：若所有有效性分数接近零、不足以区分可用 teacher，明确收敛为“Teacher Selection Signal Insufficiency”负结果；下一轮另立方法迭代，不在本轮悄悄改 DCA/损失/teacher fusion 硬凑结果。

### 12.2 仍需补充证据（按紧急程度）

| 级别 | 缺失材料/观测 | 不补会导致的结论限制 |
|---|---|---|
| **P0** | 44 个 `train_log.txt` 原始末尾完整 TEST block、各 run 总 steps/seed/环境与训练输出路径 | 只能称 Registry 记录，不能直接作为正式论文表 |
| **P0** | C1/R1/R2/R3 的完整 val 曲线与 `best_val`，同配置 start-epoch 初始模型/数据增强 trace | 无法估 val-level 可靠性/定位非确定性来源 |
| **P0** | TRAIN 数据集 label 实际像素占比/列表 hash 与本次 `signature` 数字差异说明 | 基于数据集签名的整个因果叙事可能受数据口径混杂 |
| **P0** | 固定 teacher cache/checksum、采样 ID 对应关系、cache projection 与不同 teacher 原生分辨率 | probe 是否具有真实教师能力解释可能不成立 |
| P1 | 可离线重评估的 C1/TV checkpoint + 官方 val 地块/场景划分 | 不能把 best-val 双重使用的偏差独立估计出来 |
| P1 | teacher-specific val 同配置重复（如果将来授权） | 不能计算每个教师收益严格的置信区间 |
| P1 | 反事实决策的 LODO 锁定策略 JSON、阈值 source 与模型 schema | 不能确认 Agent 是测试前制定的规则 |
| P2 | 在真实 RSML-3 两卡环境中的 smoke、real-cache dry run、显存/CPU/Cache 统计 | 部署/系统可行性只有静态推断 |

### 12.3 论文中的最终措辞建议

**推荐短动机：**

> Existing large-scale foundation teachers are not uniformly beneficial for lightweight remote sensing change detection. Teacher utility is dataset-dependent and often small relative to observed run-to-run variation. Rather than combining all teachers or selecting the largest predicted gain unconditionally, we investigate whether train-only dataset and teacher signatures can support a conservative, rejectable teacher decision for a fixed lightweight student.

**推荐可证伪创新点（必须完成实验才能使用确证语气）：**

1. **Capability Registry**：用固定全监督二值 CD 协议建立跨 SYSU/LEVIR/WHU/CDD 的 teacher utility 证据，防止“强教师必有效”的先验代替实验。
2. **Uncertainty-aware Rejectable Agent**：将效用估计和安全拒绝拆开，而不做 RL 权重融合；选择**单教师或 None**。
3. **Deployment-preserving Training Assistance**：教师只训练期进入，学生+DCA 部署图固定在约 3.28M，保留时相交换对称、部署等价。

**不要写成已证实的主张：** “跨数据集泛化强”“无需额外训练就找到最优教师”“显著优于 C1”“保证不负迁移”“SYSU 必定适合 RADIO / WHU 必定适合 MaRS”。当前证据都不足以支持这些确定性断言。

---

## 附录 A：审查核验入口（固定提交）

- [仓库完整目录与 README](https://github.com/YuqiWang-code/LS-Rep_BCD/tree/1f2ee89b6a195d177b604215ba4a0af7d7d30a5f)
- [Agent：`offline_bandit.py`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/1f2ee89b6a195d177b604215ba4a0af7d7d30a5f/models/agent/offline_bandit.py)
- [Agent 训练：`train_teacher_agent.py`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/1f2ee89b6a195d177b604215ba4a0af7d7d30a5f/models/tools/train_teacher_agent.py)
- [数据签名：`dataset_signature.py`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/1f2ee89b6a195d177b604215ba4a0af7d7d30a5f/models/agent/dataset_signature.py)
- [Teacher Probe：`capability_probe.py`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/1f2ee89b6a195d177b604215ba4a0af7d7d30a5f/models/distill/capability_probe.py)
- [数据集加载：`cd_dataset.py`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/1f2ee89b6a195d177b604215ba4a0af7d7d30a5f/models/datasets/cd_dataset.py)
- [几何同步与时相交换：`transforms.py`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/1f2ee89b6a195d177b604215ba4a0af7d7d30a5f/models/datasets/transforms.py)
- [Cache schema 与同步重放](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/1f2ee89b6a195d177b604215ba4a0af7d7d30a5f/models/distill/cache_v2.py)
- [教师能力注册表构建脚本](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/1f2ee89b6a195d177b604215ba4a0af7d7d30a5f/models/tools/build_capability_registry.py)
- [CATA-CD 训练器](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/1f2ee89b6a195d177b604215ba4a0af7d7d30a5f/models/scripts/train.py)
- [可部署 DCA](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/1f2ee89b6a195d177b604215ba4a0af7d7d30a5f/models/decoder/deployable_change_adapter.py)
- [历史 Agent LODO report](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/1f2ee89b6a195d177b604215ba4a0af7d7d30a5f/docs/temporary/CATA-CD_registry/lodo_report.json)
- [独立的 agent-train registry](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/1f2ee89b6a195d177b604215ba4a0af7d7d30a5f/docs/temporary/CATA-CD_registry/agent_train_registry.json)
- [用于研究汇总而非 Agent 输入的 research report](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/1f2ee89b6a195d177b604215ba4a0af7d7d30a5f/docs/temporary/CATA-CD_registry/research_report.json)
- [CATA-CD v2 README/实验证据](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/1f2ee89b6a195d177b604215ba4a0af7d7d30a5f/README.md)

## 附录 B：术语与数据纪律

- **pp**：百分点。例如 F1 从 0.8257 到 0.8325，差为 +0.68pp；不是 +0.68% 相对增长。
- **train-only**：Agent 的输入可使用**有标签 TRAIN**，因为是全监督任务，但不能包括任何 test 标签、指标、统计或 teacher/test 顺位。
- **LODO**：Leave-One-Dataset-Out，保持 dataset 为独立验证单位，不可把 36 teacher pair 随机行划分后叫 LODO。
- **None**：不蒸馏、不读取 Teacher Cache 的 C1 训练模式；不是一个通过 dummy Cache 注入的“空教师”。
- **单 seed 正式结果**：项目预注册 seed=2333，每臂完整 40K、同 ImageNet 初始化；不拿复现实验均值替代主结果，不做多 seed 显著性报告。
- **可证伪**：当 Agent 不能在 val-only 选择到正教师、或无法超过 `C1/固定教师+None`，承认本主张没有经实验成立。

**文档完。**
