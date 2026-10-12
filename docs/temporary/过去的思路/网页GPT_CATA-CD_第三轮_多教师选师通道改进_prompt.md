# CATA-CD v2 第三轮：如何在**保持「多教师 + 按适配性选师 Agent」创新点不变**的前提下，让指标真正提升

> **请把完整输出整理成一份结构清晰的中文 Markdown 文档，以文件形式发给我。**
> 全程中文；先给结论，再给证据与可执行方案；证据不足处标注「待验证假设」。
> 引用文献限 2024–2026 年可核验工作，区分 CCF-A 会议与 SCI 权威期刊，给题名/年份/venue/代码地址/发表状态。

---

## 0. 阅读入口（全部为固定 commit，可直接打开）

**仓库**：https://github.com/YuqiWang-code/LS-Rep_BCD
**固定版本（本 prompt 对应的代码状态）**：[`e6e9827`](https://github.com/YuqiWang-code/LS-Rep_BCD/tree/e6e9827c8f35bb93a4c85de028a6775f249addba)

| 内容 | 链接 |
|---|---|
| 项目 README（含任务、硬目标、研究纪律、全部结果与 σ 表） | [`README.md`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/e6e9827c8f35bb93a4c85de028a6775f249addba/README.md) |
| 学生模型 + DCA 挂载点 | [`models/a2net.py`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/e6e9827c8f35bb93a4c85de028a6775f249addba/models/a2net.py) |
| DCA（3 专家 × 4 尺度 + gate） | [`models/decoder/deployable_change_adapter.py`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/e6e9827c8f35bb93a4c85de028a6775f249addba/models/decoder/deployable_change_adapter.py) |
| **压缩教师 Cache v2 schema（当前知识通道）** | [`models/distill/cache_v2.py`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/e6e9827c8f35bb93a4c85de028a6775f249addba/models/distill/cache_v2.py) |
| **教师 evidence 生成方式** | [`models/distill/teacher_package.py`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/e6e9827c8f35bb93a4c85de028a6775f249addba/models/distill/teacher_package.py) |
| **原始 KD 损失（128ch cosine，单尺度）** | [`models/distill/kd.py`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/e6e9827c8f35bb93a4c85de028a6775f249addba/models/distill/kd.py) |
| J1 的任务对齐 KD（gt/gate/taskkd，1ch，65 参数） | [`models/distill/task_reliable_kd.py`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/e6e9827c8f35bb93a4c85de028a6775f249addba/models/distill/task_reliable_kd.py) |
| 能力探针 P1–P5 | [`models/distill/capability_probe.py`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/e6e9827c8f35bb93a4c85de028a6775f249addba/models/distill/capability_probe.py) |
| 选师 Agent（加权 ridge + jackknife + 安全集） | [`models/agent/safe_selector.py`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/e6e9827c8f35bb93a4c85de028a6775f249addba/models/agent/safe_selector.py) |
| 数据集签名 v1/v2（18/20 维） | [`models/agent/dataset_signature.py`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/e6e9827c8f35bb93a4c85de028a6775f249addba/models/agent/dataset_signature.py) |
| 教师元数据 | [`models/agent/teacher_metadata.py`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/e6e9827c8f35bb93a4c85de028a6775f249addba/models/agent/teacher_metadata.py) |
| Agent 的 LODO 训练/评测 | [`models/tools/train_safe_teacher_agent.py`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/e6e9827c8f35bb93a4c85de028a6775f249addba/models/tools/train_safe_teacher_agent.py) |
| 训练入口（`--dca_mode` / `--aux_task`） | [`models/scripts/train.py`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/e6e9827c8f35bb93a4c85de028a6775f249addba/models/scripts/train.py) |
| 9 个教师编码器（自包含精简实现） | [`models/thirdparty/`](https://github.com/YuqiWang-code/LS-Rep_BCD/tree/e6e9827c8f35bb93a4c85de028a6775f249addba/models/thirdparty)（`sam2 / dinov2 / dinov3 / remoteclip / mars / anysat / universat / radio`） |
| test 能力矩阵（36 组） | [`research_report.json`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/e6e9827c8f35bb93a4c85de028a6775f249addba/docs/temporary/CATA-CD_registry/research_report.json) |
| Agent 可见的 val 效用矩阵 | [`agent_train_registry.json`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/e6e9827c8f35bb93a4c85de028a6775f249addba/docs/temporary/CATA-CD_registry/agent_train_registry.json) |
| J1 机制判决结果 | [`CATA-CD_Run2_J1_结果.md`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/e6e9827c8f35bb93a4c85de028a6775f249addba/docs/temporary/CATA-CD_Run2_J1_结果.md) |
| S-PCG 四域结果 | [`CATA-CD_Run2_SPCG_结果.md`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/e6e9827c8f35bb93a4c85de028a6775f249addba/docs/temporary/CATA-CD_Run2_SPCG_结果.md) |
| Run2 的执行纪律与命令 | [`train_scripts/CATA-CD/Run2/README.md`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/e6e9827c8f35bb93a4c85de028a6775f249addba/train_scripts/CATA-CD/Run2/README.md) |
| 教师能力矩阵的复现脚本 | [`train_scripts/CATA-CD/teacher_adaptation/common.sh`](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/e6e9827c8f35bb93a4c85de028a6775f249addba/train_scripts/CATA-CD/teacher_adaptation/common.sh) |

---

## 1. 任务、约束与**研究纪律**（必须遵守，不得用「未达硬目标」否定机制线）

- 任务：推理时**极轻量化**的遥感**全监督二值变化检测**（不做半监督/多类/语义分割）；
  数据集固定四个（256×256）：SYSU-CD / LEVIR-CD / WHU-CD / CDD-CD。
- 四数据集 F1 硬目标：**CDD ≥98.00 / LEVIR ≥92.50 / SYSU ≥85.00 / WHU ≥95.00**（IoU 同方向）；
  **有效推理参数 < 5M**（当前 C1 = **3,280,274**）。
- **研究纪律 1**：硬目标是「努力方向」，不是「必达门」。未达到 ≠ 方案不行、≠ 实验失败。
  只要在受控消融下相对**同期 CTRL** 有**可复现**的提升，就是有效性证据。
  **不得**把「未达硬目标」写成「方法无效」，**也不得**把它当作放弃某条机制线的理由。
- **研究纪律 2**：有效性由「服务器 40K 训练 + 消融」判定（训练优先）。
  每轮必须上服务器跑**完整 40,000 step**（不短跑、不只做本地推断）；
  服务器为 2×RTX 5090 32GB、基本每天无人使用。
  只读诊断从宽：只要链路能跑通、机制方向不与被数据否决过的结论直接冲突，就**直接上服务器训练**。
  **最小消融范式**：`CTRL`（同期同代码、开关关闭）vs `EXP`（唯一变量打开），
  **同 seed 2333 / batch 64 / 40K / 从头训练**，比较**四库 F1**。负面结论同样要记录并归因。

---

## 2. **必须保持不变**的创新点（这是本轮的前提，请勿建议放弃或替换）

> **多个大模型作为教师（9 个 Teacher Package：SAM2 / DINOv2 / DINOv3-LVD / DINOv3-SAT / RemoteCLIP / MaRS / AnySat / UniverSat / RADIO），
> 由一个 Agent 依据「教师—数据集适配性」为每个数据集挑选合适的教师（或选择 None）。**

具体已落地且在用的组件（**都是训练期，不进部署图**）：

1. **数据集签名** `d(D)`：v1 18 维 / v2 20 维（变化率、空对比例、面积分位、小目标比例、边界密度、
   难伪变化比例、光照未变比例等）。
2. **教师 train-only 探针** `p(T,D)`：P1–P5（balanced AUROC / hard-negative separation /
   boundary contrast / size-conditioned AUROC gap / unchanged-leakage ratio）。
3. **教师元数据** `m(T)`：5 个 test-free 字段（训练域、目标、native stride、金字塔、实际 cache 分辨率）。
4. **选师器**：加权 ridge（按数据集簇均衡）+ 域级 jackknife 不确定性代理 + 安全集
   `S(D) = {T : μ − κu > δ}`，`δ=+0.20pp / κ=1`，集合为空即 `None`。
5. **教师 Cache v2**（当前唯一的知识通道，schema 见 `cache_v2.py`）：
   `{local_change [128,Ht,Wt] fp16, confidence [1,Ht,Wt], global_desc [128], meta}`，
   其中 `E_T(p) = |Norm_c F_T^A(p) − Norm_c F_T^B(p)|`，`local_change = R_0 · E_T`（`R_0` 为**固定随机投影**），
   `confidence(p) = max_c E_{T,c}(p)`。
6. **教师特征生成**：`models/thirdparty/*` 里是**自包含的精简编码器**，权重来自各 foundation model，
   **全部冻结**，offline 生成 cache。

**部署约束**：教师 / cache / agent / 辅助头**全部训练期**；
`switch_to_deploy()` 后部署 state_dict 不含它们；时间交换对称与部署前后误差都要求 `<1e-6`。

---

## 3. 已完成的实验与**全部证据**（请以此为前提，不要重复已做过的建议）

### 3.1 教师能力矩阵（44 个 40K run，test F1 相对同协议 C1，单位 pp）

| 数据集 | C1 F1 | SAM2 | DINOv2 | D3-LVD | D3-SAT | RCLIP | MaRS | AnySat | UniverSat | RADIO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SYSU | 82.57 | −0.28 | −0.12 | −0.24 | −0.67 | −0.81 | −0.03 | −0.66 | −0.27 | **+0.68** |
| WHU | 94.04 | −1.15 | −0.36 | −0.21 | +0.08 | +0.08 | **+0.56** | −0.13 | **+0.40** | −0.32 |
| CDD | 97.75 | −0.01 | −0.02 | −0.05 | −0.04 | −0.01 | −0.02 | −0.01 | +0.00 | −0.01 |
| LEVIR | 91.04 | +0.06 | −0.22 | +0.08 | +0.14 | −0.07 | −0.06 | +0.06 | −0.18 | −0.00 |

**36 组中：`ΔF1 > 0` 10 组、`ΔF1 ≤ 0` 26 组；达「强正」(ΔF1≥0.20 且 ΔIoU>0) 仅 3 组；
最大单项 +0.68pp（RADIO→SYSU）。** DCA 效应（C1−C0）：WHU +1.38 / SYSU +0.44 / LEVIR +0.12 / CDD −0.04。

### 3.2 Agent 可见的 val 效用（best-val F1 相对 C1，pp）——只有这一列能进入选师器

| 数据集 | C1 val | SAM2 | DINOv2 | D3-LVD | D3-SAT | RCLIP | MaRS | AnySat | UniverSat | RADIO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SYSU | 81.55 | −0.31 | −0.39 | −0.09 | −0.30 | −0.57 | −0.12 | **−1.08** | −0.44 | −0.24 |
| WHU | 94.82 | −0.34 | −0.42 | −0.31 | −0.20 | −0.00 | −0.15 | **+0.12** | −0.43 | −0.30 |
| CDD | 97.73 | −0.00 | −0.02 | −0.05 | −0.04 | −0.02 | −0.02 | −0.02 | −0.00 | −0.01 |
| LEVIR | 91.50 | −0.09 | −0.09 | +0.13 | −0.03 | +0.01 | +0.10 | **+0.15** | −0.18 | +0.14 |

**36 组中 6 正 30 负；正向最大仅 `+0.146pp`，负向最低 `−1.077pp`。**
**每个数据集的 oracle-val（最好教师）都没到 `δ=+0.20pp`**：
SYSU −0.0945 / WHU +0.1159 / CDD −0.0010 / LEVIR +0.1460。
→ **即使完美选师，也选不出任何动作**；旧 Agent（U-SafeAgent）在 4 折 × 5 特征阶梯上**全部输出 `None`**（F0「机制不可识别」）。

### 3.3 J1 机制判决（SYSU，四臂 40K，全部完整 TEST block）

| 臂 | 训练期辅助 | F1 | IoU |
|---|---|---:|---:|
| J1-C1 | 无 | 81.8087 | 69.2171 |
| J1-GT | 1ch GT 辅助头（全像素、正负均衡） | 82.2754 | 69.8881 |
| J1-GATE | 同 head，只在 **teacher∩GT 一致区域 G** 监督，target 仍为 GT | **82.4470** | **70.1361** |
| J1-TASKKD | 同区域 G，target 换成冻结的 **教师软响应 `q_T`** | 82.2657 | 69.8740 |

| 唯一变量 | ΔF1 | ΔIoU |
|---|---:|---:|
| J1-GT − J1-C1 | +0.467 | +0.671 |
| J1-GATE − J1-GT | +0.172 | +0.248 |
| **J1-TASKKD − J1-GATE** | **−0.181** | **−0.262** |

**即：换成教师软目标不但没帮助，还略负；`J1-TASKKD ≈ J1-GT`（−0.010pp）。**
四臂部署侧完全一致（3.2803M / 3.2432G / swap·deploy error = 0），差值不能归因于容量。

### 3.4 train-only 校准诊断（SYSU / DINOv3-LVD，512 样本）——**指出通道问题**

| 量 | 值 |
|---|---|
| `native_hw`（teacher cache 分辨率） | **16×16** |
| balanced AUROC(`confidence`, GT) | 0.852 |
| changed recall @`t_plus` | 0.517 |
| **unchanged leakage @`t_plus`** | **9.62%** |
| AUROC by size（small/medium/large） | 0.748 / 0.810 / 0.848 |
| **AUROC boundary vs interior** | **0.533 / 0.861** |
| `t_plus` / `t_minus` | 0.6412 / 0.6462（**阈值几乎无分离**） |
| ECE raw → affine refit | 0.342 → 0.064 |

### 3.5 噪声底与**可检测下限**（下一轮设计的硬约束）

同配置同 seed 重复（`moe128` / 无教师 / 40K / 同预训练）的 test F1 标准差 σ：

| 数据集 | σ | 95% χ² 区间 | 每臂 1 次可分辨下限 (2√2σ) | 分辨 δ=1.0pp 需每臂 n | δ=0.5pp | δ=0.2pp |
|---|---:|---|---:|---:|---:|---:|
| SYSU | **0.478**（n=6） | [0.298, 1.173] | 1.35pp | 2 | 8 | 116 |
| WHU | **0.561**（n=5） | [0.336, 1.612] | 1.59pp | 3 | 11 | 159 |
| LEVIR | **0.193**（n=5） | [0.116, 0.555] | 0.55pp | 1 | 2 | 19 |
| CDD | **0.011**（n=5） | [0.007, 0.031] | 0.20pp（下限主导） | 1 | 1 | 1 |

**关键事实**：同配置的 SYSU 两次运行曾相差 **1.33pp**（81.81 vs 83.14）。

### 3.6 J0：为什么同配置不可复现（已机制定位）

| 配置 | step-0 `grad_norm` 相对差 | 187 步后 loss 差 |
|---|---|---|
| TF32 开（生产协议） | 1.9e−4 | −1.04e−2 |
| + `CUBLAS_WORKSPACE_CONFIG` | 4.6e−4（**无效**） | — |
| + 关闭 TF32 | **1.6e−6（小 113×）** | **+4.64e−2（更大）** |
| + `use_deterministic_algorithms(True)` | **运行失败**（`MaxUnpool2d` 无确定性实现） | — |

- **初始化权重、数据序列、RNG、一次前向都逐位一致；不确定的是反向传播。**
- **把 step-0 扰动压小 113 倍后，一个 epoch 后的分歧反而更大 → 训练是混沌的。**

### 3.7 S-PCG 四域阶梯（结构线，已做完）

| 数据集 | S0-C1（原全局 gate） | S1-STAT（逐尺度 `GAP(\|a−b\|)`） | S2-PCG（+对称共性上下文） | S2−S1 | ΔIoU |
|---|---|---|---|---|---:|
| SYSU | 83.1391 | 82.3641 | **83.3315** | +0.967 | +1.410 |
| WHU | 93.2317 | 93.6414 | 93.6568 | +0.015 | +0.027 |
| LEVIR | 91.1783 | 90.8484 | **91.2383** | +0.390 | +0.657 |
| CDD | **97.7454** | **97.7731** | 97.7532 | −0.020 | −0.038 |

**0/4 域超过噪声阈值；四域均值 +0.338pp。**

### 3.8 当前四库最好单次 vs 硬目标

| 数据集 | 最好单次 F1 | 目标 | 缺口 |
|---|---:|---:|---:|
| SYSU | 83.3315 | 85.0 | −1.67 |
| WHU | 94.04 | 95.0 | −0.96 |
| LEVIR | 91.2383 | 92.5 | −1.26 |
| CDD | 97.7938 | 98.0 | −0.21 |

---

## 4. 我（提问者）自己认为的瓶颈，请你验证或推翻

**核心矛盾**：Agent 选师的**价值取决于"不同教师带来可辨识的不同收益"**。
但当前证据显示：

1. **9 个教师经当前通道的收益上限只有 +0.68pp（test）/ +0.146pp（val）**，
   而 SYSU/WHU 的单次可分辨下限是 1.35/1.59pp ⇒ **选择器面对的是"全都在噪声里的选项"**。
2. J1 进一步显示：**在 teacher∩GT 一致区域上，教师软目标相对硬 GT 没有额外信息**
   （`TASKKD ≈ GT`）。而教师唯一可能有额外价值的地方——**边界/伪变化**——
   它的 AUROC 只有 **0.533（≈随机）**、未变化泄漏 9.6%。
3. 我怀疑**瓶颈不在"选谁"，而在"知识通道"**，具体可疑点：
   - `local_change = R_0 · |Norm(F_A) − Norm(F_B)|`，`R_0` 是**固定随机投影**（不可学）；
   - KD 只在 **`change[2]`（c4/stride16）单尺度**上做 **方向 cosine**，丢失幅值与多尺度结构；
   - 学生**从不以教师特征为输入**，只把它当回归目标 ⇒ 教师无法在"学生自己判断不了的像素"上提供新信息；
   - `confidence = max_c E_c` 把 128 维压成 1 维，边界信息基本被抹平。

**请你判断**：以上哪一条是主因？还是另有更根本的原因（例如 3.28M 学生容量、任务本身饱和、
教师预训练域与 CD 任务不对齐）？请给**可证伪的判别方法**。

---

## 5. 请按顺序输出

### 5.1 归因
在上面 4 条可疑点 + 你补充的候选中，排序出主因，并给出**能区分它们的最小实验**
（要具体到 CTRL/EXP 的唯一变量、数据集、run 数、判据）。要求每个实验的**预期效应量级 >1pp**
（或明确说明只能在 LEVIR/CDD 这种低 σ 域上筛）。

### 5.2 **保持创新点不变**的知识通道改进方案（3–5 个，排序）
每个方案给出：
- 机制、与**当前通道**（固定随机投影 + 单尺度 cosine + max-channel 响应）的**实质区别**；
- **为什么它可能带来 >1pp**（而不是又一个 0.2pp）；
- 是否仍然"教师训练期、部署无额外负担"；
- **能否被 Agent 选择**（即：不同教师在同一通道下是否会产生**可辨识的不同收益**——
  这一点对保持我的创新点至关重要）；
- 改造涉及的文件与预估代码量；
- CTRL/EXP 消融表设计（哪几臂、哪几个域、每臂几次 run）；
- 可证伪假设与失败判据。

请优先考虑（但不限于）：
- **教师特征作为训练期"特权输入"**（privileged information / burden-free distillation 类），
  部署时移除分支；
- **多尺度 KD**（当前只对齐 c4）；
- **可学投影/translator** 替代固定随机 `R_0`；
- **任务对齐目标**（教师 logits 或 GT 校准后的软标签，而非原始特征方向）；
- **结构/关系型蒸馏**（相似度、频谱、gram 类）而非逐像素 cosine。

### 5.3 选师 Agent 的重建（在通道修好之后）
- 若某个通道能让**不同教师产生可辨识差异**（比如 ≥1pp），
  需要多少 run / 多少 (教师, 数据集) 组合，Agent 才有可学的信号？
- 在这种新信号下，`δ=+0.20pp` 应该怎么定？LODO 只有 4 个域，是否需要引入
  **跨数据集迁移的教师效应模型**（例如用签名/probe 预测效应，而非直接回归）？
- 给出 Agent 的 **CTRL 基线集合**（必须包含：固定最佳单教师、随机选、全 None）。

### 5.4 执行计划（按「训练优先、能用就用两卡」排）
给一份**可立即执行的排期表**：实验 ID、唯一变量、数据集、每臂 run 数、
预计 GPU 小时（按单臂 40K ≈ 2.4 GPU·h 估）、判据、以及"若失败则下一步什么"。
要求明确区分：
- **筛选阶段**（用低 σ 域 LEVIR/CDD，少量 run）；
- **确认阶段**（SYSU/WHU 配对重复）；
- **论文主表**（四库 40K）。

### 5.5 论文层面
- 若通道改进成功：论文主线怎么组织（多教师 + 适配性选师 + 通道创新）？
- 若仍未成功：现有的负结果（J1 + S-PCG + σ/混沌分析 + 76 个 40K run 的审计链）
  能否支撑一篇"机制归因 + 负结果"的论文？需要补什么？
- 列出**不能再宣称**的表述清单（含反向的过度断言）。

### 5.6 文献
2024–2026 年可核验工作，聚焦：教师—学生**能力/目标不匹配**、**特权信息蒸馏**、
**遥感变化检测的轻量化与蒸馏**、**多教师选择/路由**、**负结果与可复现性方法学**。
逐条给题名/年份/venue/代码地址/发表状态，并**明确说明与该项目的具体关系**（能借什么、不能借什么）。

---

## 6. 硬约束（请勿违反）

- **多教师 + 按适配性选师**这一创新点**保持不变**，不要建议换成单一教师或放弃 Agent。
- 任务、四数据集、部署约束（<5M、教师全部训练期）不变。
- 每轮必须**服务器完整 40K**（不短跑、不只做本地推断）；同 seed 2333 / batch 64 / 从头训练。
- **不得**用「未达硬目标」否定机制线；**不得**为了凑指标而堆模块。
- 不要重复建议已实现并失败的内容：U-SafeAgent 安全集、GT 一致区域门控（gate）、
  1 通道任务对齐软目标（taskkd）、DCA 的对称共性上下文 gate（S-PCG S2）、
  随机投影 128 维 cosine KD（原始通道）。

## 7. 输出格式

一份完整的中文 Markdown 文档（含目录、公式、表格、实验 ID 表、参考文献），
以文件形式发出。证据不足处标注「待验证假设」。
