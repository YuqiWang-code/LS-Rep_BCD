# SAGE-CD 结果汇报（Run1 + Run2）

> 生成时间：2026-09-24
> 数据来源：`outputs/SAGE-CD/Run1/*/*/train_log.txt`、`outputs/SAGE-CD/Run2/*/*/train_log.txt` 各自**最后一个完整** `=== TEST RESULTS ===` 块
> 汇总表：`docs/experiment_metrics.xlsx`（本次新增 SAGE-CD-Run2 共 10 行，总计 85 行）
> 固定协议：seed 2333 / batch 64 / 40000 steps；学生恒为 A2Net-LWGANet-L0（部署 2,913,094 参数 / 2.7676G FLOPs）

---

## 0. 一句话结论

**Run2 五臂全部跑完（10 个 run 全部到 40000 steps、无中断、部署硬约束全过），但主门判据未过：`R4 − R2 = SYSU −0.44 / WHU +0.18`（门要求两个数据集同时 ≥ +0.30）。**

现象与 Run1 及历史四个方向完全同构：**任何机制在 WHU 上拿到小幅正增益，代价都是 SYSU 明显下降**。更关键的是，本轮所有 ΔF1 的绝对值都落在「同协议 clean anchor 自身的复现带」之内（SYSU：极差 2.43 / sd 0.81；WHU：极差 1.41 / sd 0.55）——所以本轮**既不能主张机制有效，也不能用 −0.44 断言机制有害：这个量级测不出信号**。

---

## 1. 本轮做了什么

Run2 主方法：**Failure-Type Routed Native-Scale Dual-Expert Distillation + Symmetric Context Fusion**（SAGE-CD/Run2），5 臂消融，单一变量逐级叠加：

| ID | 配置 | 唯一变量 |
|---|---|---|
| R0 | Clean A2Net（无 SCF / gate / teacher） | clean anchor |
| R1 | R0 + joint temporal BN | joint BN |
| R2 | R1 + SCF + identity-centered decoder gate | Student 结构改革 |
| R3 | R2 + DINO 语义（1/16）+ SAM 结构（1/4），uniform 无路由 | 双教师知识 |
| R4 | R3 + failure-type 可靠性路由 | Router（主方法） |

完成状态（逐 run 核查）：

| 检查项 | 结果 |
|---|---|
| 跑到 global_step 40000 | 10 / 10 run 完成 |
| 中途 resume（`RUN RESUME` 块） | 0 个，全部一次跑完 |
| 部署参数 / FLOPs | 2.9131M / 2.7676G（全部一致，通过契约 2.7475G ± 0.03G） |
| Temporal swap max error | 0.00000000e+00（全部） |
| Deploy max error | 0.00000000e+00（全部） |
| 指标来源 | 全部为最后一个完整 `=== TEST RESULTS ===` 块 |

---

## 2. 正式测试指标（%）

### 2.1 Run2（本轮）

| Arm | Dataset | Recall | Precision | OA | F1 | IoU | Kappa |
|---|---|---:|---:|---:|---:|---:|---:|
| R0 | SYSU | 80.18 | 86.27 | 92.32 | **83.11** | 71.11 | 78.15 |
| R0 | WHU | 90.22 | 95.86 | 99.46 | **92.96** | 86.84 | 92.67 |
| R1 | SYSU | 79.22 | 86.11 | 92.08 | 82.52 | 70.24 | 77.41 |
| R1 | WHU | 91.36 | 95.52 | 99.49 | 93.39 | 87.60 | 93.12 |
| R2 | SYSU | 79.08 | 85.91 | 92.01 | 82.35 | 70.00 | 77.20 |
| R2 | WHU | 92.03 | 95.38 | 99.51 | 93.67 | 88.10 | 93.42 |
| R3 | SYSU | 77.80 | 86.07 | 91.79 | 81.72 | 69.10 | 76.45 |
| R3 | WHU | 91.92 | 95.84 | 99.52 | 93.84 | 88.39 | 93.59 |
| R4 | SYSU | 77.66 | 86.67 | 91.91 | **81.91** | 69.37 | 76.73 |
| R4 | WHU | 91.97 | 95.82 | 99.52 | **93.86** | 88.42 | 93.61 |

### 2.2 Run1（上一轮，作为背景；注意 trainer 版本不同，不可与 Run2 直接比较）

| Arm | Dataset | Recall | Precision | OA | F1 | IoU | Kappa |
|---|---|---:|---:|---:|---:|---:|---:|
| G0 | SYSU | 79.49 | 85.69 | 92.03 | 82.48 | 70.18 | 77.33 |
| G0 | WHU | 90.92 | 94.72 | 99.44 | 92.78 | 86.54 | 92.49 |
| G1 | SYSU | 81.35 | 84.13 | 91.98 | 82.72 | 70.53 | 77.50 |
| G1 | WHU | 91.29 | 96.38 | 99.52 | 93.77 | 88.26 | 93.52 |
| G2 | SYSU | 80.43 | 84.61 | 91.94 | 82.47 | 70.17 | 77.24 |
| G2 | WHU | 92.12 | 96.42 | 99.55 | 94.22 | 89.07 | 93.99 |

---

## 3. 消融差值 ΔF1

| 对比 | 变量 | ΔSYSU | ΔWHU |
|---|---|---:|---:|
| R1 − R0 | joint temporal BN | −0.59 | +0.44 |
| R2 − R1 | SCF + identity gate | −0.16 | +0.28 |
| R3 − R2 | 双教师（uniform） | −0.63 | +0.16 |
| R4 − R3 | failure-type routing | **+0.19** | **+0.02** |
| **R4 − R2** | **主门：完整主方法 vs 学生结构改革** | **−0.44** | **+0.18** |
| R2 − R0 | 学生结构改革 vs clean anchor | −0.76 | +0.72 |
| R4 − R0 | 完整主方法 vs clean anchor | −1.20 | +0.90 |

门判据判定（冻结判据，主 README §4.6）：

| 判据 | ΔSYSU | ΔWHU | 判定 |
|---|---:|---:|---|
| **主成功门：R4 − R2 ≥ +0.30（两个数据集）** | −0.44 | +0.18 | **FAIL** |
| 失败判据 R2 < R1（SCF 失败） | −0.16 | +0.28 | 触发（SYSU） |
| 失败判据 R4 ≤ R2（Teacher 无净增益） | −0.44 | +0.18 | 触发（SYSU） |
| R4 − R3 > 0（路由本身有效） | +0.19 | +0.02 | PASS（弱） |
| Reject > 95% / 单 Teacher usage > 95% | — | — | **不可测，见 §5.2** |
| Recall/Precision 极端互换 | 否 | 否 | 未触发 |
| 部署硬约束 | 通过 | 通过 | 未触发 |

**读法**：Sysu 上单调下降（R0 83.11 → R4 81.91，−1.20），WHU 上单调上升（92.96 → 93.86，+0.90）。SYSU 的损失几乎全部来自 Recall（80.18 → 77.66，−2.5 点），Precision 反而略升（86.27 → 86.67）——即机制让模型在 SYSU 这种高变化比例数据集上**变得保守、漏检增加**。

---

## 4. 机制诊断：KD 与路由确实「按设计工作」

最后 10 个 epoch 的均值（`kd` = KD loss，`lam_d/lam_s` = 实际生效的梯度预算系数，`u_d/u_s` = 路由给出的专家效用占比）：

| Arm | Dataset | kd | lam_d | lam_s | u_d | u_s | u_d+u_s |
|---|---|---:|---:|---:|---:|---:|---:|
| R3 | SYSU | 1.198 | 1.000 | 1.000 | 0.500 | 0.500 | 1.000 |
| R3 | WHU | 1.284 | 0.663 | 0.802 | 0.500 | 0.500 | 1.000 |
| R4 | SYSU | 1.520 | 0.370 | 0.303 | 0.586 | 0.411 | 0.998 |
| R4 | WHU | 1.210 | 0.127 | 0.116 | 0.395 | 0.593 | 0.989 |

- **梯度预算生效**：预算式为 `λ = clip(ρ·π·‖∇L_main‖ / ‖∇L_KD‖, 0, 1)`，`ρ = 0.25`。R4 中 λ 降到 0.37/0.30（SYSU）与 0.13/0.12（WHU），即 KD 梯度被精确压在 `ρ·π` 上限；总 KD 梯度 ≈ 主损失的 25%，与设计一致。R3 的 λ 顶到 1.0 说明 uniform 下 KD 梯度本身远小于预算，不是超预算。
- **路由非退化、且是数据集自适应的**：SYSU 上 DINO 语义专家占 `π_d = 0.586`，WHU 上反过来 SAM 结构专家占 `π_s = 0.593`。没有出现门判据担心的「单 teacher usage > 95%」。
- **KD loss 量级 1.2–1.5**：注意 v2 已换成 native-scale BCE + Dice（不再做 per-image logit z-score），与 Run1 G2 的 0.6（z-score + soft BCE）**不同定义、不可直接比较**。

---

## 5. 汇报前必须知道的三个硬问题（本轮审计发现）

### 5.1 clean anchor 本身不稳，ΔF1 的噪声带 ≥ 本轮所有效应 ★最关键

- **同协议、同 seed、data_fingerprint 完全一致的 clean A2Net**：Run1 G0 = SYSU 82.48 / WHU 92.78，Run2 R0 = SYSU **83.11** / WHU 92.96 → SYSU 差 **+0.64**。
- 两条日志在 **epoch 0 就已经不同**（val F1 0.7628 vs 0.7600；训练 loss 3.435808 vs 3.433440）→ 差异来自 **v2 代码改动**（trainer / decoder / 模块构造变化），不是数据。
- **历史同协议 clean anchor 分布**（batch 64 / 40000 steps / seed 2333，8+7 个 run）：
  - SYSU：80.81 / 81.75 / 82.28 / 82.28 / 82.48 / 83.01 / 83.11 / 83.24 → **极差 2.43，sd 0.81**
  - WHU：92.78 / 92.96 / 93.55 / 93.69 / 94.03 / 94.06 / 94.19 → **极差 1.41，sd 0.55**
- 本轮最大效应是 `R3 − R0 = −1.39`（SYSU），主门效应是 0.44 → **全部落在带内**。
- 佐证：**同一个机制 joint temporal BN**，Run1 得 SYSU +0.24、Run2 得 SYSU −0.59，符号相反（差 0.83）。
- 结论：在 SYSU 上，单 seed 单次 ΔF1 小于 ~0.8 不可解释；本轮所有消融结论都受此限制。

### 5.2 README 写的主方法含「Reject」，代码里没有实现

- `models/distill/sage_router.py::expert_utility()` 返回 `π_d, π_s` 且 **`π_d + π_s ≡ 1`**（实测 u_d+u_s = 0.998 / 0.989）→ 路由只能在两个教师之间**分配**预算，结构上**不可能**「两个都不可靠 → 拒绝」。
- 因此门判据里的 **「Reject > 95%」不可测**；Run2 README 与主 README 对主方法的 "Reject" 描述需要修正措辞，或补实现。
- （逐像素权重 `w_d / w_s` 可以趋近 0，属于隐式的区域级拒绝，但没有任何指标记录它。）

### 5.3 SAGE trainer 的 clean anchor 在 WHU 上系统性偏低约 1.0 F1

- 旧 trainer clean anchor（WHU）：94.19 / 94.03 / 93.69 / 94.06 / 93.55 → **均值 93.90**
- SAGE trainer clean anchor（WHU）：92.78 / 92.96 → **均值 92.87**
- 即 Run1 G2 的 WHU 94.22、Run2 R4 的 WHU 93.86 **都没有超过旧 trainer clean anchor 的水平**（均值 93.90 / 上界 94.19）。「WHU 正增益」很大程度是在**补 anchor 的缺口**，而不是超越既有水平。
- 样本量小（2 vs 5），需要用同代码复现确认；但它直接决定 WHU 的结果该怎么讲。

> 附带一致性问题：README §7 仍写 checkpoint 为 `format v6`，代码里是 `CHECKPOINT_FORMAT_VERSION = 7`；§1「作用域」仍把 `train_scripts/SAGE-CD/Run1` 列为唯一活动族。建议一并更正。

---

## 6. 与历史方向的对照：同一个「镜像」现象已重复 5 次

（机制臂 vs 各自 clean anchor，单位 %F1；数据来自 `docs/experiment_metrics.xlsx`）

| 方向 | 对比 | ΔSYSU | ΔWHU | 模式 |
|---|---|---:|---:|---|
| RDT-CD Run3 | R3 − B0（完整 BT-SAM-RDT） | +1.38 | −0.36 | SYSU+ / WHU− |
| RDT-CD Run3 | R3A − B0（仅 SAM） | +0.54 | −0.12 | SYSU+ / WHU− |
| RDT-CD Run3 | R3O − B0（仅 OV） | +1.02 | −0.73 | SYSU+ / WHU− |
| SCTC Run1 | S2 − S0（SCTC 主方法） | −0.25 | +0.43 | SYSU− / WHU+ |
| FA-SCRD Run1 | A1 − C1（固定教师 relation KD） | −0.33 | 未跑 | SYSU only |
| SAGE Run1 | G2 − G1（Safe-DINO 门） | −0.25 | +0.46 | SYSU− / WHU+ |
| **SAGE Run2** | **R4 − R2（主门）** | **−0.44** | **+0.18** | **SYSU− / WHU+** |

- 5 个方向里 4 个的主方法都是 SYSU/WHU **严格反号**（RDT-CD Run3 是反过来的 SYSU+/WHU−）。指向 **SYSU（变化像素比 21.1%）与 WHU 的本身属性**驱动的系统性 trade-off，而非某个模块写错。
- 唯一的「双向为正」出现在：SAGE Run2 的 routing（+0.19 / +0.02，量级太小）、SAGE Run1 的 joint BN（+0.24 / +0.98，但 Run2 复现失败，见 §5.1）。
- 历史上**没有任何机制臂同时超过两个数据集 clean anchor 的历史上界**（SYSU 83.24 / WHU 94.19）；最接近的是 RDT-CD Run1/D1（83.23 / 94.21，SYSU 差 0.01）——而它对应的 B0 anchor（80.81）恰好是 SYSU 的历史最低值。

---

## 7. 建议在汇报里怎么讲

**可以讲：**
1. Run2 机制实现完整、行为符合设计：梯度预算精确生效、路由非退化且随数据集自适应、部署硬约束（参数/FLOPs/swap/deploy 误差）全部通过。
2. 主门未过，给出确定数字：`R4 − R2 = SYSU −0.44 / WHU +0.18`。
3. 现象是跨 5 个方向重复的「SYSU−/WHU+ 镜像」，是数据集属性驱动的系统性 trade-off，不是实现错误。
4. 一个正面的方法学产出：**我们查清了本项目 ΔF1 的复现带（SYSU sd 0.81 / WHU sd 0.55），并用它证明过去若干「正增益」都在噪声内。**

**不能 / 不宜讲：**
1. 不能说「双专家蒸馏 + 失败类型路由有效」——两个数据集没有同时正增益。
2. 不能用 −0.44 断言机制有害——它落在 anchor 复现带内。
3. 不能把 WHU 的正数讲成普遍提升——它没超过历史 clean anchor 水平（§5.3）。
4. 不能把 Run1 与 Run2 互相比较——trainer 版本不同（§5.1）。
5. 不能说主方法包含 Reject——代码里没有（§5.2）。

---

## 8. 建议请老师拍板的两个问题

1. **是否接受「多数据集同时正增益」这条门在本项目内过不去**，转向别的贡献叙事（例如把「失败类型可诊断性 / 教师-学生失配分析」本身做成贡献，或调整任务/评价设定）。
2. **是否允许破例做一次「同代码 clean anchor 重复」**（不是为了扫参调优，而是为了给出 SYSU/WHU 的复现带）。当前「永远不做多 seed」的约定下，任何 ΔF1 < 0.8 的消融都无法判定有效与否——这会让后续每一轮实验都无法得出结论。

---

## 附：复现与查数命令

```bash
python -B analyse/sage_run2_summary.py        # Run2 正式指标 + ΔF1 + 门判定 + 训练诊断 + clean anchor 复现带
python -B analyse/cross_direction_deltas.py   # 跨方向 ΔF1 与符号模式
python -B analyse/extract_metrics.py          # 重建 docs/experiment_metrics.xlsx（85 行）
```

**本次产出文件**
- `docs/experiment_metrics.xlsx` — 新增 SAGE-CD-Run2 共 10 行（总 85 行），Configuration 列新增 `joint_bn/scf/gate/teacher/routing` 五个开关记录
- `docs/汇报/SAGE-CD_Run2_结果汇报.md` — 本文件
- `analyse/sage_run2_summary.py`、`analyse/cross_direction_deltas.py` — 本次新增的可复现查数脚本
