# RDT-CD Run3「BT-SAM-RDT」Teacher 增益异常：机制诊断与可证伪改进方案

## 1 现象与矛盾

### 1.1 先给结论

当前证据最支持的解释不是“SYSU 上 SAM 实例匹配更好”，也不是“WHU/LEVIR 上 residual teacher 容量不够”，而是下面三件事叠加：

1. **当前 Foundation reliability 并不等价于“这个 prior 对当前 Student 有用”的可靠性。**尤其是 SAM reliability 与 OV confidence 都可以很高，但 prior 仍远差于已经训练成熟的 Student；当前 weighted-average fusion 还会把明显更好的 OV prior 向很差的 SAM prior 拉偏。  
   **证据：**`models/distill/task_space.py::{build_ov_prior,fuse_foundation_priors}`；末 epoch `sam/ov/fusion_prior_gain`。

2. **GT positive-Brier-gain audit 在 WHU 上进入严重“信号饥饿”。**它在自己的 Brier 判据下不是“误收”，因为代码严格保证 `gain>0` 才接受；真正的问题是 **99.9% 左右像素被拒收，剩余极少的接受像素又受类别极不平衡影响，最终 KD 几乎为零且未必与 F1/IoU 的最优方向一致**。  
   **证据：**`models/distill/task_space.py::task_space_audit`、`models/distill/dynamic_teacher.py::forward`；`pixel_reject_ratio`、`aux_raw`。

3. **BT-SAM 对小建筑存在明确的几何脆弱点：小尺度边界抖动会被直接解释成结构变化。**当前 matched pair 的单边区域直接记为 change；完全 unmatched instance 的 `association` 又被直接设为 1。对小建筑，几像素的 SAM mask 抖动、轻微配准偏差、实例 split/merge 都可能生成高置信伪变化。  
   **证据：**`models/distill/task_space.py::build_bitemporal_structural_prior`。WHU 的 `match_ratio` 甚至高于 SYSU，因此问题不是简单的“WHU 匹配率低”，而更可能是**匹配后 symmetric-difference 的任务语义不可靠**。

因此，**SYSU 的 +1.38 F1 更像是“有足够 Student residual error + OV 提供额外有效信号 + audit 尚能留下足够变化像素”共同形成的累积收益；WHU/LEVIR 则因为小目标结构噪声、极低正类比例和 audit 信号枯竭，Teacher 很难形成净正作用。**

Run3 同协议首轮结果为：

| Dataset | 变化像素比 | B0 F1 | R3A F1 | R3 F1 | R3A−B0 | R3−B0 |
|---|---:|---:|---:|---:|---:|---:|
| SYSU | 21.1% | 81.75 | 82.29 | 83.13 | +0.54 | **+1.38** |
| WHU | 3.4% | 94.06 | 93.94 | 93.70 | -0.12 | **-0.36** |
| CDD | 11.9% | 97.79 | 97.78 | 97.79 | -0.01 | 0.00 |
| LEVIR | 4.1% | 91.13 | 91.03 | 90.98 | -0.10 | **-0.15** |

四数据集 train 变化像素比分别为 CDD 11.9%、LEVIR 4.1%、SYSU 21.1%、WHU 3.4%，其中 LEVIR/WHU 为建筑小目标且类别极不平衡。

### 1.2 为什么收益只明显出现在 SYSU

SYSU 有三个与当前机制匹配的条件。

第一，**Student 有更大的剩余错误空间。**末 epoch SYSU `student_abs_err=0.0048`、`error_mass=0.0108`，而 WHU 仅 `0.0010/0.0023`，difficulty-weighted error mass 约为 WHU 的 4.7 倍。Teacher 本质上是 residual corrector：

\[
q_T=
\sigma\left(
\operatorname{logit}(p_S)
+
R_F\Delta_{\max}\tanh \delta
\right)
\]

如果 Student 已经接近饱和，残差 Teacher 没有足够可修正空间。  
**证据：**`models/distill/dynamic_teacher.py::_dynamic_proposal`；日志 `student_abs_error/student_error_mass`。

第二，**SYSU 的正类占比高，audit 留下的有效 Teacher 信号更容易落到真正的变化区域。**R3/SYSU：

- `accepted_change=0.0219`
- `accepted_bg=0.0054`
- change ratio = 21.1%

按全局类别比例做仅用于诊断的粗略估算，accepted change pixel mass 与 background mass 约为：

\[
0.211\times0.0219\approx0.00462
\]

\[
0.789\times0.0054\approx0.00426
\]

两者接近均衡。这个估算不是日志里的精确 mass，因为 `accepted_change/bg` 是先逐图 masked mean 再平均，但它指出了一个很值得验证的趋势：**SYSU 的有效 KD 不容易被背景数量完全淹没。**

第三，**OV 在 SYSU 的确给 SAM-only 链增加了信息。**R3A→R3：

- F1：82.29 → 83.13，`+0.84`
- support：0.5672 → 1.0000
- fusion Brier：0.3765 → 0.1559
- `aux_raw`：0.0003 → 0.0009

说明 OV 至少使更多像素进入了可学习的 Foundation-supported 区域，并显著改善了 SAM-only prior。  
但需要强调：**这只能证明 OV 在“SAM+residual teacher”条件下有增量，不能证明 OV 单独有效，因为 Run3 缺 OV-only arm。**

### 1.3 为什么 WHU/LEVIR 更容易受害

WHU 的末期证据非常直接。

R3/WHU：

- `sam_brier=0.3788`
- `ov_brier=0.0455`
- `fusion_brier=0.0829`
- `student_brier≈0.0007`
- `sam_gain=-0.3781`
- `ov_gain=-0.0448`
- `fusion_gain=-0.0822`

也就是说，**SAM prior 极差；OV 虽远好于 SAM，但依然远差于成熟 Student；当前 fusion 又把 OV 从 0.0455 拉坏到 0.0829。**

尤其值得注意：

\[
\frac{0.0829}{0.0455}\approx1.82
\]

即 WHU 上当前融合 prior 的 Brier 约为 OV-alone 的 **1.82 倍**。这不是“互补融合”，而是明确的 **source contamination**。

与此同时：

- `ov_rel=0.8212`
- `fused_rel=0.4688`
- `foundation_support_ratio=1.0000`
- `pixel_reject_ratio=0.9991`
- `dyn_gain=0`
- `aux_raw≈0`

这组数据揭示出一个关键矛盾：

> Foundation 被 reliability 认为“到处有支撑”，但 GT audit 又认为几乎所有动态 proposal 都不比 Student 好。

因此当前 `reliability` 更像 Foundation 自身 confidence，而不是 **utility reliability，即“Teacher 比当前 Student 更值得信任”的概率**。

对 LEVIR，当前提供的诊断快照没有对应末 epoch 数值，所以不能声称已经日志确认同一机制；但 LEVIR 与 WHU 同属低变化比例建筑小目标数据集，且 F1 同样为负增益，因此**小目标几何噪声 + audit 稀疏化是最优先验证假设，而不是已经证实的结论**。

CDD 的 97.79 F1 已非常高，R3 完全持平，更符合“没有剩余可蒸馏空间”的 ceiling 情况；不能用 CDD 持平证明 Teacher 正确或错误。

---

## 2 机制假设（按可能性排序）

### H1：reliability / fusion 估计失真，是当前最强解释

**结论等级：代码事实 + 日志直接支持。**

当前 OV reliability：

\[
R_O=\frac{c_{l1}+c_{l2}}{2}
\]

它只代表 OV cache 自己的 confidence，没有检查两个 level 是否一致，也没有校准 confidence 与 GT correctness / Student advantage 的关系。

当前 fusion：

\[
F=
\frac{R_SS+R_OO}
{R_S+R_O}
\]

当两源同时存在时：

\[
R_F=
\operatorname{mean}(R_S,R_O)
\cdot
(1-|S-O|)
\]

**证据：**`models/distill/task_space.py::{build_ov_prior,fuse_foundation_priors}`。

这里存在两个机制问题。

一是**共同犯错无法被发现**：SAM 与 OV 即使一致地错，agreement 仍然很高。

二是**一好一坏时仍然强制做 weighted average**：当前 WHU 就表现为 OV 明显优于 SAM，但 SAM 仍把 fused prior 从 0.0455 拉坏到 0.0829。

日志又显示 `ov_rel=0.8212`，而 `ov_gain=-0.0448`；说明 confidence 很高不等于它比 Student 更好。

**可证伪预测：**若把 reliability 按 decile 分桶，则高 `ov_rel/sam_rel/fused_rel` 桶中的 `P(prior_gain>0)` 不会随 reliability 单调上升，甚至可能仍接近 0。

### H2：GT audit 主要问题是“拒收过度 + 类别质量失衡”，而不是 Brier 意义上的误收

**结论等级：代码事实 + 日志直接支持。**

audit 条件严格为：

\[
gain=(p-y)^2-(q-y)^2
\]

\[
accept=(R_F>0)\land(gain>\epsilon)
\]

因此按代码定义，**它不会接受一个当前像素 Brier 更差的 proposal。**

所谓“audit 误收”如果存在，只能是：

- Brier 改善但不利于最终 F1/IoU；
- batch 当前时刻改善，但长期参数更新产生副作用；
- 极端类别失衡使大量微小 background gain 的累计优化质量超过 change gain。

**证据：**`models/distill/task_space.py::task_space_audit`。

WHU `reject=0.9991`，仅约 0.09% 像素被接受；而 Student KD 的 denominator 是 **整个 importance mass**，不是 accepted mass：

\[
L_{KD}
=
\frac{
\sum KL(p,q_T)\,
R_F\,I_{accept}\,importance
}{
\sum importance
}
\]

因此 reject 越接近 1，KD 会自然趋近于 0。末期 `aux_raw=0.0000` 正符合该机制。  
**证据：**`models/distill/dynamic_teacher.py::forward`。

更重要的是 WHU change/background 比约 3.4/96.6。即使：

- change 接受率 1.23%
- background 接受率只有 0.09%

粗略质量仍为：

\[
0.034\times0.0123=0.000418
\]

\[
0.966\times0.0009=0.000869
\]

背景 accepted mass 可能约为变化区的 2 倍。这必须用精确日志验证，但已经给出了非常明确的机制方向。

**可证伪预测：**统计真实 `accepted_change_mass` 与 `accepted_bg_mass` 后，WHU/LEVIR 的 accepted KD mass 将明显偏 background，而 SYSU 更接近平衡。

### H3：BT-SAM 的结构差分对“小目标边界抖动”存在系统性偏置

**结论等级：代码事实 + WHU 日志支持，LEVIR 待验证。**

当前结构 prior 的关键定义是：

```text
corresponding overlap       -> 0
T1-only / T2-only           -> 1
non-corresponding overlap   -> 1
```

而完全 unmatched instance：

```python
association = 1
```

再乘 SAM quality / boundary confidence。

边界仅使用：

\[
quality\times(1-0.5\times boundary)
\]

即 boundary pixel 最多只降一半 reliability，并不会进入“未知区域”。

**证据：**`models/distill/task_space.py::build_bitemporal_structural_prior`。

对于大对象，2–3 px mask 漂移只影响很小面积；对于小建筑，相同漂移可以占目标面积很大比例。于是：

> unchanged building + SAM mask T1/T2 有几像素错位  
> → overlap core 判 unchanged  
> → 两侧 symmetric difference 全被判 change。

WHU 的日志还排除了一个更简单的解释：

| Dataset | match_ratio | match_iou |
|---|---:|---:|
| SYSU | 0.6648 | 0.1663 |
| WHU | **0.8093** | **0.1885** |

WHU 的 matching statistics 反而更高，但 `sam_brier=0.3788` 仍极差。

所以最值得怀疑的不是“匹不上”，而是：

> **匹到了以后，实例边界的几何差异并不等于真实语义变化。**

**可证伪预测：**将 SAM Brier 按 GT boundary/interior 和 small/large object 分层后，WHU/LEVIR 的错误应显著集中于 small-object boundary / near-boundary symmetric-difference 区域。

### H4：Residual Teacher 过拟合噪声目前不是首要解释

**结论等级：现有证据不支持为主因，但未完全排除。**

如果 residual teacher 是主要坏源，预期应该看到：

- `dynamic_shift` 较大；
- `target_residual_magnitude` 较大；
- teacher 在持续改变 Student；
- `dynamic_teacher_gain` 显著为负；
- 或 Fast Teacher 明显优于 EMA Target，仅训练集有效。

但末 epoch：

- SYSU `dyn_gain=-0.0003`
- WHU `dyn_gain=0`
- WHU `aux_raw≈0`

更像 **Teacher 已基本退化成 Student 本身 / audit 几乎切断蒸馏**，而不是一个高容量 Teacher 在强烈灌输错误。

代码也将 Teacher 限制为：

- hidden=24；
- zero-init residual；
- proposal anchored at Student；
- correction 受 `R_F × max_logit_delta` 限幅。

**证据：**`models/distill/dynamic_teacher.py::{ResidualTeacherExpert,_dynamic_proposal}`。

但**早期训练阶段**仍可能发生错误 Teacher 更新，因此不能只看末 epoch 排除。需要完整 trajectory。

### H5：CDD 主要是 ceiling，不应纳入当前 Teacher 机制优劣归因

B0 已 97.79 F1，R3 97.79。无论 prior 是否有少量有效知识，40k 后 F1 都很难明显移动。

因此 CDD 更适合作为“不得破坏已有强模型”的安全性集，而不是 Teacher 有效性的主验证集。

---

## 3 可验证的诊断量与判读

下面建议首先**不改训练机制**，只对已有日志做 epoch trajectory，并补少量 class-/region-conditioned statistics。

| 诊断量 | 看什么 | 健康 Teacher 的预期 | 异常说明 |
|---|---|---|---|
| `sam/ov/fusion_prior_gain` 随 epoch | prior 相对当前 Student 的有效窗口 | 训练前中期至少存在稳定 `gain>0` 区间；fusion 不应系统性差于更好的单源 | 从早期就全负：Foundation prior 本身无价值；后期才负：应做阶段性退出 |
| `sam/ov/fusion_prior_brier` | Foundation 自身绝对质量 | fusion 应接近或优于优质单源 | 如 WHU `0.0829 > OV 0.0455`：当前 fusion 在污染好源 |
| reliability decile → `P(gain>0)` | reliability 是否校准 | reliability 越高，positive-gain probability / mean gain 越高 | 高 reliability 仍负 gain：reliability 失真 |
| `sam_ov_conflict` × source gain | conflict 是否真的表示“不可信” | 高 conflict 时应能识别哪一源更可靠 | 一源明显正确但两源一起被降权：agreement penalty 过于粗糙 |
| `foundation_support_ratio` + reliability histogram | `r>0` 是否太宽松 | support 有明确选择性；不能只因微小 confidence 就全图 support | R3 `support=1.0` 且大部分 proposal 无 gain：support 定义失去筛选作用 |
| `pixel_reject_ratio` / `eligible_ratio` | audit 是否严重饥饿 | Teacher 有效期应有非零、可持续的 eligible mass | `>99.9%` + `aux_raw≈0`：Teacher 实际几乎不参与训练 |
| `image_reject_ratio` | 是“所有图少量像素”还是“大量图完全无 Teacher” | 大部分困难图应至少有一部分有效教师信号 | image reject 高：Teacher 仅极少样本有效，泛化叙事不成立 |
| `accepted_change_ratio / accepted_bg_ratio` | class-conditional acceptance | 低变化数据应避免 background mass 压倒 change | 必须结合类别比例计算 mass，不能只比较两个 ratio |
| **新增** `accepted_change_mass/bg_mass` | 实际 KD 质量贡献 | 与当前 Student 的 change/bg error mass 大致匹配 | WHU 若 bg mass 明显更大，验证 audit imbalance 假设 |
| `student_abs_error/error_mass` 分 change/bg | Teacher 到底该修哪里 | accepted KD 分布应追随 Student error mass | Student 错在 change，但 Teacher 主要教 background：机制错配 |
| `dynamic_teacher_gain` trajectory | residual teacher 是否真正超越 Student | 中前期应稳定 >0，并在 Student 收敛后向 0 | 全程≈0：Teacher 无新增知识；持续<0：Teacher 有害 |
| `dynamic_shift` + `dyn_gain` | Teacher “动得多”是否“动得对” | shift 增大时 gain 同步为正 | shift 大但 gain≤0：错误修正；shift≈0且gain≈0：Teacher collapse |
| `target/fast_residual_magnitude` | Teacher 是否学到 residual | 应先升后随 Student 收敛下降 | 始终≈0：Teacher 没学到；持续很大但 audit 全拒：teacher/audit 不一致 |
| `teacher_grad_norm/EMA_update/target_gap` | Fast/EMA 是否实际更新 | 非零且逐步稳定 | grad 非零但 dyn_gain≈0：容量在训练，但知识无转化 |
| `pair_match_iou/cov12/cov21` **按 instance area 分桶** | 小对象是否最差 | small/large 应无灾难性断层 | small bin IoU/cov 明显更差：小建筑几何脆弱得到直接证实 |
| **新增** SAM error 的 boundary/interior Brier | 错误是否是 mask 边缘抖动 | interior 明显优于 boundary 是正常，但差距不能巨大 | 小目标 boundary Brier 极高：验证 H3 |
| **新增** OV `|O_l1-O_l2|` | OV confidence 是否有跨尺度自洽性 | 高 confidence 同时应有低 level conflict | confidence 高而 l1/l2 disagreement 高：OV reliability 需重构 |

### 最关键的四张诊断图

建议优先生成：

1. `epoch → sam/ov/fusion/dynamic gain`
2. `reliability decile → positive-gain rate`
3. `epoch → accepted_change_mass / accepted_bg_mass / student_error_mass`
4. `SAM instance area bin → match_iou + boundary Brier`

只要这四张图出来，基本可以区分“prior 错”“reliability 错”“audit 错”“teacher 错”。

---

## 4 改进方案（含消融与失败判据）

### 4.1 必须先补的归因对照：OV-only，不把它包装成创新

当前矩阵：

```text
B0
R3A = SAM + Dynamic Teacher
R3  = SAM + OV + Dynamic Teacher
```

缺少最重要的：

```text
R3O = OV only + 同一个 Fast/EMA Teacher
```

这意味着目前只能说：

> OV 对 SAM-conditioned Teacher 在 SYSU 有 +0.84 的条件增量。

不能回答：

> OV 自己是否比 B0 有用。

**最小实验：**

| ID | SAM | OV | fusion | Fast/EMA | audit |
|---|---:|---:|---|---:|---:|
| B0 | × | × | × | × | × |
| R3A | ✓ | × | × | ✓ | advantage |
| **R3O** | × | ✓ | × | ✓ | advantage |
| R3 | ✓ | ✓ | weighted avg | ✓ | advantage |

先只跑 SYSU + WHU，40000 steps、batch 64、seed 2333。

**判读：**

- 若 `R3O > R3` 且 WHU 恢复，当前问题主要是 **SAM contamination / fusion**。
- 若 `R3O` 仍低于 B0，则问题继续下沉到 **OV reliability、audit 或 residual teacher**。
- 若 `R3O > B0` 且 R3 < R3O，则 BT-SAM 当前形式应从主方法删除。

### 4.2 (a) 修先验：Uncertainty-Aware BT-SAM，而不是继续直接 symmetric difference

#### 创新动机

当前 BT-SAM 把“实例几何差异”直接等价为“语义变化”，对小建筑不成立。

应显式区分：

\[
\text{真实结构变化}
\neq
\text{SAM 边界抖动 / 小配准误差}
\]

#### 作用机制

对 matched masks \(M_1,M_2\)，不再使用：

\[
M_1\setminus M_2,\quad M_2\setminus M_1
\]

直接作为 change。

使用现有 `boundary_radius=2` 构造 tolerance band：

\[
D_{1}^{valid}
=
M_1\setminus Dilate(M_2,\tau)
\]

\[
D_{2}^{valid}
=
M_2\setminus Dilate(M_1,\tau)
\]

只有超过几何容忍范围的 residual 才视为 structural change。

对于 completely unmatched instance，也不再：

```python
association = 1
```

而应区分：

- 与另一时相任何实例都相距较远 → high-confidence novelty；
- 只是刚好未相交、但在 \(\tau\) 范围内存在近邻 → uncertain，不作为高可靠变化。

#### 与当前模块的实质差异

当前：**mask difference = change evidence**。  
改进：**先建立 registration/segmentation uncertainty zone，再定义不可由边界抖动解释的结构变化。**

这是真正的双时相结构机制修改，不是 loss 调参。

#### 可证伪假设

若 WHU/LEVIR 的负增益主要来自 small-object geometry noise，则修改后应同时看到：

- small-object boundary SAM Brier 显著下降；
- `sam_prior_gain` 改善；
- 高可靠 SAM 像素的 positive-gain rate 上升；
- R3A 在 WHU/LEVIR 至少不再稳定低于 B0。

#### 最小消融

```text
R3A-current
vs
R3A-UASAM
```

先 SYSU + WHU seed 2333。

#### 失败判据

若：

- SAM Brier 没有明显下降；
- small/boundary error 不下降；
- WHU R3A 仍低于 B0；
- SYSU 原有 +0.54 同时消失；

则说明 **BT-SAM 结构知识本身不值得继续保留**，停止继续做 SAM 结构复杂化。

### 4.3 (a) 修可靠性/融合：Source-Selective Reliability Fusion

这是我认为比“继续堆 Teacher Agent”更优先的主改进方向。

#### 创新动机

当前 fusion 隐含假设：

> 两个 Foundation source 都有价值，只需要按 confidence 加权平均。

WHU 已直接反驳这一假设：OV 明显优于 SAM，但 fusion 反而更差。

#### 作用机制

先构造 **source-internal consistency reliability**。

SAM：

\[
R_S^\*
=
R_S
\times
G_{geometry}
\]

其中 \(G_{geometry}\) 来自前述 tolerant correspondence，而不是仅 quality/boundary。

OV：

\[
R_O^\*
=
\frac{c_1+c_2}{2}
\cdot
(1-|O_{l1}-O_{l2}|)
\]

这样高 confidence 但跨尺度不一致的 OV 会自动降可靠度。

融合时不再永远平均：

```text
低 conflict + 两源都可靠
    -> weighted fusion

高 conflict + 某一源 reliability 明显更高
    -> select dominant source

高 conflict + 两源 reliability 接近
    -> Reject / zero support
```

即：

> disagreement 首先应该触发“选择或拒绝”，而不是把一好一坏平均以后再同时降 reliability。

#### 与当前模块差异

当前：

```text
prior = weighted average
reliability = mean confidence × agreement
```

新机制：

```text
先校准每个 source
→ 冲突时 source selection / reject
→ 只有一致可靠时才 fusion
```

#### 可证伪假设

如果 current fusion 是 WHU 的主要负源，则 selective fusion 应满足：

\[
Brier(F_{new}) \le Brier(O)
\]

至少不应再出现 WHU `0.0829 >> 0.0455` 这种现象。

#### 最小消融

```text
R3O        OV-only
R3         current average fusion
R3-SRF     source-selective fusion
```

Teacher 容量、audit、batch、steps、seed 完全不动。

#### 失败判据

若 R3-SRF 的 prior Brier 不能优于/接近 OV-only，且 Student F1 不优于 R3O，则不再研究 fusion，直接采用单源 OV 或放弃 Foundation prior。

### 4.4 (b) 改 audit：Error-Mass Budgeted Advantage Audit

#### 创新动机

当前 advantage audit 解决的是：

> “该像素 Teacher 是否比 Student 更接近 GT？”

但没有解决：

> “有限的 Teacher 更新预算应该花在哪类 Student failure 上？”

在 WHU/LEVIR 中这两件事完全不同。

#### 作用机制

保留严格条件：

\[
gain>0
\]

不放宽安全门。

改变的是 accepted KD 的质量分配。

分别计算：

\[
E_{change}
=
\sum_{y=1}|p-y|\cdot importance
\]

\[
E_{bg}
=
\sum_{y=0}|p-y|\cdot importance
\]

然后让 change/bg 的实际 distillation mass 与当前 Student residual error mass 对齐，而不是与类别像素数量对齐。

即 Teacher 的训练预算回答：

> Student 当前哪里错得多，就在哪里多教；不是哪个类别像素多就在哪里多教。

这比简单的 class weight 更符合“困难画像 → Teacher correction”的 Direction-C 叙事。

#### 与当前 audit 差异

当前：

```text
gain > 0
→ 每个像素独立接受
```

新机制：

```text
gain > 0
→ 再按当前 Student 的 class-wise error mass 分配有效 Teacher budget
```

#### 可证伪假设

若 WHU 的问题主要来自 accepted background mass 偏大，则新 audit 应：

- 保持 positive-Brier safety；
- 使 `KD_change_mass / KD_bg_mass` 更接近 `student_change_error_mass / student_bg_error_mass`；
- WHU/LEVIR F1、IoU 不再下降；
- SYSU 不应明显损失。

#### 最小消融

在最佳 prior 版本上：

```text
advantage
vs
advantage + error-mass budget
```

不要同时换 prior、teacher hidden、loss 或 lr。

#### 失败判据

若 distillation mass 已明显重分配到 Student 真正困难区域，但 F1/IoU 仍没有恢复，则 audit imbalance 不是主因，停止继续设计复杂 gate/router。

### 4.5 (c) Teacher 容量：现在不要加大，先验证它有没有存在价值

目前不建议做：

```text
hidden 24 → 48/64
更深 residual head
更多 experts
多 teacher routing
```

因为当前证据根本没有显示“容量不足”。

更必要的是做一个 **Teacher necessity control**。

#### 最小实验

在 OV-only 或修正后的最佳 Foundation prior 上比较：

```text
Static:
Foundation prior
→ same GT advantage audit
→ Student KD

Dynamic:
Foundation prior
→ Fast/EMA residual teacher
→ same audit
→ Student KD
```

唯一变量：有没有 residual teacher。

#### 可证伪假设

如果 residual teacher 真正完成了“Student-aware Foundation adaptation”，则应满足：

- `dynamic_teacher_gain` 明显优于 raw prior gain；
- Dynamic F1/IoU > Static；
- 不是仅仅 `dynamic_shift≈0` 后靠 audit 退化成 Student。

#### 失败判据

如果 Dynamic 不优于 Static，或者长期：

```text
dynamic_teacher_gain ≈ 0
dynamic_shift ≈ 0
aux_raw ≈ 0
```

则删除 Fast/EMA Teacher。

这反而能形成更干净、更可信的论文机制，而不是继续增加 teacher capacity。

### 4.6 (d) 明确停止路线的条件

当前最不应该做的是在 primitive teacher 尚未证明有效时继续升级成更复杂的 Teacher Agent。

建议设置三级停止门：

| Gate | 必须回答 | 不通过怎么办 |
|---|---|---|
| G1 | OV-only 是否有独立价值？ | 否 → OV 也不能作为主 Teacher |
| G2 | 修正后的 SAM 是否有独立价值？ | 否 → 删除 BT-SAM |
| G3 | Dynamic Teacher 是否优于 Static prior？ | 否 → 删除 Fast/EMA Teacher |

只有至少一个 Foundation source 和 dynamic adaptation **分别通过独立增益验证**，才有资格继续做 Agent/routing。

否则“智能体选择 Teacher”只是在路由若干已经被证明无效的知识源。

### 4.7 工程约束

以上 A/B 类改动都只发生在：

```text
models/distill/task_space.py
models/distill/dynamic_teacher.py
models/distill/diagnostics.py
```

不改变 A2Net 主预测路径，因此应保持：

- deploy params = `2,913,094`
- 256×256 FLOPs ≈ `2.7475G`
- teacher/cache 仅训练期
- `switch_to_deploy()` 后 auxiliary 删除
- deploy 前后主预测最大误差 `<1e-6`

新实验必须使用新 ID，不要从旧 R3/R3A checkpoint 跨语义 resume；即使参数 shape 没变，prior/audit 的定义已经改变，恢复旧 Teacher 状态会造成协议污染。

---

## 5 结论与建议

### 5.1 BT-SAM 是否值得保留

**当前形式不值得作为主方法继续保留。**

证据链很明确：

- R3A 四数据集中仅 SYSU `+0.54`；
- WHU `-0.12`、CDD `-0.01`、LEVIR `-0.10`；
- SAM prior 在 SYSU/WHU 都有巨大负 gain；
- WHU `sam_brier=0.3788`，远高于 Student；
- 代码对 small-object boundary jitter 缺少 uncertainty modeling；
- unmatched instance 又被赋予 `association=1`。

因此建议定位为：

> **给 BT-SAM 最后一次“几何不确定性修正”的可证伪机会；若 UASAM 仍不能改善 WHU/LEVIR 的 prior quality 与 R3A，则从主线彻底删除。**

不建议继续在当前 BT-SAM 上堆 router、更多 SAM feature 或复杂 matching。

### 5.2 OV 是否值得保留

**值得暂时保留，而且优先级明显高于当前 BT-SAM，但尚未证明为有效 Teacher。**

支持保留的证据：

- SYSU：R3−R3A = **+0.84 F1**
- WHU/SYSU 中 OV Brier 都远好于 SAM
- R3/SYSU 相对 B0 达到 +1.38

反对直接下结论的证据：

- WHU：R3−R3A = -0.24
- LEVIR：约 -0.05
- CDD：约 +0.01
- OV gain 在成熟 Student 上仍为负
- 缺少 **OV-only** 实验

因此现在最重要的不是继续 SAM+OV，而是：

> **先跑 OV-only。**

如果 OV-only 在 SYSU 保持正增益，同时 WHU/LEVIR 比 R3 恢复，就有充分理由把论文主线从“结构+语义固定融合”转为：

> **Foundation source 的 task-/difficulty-dependent utility 不同，可靠性应基于当前任务状态选择，而不是无条件融合。**

这比当前 BT-SAM+OV averaging 的叙事更强，也更接近 Direction-C 的核心思想。

### 5.3 是否需要多 seed

**需要，但不是现在立刻把所有现有 variant 全部补多 seed。**

当前 seed 2333 已足以做两件事：

- 证明当前方法存在明显的跨数据集机制异常；
- 决定下一轮应该先做归因实验，而不是继续堆模块。

但它不足以做这些论文级结论：

- “SYSU 上稳定提升 +1.38”
- “WHU 一定下降 -0.36”
- “OV 对 SYSU 普遍有效”
- “BT-SAM 在建筑数据集普遍无效”

最终保留下来的方案至少需要 `2333/3407/5871` 三 seed；尤其 SYSU 的 +1.38 必须复现。WHU/LEVIR 的 -0.36/-0.15 数值本身可能含 seed 方差，但**两个建筑小目标数据集同方向为负，再叠加 prior/audit 诊断异常，已经足以要求先修机制。**

### 5.4 立即执行顺序

1. **先从现有 2333 日志画完整 epoch trajectory**：`prior_gain / dynamic_gain / reject / accepted class mass / dynamic_shift`，不训练新模型。
2. **补 R3O：OV-only + 同 Fast/EMA + 同 advantage audit**，只跑 SYSU、WHU。
3. 若 `R3O > R3`：优先实现 **Source-Selective Reliability Fusion**，同时把 current SAM 降为待修 source。
4. 并行做一个非常小的 **UASAM synthetic smoke**：同一小建筑 mask 平移 1–2 px 不应被大面积判为真实变化；真实 appearance/disappearance 仍应保留。
5. 只有 UASAM 明显改善 SAM Brier 后，才跑 `R3A-UASAM`。
6. 若最佳 prior 仍在 WHU 负增益，再测试 **Error-Mass Budgeted Audit**。
7. 最后才做 **Static prior vs Fast/EMA Dynamic Teacher**，决定 residual teacher 是否应该存在。
8. 通过上述机制门的最终方案，再补 `2333/3407/5871` 三 seed 和四数据集正式矩阵。

### 5.5 仍需补充证据

当前最缺的不是新的网络模块，而是五个证据：

- LEVIR/CDD 对应的完整 `prior/reliability/audit/dynamic` epoch trajectory；
- reliability 分桶后的 **positive-gain calibration curve**；
- 精确 `accepted_change_mass / accepted_bg_mass`，不能继续只用 ratio；
- SAM error 的 **small/large × boundary/interior** 分解；
- **OV-only** 与 **Static-prior** 两个关键控制组。

在这些证据出来之前，最稳妥的机制判断是：

> **当前 Run3 的核心瓶颈位于 Foundation prior 的任务适配性与 reliability/audit 的有效知识选择，而不是 Student 部署容量，也没有证据支持通过增加 residual Teacher 容量解决。BT-SAM 当前形式证据不足，应降级；OV 是唯一值得继续独立验证的 Foundation source；Teacher Agent 应暂缓，直到至少一个 teacher primitive 被同协议、多 seed 证明真正有效。**

---

**证据基线**

- GitHub 仓库：`YuqiWang-code/LS-Rep_BCD`
- 核心代码：`models/distill/task_space.py`、`dynamic_teacher.py`、`diagnostics.py`、`teacher_cache.py`
- 数据增强与 Cache 同步：`models/datasets/cache_transforms.py`、`cd_dataset.py`
- 训练协议：`models/scripts/train.py`、`train_scripts/RDT-CD/Run3/README.md`
- 数据与服务器约束：`docs/RSML-3_服务器环境与变化检测数据统一说明.md`
