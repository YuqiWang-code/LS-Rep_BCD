# SAM-HSD Run1 代码审计、失败机理与 Run2 实验方案

> 项目：LS-Rep_BCD / SAM-HSD  
> 任务：256×256 遥感二值变化检测  
> 部署学生：A2Net-LWGANet-L0  
> 文档定位：**方法审计 + 失败机理分析 + Run2 方法设计 + Run2 消融实验方案**  
> 证据优先级：**当前代码 / 实际 launcher / 正式 test 块 > AGENTS.md > 历史设计说明**

---

## 0. 结论先行

这次重新审查后，我认为 **SAM-HSD Run1 最大的问题不是“工程没有做好”，而是方法本身存在三个结构性错位**：

1. **教师结构目标与主任务的时相对称性不一致。**  
   主学生的 TFM 使用 `|F1-F2|`，最终 GT 也是 order-free 的 binary change；但 Encoder-HSD 却显式回归 appearance / disappearance 的 `E+ / E-`。H6 unsigned 在 SYSU 和 WHU 都优于 H4 directional，是最直接的警告。当前证据不支持继续把“方向性”当主创新点。

2. **Encoder-HSD 的结构表达能力太弱。**  
   当前四级 Encoder-HSD 本质上只是 `1×1 Conv(C→16)`，随后用逐像素 feature difference + shared `16→1` head 去拟合 boundary / affinity / depth / scale / compactness 聚合出来的空间结构目标。  
   这些教师信息本质上是**局部几何、邻域关系和对象尺度信息**，而 `1×1` probe 本身完全不看邻域。H2 在 SYSU 有效但幅度有限、WHU 反降，说明“encoder 结构监督有潜力”，但当前投影器可能不足以承载真正的结构知识。

3. **Decoder-HSD 对主 change probability 干预过于直接。**  
   当前 SCGR、boundary、affinity，以及 relation 的大部分 prediction-level 项，都直接作用在四个主变化概率图上。只有 `feature_relation` 单独作用在 decoder feature。  
   这种设计会把“学习 SAM 结构”与“改变二值分类校准”绑在一起。H3 在 SYSU 将 Recall 提升到 83.1638、F1 提升 +1.5325，但 WHU Precision 从 96.1101 暴跌到 92.6482、F1 −1.1006，非常符合“结构约束推动了更多 positive response，但在极不平衡的 WHU 上引发 FP 爆发”的现象。

因此 Run2 不建议继续围绕：

> directional evidence + 更多 loss + 调 lambda

做局部修修补补。

更值得形成论文主方法的是：

# **EIR-HSD：Exchange-Invariant Residual Hierarchical Structure Distillation**

核心重新定义为：

> **SAM 不直接告诉轻量学生“这是 appearance 还是 disappearance”，也不直接强迫最终 change probability 拟合 SAM 结构；SAM 提供 exchange-invariant 的结构残差编码，训练期的局部残差适配器在 Encoder 学结构变化 representation，在 Decoder 学结构恢复 representation，最后只对学生真正出错且 SAM/GT 一致的位置执行轻量 residual correction。**

这仍然属于 SAM-HSD 总框架，但比 Run1 更像一个完整、可解释、可发表的方法。

---

# 1. 证据范围与事实边界

本结论基于以下上传材料：

- `AGENTS.md`
- `models_and_metrics.txt`
- `experiment_metrics.xlsx`
- `SERVER_AND_DATASETS.md`

其中：

- 当前 `models/` 实现以 `models_and_metrics.txt` 为准；
- Run1 实际配置和正式 test 结果由 `AGENTS.md` 与 `experiment_metrics.xlsx` 交叉确认；
- 数据集属性参考 `SERVER_AND_DATASETS.md`；
- 不使用历史设计文档覆盖当前实现。

需要特别说明：

**当前上传材料包含 18 组最终正式 test 数字，但没有包含每个 run 全部逐 step / 逐 epoch 的原始 `train_log.txt` 数值轨迹。**

因此本文会严格区分：

- **[实现事实]** 当前代码明确如此；
- **[实验事实]** H0–H8 正式 test 已经观察到；
- **[合理推断]** 由代码 + 对照结果支持，但还没有直接日志证明；
- **[待验证假设]** 必须通过 Run2 消融才能确认。

---

# 2. Run1 当前方法回顾

## 2.1 部署主路径

当前学生固定为：

```text
T1 ─┐
    ├─ shared LWGANet-L0 ─ SWA ─ TFM(|F1-F2|) ─ A2Net Decoder ─ 4-scale masks
T2 ─┘
```

部署约束：

```text
Params = 2,913,094
FLOPs  = 2.7475G @ 256×256
```

SAM-HSD 全部只存在于训练期，不参与推理。

这是正确的，也是当前项目最应该保留的“burden-free deployment”基础。

---

## 2.2 SAM cache 与 PI-DTRS

缓存 T1/T2 独立提供：

```text
instance_id
boundary
quality
```

几何增强 replay 后动态生成：

```text
occupancy
affinity_r1
affinity_r2
affinity_r4
interior_depth
object_scale
compactness
```

Run1 将八种字段按固定权重聚合成 directional evidence：

\[
E^+ = \sum_k w_k \operatorname{ReLU}(R_2^k-R_1^k)
\]

\[
E^- = \sum_k w_k \operatorname{ReLU}(R_1^k-R_2^k)
\]

以及 stable evidence \(E^0\)。

这一点实现上确实保持 instance-label permutation invariance，也不需要 T1/T2 instance matching。

---

## 2.3 Encoder-HSD

当前：

```text
Stage1: 32  -> 16  (1×1)
Stage2: 64  -> 16
Stage3: 128 -> 16
Stage4: 256 -> 16
shared direction head: 16 -> 1
```

总训练期参数：

```text
7,697
```

directional mode：

\[
d^+ = \operatorname{softplus}(\beta(z_2-z_1))/\beta
\]

\[
d^- = \operatorname{softplus}(\beta(z_1-z_2))/\beta
\]

然后用一个 shared `16→1` head 预测 plus / minus。

stable prediction 用：

\[
\frac{\cos(z_1,z_2)+1}{2}
\]

stage weights：

```text
0.35 / 0.30 / 0.20 / 0.15
```

---

## 2.4 Decoder-HSD

当前 decoder loss：

\[
L_D =
0.35L_{SCGR}
+0.25L_{boundary}
+0.25L_{relation}
+0.15L_{affinity}.
\]

四个主 change probabilities 都参与 decoder structural supervision。

其中 relation 内部既包含：

- prediction pair consistency；
- stable suppression；
- prediction contrast；

也包含 decoder feature relation。

---

## 2.5 Full HSD

当前：

\[
L_{HSD}=0.5L_E+0.5L_D.
\]

外部再执行：

\[
L =
L_{main}
+
\operatorname{Cap}
\left(
\lambda(t)L_{HSD},
0.12L_{main}
\right)
\]

其中：

```text
hsd_lambda = 0.06
```

这意味着 H4 中 Encoder 和 Decoder 各自的 nominal coefficient 实际只有 H2/H3 单独训练时的一半。

---

# 3. Run1 正式结果

## 3.1 SYSU

| ID | Recall | Precision | OA | F1 | ΔF1 | IoU | Kappa |
|---|---:|---:|---:|---:|---:|---:|---:|
| H0 | 77.8818 | 86.3832 | 91.8887 | 81.9125 | 0 | 69.3660 | 76.7027 |
| H1 | 78.7398 | 85.5930 | 91.8607 | 82.0235 | +0.1110 | 69.5253 | 76.7747 |
| H2 | 80.4645 | 84.6615 | 91.9551 | 82.5097 | +0.5972 | 70.2268 | 77.2902 |
| H3 | **83.1638** | 83.7282 | **92.2181** | **83.4450** | **+1.5325** | **71.5929** | **78.3587** |
| H4 | 79.0507 | 85.3798 | 91.8673 | 82.0934 | +0.1809 | 69.6258 | 76.8424 |
| H5 | 79.9723 | 85.7890 | 92.1528 | 82.7786 | +0.8661 | 70.6173 | 77.7054 |
| H6 | 81.4440 | 84.5518 | 92.1148 | 82.9688 | +1.0563 | 70.8946 | 77.8409 |
| H7 | 80.6592 | 82.2965 | 91.3470 | 81.4697 | −0.4428 | 68.7332 | 75.8260 |
| H8 | 80.0090 | 85.7685 | 92.1548 | 82.7887 | +0.8762 | 70.6320 | 77.7161 |

---

## 3.2 WHU

| ID | Recall | Precision | OA | F1 | ΔF1 | IoU | Kappa |
|---|---:|---:|---:|---:|---:|---:|---:|
| H0 | 92.0067 | 96.1101 | 99.5351 | 94.0137 | 0 | 88.7036 | 93.7720 |
| H1 | 92.5579 | 95.3232 | 99.5246 | 93.9202 | −0.0935 | 88.5374 | 93.6729 |
| H2 | 91.6686 | 95.9332 | 99.5153 | 93.7524 | −0.2613 | 88.2396 | 93.5004 |
| H3 | **93.1796** | 92.6482 | 99.4360 | 92.9131 | **−1.1006** | 86.7643 | 92.6195 |
| H4 | 91.6034 | 96.2725 | 99.5261 | 93.8800 | −0.1337 | 88.4658 | 93.6337 |
| H5 | 91.8784 | 94.5719 | 99.4685 | 93.2057 | −0.8080 | 87.2759 | 92.9292 |
| H6 | 91.9860 | **96.3783** | **99.5449** | **94.1309** | +0.1172 | **88.9126** | **93.8943** |
| H7 | 92.8701 | 94.9294 | 99.5203 | 93.8884 | −0.1253 | 88.4808 | 93.6388 |
| H8 | 92.3736 | 94.1398 | 99.4693 | 93.2483 | −0.7654 | 87.3507 | 92.9722 |

---

# 4. Run1 最重要的五个方法证据

## 4.1 H2：Encoder structural supervision 有效，但当前形式不够稳

SYSU：

```text
+0.5972 F1
```

WHU：

```text
−0.2613 F1
```

这说明：

> 在 backbone representation 上做 SAM structural distillation 本身是有潜力的。

但当前 Encoder-HSD 不能跨数据集稳定工作。

这不应该直接得出“Encoder distillation 无效”。

更合理的问题是：

> 当前 `1×1 probe + scalar directional target` 是否太弱、太局部、太抽象？

---

## 4.2 H3：Decoder supervision 非常有力量，但副作用也最大

SYSU：

```text
Recall +5.2820
F1     +1.5325
```

WHU：

```text
Recall    +1.1729
Precision −3.4619
F1        −1.1006
```

这不是“decoder 没用”。

恰恰相反：

> Decoder-HSD 是 Run1 中作用最强的训练信号，只是它目前没有很好地区分“结构恢复”和“分类概率校准”。

在正样本较多、变化类型更复杂的 SYSU，它通过提高 Recall 获益。

在背景占绝对多数的 WHU，少量额外 FP 就会严重打击 Precision。

因此 Run2 应该**重写 Decoder-HSD，而不是删除 Decoder-HSD。**

---

## 4.3 H4：Encoder 与 Decoder 没有实现真正的 hierarchical cooperation

H4：

```text
SYSU +0.1809
WHU  −0.1337
```

明显没有继承：

```text
H2 SYSU +0.5972
H3 SYSU +1.5325
```

这里至少有两个方法问题。

### 问题 A：两个层级学的并不是同一个明确的结构表征

Encoder：

```text
directional plus / minus / stable regression
```

Decoder：

```text
SCGR + boundary + relation + affinity
```

从论文概念上说两者都叫 HSD，但实际上它们不是一个统一 structural latent 的不同层级监督。

更像：

```text
Encoder: 一套任务
Decoder: 另一套任务
```

所以“hierarchical”目前更多是位置层级，而不是表示层级。

### 问题 B：固定 0.5/0.5 直接削弱两个单支

H4 的：

\[
0.5L_E+0.5L_D
\]

导致各自有效强度相对 H2/H3 减半。

因此 H4 本身并不是一个很干净的“两个有效模块联合是否互补”的验证。

---

## 4.4 H6：directional 是当前最值得放弃或降级的设计

H6 unsigned 相对 H4：

```text
SYSU +0.8754 F1
WHU  +0.2509 F1
```

而且 H6 是 WHU 唯一超过 H0 的实验。

这个证据非常重要。

主学生的变化建模是：

\[
|F_1-F_2|.
\]

GT 也是：

```text
change / no-change
```

并没有 appearance / disappearance 标签。

因此要求 Encoder 学：

```text
T1 -> T2 appearance
T2 -> T1 disappearance
```

本质上是在引入一个主任务不需要的 temporal orientation。

### 另外存在一个实现混杂

当前 unsigned Encoder 分支：

```python
prediction_plus = prediction_minus = same_prediction
```

后面仍分别计算 plus loss 和 minus loss。

所以 H6 不是严格的纯 unsigned 消融。

Run2 必须修成：

```text
一个 symmetric prediction
一个 symmetric structural target
一个 loss
```

不能继续重复两遍。

---

## 4.5 H8：固定 robust filter 不是普适结构先验

H8 - H4：

```text
SYSU +0.6953
WHU  −0.6317
```

同一个 ±2 tolerance + 3×3 supporter filter：

- 在 SYSU 删除后更好；
- 在 WHU 删除后明显更差。

所以问题不应该理解成：

> tolerance 应该设 1 还是 2？

而应该上升为：

> **不同场景、不同对象尺度、不同 SAM quality 下，结构证据应该具有不同的可信度，而不是所有 pixel 使用同一套 hard rule。**

这可以形成 Run2 的 **structure confidence / trust modulation**，而不是继续做 H8.1/H8.2 调参。

---

# 5. 当前方法真正的“方法错漏”

下面不是代码 bug 列表，而是论文方法层面的审计。

---

# 5.1 错漏一：把“结构变化”压缩成 appearance / disappearance 两个方向并不自然

## 当前设计

对每个结构字段：

\[
R_1,R_2
\]

生成：

\[
\operatorname{ReLU}(R_2-R_1),
\quad
\operatorname{ReLU}(R_1-R_2).
\]

但二值变化检测真正需要的是：

\[
|R_2-R_1|.
\]

或者更一般地：

\[
\mathcal D(R_1,R_2)
=
\mathcal D(R_2,R_1).
\]

### 问题

SAM 的 T1/T2 instance partition 本来就是独立生成的。

例如：

```text
T1:
SAM 把一个建筑分成 2 个 instance

T2:
SAM 把同一个建筑分成 1 个 instance
```

affinity / compactness / depth 的差异并不等价于真实的“建筑出现或消失”。

directional decomposition 会把 segmentation partition inconsistency 强行解释成有向变化。

### Run2 改进

改成 **Exchange-Invariant Structural Residual**：

\[
D_k = |R_2^k-R_1^k|.
\]

交换：

\[
T_1 \leftrightarrow T_2
\]

严格满足：

\[
D_k(T_1,T_2)=D_k(T_2,T_1).
\]

这更符合：

- binary CD 标签；
- A2Net TFM；
- H6 的实验趋势。

---

# 5.2 错漏二：八个 PI-DTRS 字段过早被压成一个 scalar

当前：

```text
boundary
occupancy
r1
r2
r4
depth
scale
compactness
```

通过手工权重：

```text
0.25 / 0.15 / 0.12 / 0.10 / 0.08 / 0.10 / 0.12 / 0.08
```

直接求和。

这有两个问题。

## 第一：不同字段语义不同

boundary 表示：

```text
local geometry
```

affinity 表示：

```text
local / medium-range relation
```

scale / compactness 表示：

```text
instance-level geometry
```

它们不是一个量纲的 scalar signal。

## 第二：八种信息被压成 scalar 后，学生不知道“为什么这里发生结构变化”

例如两个 pixel 都有：

```text
delta = 0.6
```

一个可能是：

```text
strong boundary shift
```

另一个可能是：

```text
object-scale / compactness change
```

当前 Encoder head 都只能拟合同一个 1-channel target。

---

# 5.3 建议：Multi-Field Structural Residual Code

不要再直接得到一个 scalar delta。

定义四组结构 residual：

### Boundary residual

\[
D_B
\]

由 tolerance-aware 双时相边界差得到。

### Local relation residual

\[
D_L =
\alpha_1|A_1^2-A_1^1|
+\alpha_2|A_2^2-A_2^1|.
\]

### Object geometry residual

\[
D_G =
\beta_1|A_4^2-A_4^1|
+\beta_2|Depth_2-Depth_1|
+\beta_3|Scale_2-Scale_1|
+\beta_4|Compact_2-Compact_1|.
\]

### Stable structural consensus

\[
S =
C_{common}
\cdot
\left(
1-\bar D
\right).
\]

构成：

\[
T^{str}
=
[D_B,D_L,D_G,S]
\in\mathbb R^{B\times4\times H\times W}.
\]

这一改变非常重要：

> **Run1 是 scalar structure evidence；Run2 应该变成 structural code。**

这会让“hierarchical structural distillation”真正有一个可被 Encoder 和 Decoder 共同学习的结构表示。

---

# 5.4 错漏三：Encoder-HSD 只有 1×1 Conv，看不到邻域结构

当前每层：

```python
nn.Conv2d(C, 16, kernel_size=1)
```

这只做 channel projection。

对于某个位置 \(x\)：

\[
z(x)=Wf(x).
\]

它本身完全不知道：

```text
x 左边是不是同一个 instance
x 右边是不是 boundary
3×3 内部结构有没有断裂
局部对象轮廓是什么
```

但 teacher target 却包含：

```text
boundary
r1/r2/r4 affinity
depth
compactness
```

这是方法上的明显 capacity mismatch。

---

# 5.5 Run2 Encoder 改进：Spatial Residual Structure Adapter

建议把每层 probe 从：

```text
1×1 Conv
```

升级为：

```text
1×1 projection
    ↓
3×3 depthwise spatial conv
    ↓
GELU
    ↓
1×1 pointwise conv
    ↓
Residual Add
```

定义：

\[
z_i^t=P_i(F_i^t)
\]

\[
\tilde z_i^t
=
z_i^t
+
W_i^{pw}
\left(
\sigma(
DWConv_{3\times3}(z_i^t)
)
\right).
\]

其中：

```text
P_i: C_i -> 16
DWConv: 16 groups
PWConv: 16 -> 16
```

### 为什么是 3×3 depthwise？

不是为了堆参数。

它对应 Run1 缺失的核心能力：

> 将单点 feature projection 变成 local structural projection。

r1、boundary、局部 affinity 本身就是邻域结构。

而 deeper stages 已经具有更大的 receptive field，所以训练期 adapter 不需要大卷积。

---

# 5.6 Pair representation 也应该 exchange-invariant

Encoder side 不再构造：

```text
softplus(z2-z1)
softplus(z1-z2)
```

而构造：

\[
D_i^F = |\tilde z_i^2-\tilde z_i^1|
\]

以及：

\[
C_i^F = \tilde z_i^1\odot\tilde z_i^2.
\]

拼接：

\[
U_i=
[D_i^F,C_i^F]
\in\mathbb R^{B\times32\times H_i\times W_i}.
\]

显然：

\[
U_i(F_1,F_2)=U_i(F_2,F_1).
\]

然后使用 shared structure head：

```text
1×1 32 -> 16
GELU
1×1 16 -> 4
```

预测：

\[
\hat T_i^{enc}
=
[\hat D_B,\hat D_L,\hat D_G,\hat S].
\]

这样 Encoder 不再是“direction classification head”，而是真正的：

> **structural residual representation learner**

---

# 5.7 错漏四：Decoder-HSD 没有独立的结构表示空间

这是我认为 Run1 第二重要的方法问题。

当前 Decoder-HSD 里：

```text
SCGR
boundary
relation prediction terms
affinity
```

主要直接作用在：

```text
mask_p2
mask_p3
mask_p4
mask_p5
```

即最终变化分类概率。

结果就是：

> 如果 SAM structural objective 与 BCE+Dice 的 optimal calibration 不一致，它只能直接改变 change probability。

SYSU 可以通过提高 Recall 获益。

WHU 背景极多，稍微多预测一点 change 就会严重损伤 Precision。

---

# 5.8 Run2 Decoder 改进：Residual Decoder Structure Head

A2Net decoder 已经提供：

```text
p2/p3/p4/p5 feature
C = 64
```

Run2 不应该让所有 SAM structure 都直接作用在 main mask。

建议新增一个**训练期共享结构 side head**：

```text
decoder feature [B,64,H,W]
      ↓
1×1 Conv 64 -> 16
      ↓
3×3 DWConv
      ↓
GELU
      ↓
1×1 Conv 16 -> 4
      ↓
structural residual code
```

记为：

\[
\hat T_i^{dec}
=
Q(P_i).
\]

Q 在四个 decoder stages 共享，因为四层都是 64 channels。

输出同样：

\[
[\hat D_B,\hat D_L,\hat D_G,\hat S].
\]

因此 Encoder 与 Decoder 第一次真正学习**同一种 structural code**：

```text
SAM PI-DTRS
        ↓
4-channel structural residual code
        ├── Encoder structural representation
        └── Decoder structural representation
```

这才更符合 Hierarchical Structural Distillation。

---

# 5.9 为什么需要 Decoder side head，而不是继续加 loss？

因为 side head 将两个目标解耦：

## 主头

负责：

```text
binary change classification
```

## Auxiliary structure head

负责：

```text
boundary / relation / geometry / stable structure
```

结构 loss 的梯度仍然会回到 decoder feature，但不要求同一个 1-channel mask 同时承担所有 structural regression。

这个设计可以直接对应 H3 的 WHU failure：

> 我们保留 Decoder-HSD 强大的 representation shaping 能力，但减少它直接改变 change probability calibration 的机会。

---

# 5.10 错漏五：Run1 使用的是“直接结构拟合”，不是“残差纠错”

Run1 boundary 典型思想：

```text
SAM boundary 高
+ GT boundary 附近
→ predicted boundary 应该高
```

这相当于让学生复制教师认可的结构。

但 SAM 并不是变化检测教师。

SAM 只是单时相 segmentation teacher。

所以更自然的方式应该是：

> **SAM 决定哪里值得看，GT 决定正确方向，student 当前误差决定是否还需要纠正。**

---

# 5.11 Student-Error-Aware Residual Correction

令主预测：

\[
p=\operatorname{stopgrad}(P_{main})
\]

GT：

\[
y.
\]

### FN correction mask

\[
M_{FN}
=
y
\cdot
\mathbb 1[p<0.5]
\cdot
T_{change}.
\]

含义：

```text
GT 是 change
学生当前漏检
SAM structure 又支持这里存在结构变化
```

才执行 FN recovery。

### FP correction mask

\[
M_{FP}
=
(1-y)
\cdot
\mathbb 1[p>0.5]
\cdot
T_{stable}.
\]

含义：

```text
GT 是 unchanged
学生当前误报
SAM 双时相结构又相对稳定
```

才执行 FP suppression。

于是：

\[
L_{corr}
=
w_{FN}L_{FN}
+
w_{FP}L_{FP}.
\]

这与 Run1 SCGR 最大区别是：

> Run1 是根据 teacher/GT relation 全局重新加权分类；Run2 是只在“学生真的犯错”时施加 residual correction。

这更符合“residual distillation”。

---

# 5.12 对 WHU 的意义

WHU 最突出的问题是：

```text
H3 Recall 上升
Precision 大幅下降
```

因此 Run2 的 decoder correction 应该让：

\[
w_{FP} > w_{FN}
\]

但不能针对 WHU 单独设置。

建议统一：

```text
w_FP = 1.3
w_FN = 1.0
```

它表达的不是“为 WHU 调参”，而是 binary CD 的一般事实：

> SAM-induced false structural change 不应该轻易把背景推成 change。

---

# 5.13 错漏六：quality / robust filter 的角色应该从“硬过滤”变成“连续 trust”

Run1：

```text
quality gate
+ amplitude 0.15
+ supporter count
+ boundary tolerance
```

都是比较硬的 heuristic。

H8 跨数据集反转说明：

> 同样的 hard robustification 不可能同时适应 SYSU 的复杂变化与 WHU 的密集规则建筑。

Run2 建议不要继续找一个万能 threshold。

定义连续 trust：

\[
T_{change}
=
Q_\Delta^\gamma
\cdot
C_\Delta
\cdot
(1-U).
\]

其中：

\[
Q_\Delta=\max(Q_1,Q_2)
\]

coverage：

\[
C_\Delta
=
\mathbb1[Q_1>0\lor Q_2>0].
\]

stable：

\[
T_{stable}
=
\sqrt{Q_1Q_2}
\cdot
C_{both}
\cdot
(1-U).
\]

这里最重要的语义是：

```text
missing SAM coverage != stable
missing SAM coverage = abstain
```

---

# 5.14 错漏七：H4 的两个分支不只是“权重不对”，而是没有统一中间语义

如果 Run2 只是：

```text
H2 + H3
```

然后把 0.5/0.5 改成 0.6/0.4，

即使涨了，也很难形成好的方法贡献。

更好的改进是：

```text
同一个 4-channel structural residual code
   ↓
Encoder 学
   ↓
Decoder 也学
```

这样 hierarchical 的含义从：

```text
两个不同 loss 放在不同网络层
```

变成：

> **同一个结构残差知识在 representation hierarchy 中逐级对齐。**

这比单纯动态 loss weight 更方法化。

---

# 6. 推荐的 Run2 主方法

# EIR-HSD

## Exchange-Invariant Residual Hierarchical Structure Distillation

完整流程：

```text
SAM T1 structure ─┐
                  ├─ Exchange-Invariant Structural Residual Encoder
SAM T2 structure ─┘
                         │
                         ↓
          4-channel structural residual code
             [Boundary / Local / Geometry / Stable]
                  │                       │
                  │                       │
                  ↓                       ↓
       Spatial Residual            Decoder Residual
       Encoder Adapter             Structure Head
                  │                       │
                  └──────────┬────────────┘
                             ↓
                 Hierarchical Structural Loss
                             │
                     Student-error-aware
                     residual correction
                             │
                             ↓
                        Main student

deploy:
A2Net-LWGANet-L0 only
```

---

# 7. EIR-HSD 数学定义

## 7.1 Teacher structure tensor

对时相 \(t\)：

\[
R_t=
[
B_t,
O_t,
A_t^1,
A_t^2,
A_t^4,
D_t,
S_t,
C_t
].
\]

---

## 7.2 Exchange-Invariant Difference

除 boundary 外：

\[
\Delta_k=|R_2^k-R_1^k|.
\]

boundary 使用 tolerance-aware symmetric difference：

\[
\Delta_B=
\max
\left[
B_2(1-\mathcal D(B_1)),
B_1(1-\mathcal D(B_2))
\right].
\]

注意：

\[
\Delta_B(T_1,T_2)
=
\Delta_B(T_2,T_1).
\]

---

## 7.3 Structural code

建议第一版：

\[
D_B=\Delta_B.
\]

\[
D_L=
0.55\Delta_{A1}
+
0.45\Delta_{A2}.
\]

\[
D_G=
0.25\Delta_{A4}
+
0.25\Delta_{Depth}
+
0.30\Delta_{Scale}
+
0.20\Delta_{Compact}.
\]

\[
S=
C_{common}
\cdot
\left(
1-
\frac{1}{K}\sum_k\Delta_k
\right).
\]

然后：

\[
T^{str}=[D_B,D_L,D_G,S].
\]

第一版不建议直接做 learnable teacher fusion。

原因：

> 若 target mixer 本身可训练，很容易通过缩小 target 让 distillation loss 退化。

先用固定的**语义分组**，比 Run1 的八字段 scalar weighted sum 更可解释。

---

# 8. Hierarchical target routing

四个 stage 不应该完全重复同样的结构监督。

建议：

### Stage 1

重点：

```text
Boundary
Local relation
```

权重：

```text
[B,L,G,S] = [0.50, 0.35, 0.05, 0.10]
```

### Stage 2

```text
[0.30, 0.40, 0.15, 0.15]
```

### Stage 3

```text
[0.10, 0.30, 0.40, 0.20]
```

### Stage 4

```text
[0.05, 0.15, 0.50, 0.30]
```

这样：

```text
浅层 -> edge / local relation
深层 -> object geometry / stable semantics
```

仍然保留 Run1 hierarchical idea，但目标定义更明确。

---

# 9. Encoder 模块具体结构

建议类：

```text
SpatialResidualEncoderHSD
```

每层：

```python
class SpatialResidualProbe(nn.Module):
    def __init__(self, in_ch, hidden=16):
        self.proj = Conv1x1(in_ch, hidden, bias=False)
        self.dw = DWConv3x3(hidden, hidden, groups=hidden)
        self.pw = Conv1x1(hidden, hidden, bias=False)

    def forward(self, x):
        z = self.proj(x)
        r = self.pw(gelu(self.dw(z)))
        return z + r
```

pair：

```python
z1 = probe(f1)
z2 = probe(f2)

diff = torch.abs(z2 - z1)
common = z1 * z2

pair = torch.cat([diff, common], dim=1)
target_pred = shared_head(pair)
```

shared head：

```text
Conv1x1 32 -> 16
GELU
Conv1x1 16 -> 4
Sigmoid
```

### 训练参数估计

当前：

```text
+7,697
```

新 Encoder side 预计：

```text
约 +9k ~ +10k
```

即总训练参数大约：

```text
2.923M
```

数量级非常小。

部署仍：

```text
2.913094M
```

---

# 10. Decoder 模块具体结构

建议：

```text
ResidualDecoderStructureHead
```

由于 p2/p3/p4/p5 都是 64 channels，可共享：

```python
class DecoderStructureHead(nn.Module):
    def __init__(self):
        self.proj = Conv1x1(64, 16)
        self.dw = DWConv3x3(16, 16, groups=16)
        self.out = Conv1x1(16, 4)

    def forward(self, f):
        z = self.proj(f)
        z = z + gelu(self.dw(z))
        return sigmoid(self.out(z))
```

额外参数约：

```text
1.2k
```

四尺度共享后仍约 1.2k。

Full Run2 训练参数估计：

```text
约 2.924M
```

部署仍：

```text
2.9131M / 2.7475G
```

---

# 11. Decoder loss 重构

Run1：

\[
0.35SCGR
+0.25Boundary
+0.25Relation
+0.15Affinity.
\]

Run2 不建议继续沿用四个彼此重叠的 prediction-level loss。

建议改为三个层次：

## 11.1 Decoder structure-code regression

\[
L_{D,str}
=
\sum_i
\operatorname{WM}
(
\ell(
\hat T_i^{dec},
T_i^{str}
),
Trust_i
).
\]

这是主 decoder distillation。

---

## 11.2 GT-conditioned relational consistency

SAM instance 只决定：

```text
哪些 pair 值得建立关系
```

GT 决定：

```text
same-change-state / cross-change-boundary
```

而不是只凭 SAM delta 决定。

同类关系：

```text
same SAM instance
AND same GT class
AND high trust
```

才执行 feature consistency。

跨 GT change boundary：

```text
different GT class
AND SAM boundary support
```

执行 feature contrast。

这个 relation loss 应主要作用在：

```text
decoder structural feature / decoder feature
```

而不是主 probability。

---

## 11.3 Residual correction

只保留一个很轻的 main-mask correction：

\[
L_{corr}
=
L_{FN}+1.3L_{FP}.
\]

主任务仍由原始：

```text
BCE + batch Dice
```

主导。

---

# 12. Full loss

建议第一版：

\[
L=
L_{main}
+
\lambda(t)
\left[
L_E
+
L_{D,str}
+
0.5L_{rel}
+
0.5L_{corr}
\right].
\]

注意这里不是：

\[
0.5L_E+0.5L_D.
\]

每个 loss 都在自己的 valid evidence 上归一化后再组合。

outer：

```text
lambda_max = 0.06
main cap = 0.12
```

第一轮 Run2 建议先继续沿用 Run1，避免 training recipe 成为混杂因素。

如果 Full 仍然组合失败，再进入 Run3/后续优化动态 loss balancing，而不是 Run2 一开始就加入 PCGrad/CAGrad 等大量优化机制。

---

# 13. 为什么 Run2 第一版不建议直接上 PCGrad / CAGrad

这些方法可以做，但现在不应该成为主创新。

原因：

1. H4 失败还不能证明一定是 gradient conflict；
2. H4 本身存在 0.5/0.5 分支削弱；
3. Encoder 和 Decoder 当前学习的目标语义也并不统一；
4. 如果把结构目标本身重构后 H4-like full 已经互补，就没有必要增加复杂 gradient surgery。

所以 Run2 的顺序应该是：

```text
先修 structural representation
再修 decoder supervision interface
再看需不需要 gradient optimization
```

而不是反过来。

---

# 14. Run2 的论文性创新点

如果实验成功，我认为可以组织成三个方法 contribution。

## Contribution 1 — Exchange-Invariant Structural Residual Coding

将 T1/T2 独立 SAM instance partitions 转换为：

```text
boundary
local relation
object geometry
stable consensus
```

组成的 multi-field structural residual code。

特点：

- 不做 instance matching；
- permutation-invariant；
- temporal-exchange invariant；
- 不引入 appearance/disappearance 伪方向标签。

---

## Contribution 2 — Hierarchical Residual Structure Adaptation

不是简单在 backbone 四层挂 `1×1 probe`。

而是：

```text
training-only local residual spatial adapters
```

将相同 structural code 分层蒸馏到：

```text
Encoder pair representation
Decoder reconstruction representation
```

其中 shallow/deep stages 接收不同结构分量。

---

## Contribution 3 — Student-Error-Aware Residual Correction

SAM 不作为 change pseudo-label teacher。

SAM 只提供：

```text
结构可信区域
```

GT 提供：

```text
变化方向
```

Student current prediction 提供：

```text
是否需要纠错
```

形成：

```text
FN recovery
FP suppression
```

特别解决 Run1 Decoder-HSD 在高背景比例数据集上的 false-positive amplification。

---

## Contribution 4 — Burden-Free Deployment

所有：

```text
SAM cache
structure code builder
encoder residual adapters
decoder structure head
residual correction
```

训练结束全部删除。

部署仍为：

```text
A2Net-LWGANet-L0
2.9131M
2.7475G
```

这一点继续保留，但不要把它当唯一创新。

---

# 15. Run2 实验设计原则

Run2 应模仿 Run1：

- 同样 SYSU + WHU；
- batch=64；
- 40,000 optimizer steps；
- seed=2333；
- Adam；
- lr=5e-4；
- weight_decay=1e-4；
- backbone lr multiplier=1.0；
- batch Dice；
- 无 AMP；
- 无 EMA；
- 无 grad clipping；
- same H0；
- same deployment graph。

Run2 的目标不是一次堆很多模块得到最好数值。

而是：

> 每个实验回答一个明确的方法问题。

---

# 16. 推荐 Run2 H0–H8 风格矩阵

建议 Run2 使用：

```text
R0 – R8
```

共九个实验，每个 SYSU/WHU 各跑一组。

---

## R0 — Clean Anchor

与 Run1 H0 完全一致。

目的：

```text
同代码版本的公平基线
```

训练参数：

```text
2.9131M
```

部署：

```text
2.9131M / 2.7475G
```

---

## R1 — Run1 Unsigned Reference

复现当前 H6。

目的不是作为最终方法，而是：

> 验证 Run2 codebase 与 Run1 的当前最佳跨数据集 recipe 是否一致。

仍然保留 Run1 unsigned 的原实现语义。

预期：

```text
SYSU ≈ Run1 H6
WHU  ≈ Run1 H6
```

如果偏差很大，先不继续新方法。

---

## R2 — Exchange-Invariant Structural Code

相对 R1 的核心变化：

```text
删除 plus / minus
删除同一 unsigned prediction 计算两遍 loss
建立单一 exchange-invariant multi-field structural code
```

Encoder 暂时仍使用简单 1×1 projection。

目的：

> 单独验证“directional/unsigned scalar evidence”是否应该被 exchange-invariant structural residual code 替代。

### 成功信号

SYSU：

```text
不低于 R1，最好 +0.2 以上
```

WHU：

```text
Precision 不下降，F1 至少不低于 R1
```

如果 R2 明显更差：

> multi-field structural code 设计需要重审，不能继续把后面收益归给它。

---

## R3 — Spatial Residual Encoder

R2 +：

```text
1×1 probe
→
1×1 + DW3×3 + PW1×1 residual adapter
```

只改 Encoder。

目的：

> 验证 Run1 Encoder-HSD 的主要瓶颈是否是缺少 local spatial modeling。

### 关键比较

```text
R3 vs R2
```

如果 SYSU/WHU 均改善：

> 说明“structure teacher 需要 spatial adapter”获得直接支持。

如果只 SYSU 改善而 WHU 下降：

> 需要进一步检查 small/dense instance 过拟合。

---

## R4 — Residual Decoder Structure Head

使用与 R2 相同的 structural code。

关闭 Encoder。

仅启用：

```text
shared decoder structure side head
+ decoder structural-code loss
+ GT-conditioned relation
+ residual correction
```

这是对 Run1 H3 的重写。

### 核心问题

> 把 SAM structure 从主 probability 上移到独立结构 side head 后，能否保留 SYSU Recall 收益，同时恢复 WHU Precision？

这是 Run2 最重要的一组实验之一。

### 成功标准

SYSU：

```text
ΔF1 >= +0.5 vs R0
```

WHU：

```text
Precision >= R0
且 ΔF1 >= +0.2 进入候选
```

如果能达到 +0.3，则非常强。

---

## R5 — Full EIR-HSD

组合：

```text
R3 Spatial Residual Encoder
+
R4 Residual Decoder Structure Head
```

并共享同一个：

```text
4-channel structural residual code
```

这是 Run2 主方案。

注意：

不使用 Run1 的：

```text
0.5 * encoder + 0.5 * decoder
```

而是让各个 branch 在 valid evidence 上独立归一化，再统一进入 HSD loss。

### R5 必须回答

> 重构成“同一结构 code 的 hierarchical representation learning”之后，Full 是否终于能够继承 Encoder 和 Decoder 的独立收益？

如果：

```text
R5 >= max(R3,R4)
```

说明 Run1 的 H4 failure 很可能来自原来的目标异质性/组合方式。

---

## R6 — Full + Fixed 0.5/0.5

与 R5 完全相同的模块。

唯一变化：

```text
重新使用 Run1 0.5 Encoder + 0.5 Decoder
```

目的：

> 干净验证固定 0.5/0.5 是否确实造成 hierarchical combination loss。

这是 Run1 没有完成的干净实验。

### 解释

如果：

```text
R6 < R5
```

尤其 SYSU 明显下降，

则可以正式支持：

> independent normalized hierarchical objectives 比 naive fixed averaging 更有效。

---

## R7 — Full without Student-Error Residual Correction

保持：

```text
structural code
spatial encoder adapter
decoder structure head
```

但删除：

```text
student-error-aware FN / FP correction
```

只保留 structural representation losses。

目的：

> 判断 residual correction 是否真正解决 WHU Precision 问题。

### 最重要指标

WHU：

```text
Precision
FP tendency
F1
```

如果：

```text
R5 P > R7 P
```

且 SYSU Recall 没明显下降，

residual correction 就是一个有价值的贡献。

---

## R8 — Directional Restore

R5 全部结构不变。

唯一改动：

```text
exchange-invariant structural code
→
directional plus/minus version
```

必须保持：

```text
相同 spatial adapter
相同 decoder side head
相同 loss
相同 branch composition
```

目的：

> 给 directionality 一个真正公平的最终判决。

这是比 Run1 H4/H6 更干净的 directional ablation。

### 如果：

```text
R5 > R8 on both datasets
```

就可以有较强证据写：

> exchange-invariant structural residual is more suitable for binary CD than explicit directional decomposition.

如果 R8 反而更好：

> directional 不应删除，Run1 H6 的结果主要来自其他 confound。

---

# 17. Run2 矩阵汇总

| ID | 名称 | 核心变化 | 主要回答的问题 |
|---|---|---|---|
| R0 | Clean Anchor | 无 SAM | 新代码基线 |
| R1 | Run1 Unsigned Ref | 复现 H6 | 新旧 code 一致性 |
| R2 | Exchange-Invariant Code | 单 symmetric structural code | direction 是否错位 |
| R3 | Spatial Residual Encoder | R2 + 3×3 residual adapter | 1×1 probe 是否太弱 |
| R4 | Residual Decoder Head | decoder side structure learning | H3 是否因直接干预概率而失败 |
| R5 | Full EIR-HSD | R3 + R4 | hierarchical 是否恢复互补 |
| R6 | Fixed Fusion | R5 + 0.5/0.5 | H4 fixed fusion 是否有问题 |
| R7 | No Residual Correction | R5 − student-error correction | WHU Precision guard 是否必要 |
| R8 | Directional Restore | R5 改回 directional | 最终判断方向性 |

共：

```text
9 × SYSU
9 × WHU
= 18 runs
```

与 Run1 规模完全相同。

---

# 18. Run2 成功阈值

## SYSU

至少：

\[
\Delta F1 \ge +0.5
\]

相对同批 R0。

同时：

```text
IoU ↑
Kappa ↑
Recall / Precision 不明显崩塌
```

理想目标：

```text
R5 >= 83.2 F1
```

即接近或超过 Run1 H3。

---

## WHU

必须：

\[
\Delta F1\ge+0.3
\]

并且：

\[
Precision\ge P_{R0}.
\]

这是 Run2 真正应该突破的核心。

理想：

```text
F1 >= 94.3
Precision >= 96.1
```

---

# 19. 各实验失败后应该怎么解释

## R2 失败

说明：

```text
不是简单 directionality 问题
```

要检查 structural code 分组是否丢信息。

---

## R3 失败

说明：

```text
Encoder 的瓶颈不是 1×1 spatial capacity
```

不要继续扩大卷积。

---

## R4 SYSU 保住、WHU Precision 恢复

这是最理想的结果之一。

说明：

> Run1 H3 不是 decoder structural supervision 本身错误，而是“结构 supervision 直接落在 main probability 上”的接口设计错误。

这可以形成非常清晰的方法故事。

---

## R4 仍然 WHU Precision 崩塌

说明：

> 问题更可能来自 SAM structural evidence 本身在 WHU 的 false-change bias。

此时要进一步加强 trust/abstention，而不是继续堆 decoder head。

---

## R5 < R3 / R4

说明 hierarchical conflict 仍未解决。

此时才值得考虑：

```text
PCGrad
CAGrad
alternating optimization
```

进入 Run2.5 / Run3。

而不是现在提前引入。

---

## R5 > R3 与 R4

这是非常重要的 positive evidence。

说明新的：

```text
unified structural code
+ encoder/decoder residual representations
```

确实形成 hierarchy cooperation。

---

## R6 ≈ R5

说明：

```text
Run1 H4 失败不是 0.5/0.5 的主要责任
```

应该降低 fixed fusion hypothesis。

---

## R7 ≈ R5

说明：

```text
student-error residual correction 不重要
```

可以从最终方法删除，保持模型简洁。

---

## R8 > R5

说明：

```text
directionality 在公平架构下仍有价值
```

则不能为了故事强行删除。

---

# 20. 当前不建议做的 Run2 改动

## 不建议 1：直接把 probe_channels 从 16 改成 32/64

这只能回答：

```text
容量更大是否更好
```

不能形成清晰方法贡献。

应该优先加：

```text
local spatial residual modeling
```

而不是纯 channel expansion。

---

## 不建议 2：在主 decoder feature 中 concat SAM map

这会：

- 破坏训练/部署图一致性；
- 容易形成 inference dependency；
- 削弱 burden-free contribution。

---

## 不建议 3：给 A2Net 主路径加永久卷积

当前主学生固定是项目的重要控制变量。

Run2 新卷积应放在：

```text
training-only auxiliary adapter
```

而不是部署主网络。

---

## 不建议 4：继续细调 0.35/0.25/0.25/0.15

Run1 已经表明问题不是某一个 decimal weight。

应该先重构 decoder supervision 的形式。

---

## 不建议 5：同时加入 PCGrad + Tversky + dynamic weighting + uncertainty MLP

一次改变过多。

即使涨了：

```text
无法知道为什么涨
```

不利于论文消融。

---

# 21. 训练参数与部署复杂度

预计：

| 模型 | 训练参数 | 部署参数 | FLOPs |
|---|---:|---:|---:|
| R0 | 2.9131M | 2.9131M | 2.7475G |
| R1 | 2.9208M | 2.9131M | 2.7475G |
| R2 | ≈2.921M | 2.9131M | 2.7475G |
| R3 | ≈2.923M | 2.9131M | 2.7475G |
| R4 | ≈2.914–2.915M | 2.9131M | 2.7475G |
| R5–R8 | ≈2.924M | 2.9131M | 2.7475G |

具体训练参数以代码实现后实际 `sum(p.numel())` 为准。

正式报告不能使用这里的估计值替代实测值。

---

# 22. 代码层面的具体修改位置

本文不展开成完整工程 patch，但 Run2 方法落地主要只需要改以下文件。

## `temporal_evidence.py`

从：

```text
DirectionalTemporalEvidence
```

重构为：

```text
ExchangeInvariantStructuralEvidence
```

主要输出：

```text
boundary_residual
local_relation_residual
geometry_residual
stable_consensus
trust_change
trust_stable
```

保留旧 directional mode 用于 R8。

---

## `encoder_hsd.py`

新增：

```text
SpatialResidualProbe
ExchangeInvariantEncoderHSD
```

结构：

```text
1×1
DW3×3
PW1×1
residual
```

shared head：

```text
32 -> 16 -> 4
```

---

## `decoder_hsd.py`

新增：

```text
DecoderStructureHead
ResidualDecoderHSD
```

主要结构 loss 不再直接建立在四个 main probability 上。

---

## `losses.py`

重点保留/新增：

```text
structural_code_loss
gt_conditioned_relation_loss
residual_correction_loss
```

Run1 legacy losses 可以保留给 R1/reference。

---

## `sam_hsd_adapter.py`

Run2 full 应变成：

```text
structure code builder
    ↓
encoder residual HSD
decoder residual HSD
```

不要再固定：

```python
0.5 * encoder + 0.5 * decoder
```

R6 专门恢复该逻辑做消融。

---

# 23. 仍然需要保留的最少诊断

虽然这次不把工程诊断当主线，但为了让方法实验能解释，至少记录：

```text
teacher coverage
trust_change / trust_stable mean
structural code 4 channels mean
FN correction ratio
FP correction ratio
encoder loss
decoder structure loss
relation loss
correction loss
```

不需要 Run2 第一版立刻做复杂 gradient angle 系统。

只有 R5 再次组合失败时，才增加 gradient cosine 分析。

---

# 24. 对 TGRS 论文故事的建议

Run1 原故事可能是：

> SAM 提供 directional temporal structure，分别蒸馏到 Encoder 和 Decoder。

问题是 H6 已经削弱了 directional 的证据基础，而且 H4 没有支持 hierarchical synergy。

Run2 如果成功，建议改成：

> Foundation segmentation models contain rich object structure but their independent bitemporal partitions are not directly compatible with binary change detection. We therefore convert them into exchange-invariant structural residual codes and transfer the residual structure hierarchically through detachable spatial adapters. Instead of forcing change probabilities to imitate SAM structures, a residual correction mechanism selectively uses SAM-supported structure only when the lightweight student exhibits a structural error.

这个 motivation 与 Run1 的失败结果是连续的：

```text
Run1:
有结构知识
但结构表达方式和蒸馏接口不对

Run2:
重新定义 structure knowledge
重新定义 hierarchy
重新定义 correction interface
```

这比单纯“我们调了一个更好的 lambda”明显更像论文方法演进。

---

# 25. 最终推荐

## 第一优先修改

### 1. 删除正式主方法中的 appearance/disappearance regression

保留 directional 只作为 R8 对照。

---

### 2. 把 PI-DTRS 从 scalar evidence 升级为 multi-field structural residual code

这是 Run2 最重要的 teacher-side 方法改进。

---

### 3. Encoder `1×1 probe` 加 local residual spatial adapter

具体就是：

```text
1×1 + DW3×3 + PW1×1 + residual
```

而不是单纯增大 channel。

---

### 4. Decoder 新增 training-only structure side head

把结构 representation learning 从 main probability 中拆出来。

这是我认为最有希望修复：

```text
H3 SYSU 强
WHU Precision 崩
```

的关键改动。

---

### 5. SCGR 不再作为 Run2 核心模块

Run1 SCGR 可以保留为参考。

Run2 更推荐：

```text
student-error-aware residual correction
```

只纠正：

```text
SAM+GT 支持的真实 FN
SAM stable+GT 支持的真实 FP
```

---

# 26. 最值得先跑的三组

如果只能先跑三组完整 40k，我建议：

## 第一组：R0

同批 clean anchor。

---

## 第二组：R3

```text
Exchange-invariant structural code
+
Spatial Residual Encoder
```

回答：

> Encoder 方法升级本身是否成立？

---

## 第三组：R4

```text
Exchange-invariant structural code
+
Residual Decoder Structure Head
+
Residual Correction
```

回答：

> 能不能保留 H3 的 SYSU 优势，同时解决 WHU Precision？

如果 R3 和 R4 至少各自在某一方面成立，再跑 R5 Full。

如果 R4 在 WHU 仍然 Precision 明显崩：

> 先不要跑完整 Full，优先重审 trust / teacher false-change route。

---

# 27. 一句话总结 Run2

Run1 的核心问题不是：

> “SAM 没有用。”

而是：

> **当前方法把 SAM structure 过早压成 scalar directional evidence，Encoder 又没有真正的 spatial structure adapter，Decoder 则把过多结构目标直接施加到最终 change probability 上。**

Run2 最应该做的不是继续堆 loss，而是把 SAM-HSD 重构为：

\[
\boxed{
\text{Exchange-Invariant Structural Residual}
\rightarrow
\text{Spatial Residual Encoder Distillation}
\rightarrow
\text{Residual Decoder Structure Learning}
\rightarrow
\text{Selective Error Correction}
}
\]

最终仍然：

```text
Deploy = A2Net-LWGANet-L0
Params = 2.9131M
FLOPs  = 2.7475G
```

如果这条路线能够在 SYSU 保住约 +0.5～+1.0 以上增益，同时在 WHU 首次实现：

```text
ΔF1 >= +0.3
Precision >= H0
```

那么它比当前 Run1 的 directional HSD 更有资格成为 TGRS 主方法。

---

## 附：来源说明

本文方法事实与实验数字来自当前上传材料：

- `models_and_metrics.txt`：当前 `models/` 权威实现；
- `AGENTS.md`：SAM-HSD Run1 完成态、实际训练配置、正式 test 结果；
- `experiment_metrics.xlsx`：H0–H8 SYSU/WHU 指标与配置交叉核验；
- `SERVER_AND_DATASETS.md`：SYSU/WHU 数据集属性与服务器约束。

凡文中标注“建议”“推断”“假设”的内容均不是已完成实验事实，需要 Run2 消融验证。
