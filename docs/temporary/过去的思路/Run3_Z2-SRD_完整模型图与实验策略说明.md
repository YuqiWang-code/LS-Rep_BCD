# Run3 Z2-SRD：完整训练期模型图、模块解释与实验策略

> 项目：LS-Rep_BCD_RSML_3 / SAM-HSD  
> 本文只讲 **Run3 的训练期辅助机制 Z2-SRD**，不展开你已经熟悉的 A2Net + LWGANet-L0 主路径。  
> 部署结论：`switch_to_deploy()` 后仅保留原 A2Net + LWGANet-L0 主路径；Run3 的 Teacher Cache、关系结构构造、Z2 投影器、even/odd head 和全部辅助参数均被删除。

---

## 0. 本文依据的当前版本

本说明按以下两类直接证据还原：

1. GitHub 仓库 `YuqiWang-code/LS-Rep_BCD` 的 Run3 更新代码，核对的提交为 `6bfd4e17ad67b818793434eb8482380e1c62cc7a`（`Update code`，AGENTS 标注 2026-09-07 更新）。
2. 你上传的 `Run3_all_shell_scripts.txt`，其中包含 Run3 的 N0-N4、12k Stage1、40k full、smoke、dry-run 和 Teacher Cache validator 脚本。

关键代码：

- `models/a2net.py`
- `models/distill/sam_hsd/relational_structure.py`
- `models/distill/sam_hsd/temporal_evidence.py`
- `models/distill/sam_hsd/encoder_hsd.py`
- `models/distill/sam_hsd/losses.py`
- `models/distill/sam_hsd/sam_hsd_adapter.py`
- `models/datasets/cache_transforms.py`
- `models/scripts/train.py`
- `models/tools/smoke_z2_srd.py`
- `models/tools/dry_run_z2_srd.py`

GitHub 参考入口：

- [Run3 更新 commit 6bfd4e1](https://github.com/YuqiWang-code/LS-Rep_BCD/commit/6bfd4e17ad67b818793434eb8482380e1c62cc7a)
- [models/a2net.py @ 6bfd4e1](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/6bfd4e17ad67b818793434eb8482380e1c62cc7a/models/a2net.py)
- [encoder_hsd.py @ 6bfd4e1](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/6bfd4e17ad67b818793434eb8482380e1c62cc7a/models/distill/sam_hsd/encoder_hsd.py)
- [temporal_evidence.py @ 6bfd4e1](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/6bfd4e17ad67b818793434eb8482380e1c62cc7a/models/distill/sam_hsd/temporal_evidence.py)
- [sam_hsd_adapter.py @ 6bfd4e1](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/6bfd4e17ad67b818793434eb8482380e1c62cc7a/models/distill/sam_hsd/sam_hsd_adapter.py)
- [train.py @ 6bfd4e1](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/6bfd4e17ad67b818793434eb8482380e1c62cc7a/models/scripts/train.py)

---

# 1. 先用一句话说清 Run3 到底在做什么

**Z2-SRD（Swap-Group Even/Odd Structural Residual Distillation）把“交换 T1/T2”明确看成二元群 `Z2={e,s}`，把学生的双时相结构表示严格分解成：**

- **even 分量**：交换 T1/T2 后不变，负责“哪里发生了结构变化、什么结构变化、哪里稳定”；
- **odd 分量**：交换 T1/T2 后符号翻转，负责保留“结构变化的方向”。

然后让 SAM2 离线 Teacher 也生成严格对应的：

- 4 通道 **exchange-invariant even teacher code**；
- 3 通道 **exchange-anti-equivariant odd teacher code**。

二者分别监督学生的 even/odd 子空间。

最核心的研究动机不是“多加一个辅助头”，而是：

> **变化检测最终输出必须对 A/B 交换保持不变，但双时相结构差异内部确实存在方向信息。Run2 R2 用绝对差把方向全部抹掉；Run3 尝试不破坏最终交换不变性的前提下，把方向信息放进一个严格反等变的 odd 子空间中保留下来。**

这就是 Run3 相对 R2 的方法性变化。

---

# 2. 为什么是 Z2：先理解“交换群”

双时相输入只有一种非平凡操作：交换时相。

记：

- 恒等操作：`e(A,B)=(A,B)`
- 交换操作：`s(A,B)=(B,A)`

并且：

```text
s(s(A,B)) = (A,B)

=> s^2 = e

=> {e, s} 正好构成二元群 Z2
```

变化检测的最终二值变化图没有“从 A 到 B”和“从 B 到 A”两种不同答案，因此主预测理想上满足：

\[
P(A,B)=P(B,A).
\]

但是内部的“方向残差”满足另一种关系，例如：

\[
D(A,B)=F_B-F_A,
\]

交换后：

\[
D(B,A)=F_A-F_B=-D(A,B).
\]

所以同一个双时相问题天然有两种表示：

```text
EVEN / invariant
----------------
E(A,B) = E(B,A)

ODD / anti-equivariant
----------------------
O(A,B) = -O(B,A)
```

Run3 的核心就是**不再把这两种性质混在一个表示里，而是显式投影到两个子空间。**

---

# 3. Run3 总训练图

下面这张图只保留主路径的“接口”，不展开 A2Net/LWGANet 内部。

```text
                         TRAINING ONLY: Z2-SRD
                         =====================

   Image A -----------------------------------------------+
                                                          |
                                                          v
                              +------------------------------------------+
   Image B -----------------> | joint Siamese feature extraction        |
                              |                                          |
                              | [A;B] -> shared LWGANet once as 2B      |
                              |       -> split back to FA_i / FB_i       |
                              +-------------------+----------------------+
                                                  |
                     +----------------------------+----------------------------+
                     |                                                         |
                     |                                                         |
                     v                                                         v
          +----------------------+                              +------------------------------+
          | MAIN STUDENT PATH    |                              | Z2-SRD encoder auxiliary     |
          |                      |                              | training only                |
          | A2Net main path      |                              |                              |
          | (not expanded here)  |                              | Stage1: FA1, FB1             |
          |                      |                              | Stage2: FA2, FB2             |
          | produces CD masks    |                              | Stage3: FA3, FB3             |
          +----------+-----------+                              | Stage4: FA4, FB4             |
                     |                                          +---------------+--------------+
                     |                                                          |
                     v                                                          |
             +---------------+                                                  |
             | Main CD Loss  |                                                  |
             | GT supervised |                                                  |
             +-------+-------+                                                  |
                     |                                                          |
                     |                                                          |
                     |             SAM2 OFFLINE TEACHER CACHE                   |
                     |             ==========================                   |
                     |                                                          |
                     |   Cache T1 -------------------+                          |
                     |                                |                          |
                     |   Cache T2 -------------------+                          |
                     |                                v                          |
                     |                  +-----------------------------+          |
                     |                  | replay same geometry        |          |
                     |                  | scale/crop/flip/exchange    |          |
                     |                  +-------------+---------------+          |
                     |                                |                          |
                     |                                v                          |
                     |                  +-----------------------------+          |
                     |                  | Relational Structure       |          |
                     |                  | Builder                    |          |
                     |                  +-------------+---------------+          |
                     |                                |                          |
                     |                                v                          |
                     |                  +-----------------------------+          |
                     |                  | Structural Evidence Builder |          |
                     |                  +------+----------------------+          |
                     |                         |                                 |
                     |             +-----------+-----------+                     |
                     |             |                       |                     |
                     |             v                       v                     |
                     |      +-------------+          +-------------+             |
                     |      | EVEN teacher|          | ODD teacher |             |
                     |      | 4 channels  |          | 3 channels  |             |
                     |      | [0,1]       |          | [-1,1]      |             |
                     |      +------+------+          +------+------+             |
                     |             |                        |                    |
                     |             +-----------+------------+                    |
                     |                         |                                 |
                     |                         v                                 v
                     |              +----------------------------------------------+
                     |              | per-stage Z2 Structural Encoder HSD           |
                     |              |                                              |
                     |              | Probe -> Z2 Projector -> Even/Odd Heads      |
                     |              +---------------------+------------------------+
                     |                                    |
                     |                                    v
                     |                         +-------------------------+
                     |                         | L_even + L_odd          |
                     |                         | = L_Z2_raw              |
                     |                         +-----------+-------------+
                     |                                     |
                     |                                     v
                     |                         +-------------------------+
                     |                         | lambda=0.06             |
                     |                         | schedule factor         |
                     |                         | cap <= 0.12*L_main      |
                     |                         +-----------+-------------+
                     |                                     |
                     +--------------------+----------------+
                                          |
                                          v
                               +-----------------------+
                               | L_total               |
                               | = L_main + L_Z2       |
                               +-----------------------+
```

## 最重要的三点

### 3.1 Z2-SRD 只从 encoder 四层特征上取信息

Run3 不再使用 Run2 的 decoder correction，也不需要 decoder feature。

代码上 `a2net.py` 只有 `sam_hsd/eir_hsd` 会把 decoder feature 返回给 auxiliary；`z2_srd` 不在这个集合里。

所以 Run3 的方法本质是：

```text
backbone feature supervision
        |
        +-- Stage 1
        +-- Stage 2
        +-- Stage 3
        +-- Stage 4

NO decoder correction
NO teacher map injected into main feature
NO extra inference branch
```

### 3.2 Z2 auxiliary 不用 GT 做路由

`Z2SRDAdapter.forward()` 中直接：

```python
del target
```

含义是：

- GT 仍然用于主变化检测损失；
- 但 Z2-SRD 辅助监督本身只转移 SAM2 teacher structure；
- 它不使用 GT 去挑 teacher 像素，不把 GT 参与 teacher routing。

这使“结构知识转移”与主任务标签监督在机制上更干净。

### 3.3 Teacher 从始至终不进入主预测路径

Teacher 只决定训练期的 auxiliary loss。

```text
Teacher evidence ---> auxiliary loss ---> gradient ---> student encoder

Teacher evidence -X-> main feature tensor
Teacher evidence -X-> main decoder
Teacher evidence -X-> deployed model
```

---

# 4. 一个容易忽略但非常关键的模块：Joint Siamese BN

如果训练态下分别做：

```text
FA = Backbone(A)
FB = Backbone(B)
```

共享 BatchNorm 的 running statistics 会先被 A 更新，再被 B 更新。

交换输入后变成：

```text
FB = Backbone(B)
FA = Backbone(A)
```

BN 状态更新顺序也交换了。

这样即使你后面的数学表达是严格对称的，训练态仍可能出现“顺序泄漏”。

Run3 为 Z2-SRD 专门改成：

```text
             +-----------+
A ---------->|           |
             | concat    |----> [A ; B]  shape = 2B
B ---------->|           |
             +-----------+
                    |
                    v
          +--------------------+
          | shared LWGANet     |
          | ONE joint pass     |
          +---------+----------+
                    |
              feature batch 2B
                    |
             +------+------+
             |             |
             v             v
          FA[0:B]       FB[B:2B]
```

交换 A/B 后只是 2B batch 中样本排列变化，而不是先后执行两个 BN pass。

`SWA` 中共享 BN 也采用同样思路：四层 A/B feature 先在 batch 维合并，再一次经过共享 SWA，最后重新切开。

### 它解决什么？

它不是论文主创新模块，但它是**让 Z2 对称性实验成立的必要实现条件**：

> 如果训练图本身因为 BN 更新顺序产生 A/B 顺序依赖，那么后面讨论 even/odd 群表示就会被一个工程层面的非对称混杂因素污染。

### 它会不会增加部署成本？

不会。

它只改变 Z2-SRD **训练态的 batch 组织方式**，没有新增部署模块，也没有改变部署参数/FLOPs。

---

# 5. Teacher Cache 与学生增强重放图

Teacher Cache 是离线生成的，因此训练时必须让 teacher maps 和当前学生样本经历完全一致的几何变化。

```text
 RAW SAMPLE                          CACHED SAM2 STRUCTURE
 ==========                          =====================

 A image ----+                       T1 instance_id
             |                       T1 boundary
 B image ----+---- paired aug        T1 quality
             |                       T2 instance_id
 GT label ---+                       T2 boundary
                                     T2 quality
      |                                    |
      |                                    |
      v                                    v
 +---------------------+          +-----------------------+
 | Normalize           |          | load Teacher Cache    |
 | Scale               |          +-----------+-----------+
 | RandomCropResize    |                      |
 | RandomFlip          |                      |
 | RandomExchange      |                      |
 +----------+----------+                      |
            |                                 |
            | save aug_state                  |
            +---------------------+-----------+
                                  |
                                  v
                      +-------------------------+
                      | replay_teacher_pack     |
                      |                         |
                      | Scale       -> replay   |
                      | CropResize  -> replay   |
                      | Flip V/H    -> replay   |
                      | Exchange    -> swap T1/T2
                      +------------+------------+
                                   |
                                   v
                      geometrically aligned pack
```

## 插值语义

当前代码区分：

```text
instance_id : nearest interpolation
              -> 不能把离散 instance ID 插值成小数

other maps  : bilinear interpolation
```

时相交换时：

```text
if exchange == True:
    pack["t1"], pack["t2"] = pack["t2"], pack["t1"]
```

所以学生输入交换与 Teacher Cache 的 T1/T2 也同步交换。

---

# 6. Relational Structure Builder：SAM2 原始缓存怎样变成结构字段

原始 cache 核心只要求：

```text
T1/T2:
  instance_id
  boundary
  quality
```

几何 replay 结束之后，再由 `prepare_relational_pack()` 动态派生关系结构。

```text
                   replayed SAM2 cache
                           |
                           v
             +---------------------------+
             | instance_id               |
             | boundary                  |
             | quality                   |
             +-------------+-------------+
                           |
                           v
             +---------------------------+
             | Relational derivation     |
             +-------------+-------------+
                           |
       +-------------------+-------------------------------+
       |                   |               |               |
       v                   v               v               v
  occupancy           affinity_r1     affinity_r2     affinity_r4
 instance > 0        same-instance    same-instance   same-instance
                    relation r=1     relation r=2    relation r=4
       |
       +----------------------+----------------+-------------------+
                              |                |                   |
                              v                v                   v
                       interior_depth     object_scale       compactness
                       normalized         log area           4*pi*A/P^2
                       inside object
```

最终每个时相有：

```text
Structure(T)
|
+-- instance_id
+-- boundary
+-- quality
+-- occupancy
+-- affinity_r1
+-- affinity_r2
+-- affinity_r4
+-- interior_depth
+-- object_scale
+-- compactness
```

这些不是重新跑 SAM2，而是从已经 replay 对齐的 SAM2 instance map 派生。

## 每个字段在直觉上是什么

### occupancy

某像素是否被某个 SAM instance 覆盖。

```text
instance_id > 0 -> 1
background      -> 0
```

### affinity_r1 / r2 / r4

查看像素与不同距离的邻居是否属于同一个 instance。

```text
r=1 : 最局部关系
r=2 : 中近程关系
r=4 : 更粗结构关系
```

所以它表达的不是 RGB，而是“这个区域是否属于同一个结构实体”。

### interior_depth

对 instance 内部做距离变换，再按该 instance 最大距离归一化。

可理解为：

```text
boundary                         object center/interior
  0.0  ------------------------------->  1.0
```

### object_scale

由 instance 面积计算的归一化尺度表示。

### compactness

近似：

\[
C=\frac{4\pi A}{P^2}.
\]

更接近紧凑规则形状时更大，细长/不规则结构时更小。

---

# 7. Teacher Evidence Builder：从 T1/T2 结构生成 even/odd 教师

这是 Run3 Teacher 侧最重要的一层。

```text
        Structure T1                       Structure T2
             |                                  |
             +----------------+-----------------+
                              |
                              v
            +----------------------------------------+
            | ExchangeInvariantStructuralEvidence    |
            +------------------+---------------------+
                               |
                +--------------+--------------+
                |                             |
                v                             v
      +-------------------+          +--------------------+
      | EVEN evidence     |          | ODD evidence       |
      | swap invariant    |          | sign flips on swap |
      | 4 channels [0,1]  |          | 3 channels [-1,1]  |
      +-------------------+          +--------------------+
```

---

# 8. Even Teacher：4 通道对称结构码

默认 `directional_restore=False` 时：

```text
EVEN teacher structural_code

channel 0 : Boundary residual
channel 1 : Local relation residual
channel 2 : Geometry residual
channel 3 : Stable consensus
```

## 8.1 Boundary residual

先允许 `boundary_tolerance=2` 的空间容差，减少轻微边界错位造成的假变化。

概念上：

```text
B1 -------------------+
                       +--> tolerant match --> unmatched B2 = plus
B2 -------------------+                   \-> unmatched B1 = minus

boundary_even = max(plus, minus)
```

交换 T1/T2 只会交换 plus/minus，所以：

\[
\max(plus,minus)
\]

不变。

## 8.2 Local residual

\[
E_{local}
=0.55\,|A^{r1}_2-A^{r1}_1|
+0.45\,|A^{r2}_2-A^{r2}_1|.
\]

表达局部/中近程 instance 关系发生了多少变化。

## 8.3 Geometry residual

\[
\begin{aligned}
E_{geo}={}&0.25|A^{r4}_2-A^{r4}_1|\\
&+0.25|D_2-D_1|\\
&+0.30|S_2-S_1|\\
&+0.20|C_2-C_1|.
\end{aligned}
\]

其中：

- `A^r4`：较粗 relation affinity；
- `D`：interior depth；
- `S`：object scale；
- `C`：compactness。

这比单纯 boundary difference 更偏“对象几何结构变化”。

## 8.4 Stable consensus

代码先求双方共同 occupancy：

\[
O_{common}=\min(O_1,O_2),
\]

再由所有结构残差的均值抑制稳定度：

\[
E_{stable}=O_{common}(1-\overline{\Delta}).
\]

所以它回答的是：

> 两个时相都存在结构覆盖，并且多种结构属性都一致的地方在哪里？

## Even Teacher 总图

```text
 T1/T2 structural fields
          |
          +--> tolerant boundary difference ------> Boundary
          |
          +--> |affinity_r1_2-r1_1| --+
          |                            +-----------> Local
          +--> |affinity_r2_2-r2_1| --+
          |
          +--> |affinity_r4_2-r4_1| ---+
          +--> |interior_2-interior_1|  |
          +--> |scale_2-scale_1|         +---------> Geometry
          +--> |compact_2-compact_1| ----+
          |
          +--> common occupancy + mean residual ---> Stable

          => concat [Boundary, Local, Geometry, Stable]
          => E_teacher in [0,1]^4
```

并严格满足：

\[
E_T(A,B)=E_T(B,A).
\]

---

# 9. Odd Teacher：3 通道有符号结构残差

Run3 新增的是 `signed_structural_code`。

```text
ODD teacher

channel 0 : signed boundary residual
channel 1 : signed local residual
channel 2 : signed geometry residual

range     : [-1, 1]
```

这里不再做绝对值，而保留：

\[
R = field_2-field_1.
\]

交换后：

\[
R' = field_1-field_2=-R.
\]

所以每个 odd teacher channel 都满足：

\[
O_T(B,A)=-O_T(A,B).
\]

## Odd Teacher 模型图

```text
 T1/T2 structural fields
          |
          +--> signed tolerant boundary residual --> Signed Boundary
          |
          +--> (aff_r1_2-aff_r1_1) --+
          |                            +------------> Signed Local
          +--> (aff_r2_2-aff_r2_1) --+
          |
          +--> (aff_r4_2-aff_r4_1) ---+
          +--> (depth_2-depth_1)       |
          +--> (scale_2-scale_1)        +----------> Signed Geometry
          +--> (compact_2-compact_1) --+

          => concat [Signed Boundary,
                     Signed Local,
                     Signed Geometry]

          => O_teacher in [-1,1]^3
```

### 为什么 odd teacher 没有 Stable channel？

因为“稳定”本身是对称概念：

```text
stable(A,B) = stable(B,A)
```

它没有一个合理的“交换后应该取负号”的稳定方向，因此属于 even 子空间，不属于 odd 子空间。

---

# 10. Teacher Trust：不是所有 SAM 结构都一样可信

Teacher 不直接把所有像素等权监督。

```text
                    Q1 quality       Q2 quality
                         \              /
                          \            /
                           v          v
                    +---------------------+
                    | coverage + quality  |
                    +----------+----------+
                               |
         structural channels ->| disagreement
                               v
                    +---------------------+
                    | uncertainty         |
                    | symmetric           |
                    +----------+----------+
                               |
                +--------------+--------------+
                |                             |
                v                             v
          trust_change                  trust_stable
```

当前：

\[
T_{change}
=\max(Q_1,Q_2)^\gamma
\cdot coverage_{any}
\cdot(1-U),
\]

\[
T_{stable}
=\sqrt{Q_1Q_2}
\cdot coverage_{both}
\cdot(1-U),
\]

Run3 固定：

```text
trust_gamma = 1.0
```

`uncertainty` 来自 Boundary / Local / Geometry 三个 symmetric change channel 的语义不一致程度，并做 3x3 平滑。

### 对称性为什么重要？

`trust_change`、`trust_stable`、`uncertainty` 都必须满足交换不变：

\[
T(A,B)=T(B,A).
\]

否则即使 odd teacher/prediction 都正确取负，loss 权重本身若随时相顺序变化，整体辅助目标仍会破坏交换对称性。

---

# 11. 学生端 Stage Probe

学生 backbone 有四个 stage：

```text
Stage 1 : C=32
Stage 2 : C=64
Stage 3 : C=128
Stage 4 : C=256
```

每个 stage 的 A/B feature 各自经过**同一个 probe**压到 16 channel。

```text
          FA_i [C_i,H_i,W_i]
                    |
                    v
              +-----------+
              | 1x1 Conv  |
              | C_i -> 16 |
              +-----+-----+
                    |
                    v
                 zA_i

          FB_i [C_i,H_i,W_i]
                    |
                    v
              +-----------+
              | SAME probe|
              | C_i -> 16 |
              +-----+-----+
                    |
                    v
                 zB_i
```

代码类：`SpatialResidualProbe`。

但是 Run3 所有 N0-N4 都固定：

```text
spatial_adapter = False
```

所以当前实际执行的是：

```text
feature -> 1x1 projection -> z
```

**没有** Run2 R3 那种 depthwise 3x3 + pointwise spatial residual adapter。

这是有意控制变量：Run3 不再把“空间适配器是否有效”混进 Z2 表示问题。

---

# 12. Z2PairProjector：整篇 Run3 最核心的学生模块

输入是同一 stage 的 `zA, zB`。

首先不是直接做差，而是构造一个“有序对编码输入”：

\[
\phi(z_L,z_R)
=[z_L,z_R,z_R-z_L,z_L\odot z_R].
\]

然后用**同一个** ordered encoder `h(.)` 分别看两个顺序：

\[
u_{AB}=h(\phi(z_A,z_B)),
\]

\[
u_{BA}=h(\phi(z_B,z_A)).
\]

其中：

```text
h(.) = Conv1x1(64 -> 16) + GELU
```

因为 `zA,zB` 各 16 ch，四块拼接后是 64 ch。

## 完整 ASCII 图

```text
                   zA (16ch)                       zB (16ch)
                       |                                |
             +---------+---------+            +---------+---------+
             |                   |            |                   |
             |                   |            |                   |
             v                   v            v                   v
      ordered pair AB                              ordered pair BA

 +--------------------------------+      +--------------------------------+
 | [ zA, zB, zB-zA, zA*zB ]      |      | [ zB, zA, zA-zB, zB*zA ]      |
 |            64 ch               |      |            64 ch               |
 +---------------+----------------+      +---------------+----------------+
                 |                                       |
                 v                                       v
       +---------------------+                 +---------------------+
       | shared h(.)         |                 | SAME h(.)           |
       | Conv1x1 64->16      |                 | Conv1x1 64->16      |
       | GELU                |                 | GELU                |
       +----------+----------+                 +----------+----------+
                  |                                       |
                  v                                       v
                uAB                                     uBA
                  |                                       |
                  +-------------------+-------------------+
                                      |
                         +------------+------------+
                         |                         |
                         v                         v
              +-------------------+      +-------------------+
              | EVEN projection   |      | ODD projection    |
              | (uAB + uBA) / 2   |      | (uAB - uBA) / 2   |
              +---------+---------+      +---------+---------+
                        |                          |
                        v                          v
                    e_i (16ch)                 o_i (16ch)
```

交换 A/B 后：

```text
uAB <-> uBA
```

因此：

\[
e(B,A)=\frac{u_{BA}+u_{AB}}2=e(A,B),
\]

\[
o(B,A)=\frac{u_{BA}-u_{AB}}2=-o(A,B).
\]

这不是“希望网络自己学到”的性质，而是**由投影公式严格保证**。

---

# 13. 为什么不直接用 `abs(zB-zA)`？

这正是 N3/R2 与 N0/N1 的核心区别。

Run2 R2 / Run3 N3：

```text
zA,zB
  |
  +--> abs(zB-zA) ----+
  |                    |
  +--> zA*zB ----------+--> head
```

交换当然不变，但问题是：

```text
+0.8 和 -0.8
在 abs(.) 后都变成 0.8
```

方向被不可逆地丢掉。

Z2 projector 则是：

```text
ordered representation
         |
         +--> even = symmetric information
         |
         +--> odd  = antisymmetric information
```

它不是“为了方向而破坏不变性”，而是：

> **把不变信息和方向信息同时保留，但放在不同表示子空间中，分别遵守正确的交换规律。**

---

# 14. Even Head

当 `use_even=True`：

```text
 even latent e_i [16,H,W]
           |
           v
   +-------------------+
   | Conv1x1 16 -> 4   |
   | bias = True       |
   +---------+---------+
             |
             v
          sigmoid
             |
             v
  Pred even code [0,1]^4
             |
             v
   +--------------------+
   | structural_code_loss|
   +--------------------+
             ^
             |
 Teacher even code [0,1]^4
```

由于输入 `e_i` 本身严格 invariant，所以 even head 输出自然仍然 invariant。

四个预测通道：

```text
Boundary / Local / Geometry / Stable
```

---

# 15. Odd Head：为什么必须 bias-free

当 `use_odd=True`：

```text
 odd latent o_i [16,H,W]
           |
           v
   +-------------------+
   | Conv1x1 16 -> 3   |
   | bias = FALSE      |
   +---------+---------+
             |
             v
           tanh
             |
             v
  Pred odd code [-1,1]^3
             |
             v
 +-----------------------------+
 | signed_structural_code_loss |
 +-----------------------------+
             ^
             |
 Teacher odd code [-1,1]^3
```

为什么 `bias=False` 是硬要求？

假设 odd latent 交换后：

\[
o'=-o.
\]

如果 head 为：

\[
y=Wo+b,
\]

交换后：

\[
y'=W(-o)+b=-Wo+b,
\]

一般不等于：

\[
-(Wo+b)=-Wo-b.
\]

所以 bias 会破坏反等变性。

当前代码使用：

\[
y=\tanh(Wo),
\]

则：

\[
\tanh(W(-o))=-\tanh(Wo).
\]

因此 odd prediction 的反等变性从 latent 一直保持到最终 3-channel prediction。

---

# 16. Even Loss

`structural_code_loss()` 是每个 channel 的 trust-weighted Smooth L1。

```text
Pred even 4ch --------------------+
                                  |
Teacher even 4ch -- detach -------+--> SmoothL1 per channel
                                  |
trust_change ------ detach -------+    Boundary/Local/Geometry
                                  |
trust_stable ------ detach -------+    Stable
                                  |
stage channel route --------------+
```

形式上：

\[
L_{even}^{(i)}
=\sum_{c=1}^{4}r_{i,c}
\frac{\sum_x w_c(x)\,\ell_{SL1}(\hat E_c(x),E_c(x))}
{\sum_x w_c(x)+\epsilon}.
\]

其中：

```text
Boundary / Local / Geometry -> trust_change
Stable                      -> trust_stable
```

Teacher 和 trust 都 detach，不会反向进入 Teacher 构造。

---

# 17. Odd Loss

`signed_structural_code_loss()` 同样是 Smooth L1，但只有三个 change channels：

\[
L_{odd}^{(i)}
=\sum_{c=1}^{3}\tilde r_{i,c}
\frac{\sum_x T_{change}(x)\,\ell_{SL1}(\hat O_c(x),O_c(x))}
{\sum_x T_{change}(x)+\epsilon}.
\]

```text
Pred odd 3ch --------------------+
                                 |
Teacher odd 3ch -- detach -------+--> SmoothL1
                                 |
trust_change ---- detach --------+
                                 |
route first 3 channels ----------+
```

### 为什么 trust 可以是 invariant，而 teacher 是 odd？

因为 trust 只表达“这个位置的方向结构证据有多可信”，不是方向本身。

交换 A/B 后：

```text
teacher odd     : O -> -O
student odd     : O_hat -> -O_hat
trust magnitude : T -> T
```

Smooth L1 的误差大小不变，因此整个 odd loss 仍保持 exchange invariant。

---

# 18. 四个 stage 怎样加权

学生四个 encoder stage 的 loss 权重固定：

| Stage | loss 权重 |
|---|---:|
| Stage 1 | 0.35 |
| Stage 2 | 0.30 |
| Stage 3 | 0.20 |
| Stage 4 | 0.15 |

所以：

\[
L_{Z2,raw}
=\sum_{i=1}^{4}\alpha_i
\left(L^{(i)}_{even}+\lambda_{odd}L^{(i)}_{odd}\right).
\]

Run3：

```text
odd_weight = 1.0
```

不是通过调一个特殊 odd loss 权重制造收益。

---

# 19. 不同 stage 还会强调不同结构语义

同一份 teacher code 在不同深度有不同 channel routing：

| Stage | Boundary | Local | Geometry | Stable |
|---|---:|---:|---:|---:|
| 1 | 0.50 | 0.35 | 0.05 | 0.10 |
| 2 | 0.30 | 0.40 | 0.15 | 0.15 |
| 3 | 0.10 | 0.30 | 0.40 | 0.20 |
| 4 | 0.05 | 0.15 | 0.50 | 0.30 |

直觉是：

```text
shallow ------------------------------------------> deep

Boundary          Local relation       Geometry / Stable
fine detail       meso structure       coarse structure
```

odd branch 使用各 stage route 的前三项再归一化，因为它没有 Stable channel。

这部分继承自 EIR-HSD 的层级结构语义，不是 Run3 新增的调参点。

---

# 20. N0 的单个 stage 完整图

这是主方法 `N0_Z2_SRD_Full`。

```text
       FA_i                              FB_i
        |                                 |
        v                                 v
   +----------+                      +----------+
   | Probe_i  |                      | SAME     |
   | C_i->16  |                      | Probe_i  |
   +----+-----+                      +----+-----+
        |                                 |
       zA_i                              zB_i
        |                                 |
        +----------------+----------------+
                         |
                         v
             +-------------------------+
             | Z2PairProjector         |
             |                         |
             | uAB = h(phi(zA,zB))     |
             | uBA = h(phi(zB,zA))     |
             +-----------+-------------+
                         |
              +----------+----------+
              |                     |
              v                     v
        even=(uAB+uBA)/2      odd=(uAB-uBA)/2
              |                     |
              v                     v
      +---------------+       +---------------+
      | Even Head     |       | Odd Head      |
      | Conv 16->4    |       | Conv 16->3    |
      | + sigmoid     |       | no bias+tanh  |
      +-------+-------+       +-------+-------+
              |                       |
              v                       v
        Pred Even 4ch            Pred Odd 3ch
              |                       |
              v                       v
        L_even^(i)                L_odd^(i)
              |                       |
              +-----------+-----------+
                          |
                          v
             L_stage_i = L_even_i + L_odd_i
```

再在四个 stage 上加权求和。

---

# 21. N0-N4：每个实验到底在验证什么

Run3 不是五个随意变体，而是一条比较清晰的机制链：

```text
N3  R2-style invariant control
 |
 |  "显式 Z2 group projection 有必要吗？"
 v
N1  Z2 group + EVEN only
 |
 |  "在严格 invariant 表示上加入分离的 ODD 有价值吗？"
 v
N0  Z2 group + EVEN + ODD      <-- proposed full method

另外两条诊断：

N2  mixed signed control
    "如果不分子空间，只把 signed 信息混进一个 head，会怎样？"

N4  ODD only
    "方向结构单独使用是否足够？它与 EVEN 是互补还是替代？"
```

下面逐个讲。

---

# 22. N3：No Group Projection

配置：

```text
pair_mode = invariant
use_even  = True
use_odd   = False
```

训练参数：

```text
2,921,370
```

部署参数：

```text
2,913,094
```

## 模型图

```text
       zA                         zB
        |                          |
        +-----------+--------------+
                    |
        +-----------+-----------+
        |                       |
        v                       v
   abs(zB-zA)                 zA*zB
        |                       |
        +-----------+-----------+
                    |
                    v
           concat -> 32ch
                    |
                    v
      +---------------------------+
      | control head              |
      | Conv 32->16 + GELU        |
      | Conv 16->4                |
      +-------------+-------------+
                    |
                  sigmoid
                    |
                    v
             Pred even 4ch
                    |
                    v
            Even teacher loss
```

这是 **Run2 R2 的最小表示祖先**：

\[
[|z_B-z_A|, z_A\odot z_B].
\]

### N3 回答的问题

> 如果不做任何显式群投影，只用最直接的绝对差 + 共性乘积，结果能到什么程度？

它是 Run3 最重要的 ancestor control。

### 和 N1 比较

`N1 - N3` 主要回答：

> **“严格的 learned ordered-pair + group projection”是否比手工 abs/product invariant 表示更好？**

注意：N1 的训练期参数略多，因此这不是严格参数量匹配的结构比较；但部署参数完全相同，且参数差仅存在于训练辅助。

---

# 23. N1：Z2 Even Only

配置：

```text
pair_mode = group
use_even  = True
use_odd   = False
```

训练参数：

```text
2,921,882
```

## 模型图

```text
zA,zB
  |
  v
+---------------------+
| shared ordered h(.) |
| uAB / uBA           |
+----------+----------+
           |
           v
 even=(uAB+uBA)/2
           |
           v
   +---------------+
   | Even Head 4ch |
   | sigmoid       |
   +-------+-------+
           |
           v
      Even teacher
           |
           v
        L_even

odd branch: NOT CREATED
```

### N1 回答的问题

它先把“方向信息”完全拿掉，只验证 group projector 对 invariant representation 本身有没有价值。

比较：

```text
N1 vs N3
```

两者都只监督 4ch invariant teacher，区别主要是学生端的 pair representation：

```text
N3 : hand-crafted abs/product invariant
N1 : shared ordered encoder + exact Z2 even projection
```

如果 N1 明显优于 N3，支持：

> 手工绝对差不是最优的不变编码；先学习有序对，再通过群平均得到不变分量更有表达力。

如果 N1 不优于 N3，则需要谨慎：

> learned group projection 可能只是增加训练参数，并没有给 invariant branch 带来额外表示收益。

---

# 24. N0：Z2-SRD Full

配置：

```text
pair_mode = group
use_even  = True
use_odd   = True
```

训练参数：

```text
2,921,930
```

部署参数：

```text
2,913,094
```

## 模型图

```text
                    zA,zB
                      |
                      v
           +-----------------------+
           | Z2 Pair Projector     |
           +----------+------------+
                      |
          +-----------+-----------+
          |                       |
          v                       v
       EVEN                     ODD
  swap invariant          swap sign-flip
          |                       |
          v                       v
   Even Head 4ch            Odd Head 3ch
    + sigmoid              no bias + tanh
          |                       |
          v                       v
   Even Teacher             Odd Teacher
  B/L/G/Stable              B/L/G signed
          |                       |
          v                       v
       L_even                  L_odd
          |                       |
          +-----------+-----------+
                      |
                      v
             L_stage = L_even + L_odd
```

### N0 回答的问题

这是 Run3 的核心假设：

> 在保证最终辅助目标对时相交换保持一致的前提下，**单独保留一个反等变方向子空间**，是否比只有 invariant representation 更有利于学生学习变化结构？

最直接比较：

```text
N0 vs N1
```

二者有相同的 group projector 和 even branch；N0 唯一新增的核心语义是 odd supervision。

因此如果 N0 在两数据集上稳定优于 N1，且 odd branch 非退化，就形成最核心的因果证据：

> **方向信息不是应该被完全抹除，而应该被放在正确的 anti-equivariant 子空间中。**

---

# 25. N2：Mixed Signed Control

配置：

```text
pair_mode = mixed
use_even  = True
use_odd   = False
```

训练参数：

```text
2,921,370
```

这是一个**故意不满足严格 group 分离**的 negative/control，不应包装成主方法。

## 学生图

```text
       zA                         zB
        |                          |
        +------------+-------------+
                     |
         +-----------+----------+
         |                      |
         v                      v
      (zB-zA)                 zA*zB
      signed
         |                      |
         +-----------+----------+
                     |
                     v
            concat -> 32ch
                     |
                     v
             control head
          Conv-GELU-Conv 4ch
                     |
                   sigmoid
                     |
                     v
          one mixed 4ch code
```

Teacher 侧在 N2 中 `directional_restore=True`：

前三个 change channel 不再是 `abs residual`，而是把 signed teacher 从 `[-1,1]` 映射到 `[0,1]`：

\[
C_{mixed}=0.5+0.5\,O_T.
\]

Stable 仍是 symmetric。

交换后前三通道满足的是“互补”：

\[
C_{mixed}(A,B)+C_{mixed}(B,A)=1,
\]

而不是：

\[
C_{mixed}(A,B)=C_{mixed}(B,A).
\]

## N2 回答的问题

> 方向信息的收益是否只因为“有 sign”，还是必须通过严格 even/odd 子空间分离才能得到？

关键比较：

```text
N0 vs N2
```

如果 N0 优于 N2，支持：

> 关键不是简单恢复 signed residual，而是**用正确的群表示约束把 symmetric 与 antisymmetric 信息分离监督**。

如果 N2 与 N0 一样好甚至更好，则论文叙事要警惕：

> 可能“显式 Z2 分解”不是必要机制，普通 signed mixed representation 就已经足够。

当前 smoke test 也明确：N2 是唯一**不要求 auxiliary total 在交换后严格相等**的 Run3 实验。

---

# 26. N4：Z2 Odd Only

配置：

```text
pair_mode = group
use_even  = False
use_odd   = True
```

训练参数：

```text
2,921,862
```

## 模型图

```text
zA,zB
  |
  v
+---------------------+
| Z2 Pair Projector   |
+----------+----------+
           |
           +--------------------X EVEN head disabled
           |
           v
          ODD
    (uAB-uBA)/2
           |
           v
 +------------------+
 | bias-free 3ch    |
 | tanh             |
 +--------+---------+
          |
          v
     Odd Teacher
          |
          v
        L_odd
```

### N4 回答的问题

N4 不是为了证明 odd 可以代替 even，而是看：

> **odd directional structure 单独是否含有足够有效的训练信号，以及它和 even 是否是互补关系。**

理想的论文级现象通常不是“N4 最好”，而更可能是：

```text
N0 > N1
N0 > N4
```

同时 N4 本身不是完全失效。

这会支持：

> even 负责变化存在性/稳定结构，odd 负责方向残差；两者互补而非互相替代。

如果 N4 > N0，则要检查 even branch 是否引入冲突监督。

如果 N4 接近随机/明显退化，则说明 odd teacher 可能噪声过大，或者方向结构不能独立支撑有效表示。

---

# 27. N0-N4 一张表看懂

| Exp | Pair representation | Even | Odd | Teacher target | 严格 Z2 分离 | 主要研究问题 |
|---|---|---:|---:|---|---:|---|
| N3 | `abs(zB-zA), zA*zB` | Yes | No | invariant 4ch | No | R2-style ancestor 到底多强？ |
| N1 | ordered encoder + Z2 projection | Yes | No | invariant 4ch | Yes | 显式 group projection 对 invariant 表示有价值吗？ |
| N0 | ordered encoder + Z2 projection | Yes | Yes | even 4ch + odd 3ch | **Yes** | 分离的 odd 方向结构是否提供互补收益？ |
| N2 | signed `(zB-zA), zA*zB` | mixed | No | signed-mapped 4ch | **No** | “只是加入 sign”是否已经足够？ |
| N4 | ordered encoder + Z2 projection | No | Yes | odd 3ch | Yes | odd 单独是否有用/是否必须与 even 配合？ |

---

# 28. 训练参数和部署参数

当前 smoke recipe 固定的训练参数量：

| Exp | 训练参数 |
|---|---:|
| N0 | 2,921,930 |
| N1 | 2,921,882 |
| N2 | 2,921,370 |
| N3 | 2,921,370 |
| N4 | 2,921,862 |

所有实验部署后：

```text
2,913,094 params
```

并要求：

```text
switch_to_deploy() before/after main prediction max error < 1e-6
```

部署图：

```text
TRAIN
=====
Main student path
  + Z2SRDAdapter
  + teacher structure builder
  + evidence builder
  + probes
  + pair projector
  + even/odd heads

              |
              | switch_to_deploy()
              v

DEPLOY
======
Main student path ONLY

training_auxiliary : deleted
auxiliary_mode     : "none"
Teacher Cache       : absent
```

因此 Run3 的论文定位仍是：

> **训练期结构表征增强，部署零额外参数/FLOPs。**

---

# 29. 总损失是怎样接回主网络的

Run3 auxiliary raw loss：

\[
L_{Z2,raw}=\sum_i\alpha_iL_i.
\]

shell 固定：

```text
hsd_lambda    = 0.06
hsd_max_ratio = 0.12
```

训练器首先：

\[
L'_{Z2}=0.06\cdot m(t)\cdot L_{Z2,raw},
\]

然后做 cap：

\[
L_{Z2}=\min_{scale}\left(L'_{Z2},0.12L_{main}\right),
\]

实现上是 detach 后算 scale，再乘回 raw auxiliary，以避免 cap 比例反向形成额外梯度路径。

总损失：

\[
L_{total}=L_{main}+L_{Z2}.
\]

注意：这套权重与 schedule 是训练稳定控制，不应作为论文创新点。

---

# 30. Auxiliary schedule

`hsd_multiplier()`：

```text
training progress
0%        5%         15%                         80%       100%
|---------|-----------|---------------------------|----------|

OFF       linear      plateau = 1.0              cosine decay
0         warmup                                 -> 0.30
```

即：

```text
0 - 5%     : auxiliary off
5 - 15%    : linear warm-up
15 - 80%   : full factor 1.0
80 - 100%  : cosine decay from 1.0 to 0.3
```

直觉是：

- 开始先让学生主任务建立基本表示；
- 中期主要接受结构教师约束；
- 后期降低辅助强度，让最终参数更贴近主变化检测目标。

但再次强调：它是控制项，不是方法核心。

---

# 31. Run3 smoke_test 到底验证了什么

`models/tools/smoke_z2_srd.py` 不只是“跑通 forward”。

它实际上在做一套机制验收。

## 31.1 Teacher even invariance

```text
E_teacher(A,B) == E_teacher(B,A)
```

## 31.2 Teacher odd anti-equivariance

```text
O_teacher(A,B) == -O_teacher(B,A)
```

## 31.3 trust symmetry

```text
trust_change(A,B) == trust_change(B,A)
trust_stable(A,B) == trust_stable(B,A)
uncertainty(A,B)  == uncertainty(B,A)
```

## 31.4 Z2 projector exact property

```text
E_student(zA,zB) == E_student(zB,zA)
O_student(zA,zB) == -O_student(zB,zA)
```

## 31.5 分支激活语义

```text
N0 : even > 0, odd > 0
N1 : even > 0, odd = 0
N2 : odd = 0
N3 : odd = 0
N4 : even = 0, odd > 0
```

## 31.6 auxiliary 不改变主输出定义

同一模型：

```text
compute_auxiliary=False
vs
compute_auxiliary=True
```

主预测要求完全一样。

## 31.7 梯度真的到达学生 backbone 和 auxiliary

检查：

```text
nonzero finite gradient -> backbone
nonzero finite gradient -> training_auxiliary
```

所以不是“branch 被创建了但没有训练”。

## 31.8 主输出 A/B swap invariance

smoke 对 N0-N4 检查：

```text
main(A,B) ~= main(B,A)
```

容差 `5e-6`。

## 31.9 auxiliary loss swap invariance

除故意作为 mixed control 的 N2 之外：

```text
L_aux(A,B) ~= L_aux(B,A)
```

容差 `2e-6`。

## 31.10 train-mode BN state symmetry

专门检查 joint Siamese pass 后：

- output；
- auxiliary；
- BN buffers；
- gradients；

在交换时相后保持一致到训练态可接受容差。

Blackwell CUDA reduction 次序存在浮点差异，因此 train-state diagnostic 的容差比纯数学 projector 检查宽；这不等同于放宽 deploy 等价要求。

## 31.11 deploy

所有 N0-N4：

```text
deploy params = 2,913,094
max_error < 1e-6
```

---

# 32. Real-cache dry run 在验证什么

`dry_run_sysu.sh` 默认可以用：

```text
N0 / batch 64 / 30 steps / real SYSU cache
```

它比 synthetic smoke 多验证一层真实数据链：

```text
Dataset list
   |
   v
TeacherCache manifest
   |
   +--> source dataset match?
   +--> teacher_type == sam2_struct_v2?
   +--> cache coverage exactly matches train list?
   |
   v
replay augmentation
   |
   v
prepare_relational_pack
   |
   v
real Z2-SRD forward/backward
```

并记录：

```text
z2_even
z2_odd
z2_even_feature
z2_odd_feature
z2_even_odd_cosine
teacher_odd_abs_mean
teacher_odd_signed_mean
code_boundary/local/geometry/stable
trust_change / trust_stable
coverage
uncertainty
aux cap ratio
memory / timing
```

所以 Stage 0 最有价值的不是看 30 step F1，而是确认：

> **真实 Teacher Cache 中 odd 信号确实存在、梯度非零、内存正常、replay 无错、loss 没有退化。**

---

# 33. 当前 Run3 实验协议

从上传的 `Run3_all_shell_scripts.txt` 可直接确定：

```text
Datasets     : SYSU-CD-256, WHU-CD-256
Batch size   : 64
Full steps   : 40000
Stage1 steps : 12000
Seed default : 2333
Optimizer    : Adam
LR           : 5e-4
Weight decay : 1e-4
Dice         : batch reduction
hsd_lambda   : 0.06
hsd_max_ratio: 0.12
Teacher      : validated SAM2 structural cache
```

Run3 每个运行的输出目录把 `max_steps` 和 `seed` 编进路径：

```text
<experiment>/<dataset>/steps_<max_steps>/seed_<seed>/
```

所以：

```text
12k screening
```

不会误恢复成：

```text
40k full run
```

这是很重要的实验隔离设计。

---

# 34. 实验执行策略总图

```text
                   +-------------------------+
                   | Cache Validator         |
                   | SYSU + WHU              |
                   +------------+------------+
                                |
                                v
                   +-------------------------+
                   | Z2 smoke test           |
                   | algebra/gradient/deploy |
                   +------------+------------+
                                |
                                v
                   +-------------------------+
                   | Real-cache dry run      |
                   | SYSU, e.g. N0 30 steps  |
                   +------------+------------+
                                |
                                v
            +------------------------------------------+
            | Stage 1: 12k mechanism screening        |
            |                                          |
            | SYSU: N3 -> N1 -> N0 -> N2              |
            | WHU : N3 -> N1 -> N0 -> N2              |
            +----------------------+-------------------+
                                   |
                         mechanism survives?
                                   |
                    +--------------+--------------+
                    |                             |
                   NO                            YES
                    |                             |
                    v                             v
            stop / diagnose              +--------------------------+
                                         | Full 40k matrix          |
                                         | N3 -> N1 -> N0 -> N2 -> N4
                                         | SYSU + WHU               |
                                         +------------+-------------+
                                                      |
                                                      v
                                         +--------------------------+
                                         | formal final test block  |
                                         +------------+-------------+
                                                      |
                                                      v
                                         publication-level repeats
                                         for shortlisted comparisons
```

---

# 35. 为什么 Stage1 顺序是 N3 -> N1 -> N0 -> N2

上传脚本对 SYSU 和 WHU 都固定：

```text
N3 -> N1 -> N0 -> N2
```

不是随机顺序。

## 第一步 N3：先确定最小祖先

```text
N3 = R2-style invariant representation
```

先知道没有 group projection 时，当前代码/当前服务器/当前 Teacher Cache 下的学习轨迹。

## 第二步 N1：只改变 group projection

```text
N3 -> N1
```

先回答最干净的问题：

> group-based even representation 是否比 abs/product 有价值？

## 第三步 N0：再加入 odd

```text
N1 -> N0
```

只在已经有严格 even representation 的基础上加入 odd branch，回答：

> anti-equivariant directional information 是否提供额外信息？

## 第四步 N2：最后跑 mixed signed control

这一步是防止论文出现一个很危险的替代解释：

> “N0 变好其实不需要 Z2 分解，只要把 signed difference 喂进去就行。”

N2 就是专门反驳或验证这个替代解释。

---

# 36. 为什么 Stage1 没有 N4

Stage1 脚本只跑：

```text
N3 N1 N0 N2
```

没有 N4。

这是合理的，因为早期筛选首先要回答三件核心问题：

```text
1. explicit group projection 是否有价值？   N1 vs N3
2. separated odd 是否有价值？              N0 vs N1
3. separation 是否比 mixed sign 更重要？   N0 vs N2
```

N4 的问题是：

```text
odd alone 是否足够？
```

它是第二层机理诊断，而不是判断主方法是否值得继续训练的第一优先级问题。

因此 N4 只放进完整 40k 矩阵更省实验预算。

---

# 37. 12k Stage1 应该怎么看，不要只盯 12k 的最终 F1

Stage1 是**机制筛选**，不是最终论文结果。

应该同时看：

## 主任务轨迹

```text
F1
IoU
Kappa
Precision
Recall
```

重点是不同模型在同样 12k budget 下的**学习趋势和跨数据集方向**。

## Z2 内部诊断

### `z2_even`

even teacher supervision 是否正常下降。

### `z2_odd`

N0/N4 的 odd supervision 是否真正学习。

### `z2_even_feature`

Even latent 的平均绝对幅度。

### `z2_odd_feature`

Odd latent 的平均绝对幅度。

如果长期接近 0，要警惕 odd collapse。

### `z2_even_odd_cosine`

是 even/odd latent 的相关性诊断。

注意：

> Z2 投影保证的是**交换作用下的 even/odd 变换性质**，并没有数学上强制两个张量在欧氏内积下正交。

因此 cosine 不是“必须越接近 0 越好”的硬目标，代码也没有拿它做 loss。它只是看两个分支是否出现高度同质化的经验诊断。

### `teacher_odd_abs`

Teacher odd 方向信号的平均绝对强度。

如果几乎为 0，说明数据/Teacher 在当前 batch 中没有足够方向结构信号，N0 很难从 odd branch 得益。

### `teacher_odd_signed`

有符号均值。

它接近 0 **不代表 odd teacher 没信号**，因为正负方向会互相抵消。必须与 `teacher_odd_abs` 一起看。

### `aux_ratio`

若长期接近 0.12，表示 auxiliary 经常触发主损失 12% cap。

这时即使 raw odd loss 很大，实际回传的 auxiliary 总强度也被截断，解释 N0/N1 差异时要结合 cap 频率。

---

# 38. Stage1 的最关键判定逻辑

## 假设 H1：group projection 有效

比较：

```text
N1 vs N3
```

支持 H1 的理想证据：

- SYSU/WHU 收益方向一致；
- 不只是 Recall 单边拉高导致 Precision 大降；
- IoU/Kappa 与 F1 同方向；
- 训练曲线不是单次抖动。

如果 N1 ≈ N3：

> 显式 group projector 对 invariant branch 的增益证据弱。

如果 N1 < N3：

> learned ordered-pair projection 可能引入不必要自由度，或者 12k 下更难优化。

---

# 39. 假设 H2：分离的 odd 方向表示有效

比较：

```text
N0 vs N1
```

如果 N0 稳定优于 N1，并且：

```text
z2_odd > 0
odd feature non-collapse
teacher_odd_abs non-trivial
```

则最直接支持 Run3 主假设。

如果 N0 ≈ N1：

> odd 信息可能没有给 binary change detection 提供额外有效监督，或者 Teacher odd 质量/覆盖不足。

如果 N0 < N1：

可能有三种主要解释：

1. odd teacher 含有过多方向噪声；
2. even/odd 同时优化发生梯度竞争；
3. auxiliary cap 使新增 odd 只是重新分配固定辅助梯度预算，并没有增加有效信息。

这时不要先调 `odd_weight` 包装成新方法，应先用日志诊断是哪一种。

---

# 40. 假设 H3：真正重要的是“分离”，不是简单 signed residual

比较：

```text
N0 vs N2
```

如果：

```text
N0 > N2
```

而 N2 已经有 signed input，那么可以排除一个关键替代解释：

> “只要把方向符号恢复回来就行。”

这时论文可以更有底气地说：

> **结构方向信息需要在与交换群一致的反等变子空间中被建模，而不是与 invariant evidence 混合编码。**

如果：

```text
N2 >= N0
```

则必须承认：

> 当前实验不能证明显式 group separation 比普通 signed mixed representation 更必要。

这是 Run3 最关键的可证伪设计之一。

---

# 41. Full 40k 策略

上传脚本对两个数据集完整队列都为：

```text
N3 -> N1 -> N0 -> N2 -> N4
```

即每个 dataset 5 个实验。

单 seed full matrix：

```text
5 experiments x 2 datasets = 10 full runs
```

Stage1：

```text
4 experiments x 2 datasets = 8 screening runs
```

如果 Stage1 和 full 都执行：

```text
总计 18 个训练运行
```

不包括 smoke / dry run / cache validator。

---

# 42. Full 40k 后最重要的五组比较

建议按这个顺序解读，而不是按 F1 排名讲故事。

## Comparison A：N1 vs N3

```text
learned exact group-even
vs
hand-crafted abs/product invariant
```

回答 group projection 的价值。

## Comparison B：N0 vs N1

```text
even + odd
vs
even only
```

回答 odd 的增量价值。

## Comparison C：N0 vs N2

```text
separated group even/odd
vs
mixed signed representation
```

回答“群分解是否必要”。

## Comparison D：N0 vs N4

```text
even + odd
vs
odd only
```

回答 even/odd 是互补还是替代。

## Comparison E：N3 与 Run2 R2

如果训练协议/代码版本/服务器条件严格可比，这可以用于检查 Run3 ancestor 是否忠实复现 R2-style mechanism。

但如果 Run2 旧归档与 Run3 当前 RSML-3 条件不同，只能作为机制历史参考，不能直接混成同一正式统计表宣称因果提升。

---

# 43. 最理想的论文结果模式是什么

不是简单的：

```text
N0 F1 最高
```

而是形成一条因果链：

```text
N3  <  N1  <  N0
 |       |      |
 |       |      +-- separated odd gives extra signal
 |       +--------- exact group-even is useful
 +----------------- minimal invariant ancestor

同时：
N2 < N0            -> sign alone is insufficient
N4 < N0            -> even and odd are complementary
```

并且这一方向最好：

```text
SYSU : consistent
WHU  : consistent
```

这才是完整的方法论证闭环。

---

# 44. 几种非常重要的反例结果，以及应该怎么解释

## 情况 1：N1 > N3，但 N0 <= N1

说明：

```text
exact group-even 有价值
odd directional supervision 没有增量价值
```

此时可以考虑把论文主线收缩为**learned group invariant projection**，而不是强行保留 odd。

## 情况 2：N0 > N1，但 N2 >= N0

说明方向信息有价值，但：

```text
严格 even/odd 分离的必要性没有被证明
```

此时 Z2 分解的新颖性叙事会变弱。

## 情况 3：N0 在 SYSU 好、WHU 差

不能说“Z2-SRD 普适提升”。

优先检查：

- teacher_odd_abs；
- trust_change；
- coverage；
- object scale 分布；
- Precision/Recall trade-off；
- auxiliary cap 使用率。

这可能意味着方向结构教师对数据集形态敏感。

## 情况 4：N4 >= N0

说明 even branch 可能与 odd branch 竞争，或 even teacher 带来冗余/冲突。

这会推翻“even+odd complementary”假设，需要重新解释。

## 情况 5：N0/N1/N3 都接近

说明 Run3 当前 representation change 的效应小于单 seed 随机波动可能范围。

不能通过挑 best checkpoint 文件名来扩大差异，必须增加 seed 才能判断。

---

# 45. 当前脚本里的 GPU 策略要特别注意

文件名还保留：

```text
run_gpu0_sysu.sh
run_gpu1_whu.sh
```

但当前脚本注释和默认参数实际上都使用：

```text
physical GPU 1
```

而最新 AGENTS 更新也记录：

```text
GPU count = 2
currently_available_ids = [1]
```

所以这里必须区分：

```text
script filename != actual currently assigned GPU
```

当前如果只有 GPU 1 可用，不应同时把 SYSU 和 WHU 两条 batch=64 队列都无约束塞到同一块卡上。

更稳妥的当前策略：

```text
GPU 1:
  Stage1 SYSU queue
  -> Stage1 WHU queue

或者按你当前服务器实际空闲情况分时运行
```

如果之后 GPU 0 重新可用，则最自然的是：

```text
GPU 0 : SYSU queue
GPU 1 : WHU queue
```

保持两个数据集各自串行，避免单卡同时跑两个大 batch 正式训练。

---

# 46. Resume 与完成判定

Run3 目录结构把 checkpoint 与 log 分开，但运行标识一致。

如果存在：

```text
last_checkpoint.pth
```

则精确恢复。

如果只有：

```text
best_model_F1=*.pth
```

而没有 `last_checkpoint.pth`，脚本直接拒绝不精确恢复。

一个运行只有同时出现：

```text
=== TEST RESULTS ===
=== END TEST RESULTS ===
```

才判定为 complete。

因此：

```text
训练到 40000/40000
!=
正式实验完成
```

最终论文指标只能从完整 formal test block 中提取。

---

# 47. 论文级结果最终应该看什么

主指标至少：

```text
Recall
Precision
OA
F1
IoU
Kappa
```

不要只看 F1。

尤其 Run3 要看：

## Precision/Recall 是否失衡

Odd directional supervision 可能增强变化敏感性，但也可能引入更多 FP。

如果：

```text
Recall +1.0
Precision -1.5
F1 only +0.05
```

不能简单写成“有效提升”。

## IoU/Kappa 是否同向

如果 F1 小幅升而 IoU/Kappa 不升，需要检查是不是阈值附近 trade-off，而不是整体分割质量提升。

## 跨数据集方向

至少 SYSU/WHU 都支持同一机制，论文叙事才更强。

## 多 seed

当前脚本默认 seed 2333，但支持外部传入 seed。

当前一轮 N0-N4 可以作为机制筛选；真正准备投稿时，建议对 shortlist 至少做 3 seeds，报告均值±标准差。

优先重复：

```text
N3  minimal ancestor
N1  group-even
N0  full Z2-SRD
```

如果 N2 是论文中用于排除替代解释的关键 control，也应进入多 seed；N4 可根据 full 40k 的诊断价值决定。

这里不替当前 shell 虚构另外两个 seed 数字；应在正式统计前统一预注册并对所有对照使用完全相同的 seeds。

---

# 48. Z2-SRD 的论文叙事可以怎样讲

可以把逻辑压缩成四步。

## Step 1：问题

二时相 binary change detection 的输出天然应满足 temporal exchange invariance：

\[
P(A,B)=P(B,A).
\]

## Step 2：Run2 R2 的不足

绝对差：

\[
|F_B-F_A|
\]

保证不变，但把：

\[
F_B-F_A
\]

中的方向信息完全抹掉。

## Step 3：Run3 的关键观察

方向信息并不是“非法信息”。

真正需要的是：

```text
final task semantics -> invariant
internal directional residual -> anti-equivariant
```

因此应该把两者分开，而不是二选一。

## Step 4：方法

学习共享的 ordered-pair representation，然后用 Z2 group projection：

\[
E=\frac{h(A,B)+h(B,A)}2,
\qquad
O=\frac{h(A,B)-h(B,A)}2.
\]

再用 SAM2 teacher 的 symmetric structural code 和 signed structural residual 分别监督两个子空间。

训练结束全部删除，因此部署零开销。

---

# 49. 与 Run2 R2 的最本质差异

## R2

```text
zA,zB
 |
 v
abs(zB-zA) + zA*zB
 |
 v
one invariant structural representation
 |
 v
4ch teacher structure
```

核心是假设：

> 对变化检测有用的 teacher structural difference 应该全部变成 exchange invariant evidence。

## Run3 N0

```text
zA,zB
 |
 v
shared ordered pair encoder
 |
 +--> EVEN: invariant structure
 |
 +--> ODD : anti-equivariant directional structure
 |
 v
separate teacher supervision
```

核心假设变成：

> **最终变化语义应该不变，但表示空间不必只含 invariant 信息；方向残差可以保留，只要它被隔离在具有正确群作用的 odd 子空间。**

这比“R8 把 signed residual 重新塞回一个 head”更有明确的数学结构。

---

# 50. 为什么 N2 非常重要

如果论文只有：

```text
R2 abs diff
vs
N0 even+odd
```

审稿人可以问：

> “你这个收益是不是因为重新用了 signed difference？为什么一定需要 group theory / even-odd decomposition？”

N2 正是回答这个问题。

```text
N2 : 有 sign，但没有严格分离
N0 : 有 sign，而且放在严格 odd subspace
```

所以：

```text
N0 > N2
```

比单独的：

```text
N0 > N3
```

对“方法为什么有效”的论证价值更高。

---

# 51. 为什么 N1 也不可缺

如果只有 N0 vs N3，变化同时包含：

```text
1. hand-crafted invariant -> learned group projection
2. no odd -> odd branch
```

无法知道提升来自哪一个。

N1 把两步拆开：

```text
N3 -> N1 : projection effect
N1 -> N0 : odd effect
```

这才是可归因的消融链。

---

# 52. 为什么 N4 是“必要但次优先级”的诊断

N0 > N1 只能说明加 odd 后变好了。

还不能完全回答：

> odd 是作为一个独立强监督替代 even，还是与 even 真正互补？

N4 把 even 拿掉。

如果最终出现：

```text
N0 > N1
N0 > N4
N1 and N4 both useful
```

就更符合“两个 irreducible components 互补”的叙事。

---

# 53. 当前方法不应该怎样讲

不建议把 Run3 描述成：

> “我们加入两个辅助头，一个学习不变特征，一个学习方向特征。”

这会显得非常像普通模块堆叠。

更准确的表述是：

> **我们把 temporal exchange 明确定义为 Z2 group action，通过共享 ordered-pair encoder 与解析式 group projection 将双时相结构表示分解为 trivial/even 和 sign/odd 两个表示分量，并让 SAM2 teacher 的 symmetric / signed structural residual 分别监督它们。**

区别在于：

- even/odd 不是两个随便设计的 branch；
- 它们的交换性质由公式严格保证；
- Teacher targets 也具有匹配的群变换性质；
- odd head 甚至必须 bias-free 才不破坏反等变；
- N2 专门验证普通 signed mixing 是否足以替代这种分解。

这才是 Run3 的方法贡献核心。

---

# 54. 一张“从输入到 loss”的最终总图

```text
+=============================================================================+
|                               RUN3 Z2-SRD TRAIN                             |
+=============================================================================+

 Images
 ------
 A -------------------------+
                            +--> joint 2B Siamese feature extraction
 B -------------------------+             |
                                          +--> FA1 FB1
                                          +--> FA2 FB2
                                          +--> FA3 FB3
                                          +--> FA4 FB4
                                                  |
                                                  |
 Teacher cache                                     | Student structural branch
 -------------                                     |
 T1 instance/boundary/quality                      |
 T2 instance/boundary/quality                      |
          |                                        |
          v                                        |
 replay exact augmentation                         |
 scale/crop/flip/exchange                          |
          |                                        |
          v                                        |
 relational fields                                 |
 occ/aff_r1/r2/r4/depth/scale/compact             |
          |                                        |
          v                                        |
 +---------------------------+                     |
 | Structural Evidence       |                     |
 +-------------+-------------+                     |
               |                                   |
       +-------+--------+                          |
       |                |                          |
       v                v                          |
 EVEN teacher       ODD teacher                    |
 4ch [0,1]          3ch [-1,1]                    |
 invariant          anti-equivariant               |
       |                |                          |
       |                |       for each stage i   |
       |                |                 +--------+--------+
       |                |                 |                 |
       |                |                 v                 v
       |                |               FA_i              FB_i
       |                |                 |                 |
       |                |               probe             probe
       |                |                 |                 |
       |                |                zA_i              zB_i
       |                |                 +--------+--------+
       |                |                          |
       |                |                          v
       |                |              shared ordered encoder h
       |                |                 uAB              uBA
       |                |                  |                |
       |                |          +-------+--------+-------+
       |                |          |                |
       |                |          v                v
       |                |      EVEN latent       ODD latent
       |                |     (uAB+uBA)/2       (uAB-uBA)/2
       |                |          |                |
       |                |          v                v
       |                |       sigmoid           tanh
       |                |       4ch head       bias-free 3ch
       |                |          |                |
       +----------------+----------+                |
                        |                           |
                     L_even                     L_odd
                        |                           |
                        +------------+--------------+
                                     |
                                     v
                           weighted 4-stage L_Z2_raw
                                     |
                             lambda * schedule
                                     |
                              cap <= 0.12 L_main
                                     |
                                     v
                             +----------------+
 Main CD loss -------------->| L_total        |
                             +----------------+
```

---

# 55. 最后用最直白的话总结整个方法

你可以把 Z2-SRD 想成这样：

以前 R2 对 A/B 做：

```text
"我只关心变化大小，不关心方向"
```

所以直接：

```text
abs(B-A)
```

方向被彻底扔掉。

Run3 说：

```text
"最终变化图确实不应该关心 A/B 谁在前，
 但这不代表中间的方向结构信息没用。"
```

于是它把信息拆成两盒：

```text
BOX 1: EVEN
A/B 换位置也不变
负责：变化强度、局部关系、几何变化、稳定结构

BOX 2: ODD
A/B 换位置就整体变号
负责：方向性的 boundary/local/geometry residual
```

Student 和 SAM2 Teacher 都按同一个规则拆。

然后：

```text
Teacher EVEN -> 教 Student EVEN
Teacher ODD  -> 教 Student ODD
```

这样做的目标是：

> **既不牺牲变化检测任务要求的交换不变性，又不必像绝对差那样把所有方向结构信息丢掉。**

训练结束后，这两个盒子、Teacher Cache 和所有辅助 head 全部拆掉，留下原来的 A2Net + LWGANet-L0。

---

# 56. 当前实验策略的最简记忆版

如果你之后给导师讲，建议记住下面五句话：

```text
N3：不用群投影，只做 R2 式 abs/product，给最小祖先。

N1：加入严格 Z2 投影，但只保留 even，验证“群不变表示”本身。

N0：在 N1 上加入独立 odd，验证“方向结构的互补价值”，这是主方法。

N2：有 signed difference 但不做 even/odd 分离，排除“只是 sign 有用”的替代解释。

N4：只保留 odd，验证 odd 是否独立有信息以及 even/odd 是否互补。
```

Stage1：

```text
N3 -> N1 -> N0 -> N2
12k, SYSU + WHU
```

Full：

```text
N3 -> N1 -> N0 -> N2 -> N4
40k, SYSU + WHU
```

最终最有论文价值的证据链不是单纯 `N0 best`，而是：

```text
N1 > N3     : exact group-even representation 有效
N0 > N1     : separated odd 提供增量信息
N0 > N2     : 严格分离比简单 mixed sign 更重要
N0 > N4     : even 与 odd 是互补，而非 odd 单独替代
```

并且这些方向在 SYSU/WHU、F1/IoU/Kappa、Precision/Recall 上尽量保持一致，随后用多 seed 确认方差。

---

# 57. 源码定位速查

| 功能 | 文件 | 关键类/函数 |
|---|---|---|
| Run3 接入主模型 | `models/a2net.py` | `A2Net_LWGANet_L0`, `extract_pair_features`, `_forward_main_path`, `switch_to_deploy` |
| Cache replay | `models/datasets/cache_transforms.py` | `replay_teacher_pack`, `_replay_tensor` |
| 动态关系字段 | `models/distill/sam_hsd/relational_structure.py` | `prepare_relational_pack`, `RelationalStructureBuilder` |
| even/odd teacher | `models/distill/sam_hsd/temporal_evidence.py` | `ExchangeInvariantStructuralEvidence` |
| probe / Z2 投影 / heads | `models/distill/sam_hsd/encoder_hsd.py` | `SpatialResidualProbe`, `Z2PairProjector`, `Z2StructuralEncoderHSD` |
| even/odd loss | `models/distill/sam_hsd/losses.py` | `structural_code_loss`, `signed_structural_code_loss` |
| Run3 总适配器 | `models/distill/sam_hsd/sam_hsd_adapter.py` | `Z2SRDAdapter` |
| N0-N4 配方与总损失 | `models/scripts/train.py` | `EXPERIMENTS`, `build_model`, `hsd_multiplier`, `capped_auxiliary` |
| 数学/梯度/deploy smoke | `models/tools/smoke_z2_srd.py` | `evidence_acceptance`, `projector_acceptance`, `recipe_acceptance`, `train_state_acceptance` |
| 真缓存短训练 | `models/tools/dry_run_z2_srd.py` | real-cache dry run |
| 当前实验脚本 | `train_scripts/SAM-HSD/Run3/` | `run_experiment.sh`, Stage1/full queues, validators |

---

# 58. 最终结论

Run3 已经从 Run2 的“**如何构造更好的结构监督组合**”收缩成一个更集中、也更像论文方法问题的研究主线：

> **双时相交换不变任务中，是否应该把结构知识严格分解为 Z2-even 与 Z2-odd 两类表示，而不是用绝对差丢弃方向、或把 signed 信息混在一个普通表示里？**

N0-N4 的设计正好围绕这一个问题逐级拆解：

```text
hand-crafted invariant
        -> learned exact even
        -> separated even + odd
        -> mixed signed negative control
        -> odd-only complementarity control
```

因此下一步最重要的不是继续加模块，而是让现有实验真正回答：

1. **N1 是否稳定优于 N3？**
2. **N0 是否稳定优于 N1？**
3. **N0 是否优于 N2？**
4. **N4 是否证明 odd 有信息但不能替代 even？**
5. **这些结论是否在 SYSU 与 WHU 上一致，并经多 seed 复现？**

只要这条证据链成立，Run3 的论文叙事会比 Run2 “多个辅助模块组合”更集中：核心贡献不是损失权重，也不是部署网络变大，而是**利用 temporal swap group 的表示结构，在训练期把 SAM2 的对称结构知识和反对称方向结构知识放到正确的子空间中蒸馏，并保持部署零开销。**
