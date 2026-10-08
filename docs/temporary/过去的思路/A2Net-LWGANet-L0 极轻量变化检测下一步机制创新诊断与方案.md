# A2Net-LWGANet-L0 极轻量变化检测下一步机制创新诊断与方案

## 1 现状诊断

### 1.1 结论

当前最值得解决的已经不是“如何设计更强 Teacher”，而是 **Student 主路径的双时相表征在发生有效交互之前，就被过早压缩为绝对差分**。

现有部署主图为：

```text
T1 ── Shared LWGANet ── SWA ──┐
                               ├─ TFM ── Decoder ── Change Map
T2 ── Shared LWGANet ── SWA ──┘
```

其中 `T1/T2` 分别独立执行 backbone 与 `NeighborFeatureAggregation`；真正第一次发生跨时相交互是在 `TemporalFeatureFusionModule`。而该模块的第一步直接是：

```python
x = torch.abs(x1 - x2)
```

之后才进行 dilation=7/5/3/1 的卷积链。也就是说，在现有 A2Net 中，**没有任何显式的跨时相分布校准、语义对应或交互发生在差分之前**。一旦光照、季节、纹理、成像条件或局部表观差异已经进入 `F1/F2`，它们和真实变化会一起被转换成“差分能量”。后续 TFM 只能学习“这个差值像不像变化”，却无法再知道该差值究竟来自跨时相域偏移还是语义变化。

**证据来源：** `[代码：models/a2net.py::_forward_main_path / extract_pair_features；models/decoder/a2net_decoder.py::TemporalFeatureFusionModule.forward]`

### 1.2 Teacher 方向失败并非“Teacher 不够复杂”，而是 Teacher 实际已经接近失效

Run3 同协议、seed=2333 的结果为：

| Dataset | B0 | R3A（SAM） | R3O（OV） | R3（SAM+OV） | R3−B0 |
|---|---:|---:|---:|---:|---:|
| SYSU | 81.75 | 82.29 | 82.76 | 83.13 | +1.38 |
| WHU | 94.06 | 93.94 | 93.33 | 93.70 | -0.36 |
| CDD | 97.79 | 97.78 | — | 97.79 | 0.00 |
| LEVIR | 91.13 | 91.03 | — | 90.98 | -0.15 |

因此，当前证据只能支持“Teacher 在 SYSU 上有单数据集正信号”，不能支持稳定增益。尤其 WHU 上 SAM-only、OV-only、SAM+OV 三个 Teacher 臂全部低于 B0，并且 OV-only 最差。这已经排除了“只是选错了某个 Foundation prior”的简单解释。

**证据来源：** `[指标：README.md §4 Run3 首轮正式结果与 R3O 消融；train_scripts/RDT-CD/Run3/README.md]`

更关键的是日志诊断：

| Run | `aux_raw` | `dynamic_teacher_gain` | `pixel_reject_ratio` |
|---|---:|---:|---:|
| R3 / SYSU | 0.0009 | -0.0003 | 0.9961 |
| R3A / SYSU | 0.0003 | -0.0001 | 0.9978 |
| R3 / WHU | ≈0 | ≈0 | 0.9991 |
| R3A / WHU | ≈0 | ≈0 | 0.9993 |

这说明实际进入 Student 的 Teacher 监督几乎为零。代码上 Fast Teacher 又采用“初始 residual=0 → Teacher 初始等于 Student”的零初始化，并在其后使用 positive-Brier-gain audit；当 Foundation prior 本身比当前 Student 差时，大量像素自然会被 audit 拒绝。于是形成：

```text
Foundation prior 与 Student 差距大
        ↓
Residual Teacher 很快接近 Student
        ↓
gain ≈ 0
        ↓
99.6%~99.9% pixel 被拒绝
        ↓
KD 梯度接近消失
```

这不是继续增加 router、Teacher 数量或 KD loss 能解决的问题。

**证据来源：** `[代码：models/distill/dynamic_teacher.py::ResidualTeacherExpert / BTSAMRDT；models/distill/task_space.py；指标：train_scripts/RDT-CD/Run3/README.md]`

### 1.3 当前 Student 的主要机制瓶颈

| 优先级 | 诊断 | 直接证据 | 研究含义 |
|---|---|---|---|
| P0 | 部署隔离本身正确 | Teacher/Cache 只观察 decoder feature 与 prediction，不进入 `_forward_main_path`；`switch_to_deploy()` 删除训练辅助 | 无需重构部署框架 |
| P1 | **跨时相交互发生过晚** | Backbone、SWA 对两时相独立执行 | Student 在差分前没有 pair-aware 表征 |
| P1 | **第一种跨时相运算就是绝对差分** | `x=torch.abs(x1-x2)` | 成像域偏移与真实变化一起进入 Difference |
| P1 | Teacher 已无可靠剩余空间 | `dyn_gain≈0`、`reject≈0.999` | 不应继续 Teacher/KD |
| P2 | 数据集特征差异极大 | SYSU 变化像素 21.1%，WHU 3.4%，LEVIR 4.1%，CDD 11.9% | 方法必须同时经受高变化率与稀疏小目标测试 |
| P2 | CDD 已接近饱和 | B0 F1=97.79 | 不能围绕 CDD 的零点几提升设计机制 |
| P2 | 当前关键结论仍主要来自单 seed | seed=2333 | 最终论文结论仍需多 seed |

这里尤其值得注意 WHU/LEVIR：两者变化比例仅约 3%–4%，真实变化本身就是稀疏信号。相比 SYSU，它们更容易出现“少量伪变化 FP 就足以抵消大量 TP”的情况。**这并不能直接证明当前误差一定全部来自时相域偏移，但它与当前 `abs(F1-F2)` 缺乏差分前校准的结构瓶颈是吻合的，因此形成了一个可以直接证伪的研究假设。**

**证据来源：** `[代码：models/decoder/a2net_decoder.py；数据：RSML-3 数据统一说明；指标：B0 四数据集正式结果]`

---

## 2 文献调研

### 2.1 双时相域差异、伪变化与差分前对齐

| 工作 | 年份 / Venue | 层级与状态 | 代码 | 与本项目的相关性 |
|---|---|---|---|---|
| **Feature Spectrum Learning for Remote Sensing Change Detection (FeaSpect)** | 2025, CVPR | **CCF-A，已正式发表** | 截至 2026-09-20，CVF 官方页面仅给出 paper/supp，未检索到作者官方公开代码 | 直接把 imaging-environment 导致的 pseudo-change 视为核心问题，并在 feature spectrum 中做 style alignment；说明“先抑制时相域偏移再检测变化”是 2025 CCF-A 明确研究方向。 |
| **Exploring the Cross-Temporal Interaction: Feature Exchange and Enhancement for Remote Sensing Change Detection (ExNet)** | 2024, IEEE JSTARS | **SCI 期刊，非 CCF-A；已正式发表** | 未检索到作者官方 GitHub | 使用跨时相 feature statistics exchange、domain-correlated embedding 交换和频域增强缓解双时相 heterogeneity；与当前 A2Net“先独立提取、后直接 abs”形成直接对照。 |
| **BiFA: Remote Sensing Image Change Detection With Bitemporal Feature Alignment** | 2024, IEEE TGRS | **SCI 权威期刊，非 CCF-A；已正式发表** | [官方 GitHub：BiFA](https://github.com/zmoka-zht/BiFA?utm_source=chatgpt.com) | 从 channel、spatial、multi-scale 三个层面对双时相特征做 alignment，目标就是降低 illumination / perspective 等 irrelevant factors；证明“alignment before decision”具有直接 CD 依据，但其完整 flow/implicit alignment 对 2.9M 模型过重。 |
| **Enhancing VMamba for Change Detection via Lightweight Feature Interaction and Selection (VMI-CD)** | 2026, Pattern Recognition | **SCI 权威期刊，非 CCF-A；2026 Volume 172 正式发表** | [作者 GitHub：VMI-CD](https://github.com/ptdoge/VMI-CD?utm_source=chatgpt.com)；仓库已建立，但当前尚为空仓库 | 2026 工作进一步把问题归结为 bi-temporal domain gap，并设计 **parameter-free BFIM**；这说明在极轻量场景中，跨时相交互完全可以优先于“加更大的网络”。 |

### 2.2 极轻量变化检测与表征能力

| 工作 | 年份 / Venue | 层级与状态 | 代码 | 与本项目的相关性 |
|---|---|---|---|---|
| **Robust Feature Aggregation Network for Lightweight and Effective Remote Sensing Image Change Detection (RFANet)** | 2024, ISPRS JPRS | **SCI 权威期刊，非 CCF-A；已正式发表** | [官方 GitHub：RFANet](https://github.com/Youzhihui/RFANet?utm_source=chatgpt.com) | 面向 lightweight CD，用 FRM 做跨尺度细节/语义补偿，并通过轻量 decoder 抑制 background/pseudo-change；其官方实现还明确参考了 A2Net，并覆盖 LEVIR/WHU/CDD/SYSU，因此与本项目非常接近。 |
| **RepViT: Revisiting Mobile CNN From ViT Perspective** | 2024, CVPR | **CCF-A，已正式发表** | [官方 GitHub：RepViT](https://github.com/THU-MIG/RepViT?utm_source=chatgpt.com) | 说明轻量网络的优势不来自无约束堆 attention，而来自针对移动端重新设计高效 block。对本项目的启示是：若瓶颈可以用零参数/低算力操作修复，就没有必要用 Transformer 或大模块替代 2.9M backbone。 |
| **ChangeMamba: Remote Sensing Change Detection With Spatiotemporal State Space Model** | 2024, IEEE TGRS | **SCI 权威期刊，非 CCF-A；已正式发表** | [官方 GitHub：ChangeMamba](https://github.com/ChenHongruixuan/ChangeMamba?utm_source=chatgpt.com) | 从显式 spatio-temporal modeling 方向证明时序建模的重要性；但替换成 Mamba 会同时改变 backbone、参数结构和 CUDA extension 依赖，不适合作为当前 2.9M 固定 Student 的第一步。 |

### 2.3 文献证据的共同指向

上述 2024–2026 工作虽然实现不同，但高相关部分高度集中于三点：

1. **双时相输入并非天然处在可直接做差的统一特征域。** FeaSpect、ExNet、BiFA、VMI-CD 都从不同角度处理 domain/style/pseudo-change。  
2. **跨时相 interaction/alignment 应出现在差异判别之前，而不仅是差分后的 decoder 中。**
3. 对极轻量模型，**parameter-free 或极低成本 interaction 比增加大模块更符合部署约束**；VMI-CD 的 parameter-free interaction 与 RFANet 的 lightweight 设计尤其支持这一判断。

因此，文献与当前代码诊断共同支持下一步从：

```text
“如何给已有 difference 添加更多知识”
```

转向：

```text
“如何让进入 difference 的两个 temporal feature
首先处在更可比较的表示域中”
```

---

## 3 唯一最可行的下一步方案

### 3.1 方案名称

**SCTC：Unchanged-Aware Symmetric Cross-Temporal Calibration**

中文工作名：

**不变区域感知的对称跨时相校准**

核心思想只有一句：

> **不再让 A2Net 直接对两个独立特征做绝对差；先利用双时相自身的高一致区域估计“成像域偏移”，把 T1/T2 对称地映射到同一个公共特征域，再交给原有 TFM。**

这不是 Teacher、不是蒸馏、不是调 loss、不是增加 decoder，也不改变 LWGANet。

### 3.2 创新动机

当前 A2Net：

\[
F_1^s,F_2^s
\overset{\text{TFM}}{\longrightarrow}
|F_1^s-F_2^s|
\]

隐含假设是：

\[
F_1^s \text{ 与 } F_2^s
\]

已经具有充分可比的 feature distribution。

但当前代码没有显式机制保证这一点。

对于 WHU/LEVIR 这类变化像素只占约 3%–4% 的数据，大多数区域实际上是不变背景，因此可以反过来利用这一 CD 特性：

> **高跨时相一致区域不是需要检测的对象，而可以作为当前 image pair 的“内部标尺”，用于估计两个时间点之间的 nuisance/domain shift。**

这是二时相变化检测特有的可利用信息，而普通单图分割不存在这一条件。

### 3.3 机制定义

对于 SWA 输出的第 \(s\) 个尺度：

\[
F_1^s,F_2^s\in
\mathbb{R}^{B\times C\times H_s\times W_s}
\]

#### 第一步：构造无参数、交换对称的“不变可信度”

对每个空间位置计算两时相 channel vector 的 cosine agreement：

\[
a^s(h,w)
=
\frac{1+\cos
\left(
F_1^s(:,h,w),
F_2^s(:,h,w)
\right)}{2}.
\]

因此：

\[
a^s\in[0,1].
\]

高值表示两时相当前特征方向高度一致，更可能属于稳定区域；低值表示真实变化、配准误差或强域差异。

使用：

\[
w^s=\operatorname{stopgrad}(a^s)
\]

作为统计权重，避免 backbone 通过主动操纵权重来“钻空子”。

注意：

\[
w(F_1,F_2)=w(F_2,F_1)
\]

因此整个过程天然满足时间交换对称。

#### 第二步：只利用高一致区域估计两时相统计量

对每个 channel：

\[
\mu_t^s
=
\frac{
\sum_{h,w}w^s(h,w)F_t^s(h,w)
}{
\sum_{h,w}w^s(h,w)+\epsilon
},
\quad t\in\{1,2\}
\]

以及：

\[
(\sigma_t^s)^2
=
\frac{
\sum_{h,w}
w^s(h,w)
\left(
F_t^s(h,w)-\mu_t^s
\right)^2
}{
\sum_{h,w}w^s(h,w)+\epsilon
}.
\]

与普通 InstanceNorm 不同，这里的统计量不是由整张 feature map 等权计算，而是优先由**双时相共同稳定区域**确定。

#### 第三步：建立公共、无方向偏置的 pair domain

不把 T1 强行变成 T2，也不把 T2 强行变成 T1，而使用二者中点：

\[
\mu_*^s
=
\frac{
\mu_1^s+\mu_2^s
}{2},
\]

\[
\sigma_*^s
=
\sqrt{
\frac{
(\sigma_1^s)^2+
(\sigma_2^s)^2
}{2}
+\epsilon
}.
\]

然后：

\[
\widetilde F_t^s
=
\sigma_*^s
\frac{
F_t^s-\mu_t^s
}{
\sigma_t^s+\epsilon
}
+\mu_*^s.
\]

最终只把：

\[
\widetilde F_1^s,
\widetilde F_2^s
\]

送入**原有、完全不改结构的 TFM**：

\[
D^s=
\left|
\widetilde F_1^s
-
\widetilde F_2^s
\right|.
\]

后面的 dilation=7/5/3/1 分支和 Decoder 全部保持不变。

### 3.4 新的数据流

```text
T1 ── LWGANet ── SWA ───────────────┐
                                     │
                       ┌─────────────▼─────────────┐
                       │  Unchanged-Aware SCTC    │
                       │                           │
                       │  cosine agreement         │
                       │        ↓                  │
                       │ stable-region statistics  │
                       │        ↓                  │
                       │ shared symmetric domain   │
                       └───────┬───────────┬───────┘
                               │           │
                              F̃1          F̃2
                               └─────┬─────┘
                                     ↓
                            Original A2Net TFM
                              |F̃1 - F̃2|
                                     ↓
                              Original Decoder
                                     ↓
                                Change Map

T2 ── LWGANet ── SWA ───────────────┘
```

### 3.5 为什么这个方案最适合当前硬约束

**参数量：**

SCTC 只有 cosine、weighted mean/std 与 affine transform：

\[
\Delta Params=0.
\]

因此部署参数仍应严格保持：

\[
2,913,094.
\]

**FLOPs：**

SCTC 只作用于 SWA 后四个 64-channel feature map，新增的是 element-wise reduction 与 affine 运算，而不是卷积/attention。按当前四尺度空间尺寸估算，新增运算属于**数百万标量运算量级，远低于 0.01G 的量级**；这是工程估算而不是正式 FLOPs 结果。正式实现必须同时执行 THOP 与 functional-op 手工审计，要求最终仍落在项目约定的约 `2.7475G ± 0.03G` 范围，否则直接判定不满足部署约束。

**Teacher / Cache：**

```text
SAM Cache:       不需要
OV Cache:        不需要
Foundation Model:不需要
EMA Teacher:     不需要
```

**部署图：**

SCTC 是 Student 本身的零参数 temporal operation，不属于训练辅助，因此训练、验证、测试、部署都使用同一 SCTC。

**恢复兼容性：**

SCTC 没有 state_dict 参数，因此旧 B0 权重理论上可以无新增 missing/unexpected parameter 加载；但实验模式必须写入 checkpoint/log，恢复时必须核验 `temporal_calibration_mode`，禁止把 B0 与 SCTC checkpoint 静默互换实验语义。

### 3.6 时间交换一致性

这是该方案必须坚持的硬性质。

对于交换输入：

\[
(F_1,F_2)\rightarrow(F_2,F_1)
\]

agreement 不变：

\[
a(F_1,F_2)=a(F_2,F_1),
\]

公共统计量也不变，而两个 calibrated branch 只发生交换：

\[
(\widetilde F_1,\widetilde F_2)
\rightarrow
(\widetilde F_2,\widetilde F_1).
\]

因此：

\[
|
\widetilde F_1-\widetilde F_2
|
=
|
\widetilde F_2-\widetilde F_1
|.
\]

正式 smoke 必须验证：

```text
max | Model(A,B) - Model(B,A) | < 1e-6
```

这比引入 directional cross-attention 更符合当前二值 CD 的任务对称性。

### 3.7 与已有工作的实质区别

**与 ExNet 不同。** ExNet 将一个时间点的统计特性交换给另一个时间点，并进一步提取/交换 DCE，再增加 frequency enhancement；SCTC 不执行 T1→T2 或 T2→T1 的方向性交换，而是通过不变区域估计当前 pair 的 nuisance statistics，并同时映射到**共同中点域**，从结构上保证 swap symmetry。

**与 FeaSpect 不同。** FeaSpect 在频域学习 feature spectrum transformation，并需要可学习 extraction mechanism；SCTC 不做 FFT、不学习 spectrum、不增加可学习模块，只针对 A2Net 当前“difference 前缺少 temporal calibration”这一具体缺陷。

**与 BiFA 不同。** BiFA 解决 channel/spatial/multiscale alignment，并包含 differential flow / implicit alignment 思路；SCTC 明确不处理 spatial warp，只解决更轻量的 **cross-temporal distribution nuisance**，因而更符合固定 2.9M 的部署条件。

**与 VMI-CD 不同。** VMI-CD 的 BFIM 服务于 Siamese VMamba encoder，并另有 CS2M 做 channel-spatial selection；SCTC 不替换当前 CNN backbone，而是在 **A2Net 的 SWA 与 TFM 之间**显式利用“unchanged-majority”这一 BCD 任务结构估计公共时相域。

因此论文叙事不是：

```text
“我们也设计了一个 feature interaction module”
```

而应当是：

> **现有极轻量 Siamese CD 在独立特征直接差分时，会把跨时相域偏移编码成伪变化；我们利用变化检测中大量空间位置跨时相保持稳定这一任务属性，从稳定区域中自估计 nuisance statistics，并以严格交换对称、零参数方式在差分前建立 pair-specific common feature domain。**

### 3.8 可证伪假设

**H1 — 伪变化抑制假设**

如果当前 WHU/LEVIR 的瓶颈确实部分来自 difference 前的 feature-domain mismatch，则 SCTC 应主要降低背景 FP：

\[
Precision\uparrow
\]

且 Recall 不应明显下降。

第一阶段在 WHU 上要求：

```text
ΔF1 ≥ +0.30
Precision 明显提高
Recall 下降 ≤ 0.20
```

否则不支持核心机制。

**H2 — 稀疏变化特异性假设**

由于 weighted statistics 主要依赖稳定区域，低变化比例 WHU 应比高变化比例 SYSU 获益更明显。

若再次出现：

```text
SYSU 明显提升
WHU 明显下降
```

则说明它仍只是 dataset-specific trick，应停止。

**H3 — unchanged-aware 必要性假设**

如果收益只是来自普通 normalization，则“不变区域权重”没有创新价值。

因此必须比较：

```text
S1: 全像素对称统计校准
S2: unchanged-aware 对称统计校准
```

若两个数据集上：

```text
|F1(S2)-F1(S1)| < 0.10
```

则不能把 unchanged-aware estimation 作为有效贡献点。

**H4 — 多数据集机制一致性**

第一阶段 SYSU + WHU 通过后扩展 CDD + LEVIR。最终至少应满足：

```text
4 个数据集中 ≥3 个正增益
且任一数据集 ΔF1 不低于 -0.20
```

否则不得表述为稳定改进。

### 3.9 最小消融

只需要三个 Student-only 实验臂，不再跑任何 Teacher：

| ID | Difference 前操作 | 唯一变量 | 目的 |
|---|---|---|---|
| S0 | 无 | 原始 B0 | clean anchor |
| S1 | 全像素 symmetric calibration | 是否做基础 pair calibration | 排除“普通 normalization 就够了” |
| S2 | **Unchanged-Aware SCTC** | 加入 agreement-weighted statistics | 完整主方法 |

所有其它条件保持：

```text
batch       = 64
steps       = 40000
seed        = 2333
optimizer   = 与 clean B0 相同
loss        = 原始 BCE + Dice
backbone    = LWGANet-L0
decoder     = 原始 A2Net decoder
Teacher     = OFF
Cache       = OFF
```

启动顺序应为：

```text
第一阶段：
S0/S1/S2 × WHU
S0/S1/S2 × SYSU

机制门通过
        ↓

第二阶段：
S0/S1/S2 × LEVIR
S0/S1/S2 × CDD

四数据集通过
        ↓

2333 / 3407 / 5871 多 seed
```

### 3.10 失败判据

出现以下任一情况就应停止继续扩展 SCTC，而不是继续加模块：

- WHU `S2 ≤ S0`；
- SYSU 正、WHU/LEVIR负，重现此前 Teacher 的数据集方向性；
- Precision 上升但 Recall 明显下降，最终 F1 无提升；
- `S2≈S1`，说明 unchanged-aware weighting 没有独立贡献；
- 只有一个 seed 出现 `<0.2` 的微小正差；
- swap test 不满足 `<1e-6`；
- 部署参数不再等于 `2,913,094`；
- 实际 arithmetic audit 使 256×256 FLOPs 超出既定约束。

### 3.11 为什么不选其它候选

**不选继续 Teacher/KD：**已有同协议结果和 `dynamic_teacher_gain≈0 / reject≈0.999` 已经给出直接否定证据，再增加 Teacher、router 或 loss 缺乏前置有效性。

**不直接做 FeaSpect 式频域模块：**FeaSpect 的学术依据很强，但 FFT + learnable spectrum extraction 会把一个局部瓶颈变成新的部署模块；对 2.7475G 固定模型不是当前风险最低的第一步。

**不直接做 BiFA 式 flow alignment：**空间对齐能力更强，但新增 alignment network/flow machinery 与固定 2.9M 的冲突更明显。

**不换 ChangeMamba/VMI-CD backbone：**这会同时改变 backbone 表征、时序机制、参数结构和环境依赖，无法回答“原 A2Net 究竟缺什么”；当前 `lsrep` 也没有必要为第一次机制验证重新引入 Mamba CUDA 路径。

**不继续加强 RFANet 式跨尺度聚合：**当前 A2Net 已经存在 `NeighborFeatureAggregation`，继续增加 multiscale aggregation 更可能形成模块叠加，而不是修复已定位到的“temporal interaction before difference”缺口。RFANet 更适合作为轻量设计对照，而不是下一步直接复制。

因此，当前唯一优先方案应当是：

> **先用零参数、交换对称、unchanged-aware 的 SCTC 修正现有 A2Net 的“独立表征 → 直接绝对差分”瓶颈；只有该机制在 WHU/SYSU 两个相反数据分布上同时通过，才值得向四数据集和多 seed 扩展。**

---

## 4 需要修改的代码清单

| 路径 |
|---|
| `models/decoder/temporal_calibration.py` |
| `models/decoder/a2net_decoder.py` |
| `models/a2net.py` |
| `models/scripts/train.py` |
| `models/tools/smoke_temporal_calibration.py` |