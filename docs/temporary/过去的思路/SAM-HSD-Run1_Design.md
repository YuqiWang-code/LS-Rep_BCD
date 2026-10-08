# SAM-HSD Run1 升级版：Directional Temporal Relational Structure Distillation

> **文档版本**: SAM-HSD Run1 完整升级版  
> **核心设计**: 方向化关系结构证据场（PI-DTRS）+ 非对称层次结构蒸馏（A-HSD）+ 结构冲突感知梯度路由（SCGR）  
> **部署约束**: 训练期辅助，部署时严格恢复原始 A2Net_LWGANet_L0 计算图，保持 **2.9131M / 2.7475G** 推理开销

---

## 目录

1. [最终三项核心贡献](#一最终三项核心贡献)
2. [完整框架](#二完整框架)
3. [PI-DTRS 详细构造](#三pi-dtrs-的详细构造)
4. [Encoder-HSD 设计](#四encoder-hsd-设计)
5. [Decoder-HSD 设计](#五decoder-hsd-设计)
6. [总损失与训练设置](#六总损失与训练设置)
7. [部署验证与工程实现](#七部署验证与工程实现)
8. [实验设计](#八实验设计)
9. [与现有工作的差异](#九与现有工作的差异)
10. [总结](#十总结)

---

## 一、最终三项核心贡献（论文级定义）

### Contribution 1 — PI-DTRS: Permutation-Invariant Directional Temporal Relational Structure

将独立生成的 SAM instance partitions 转换为与实例编号无关的关系结构场，再分解为：

**E⁺, E⁻, E⁰, Eᵤ** 分别表示：

- **E⁺**: T2 相比 T1 **出现**的结构证据
- **E⁻**: T1 相比 T2 **消失**的结构证据
- **E⁰**: **稳定**结构证据
- **Eᵤ**: 低质量/冲突/不确定结构

它不是简单的 `SAM(T1) - SAM(T2)` 变化先验，而是一个**双时相结构证据张量**，保留了方向信息。

### Contribution 2 — A-HSD: Asymmetric Hierarchical Structural Distillation

**Encoder**: 学习**方向性结构变化**（E⁺, E⁻, E⁰），保留 richer directional knowledge

**Decoder**: 由于 A2Net TFM 本身计算 `|F₁ - F₂|` 后融合，天然时间交换对称，因此 Decoder 学习折叠后的**对称变化证据** `E_Δ = max(E⁺, E⁻)` 以及稳定证据 `E⁰`

这种非对称设计完全适配 **Siamese encoder + difference decoder** 的架构特性。

### Contribution 3 — SCGR: Structural Conflict-aware Gradient Routing

不丢弃 SAM 与 GT 不一致的样本，而是将像素分成四种训练语义：

1. **可靠正样本**: GT=change & SAM 高变化证据
2. **可靠负样本**: GT=unchanged & SAM 高稳定证据
3. **硬假变化负样本**: GT=unchanged & SAM 高变化证据（hard false-change negative）
4. **SAM 遗漏硬正样本**: GT=change & SAM 低变化证据（SAM-missed hard positive）

**SAM 正确时作为教师，冲突时转化为硬样本梯度路由**，提升对 SYSU 等数据集的鲁棒性。

---

**附带特性**: 所有 PI-DTRS generator、Encoder probes、结构损失仅存在于训练期，部署严格恢复原始 A2Net_LWGANet_L0。**Burden-free** 成为整个方法的设计属性，而非独立贡献。

---

## 二、完整框架

```
Stage A: Offline SAM2 AMG
────────────────────────────────────
T1 ──→ SAM2 AMG ──→ instance_id1 / boundary1 / quality1
T2 ──→ SAM2 AMG ──→ instance_id2 / boundary2 / quality2
                          │
                          ↓
                  cached structure only

Stage B: Student Training
────────────────────────────────────
cached T1/T2 SAM structure
          │
          ↓
┌─────────────────────────────────────┐
│ PI-DTRS Generator                   │
│                                     │
│ instance partition                  │
│ → boundary                          │
│ → affinity                          │
│ → interior depth                    │
│ → scale / shape                     │
│                                     │
│ temporal relational comparison      │
│ → E+ / E- / E0 / Eu                │
└─────────────────────────────────────┘
          │
     ┌────┴─────┐
     ↓          ↓
Encoder-HSD   Decoder-HSD
     │          │
     └─────loss─┘
          ↓
       gradient
          ↓

T1 ─┐
    ├→ Shared LWGANet-L0
T2 ─┘
       ↓
      SWA
       ↓
      TFM
       ↓
 A2Net Decoder
       ↓
 change prediction
```

**关键约束**: PI-DTRS 不注入主特征、不写入 TFM、不直接参与 Decoder 输入。仅通过 **auxiliary loss → gradient** 影响训练。

---

## 三、PI-DTRS 的详细构造

### 3.1 单时相结构场：Relational Structural Field

对于时相 `t ∈ {1, 2}`，构造：

```
Rₜ = {Bₜ, Oₜ, Aₜ¹, Aₜ², Aₜ⁴, Dₜ, Sₜ, Cₜ}
```

其中：

- **Bₜ**: 实例边界（已有 cache 提供）
- **Oₜ**: occupancy，`Oₜ(p) = 1[IDₜ(p) > 0]`
- **Aₜʳ**: permutation-invariant affinity（详见下）
- **Dₜ**: interior depth（距离边界归一化）
- **Sₜ**: object scale（面积对数归一化）
- **Cₜ**: shape compactness（`4πA/P²`）

### 3.2 Permutation-Invariant Affinity

不能比较 `ID₁(p) = ID₂(p)`，因为两时相 instance 编号无对应关系。定义单时相内部关系：

```
Aₜᵟ(p) = 1[IDₜ(p) = IDₜ(p+δ) > 0]
```

取三个尺度 `r ∈ {1, 2, 4}`（水平、垂直、对角邻域平均），得到 `Aₜ¹, Aₜ², Aₜ⁴`。这样实例编号的排列不会影响结果。

### 3.3 Interior Depth

从实例区域计算 distance-to-boundary：

```
Dₜ(p) = dist(p, ∂Iₜ) / (d_max + ε)
```

clamp 到 `[0, 1]`。描述**边界 → 近内部 → 深内部**的结构变化。

### 3.4 Object Scale Field

对 replay 后的每个实例计算面积 `aₖ`，像素赋值为：

```
Sₜ(p) = log(1 + aₖ) / log(1 + H×W)
```

必须在 crop/resize replay 后重新计算，不缓存原始面积。

### 3.5 Shape Compactness

对于实例 `k`，计算：

```
Cₖ = 4πAₖ / (Pₖ² + ε)
```

然后赋值给该实例的所有像素。

### 3.6 时序关系结构证据的构造

#### Step 1: 单尺度比较

对于每个结构场分量，定义方向比较：

```
δ⁺(X) = max(X₂ - X₁, 0)  # 出现证据
δ⁻(X) = max(X₁ - X₂, 0)  # 消失证据
δ⁰(X) = min(X₁, X₂)      # 稳定证据
```

#### Step 2: 多尺度融合

分别对 `{B, O, A¹, A², A⁴, D, S, C}` 计算 `δ⁺, δ⁻, δ⁰`，得到 24 个单尺度证据。

#### Step 3: 加权聚合

```
E⁺ = w_B·δ⁺(B) + w_O·δ⁺(O) + w_A1·δ⁺(A¹) + w_A2·δ⁺(A²) + w_A4·δ⁺(A⁴) 
     + w_D·δ⁺(D) + w_S·δ⁺(S) + w_C·δ⁺(C)

E⁻ = w_B·δ⁻(B) + w_O·δ⁻(O) + ...

E⁰ = w_B·δ⁰(B) + w_O·δ⁰(O) + ...
```

**初始权重建议**:
- `w_B = 0.25` (边界最重要)
- `w_O = 0.15` (occupancy 基础)
- `w_A1 = 0.12`, `w_A2 = 0.10`, `w_A4 = 0.08` (多尺度亲和)
- `w_D = 0.10` (内部深度)
- `w_S = 0.12`, `w_C = 0.08` (尺度和形状)

#### Step 4: 不确定性估计

```
Eᵤ = quality_conflict(Q₁, Q₂) + topology_conflict(ID₁, ID₂)
```

其中：

```python
quality_conflict = max(1 - Q₁, 1 - Q₂)
topology_conflict = smoothed_mismatch(E⁺, E⁻, E⁰)
```

#### Step 5: Tolerance & Support 鲁棒滤波

```
E⁺ ← E⁺ · 1[E⁺ > τ_tolerance] · 1[support_votes(E⁺) > 2]
E⁻ ← E⁻ · 1[E⁻ > τ_tolerance] · 1[support_votes(E⁻) > 2]
```

其中 `τ_tolerance = 0.15`，`support_votes` 计算 3×3 邻域内有多少像素同样为高证据。

### 3.7 对称变化证据（Decoder 用）

```
E_Δ = max(E⁺, E⁻)
```

Decoder 只关心"是否变化"，不关心"出现还是消失"。

---

## 四、Encoder-HSD 设计

### 4.1 核心思想

LWGANet-L0 四个 stage 输出 `F₁ⁱ, F₂ⁱ` (i ∈ {1,2,3,4})，它们是**独立编码**的，不知道 T1/T2 时序关系。

**目标**: 让 Encoder 在每个 stage 学习**方向性结构变化**：
- 哪里是 E⁺（出现）区域
- 哪里是 E⁻（消失）区域
- 哪里是 E⁰（稳定）区域

### 4.2 结构 Probe

对每个 stage `i`，定义轻量 probe：

```python
probe_i = nn.Conv2d(C_i, 16, 1, bias=False)  # 1×1 conv
```

**参数量**: 4 个 stage × (32+64+128+256) × 16 = **7,680 参数**

### 4.3 方向性蒸馏目标

对于 stage `i`，构造三个教师目标：

```python
T⁺_i = F.interpolate(E⁺, size=F₁ⁱ.shape[-2:], mode='bilinear')
T⁻_i = F.interpolate(E⁻, size=F₁ⁱ.shape[-2:], mode='bilinear')
T⁰_i = F.interpolate(E⁰, size=F₁ⁱ.shape[-2:], mode='bilinear')
```

### 4.4 分离时相预测

```python
S₁ⁱ = probe_i(F₁ⁱ)  # [B, 16, H_i, W_i]
S₂ⁱ = probe_i(F₂ⁱ)
```

**三个预测分支** (共享 probe 参数):

```python
# 分支 1: 出现预测（T2 应该强，T1 应该弱）
P⁺_i = σ(channel_pool(S₂ⁱ)) - σ(channel_pool(S₁ⁱ))

# 分支 2: 消失预测（T1 应该强，T2 应该弱）
P⁻_i = σ(channel_pool(S₁ⁱ)) - σ(channel_pool(S₂ⁱ))

# 分支 3: 稳定预测（T1 和 T2 都应该强）
P⁰_i = σ(channel_pool(S₁ⁱ)) · σ(channel_pool(S₂ⁱ))
```

其中 `channel_pool` 是 `mean(dim=1)` 或可学习的 1×1 conv 到 1 通道。

### 4.5 Encoder 损失

```python
L_E = Σᵢ [w_i · (ℓ_smooth(P⁺_i, T⁺_i) + ℓ_smooth(P⁻_i, T⁻_i) + ℓ_smooth(P⁰_i, T⁰_i))]
```

其中：
- `w₁ = 0.10, w₂ = 0.20, w₃ = 0.30, w₄ = 0.40` (深层权重更高)
- `ℓ_smooth` 是 smooth L1 loss

### 4.6 小实例平衡

对于 scale field `S < 0.3` 的区域，权重 ×1.5。

---

## 五、Decoder-HSD 设计

### 5.1 核心思想

A2Net Decoder 输出四个尺度的 `(pᵢ, mᵢ)`，其中：
- `pᵢ`: progressive prediction
- `mᵢ`: final mask

Decoder 天然计算 `|F₁ - F₂|`，因此**时间对称**。

**目标**: 让 Decoder 学习：
- 变化区域的**边界精度**（boundary）
- 变化区域的**关系一致性**（relational）
- 实例内的**亲和性**（affinity）

### 5.2 SCGR: Structural Conflict-aware Gradient Routing

将像素分为 4 类：

#### Type 1: 可靠正样本

```
M_rp = (GT = 1) & (E_Δ > 0.5) & (Eᵤ < 0.3)
```

**损失**: 标准 BCE

#### Type 2: 可靠负样本

```
M_rn = (GT = 0) & (E⁰ > 0.5) & (Eᵤ < 0.3)
```

**损失**: 标准 BCE

#### Type 3: 硬假变化负样本

```
M_hn = (GT = 0) & (E_Δ > 0.5)
```

**损失**: 加权 BCE，权重 `w_hard_neg = 1.5`

#### Type 4: SAM 遗漏硬正样本

```
M_hp = (GT = 1) & (E_Δ < 0.3)
```

**损失**: 加权 BCE，权重 `w_hard_pos = 2.0`

**SCGR 损失**:

```python
L_SCGR = BCE(pred, GT, weight=pixel_weight)
```

其中：

```python
pixel_weight = 1.0 · M_rp + 1.0 · M_rn + 1.5 · M_hn + 2.0 · M_hp
```

### 5.3 边界损失

```python
L_boundary = Σᵢ w_scale_i · boundary_dice_loss(pᵢ, mᵢ, GT, E_Δ, B_teacher)
```

其中：

```python
B_teacher = max(B₁, B₂)  # 两时相边界 union
B_pred = soft_boundary(pred)
B_GT = soft_boundary(GT)

# 只在 GT 边界 band 内监督
Band_GT = dilate(B_GT, kernel=7)
weight = B_teacher · Band_GT · E_Δ

loss = smooth_l1(B_pred, 1.0, reduction='none') · weight
```

### 5.4 关系损失（新增）

**目标**: 约束变化区域内的**空间一致性**。

如果两个像素：
- 都在 GT 变化区域内
- SAM 说它们属于同一个实例（`A₁ > 0.8` 或 `A₂ > 0.8`）
- 质量高（`Q > 0.7`）

则它们的预测应该接近：

```python
L_rel = Σ_4-neighbors |pred(p) - pred(q)| · weight(p, q)
```

其中：

```python
weight(p, q) = (GT(p) = 1) · (GT(q) = 1) · max(A₁(p,q), A₂(p,q)) · min(Q₁, Q₂)
```

### 5.5 亲和性损失

**目标**: 约束同一 SAM 实例内部的预测一致性。

与 Run4 的 affinity loss 类似，但增加了方向判断：

```python
# 只在变化证据 E_Δ 高的区域约束
L_affinity = affinity_loss(pred, E_Δ, A₁, A₂, Q₁, Q₂)
```

### 5.6 Decoder 总损失

```python
L_D = 0.35·L_SCGR + 0.25·L_boundary + 0.25·L_rel + 0.15·L_affinity
```

---

## 六、总损失与训练设置

### 6.1 总损失

```python
L = L_main + λ_HSD(t) · L_HSD
```

其中：

```python
L_HSD = 0.50·L_E + 0.50·L_D
```

`λ_HSD` 采用 warmup + cosine decay 策略。

### 6.2 权重与 Cap

- **λ_max = 0.06** (可消融 0.04 / 0.06 / 0.08)
- **Aux ratio cap**: `L_HSD_eff ≤ 0.12·L_main`

### 6.3 训练 Schedule

- **0–5%**: 仅主任务
- **5–15%**: linear warmup
- **15–80%**: full strength
- **80–100%**: cosine decay 到 `0.3·λ_max`

### 6.4 训练参数

- **Encoder probes**: 约 **7,680 参数** (4 个 scale 的 1×1 conv 到 16 通道)
- **Decoder relation loss 参数**: **0**
- **总训练参数**: < 10K
- **训练模型**: ≈ 2.92M (实现后需精确统计)

---

## 七、部署验证与工程实现

### 7.1 缓存复用

无需重新运行 SAM2 AMG。现有缓存已包含 `instance_id / boundary / quality`，所有新增结构（affinity, occupancy, depth, scale, compactness）在 augmentation replay 后动态计算。

### 7.2 代码组织

```
models/distill/sam_hsd/
├── __init__.py
├── sam_hsd_adapter.py
├── relational_structure.py
├── temporal_evidence.py
├── encoder_hsd.py
├── decoder_hsd.py
└── losses.py
```

**核心类**:
- `SAMHSDAdapter`
- `RelationalStructureBuilder` (0 参数)
- `DirectionalTemporalEvidence` (0 参数)
- `EncoderHSD` (tiny probe)
- `DecoderHSD` (tiny probe / 0 参数)

### 7.3 主路径接口

训练 forward 概念：

```python
feats1 = tuple(self.backbone(x1))
feats2 = tuple(self.backbone(x2))

if training and compute_aux and use_sam_hsd:
    struct = self.sam_hsd.build_teacher_evidence(teacher_pack)
    sam_enc_loss = self.sam_hsd.encoder_loss(feats1, feats2, target, struct)

x1 = self.swa(*feats1)
x2 = self.swa(*feats2)
change = self.tfm(*x1, *x2)
p2, p3, p4, p5, m2, m3, m4, m5 = self.decoder(*change)

if training and compute_aux and use_sam_hsd:
    sam_dec_loss = self.sam_hsd.decoder_loss(
        (p2, p3, p4, p5), (m2, m3, m4, m5), target, struct
    )
```

**关键**: `sam_enc_loss` 不改变 `feats`，`sam_dec_loss` 不改变 `p/mask`，只产生梯度。

### 7.4 switch_to_deploy

```python
def switch_to_deploy(self):
    if self.use_sam:
        del self.sam
        self.use_sam = False
        self.sam_mode = "none"
    return self
```

删除所有训练辅助模块，包括 0 参数 generator。

### 7.5 三层部署验证

1. **辅助开关一致性**: `out_aux == out_noaux` (同 mode、同 BN)
2. **部署前后一致性**: `max |out_before - out_after| < 1e-6`
3. **参数与 FLOPs**: 部署后必须为 `params = 2,913,094`, `FLOPs = 2.7475G`

---

## 八、实验设计

### 8.1 消融实验组

| Exp | PI-DTRS | Encoder HSD | Decoder HSD | SCGR | 目的 |
|-----|:-------:|:-----------:|:-----------:|:----:|------|
| **H0** | × | × | × | × | fair clean anchor |
| **H1** | Run4 old | old final-mask | × | × | 旧 SAMStruct 复现 |
| **H2** | ✓ | ✓ | × | × | Encoder directional structure |
| **H3** | ✓ | × | ✓ | ✓ | Decoder structure calibration |
| **H4** | ✓ | ✓ | ✓ | ✓ | **Full SAM-HSD** |
| **H5** | scalar (B₁-B₂) | ✓ | ✓ | ✓ | 验证 directional decomposition |
| **H6** | unsigned E_Δ | ✓ | ✓ | ✓ | 验证 directional 必要性 |
| **H7** | ✓ | ✓ | ✓ | × | 验证 conflict routing |
| **H8** | ✓ (无 tolerance/support) | ✓ | ✓ | ✓ | 验证鲁棒滤波 |

**核心对比**:
- **H4 - H5**: 证明不是简单边界相减
- **H4 - H6**: 证明保留出现/消失方向的价值
- **H4 - H7**: 证明 SAM-GT 冲突可转化为 hard-sample 信号

### 8.2 数据集与训练

- **第一轮**: SYSU + WHU
- 统一 batch、seed、优化器、学习率等
- 特别需要重新跑 SYSU 的 H0（消除 batch confound）

### 8.3 成功门槛

**SYSU**: 相对公平 H0
- `ΔF1 ≥ +0.5`
- `ΔIoU > 0`
- `ΔKappa > 0`
- Recall/Precision 不能明显牺牲

**WHU**: 
- `ΔF1 ≥ +0.3`
- Precision `≥` H0

### 8.4 指标与机制对应

| 指标 | 主要收益机制 |
|------|-------------|
| **Recall** | E⁺/E⁻、S₁/S₂ 方向蒸馏、小实例平衡 |
| **Precision** | E⁰、hard-negative SCGR、negative boundary |
| **OA** | stable unchanged suppression |
| **F1** | Recall + Precision 联合 |
| **IoU** | object completeness + precise boundary |
| **Kappa** | changed/unchanged 混淆同时下降 |

---

## 九、与现有工作的差异

| 工作 | 关键设计 | 我们的区别 |
|------|---------|-----------|
| **SAM-CD** (ICASSP 2024) | semantic prompt, masked attention | 无类别 prompt，AMG 结构场，方向差分，仅梯度指导 |
| **SGAMFNet** | region label → SAM, difference features, FBSM, pseudo label | 无 box prompt，不生成 pseudo label，不参与 forward |
| **BFD** | foundation feature → DFM → PCD → student | 使用 SAM instance topology → 方向结构证据 → 层次蒸馏，且不直接对齐特征 |

---

## 十、总结

### 升级版的核心逻辑链

```
SAM instance 
    → Relational Structural Field 
    → E⁺, E⁻, E⁰, Eᵤ 
    → Directional Encoder HSD 
    → Symmetric Decoder HSD 
    → Conflict Aware Gradient Routing
```

### 最值得坚持的三点

1. **方向化关系结构** E⁺/E⁻
2. **Encoder/Decoder 非对称蒸馏**
3. **SAM-GT 冲突转 hard sample**

### 部署不变

```
T1, T2 → LWGANet-L0 → SWA → TFM → Decoder
```

**参数**: 2.9131M  
**FLOPs**: 2.7475G

---

**文档整理完毕** ✅

