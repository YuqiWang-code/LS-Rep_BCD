# 极轻量遥感变化检测下一步改进方案  
## ——基于 LS-Rep_BCD 全历史结果、当前代码审查与 2024–2026 文献调研

> **仓库快照**：`YuqiWang-code/LS-Rep_BCD`，`main`，最新审查提交 `9d5096c7aa93b000e03c9e6dd3567504d17bd6eb`（2026-09-22）。  
> **任务范围**：CDD-CD-256、LEVIR-CD-256、SYSU-CD-256、WHU-CD-256，二时相、全监督、二值变化检测。  
> **学生**：A2Net-LWGANet-L0；部署参数固定 **2,913,094**；256×256 FLOPs 约 **2.75–2.77G**。  
> **实验约束**：seed=2333；batch=64；40k steps；不做多 seed；Teacher / Cache / Router / KD 头全部训练期删除；保持 T1/T2 交换对称。  
> **本文只给方案，不改代码。**

---

# 0. 最终结论

## 0.1 唯一建议方向

**建议下一步只做：SAGE-CD（Student-Aware Gated Expert Distillation）——“学生失败驱动的 DINOv3 语义专家 + SAM2 结构专家软路由蒸馏”。**

一句话核心思想：

> **不再把多个 Foundation Teacher 的 feature 直接强行对齐给 2.9M 学生，而是先让学生暴露“哪里错、错在语义还是边界”，再由 DINOv3 与 SAM2 分别提供异构知识；用带 Reject 的软路由只在各自可靠区域蒸馏，并把 KD 放到训练期独立辅助头上，同时用梯度预算限制 KD 对主任务梯度的最大占比。**

这个方向不是“DINO+SAM 简单融合”，也不是“再换一种 relation loss”，核心有四点：

1. **异构专家分工而不是 feature 融合**  
   - DINOv3：负责语义变化、区域内部真假变化判断；
   - SAM2：负责边界/实例结构变化；
   - 两者不要求处于同一 feature space，也不直接求和。

2. **学生先诊断，教师再被选择**  
   - 当前学生的语义误差与边界误差构成 failure profile；
   - 路由器按“当前学生在哪一类困难上失败 + 对应教师是否比学生更可靠”选择 DINO / SAM / Reject；
   - 不再假设任何 Teacher 对所有像素都可靠。

3. **教师也受学生失败画像驱动，但绝不被学生复制**  
   - 先固定 clean student；
   - 只微调 DINOv3 的 LoRA/change head 与 SAM2 的轻量结构校准器，使其重点学习学生困难区域；
   - 教师仍由 GT 监督，不用 `Teacher ← Student` 的 residual/EMA 复制，避免 RDT-CD 的“教师被学生拉平”。

4. **KD 与主预测头解耦 + 梯度预算**  
   - 主 decoder 仍只由 GT 主损失训练；
   - DINO/SAM KD 只走训练期 auxiliary heads；
   - 每个 KD loss 先归一化，再按共享特征上的梯度范数自动限幅，使总 KD 梯度始终低于主损失设定比例；
   - 从机制上阻止 FA-SCRD 中 `kd≈12–15`、`main≈3.6`、`0.5×KD > main` 的情况再次发生。

**部署图完全不变**：Teacher、Router、Aux-KD heads 全删，最终仍是现有 A2Net-LWGANet-L0，参数 **2,913,094**，FLOPs 保持约 **2.75–2.77G**。

---

# 1. 先给证据：历史结果说明真正的问题不是“Teacher 不够多”

我读取了仓库 `README.md §6`、`docs/experiment_metrics.xlsx`、SCTC Run1 说明和当前 FA-SCRD 代码。当前历史结果已经形成很清楚的模式。

## 1.1 历史实验核心证据表

下表只列同一实验族内最有解释力的对照；完整指标仍以 `experiment_metrics.xlsx` 与每个 `train_log.txt` 最后完整 `=== TEST RESULTS ===` 区块为准。

| 实验族 | 关键对照 | CDD ΔF1 | LEVIR ΔF1 | SYSU ΔF1 | WHU ΔF1 | 可支持的结论 |
|---|---|---:|---:|---:|---:|---|
| RDT-CD Run1 | D1 − B0 | +0.00 | −0.11 | **+2.43** | +0.19 | 双教师方向高度数据集依赖 |
| RDT-CD Run1 | D2 − B0 | +0.01 | −0.11 | **+1.83** | −0.45 | Teacher 叠加并未带来一致收益 |
| RDT-CD Run2 | D1NG − B0 | −0.05 | +0.06 | −0.57 | **−0.99** | regional/gate 并未解决负迁移 |
| RDT-CD Run3 | R3 − B0 | +0.00 | −0.15 | **+1.38** | −0.36 | SAM+OV 仍是 SYSU-only |
| RDT-CD Run3 | R3O − B0 | — | — | **+1.02** | **−0.73** | “SAM 污染 OV”被否定，单 OV 也会负迁移 |
| SCTC | S2 − S0 | −0.20 | −0.10 | −0.25 | **+0.43** | 零参数 temporal trick 变成 WHU-only |
| FA-SCRD | C1 − C0 | +0.01 | — | +0.06 | — | joint temporal BN 在已跑数据上影响很小 |
| FA-SCRD | A1 − C1 | **−0.33** | — | **−0.33** | — | 固定 DINOv3 relation KD 直接负迁移 |
| FA-SCRD | M1 − C1 | −0.00 | — | **−0.58** | — | failure-aware soft weight 没救回 KD |

### 直接事实

- RDT-CD Run3 日志归档显示 `dynamic_teacher_gain≈0`、`pixel_reject_ratio≈0.999`：Fast/EMA Teacher 很快接近 Student，Brier hard gate 又把绝大多数像素拒绝，KD 实际近乎饿死。
- SCTC 只在 WHU 正增益，SYSU/CDD/LEVIR 中性到负。
- FA-SCRD 中 A1 在 SYSU、CDD 均比 C1 低约 0.33pp；M1 没有修复。
- FA-SCRD 首轮训练的 `kd≈12–15`，而主损失约 `3.6`，`kd_lambda=0.5` 后 KD 的有效量级仍可大于主损失。

### 证据支持的推断

前三轮不是三个互不相关的失败，而是同一个问题的不同表现：

> **Teacher supervision 没有被限制在“Teacher 真正比 Student 更可靠且 Student 有容量吸收”的知识子空间。**

- RDT：Teacher 被 Student 同化，随后 gate 又把监督饿死；
- SCTC：完全放弃 Teacher 后，固定的 temporal 校准只适合部分数据分布；
- FA-SCRD：Teacher 固定了，但把 768-channel Foundation relation 直接强压给 64-channel change feature，且 loss 尺度失衡，变成过强监督。

因此，下一步不应该问“再加一个 Teacher 会不会更强”，而应该问：

> **“哪种 Teacher 知识，在什么失败类型、什么区域、什么强度下，值得进入 2.9M 学生？”**

---

# 2. 代码审查：先修正确性，再谈新方法

下面严格区分 **P0 正确性 / P1 方法瓶颈 / P2 工程或次要结构问题**。

---

# 3. P0：FA-SCRD 存在一个会混淆结论的 Teacher Cache 几何重放问题

## 3.1 1/16 DINO feature 的 RandomCropResize 实际上几乎没有被 crop

涉及文件：

- `models/datasets/transforms.py:34–62`
- `models/distill/cache_dataset.py:19–35`
- `models/scripts/train.py:267–276`
- DINO cache feature 尺寸由 `models/distill/fa_scrd_teacher.py:143–158` 确认为 16×16。

训练增强中：

```python
RandomCropResize(int(7.0 / 224.0 * trainsize))
```

在 256×256 下最大 margin 为约 8 像素。

cache 重放中对 16×16 feature 使用：

```python
t = int(round(top * H / src_h))
l = int(round(left * W / src_w))
```

因此：

\[
t=\operatorname{round}(top\times16/256)
=\operatorname{round}(top/16)
\]

而 `top ∈ [0,8]`。

Python 的 `round(0.5)` 为 0，因此 `top=0…8` 时，映射到 16×16 feature grid 的 crop margin 实际全部为 0；`left` 同理。

### 后果

约 50% 启用 RandomCropResize 的训练样本中：

- A / B / label：确实 crop 后 resize；
- `teacher_change_logit`（实际是 256×256 probability）：能正确 crop；
- **DINO `t1_mid/t2_mid/t1_deep/t2_deep` 16×16 relation feature：基本没 crop。**

所以 FA-SCRD 的 relation KD 在这些样本上存在**空间视图不一致**。

### 结论

**FA-SCRD 的负结果仍然是有效实验事实，但不能把全部负增益都归因于“DINO Teacher 本身无效”或“relation KD 必然无效”。**

它至少混有：

1. KD loss 尺度过大；
2. 低分辨率 teacher relation target 与学生增强视图错位；
3. 异构 teacher/student relation space 强制匹配。

### 下一轮必须先做的修复

**不要再对 16×16 Foundation feature 做随机 crop 的整数重放。**

SAGE-CD 只缓存/蒸馏 **256×256 task target map**：

- DINO semantic change probability/logit；
- SAM structural boundary-change probability；
- teacher reliability map。

这些全分辨率 target map 可以和 A/B/label 做严格一致的 crop / resize / flip；不再用低分辨率 relation feature 当最终 KD target。

---

# 4. P1：当前 relation KD 的数学尺度天然偏大

文件：`models/distill/relation_kd.py:48–72`

当前每个 4×4 patch 有 16 个 token，构造 16×16 affinity。

核心：

```python
kl = (a_t * (log_a_t - log_a_s)).sum(dim=(-1, -2))
return (kl * w_p).mean()
```

这里对：

- 每一行分布的 KL；
- 16 个 query row；

全部求和，却没有再除以 query 数 `pp=16`。

因此 relation loss 的自然量级本来就会比“逐像素平均”的主 BCE/Dice 大很多。

随后 `train.py:313–316` 又把 mid + deep 两个 relation KD 相加：

```python
l_mid + l_deep
```

最后 `train.py:367`：

```python
total = main_loss + 0.5 * kd_loss
```

这与首轮观察到的 `kd≈12–15` 完全一致。

## 4.1 这不是简单“把 lambda 从 0.5 改 0.05”就应该结束

只调 lambda 能把数值压小，但无法解决：

- Teacher target 是否可靠；
- mid/deep 两套 relation 是否互相冲突；
- Teacher 与 Student 是否在同一种结构关系空间；
- KD 与 GT 对同一 student feature 是否给相反梯度。

所以新方案应该换掉“直接 relation imitation”，而不是只调权重。

---

# 5. D：学生模型逐文件代码审查

## 5.1 `models/a2net.py`

| 严重度 | 行 | 审查点 | 判断 | 建议 |
|---|---:|---|---|---|
| P1 | 99–104 | `joint_temporal_bn=True` 时 T1/T2 concat 后一次 backbone forward | 逻辑合理；可以让共享 BN 同时看两时相。但历史 C1−C0 只有约 +0.01/+0.06，不能当主要创新 | 保留作 normalization control，不包装成方法贡献 |
| P2 | 75–78 | `NeighborFeatureAggregation([32,32,64,128,256],64)` 传 5 个维度，但 NFA forward 只接 c2–c5，内部实际从 `in_d[1]` 开始用 | 历史接口冗余，不是功能 bug | 后续重构时清理，不值得单独实验 |
| 正常 | 119–124 | NFA → TFM → Decoder | 数据流清楚 | 不建议同时重写 backbone/decoder，避免与新 KD 机制混杂 |
| 正常 | 128–138 | 多尺度 logits interpolate 后 sigmoid | 与现有 BCE+Dice probability loss 匹配 | 保持 |
| 正常 | 173–180 | `switch_to_deploy()` no-op | 当前 SCTC 已归档、main graph 无训练辅助时合理 | SAGE-CD 的辅助模块应放在 `models/distill/`，不要塞进 deploy student；这样这里仍可 no-op |

### 结论

`a2net.py` 本身没有发现会导致当前指标异常的残差/BN/激活错误。

---

## 5.2 `models/decoder/a2net_decoder.py`

### P1-1：TFM 只看到 `|F1-F2|`，语义上下文在进入 change encoder 前被彻底丢掉

关键行：`237–253`

```python
x = torch.abs(x1 - x2)
```

然后所有分支都只处理 `x`。

后续 dilation 7 → 5 → 3 → 1 能扩大局部上下文，但它回答的是：

> “差异纹理长什么样？”

而不是：

> “这个差异发生在什么语义对象/结构上，它应不应该算变化？”

这与项目现象高度一致：

- CDD：伪变化需要语义判断；
- WHU/LEVIR：建筑小目标更需要结构边界；
- SYSU：变化类型更多，语义 Teacher 更可能发挥作用。

### 但当前不建议立刻替换 TFM

理由：

1. SCTC 已证明“再加一个零参数 temporal trick”很容易成为单数据集特例；
2. 当前首要科学问题仍是“Teacher 能否有效传给 2.9M Student”；
3. 若同时改 TFM，会无法区分收益来自 Student deploy graph 还是 Teacher transfer。

**因此第一轮 SAGE-CD 保留现有 TFM，只把 teacher supervision 挂到 TFM 的不同尺度输出上。**

- `c4`：DINO semantic auxiliary head；
- `c2/c3`：SAM boundary auxiliary head。

这样可以利用语义/结构监督重新塑造 TFM representation，但**推理结构与参数完全不变**。

---

### P1-2：SupervisedAttentionModule 是纯乘法抑制，没有 identity bypass

`304–328`：

```python
context = self.conv_context(context)
x = x.mul(context)
x_out = self.conv2(x)
```

如果 `context` 在困难边界/小目标区域偏低，原始 feature 会被直接压低。

一个参数零增加的候选是：

```python
x = x + x * context
```

或有界 residual gate。

但这是**待验证结构假设**，当前没有历史证据表明它是主要瓶颈。

**结论：不进入下一轮主方案。**  
只有当 SAGE-CD 已证明 Teacher transfer 有效、但主要表现为 Recall/小目标不足时，再单独做 decoder residual-gating ablation。

---

### P2-1：TFFM 中 1×1 residual branches 与 dilation branches 的 normalization 不完全对称

例如：

- `conv_branch1`: Conv3×3(d=7)+BN；
- `conv_branch2`: Conv1×1，无 BN；
- `conv_branch2_f`: Conv3×3(d=5)+BN；
- 后续类似。

这可能让相加两支的尺度不同，但当前每级相加后又经过后续 BN，且模型已能稳定训练，**不能把它定性为 BN 放错或缺失 bug**。

如果未来要改，应单独做：

- affine-free normalization；
- 或整体 PreNorm/PostNorm 重构；

不能混到 Teacher 实验中。

---

### 激活函数检查

- backbone：GELU；
- decoder/NFA/TFM：ReLU；
- attention probability 与预测概率：sigmoid。

这套组合没有明显语义错误：

- GELU 与 LWGANet 预训练结构一致；
- decoder 从头训练，ReLU 合理；
- sigmoid 用于二值 mask / attention gate 合理。

**不要为了“统一激活”把 GELU/ReLU 全部替换。**

---

## 5.3 `models/backbone/lwganet.py`

### P1-候选，但不建议第一轮改：四分支局部残差形式不一致

`306–323`：

```python
x1 = x1 + self.PA(x1)
x2 = self.LA(x2)
x3 = self.MRA(x3)
...
x4 = x4 + ...
...
x = shortcut + self.norm1(self.drop_path(self.mlp(x_att)))
```

x1/x4 有支内 residual，x2/x3 没有；但最后整个 block 还有外层 `shortcut`。

因此：

- **不是“残差连接写错”**；
- 这是 LWGANet 原始分支设计的一部分；
- 直接给 x2/x3 再加 residual 会破坏 backbone 预训练函数。

在本项目中 backbone 是有预训练权重的，**不建议随意改动这一处**。

---

### P2：`drop_path` 被重复赋值

`279`：

```python
self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
```

`304` 又：

```python
self.drop_path = DropPath(drop_path)
```

L0 工厂 `drop_path_rate=0.0`，所以当前实际无影响，只是代码冗余。

---

### P2：DRFD fusion 后没有 norm/activation

`57–78`：

```python
x = torch.cat([conv, max], dim=1)
x = self.fusion(x)
```

理论上可能出现两支统计分布不一致，但：

- 这是 backbone 预训练结构；
- 任意加 BN/activation 都会改变预训练函数；
- 没有历史证据证明这是当前性能瓶颈。

**不进入第一轮。**

---

## 5.4 D 部分最终判断

### 可以确认的问题

1. **没有发现学生主干存在明显“ReLU/GELU/sigmoid 用错”或致命 BN 漏放。**
2. LWGANet 残差看似不对称，但外层 residual 存在，不能当 bug。
3. TFM 的主要局限不是卷积写错，而是**变化建模只基于绝对差，缺少语义/结构条件**。

### 下一轮最合理的“学生改进”

不是现在去重写 deploy TFM，而是：

> **保留 2.913M deploy student，给 TFM 多尺度 change features 加训练期语义/结构 auxiliary distillation heads，让 Teacher 在训练时改变表示学习，但推理图完全不变。**

这是当前风险最低、因果最清楚的做法。

---

# 6. A：联合教师文献——为什么要“专家分工”，不能简单 DINO+SAM feature fusion

## A1. AM-RADIO

**Mike Ranzinger, Greg Heinrich, Jan Kautz, Pavlo Molchanov.  
“AM-RADIO: Agglomerative Vision Foundation Model Reduce All Domains Into One.”  
CVPR 2024，CCF-A。**

- Paper: https://openaccess.thecvf.com/content/CVPR2024/html/Ranzinger_AM-RADIO_Agglomerative_Vision_Foundation_Model_Reduce_All_Domains_Into_One_CVPR_2024_paper.html
- Official code: https://github.com/NVlabs/RADIO

### 核心机制

将 CLIP、DINOv2、SAM 等具有不同专长的 VFM 做 multi-teacher distillation，目标是让一个学生吸收多类知识，而不是认为所有 Teacher feature 天然同分布。

### 对本项目的直接启示

DINO 与 SAM 的“互补”是有文献基础的，但启示不是：

```text
DINO feature + SAM feature -> concat -> KD
```

而是：

> **保留教师差异性，分别定义可学习的知识接口。**

在 CD 中最自然的差异是：

- DINO：semantic / appearance invariance；
- SAM：instance / boundary structure。

---

## A2. MoVE-KD

**Jiajun Cao, Yuan Zhang, Tao Huang, Ming Lu, Qizhe Zhang, Ruichuan An, Ningning Ma, Shanghang Zhang.  
“MoVE-KD: Knowledge Distillation for VLMs with Mixture of Visual Encoders.”  
CVPR 2025，CCF-A。**

- Paper: https://arxiv.org/abs/2501.01709
- Official code: https://github.com/hey-cjj/MoVE-KD

### 核心机制

- 多视觉编码器具有不同优势；
- 用 LoRA/MoE 保存各 Teacher 的 specialist knowledge；
- attention-based distillation 对不同 encoder 与 token 自适应加权；
- 重点解决多教师冲突与“学生不需要复制所有 Teacher feature”的问题。

### 对本项目的直接启示

这篇最直接支持：

> **Teacher selection 应该发生在知识层面，而不是先把 Teacher 混成一个统一 target。**

因此 SAGE-CD 的 router 输出的是：

```text
DINO semantic KD weight
SAM structural KD weight
Reject weight
```

而不是一个融合后的“超级 Teacher feature”。

---

## A3. JL1-CD / MTKD

**Ziyuan Liu, Ruifei Zhu, Long Gao, Yuanxiu Zhou, Jingyu Ma, Yuantao Gu.  
“JL1-CD: A New Benchmark and a Robust Multi-Teacher Knowledge Distillation Framework for Remote Sensing Change Detection.”  
IEEE Transactions on Instrumentation and Measurement, 2026。SCI 期刊，非 CCF-A。**

- Paper: https://arxiv.org/abs/2502.13407
- Official code: https://github.com/circleLZY/JL1-CD

### 核心机制

按 change-area ratio 划分 small / medium / large specialist teachers，再让 student 从多个 specialist teacher 学习。

### 对本项目的直接启示

它证明 CD 里“不同样本需要不同 Teacher”本身是合理问题，但它的分工轴主要是**变化面积大小**。

SAGE-CD 与它的实质区别应明确写成：

> **本项目按“学生失败类型 × Teacher 知识类型”路由，而不是按样本 change-area ratio 预先切教师。**

这更符合已有失败：

- WHU/LEVIR：边界、小目标、稀疏；
- CDD：伪变化；
- SYSU：语义类型更丰富。

---

# 7. B：教师—学生互选互促文献——不做“Teacher 永远静态、Student 永远模仿”

## B1. DIFO

**Song Tang, Wenxin Su, Mao Ye, Xiatian Zhu.  
“Source-Free Domain Adaptation with Frozen Multimodal Foundation Model.”  
CVPR 2024，CCF-A。**

- Paper: https://openaccess.thecvf.com/content/CVPR2024/html/Tang_Source-Free_Domain_Adaptation_with_Frozen_Multimodal_Foundation_Model_CVPR_2024_paper.html
- Official code: https://github.com/tntek/source-free-domain-adaptation

### 核心机制

交替执行：

1. 用当前 target model 的信息去定制 foundation model；
2. 再把定制后的 foundation knowledge 蒸馏回 target model。

### 对本项目的直接启示

Foundation Teacher 不必永远是离线固定知识源。

但本项目不应把 DIFO 原样搬来，因为我们有 GT、且 Teacher 很大。

更合适的形式是：

> **固定 clean student 产生 failure map → 用 failure map 加权 Teacher 的有监督校准 → 固定 Teacher → 再教 Student。**

这样完成“Student 影响 Teacher 学什么”，但不会像 RDT-CD 那样把 Teacher 数值直接拉向 Student。

---

## B2. DAMP

**Zhekai Du, Xinyao Li, Fengling Li, Ke Lu, Lei Zhu, Jingjing Li.  
“Domain-Agnostic Mutual Prompting for Unsupervised Domain Adaptation.”  
CVPR 2024，CCF-A。**

- Paper: https://openaccess.thecvf.com/content/CVPR2024/html/Du_Domain-Agnostic_Mutual_Prompting_for_Unsupervised_Domain_Adaptation_CVPR_2024_paper.html
- Official code: https://github.com/TL-UESTC/DAMP

### 核心机制

视觉与文本两路不是单向注入，而是互相生成/约束 prompt，通过 cross-attention 和一致性约束共同适应。

### 对本项目的直接启示

虽然任务不是 KD/CD，但它支持一个重要设计原则：

> **互促不等于参数 EMA；互促可以表现为“一方的状态决定另一方该关注什么”。**

SAGE-CD 就用 Student failure profile 决定：

- DINO teacher calibration 的难区；
- SAM teacher calibration 的难边界；
- 后续每个区域使用哪位 Teacher。

---

## B3. Your Student is Better Than Expected

**Nikita Starodubcev, Artem Fedorov, Artem Babenko, Dmitry Baranchuk.  
“Your Student is Better Than Expected: Adaptive Teacher-Student Collaboration for Text-Conditional Diffusion Models.”  
CVPR 2024，CCF-A。**

- Paper: https://arxiv.org/abs/2312.10835
- Official code: https://github.com/yandex-research/adaptive-diffusion

### 核心机制

先让 Student 产生结果，再由质量估计器判断是否值得调用 Teacher；不是默认 Teacher 一定优于 Student。

### 对本项目的直接启示

这与本项目三轮失败极其契合：

> **Teacher “更大”不代表局部一定更可靠。**

SAGE-CD 必须保留 **Reject**：

```text
a_i ∈ {DINO_semantic, SAM_structure, Reject}
```

当两位 Teacher 都不比 Student 可靠时，不蒸馏。

注意：该论文在推理时会调用 Teacher，**这一点不能照搬**；本项目只借鉴“Student-first + quality-gated teacher selection”的思想，所有选择都发生在训练期。

---

# 8. C：KD 本身怎么改——为什么 relation KD 会带偏，以及换成什么

## C1. CrossKD

**Jiabao Wang, Yuming Chen, Zhaohui Zheng, Xiang Li, Ming-Ming Cheng, Qibin Hou.  
“CrossKD: Cross-Head Knowledge Distillation for Object Detection.”  
CVPR 2024，CCF-A。**

- Paper: https://openaccess.thecvf.com/content/CVPR2024/html/Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection_CVPR_2024_paper.html
- Official code: https://github.com/jbwang1997/CrossKD

### 核心机制

直接对 Student 主输出做 GT supervision + Teacher mimicking 会产生 target conflict。

CrossKD 把 Student 中间 feature 送到独立 teacher head 路径：

- 主 Student head：只收 GT；
- KD branch：单独收 Teacher target。

### 对本项目的直接启示

FA-SCRD 当前：

```text
Student c4
  ├── decoder -> GT main loss
  └── relation KD -> DINO relation
```

两个损失直接竞争同一 change feature。

SAGE-CD 改成：

```text
Student change features
  ├── 原 Decoder -> 只做 GT 主损失
  ├── train-only semantic head -> DINO target
  └── train-only boundary head -> SAM target
```

这就是最适合本项目的 CrossKD 思路迁移。

---

## C2. Logit Standardization in Knowledge Distillation

**Shangquan Sun, Wenqi Ren, Jingzhi Li, Rui Wang, Xiaochun Cao.  
“Logit Standardization in Knowledge Distillation.”  
CVPR 2024 Highlight，CCF-A。**

- Paper: https://openaccess.thecvf.com/content/CVPR2024/html/Sun_Logit_Standardization_in_Knowledge_Distillation_CVPR_2024_paper.html
- Official code: https://github.com/sunshangquan/logit-standardization-KD

### 核心机制

Teacher/Student 容量不同时，强迫 logit 绝对幅度和方差完全一致会伤害小模型；先做 z-score，使 Student 更关注相对关系而非 magnitude。

### 对本项目的直接启示

DINOv3 Teacher 与 2.9M Student 容量差巨大。

新 semantic KD 不应直接 MSE/硬 KL 对齐 raw logit magnitude。

建议对每个样本的空间 logits：

\[
\hat z = \frac{z-\mu_{HW}}{\sigma_{HW}+\epsilon}
\]

再做 soft Bernoulli target distillation。

---

## C3. VkD

**Roy Miles, Ismail Elezi, Jiankang Deng.  
“VkD: Improving Knowledge Distillation using Orthogonal Projections.”  
CVPR 2024，CCF-A。**

- Paper: https://openaccess.thecvf.com/content/CVPR2024/html/Miles_VkD_Improving_Knowledge_Distillation_using_Orthogonal_Projections_CVPR_2024_paper.html
- Official code: https://github.com/roymiles/VkD

### 核心机制

异构模型间直接 feature imitation 不稳定；通过约束 projection 与 task-specific normalization 缩小表示空间错配。

### 对本项目的直接启示

DINO 768ch 与 Student 64ch 的 feature relation 不应该被当作天然可比。

SAGE-CD 因而**优先不蒸馏 Foundation raw feature**；若后续确实需要 feature KD，只允许在 training-only projector 后做，并作为额外消融，而不是主方案第一版。

---

# 9. 为什么不选另外三个看似自然的方向

## 9.1 不选“DINO+SAM feature concat / average”

原因：

- AM-RADIO / MoVE-KD 都说明多 Teacher 冲突需要显式处理；
- DINO semantic 与 SAM boundary 的统计含义不同；
- 直接融合会把“谁可靠”重新隐藏掉；
- RDT-CD 已经验证“更多 Teacher”本身不解决问题。

---

## 9.2 不选“再做在线 EMA / reciprocal teacher”

原因：

- RDT-CD 已经实际出现 Teacher 被 Student 拉平；
- `dynamic_teacher_gain≈0` 是直接日志证据；
- 再做 EMA 只是在同一失败机制上换参数。

---

## 9.3 不选“先大改 TFM / Decoder”

原因：

- SCTC 已经说明 student-only temporal trick 容易数据集特化；
- 当前尚未正面回答“强 Teacher 是否能安全转移”；
- 同时改 deploy student 会让实验因果无法解释。

---

# 10. SAGE-CD：具体数学定义

---

## 10.1 Teacher Expert 1：DINOv3 Semantic Change Expert

沿用 DINOv3 ViT-B/16：

\[
F_D^1 = E_D(I_1),\qquad F_D^2 = E_D(I_2)
\]

共享 encoder，保证交换对称。

构造：

\[
D_D=|N(F_D^1)-N(F_D^2)|
\]

经训练期 change head 得到 semantic change logit：

\[
z_D=H_D(D_D)
\]

\[
p_D=\sigma(z_D)
\]

### 与 FA-SCRD 的关键区别

FA-SCRD 蒸馏的是 `D_D` 的 local relation。

SAGE-CD **只把最终 task semantic target `z_D/p_D` 蒸馏给 Student**。

这样：

- 不要求 768ch 与 64ch feature relation 同构；
- target 与 binary CD 主任务一致；
- full-resolution map 能严格重放 augmentation。

---

## 10.2 Teacher Expert 2：SAM2 Structural Change Expert

不把 SAM2 mask 直接相减当最终 change mask。

已有 cache：

```text
T1: instance_id / boundary / quality
T2: instance_id / boundary / quality
```

第一版只使用最稳定且交换对称的 boundary/quality：

\[
s_S =
\left|
q_1 B_1-q_2 B_2
\right|
\]

其中：

- \(B_1,B_2\)：SAM boundary map；
- \(q_1,q_2\)：SAM quality map。

再加一个**训练期、极小的 SAM structural calibrator**：

\[
p_S=C_S(s_S)
\]

监督目标不是整幅变化 mask，而是 GT change boundary：

\[
B_Y=\operatorname{MorphGradient}(Y)
\]

即：

\[
\mathcal L_{T,S}
=
\operatorname{BCE}(p_S,B_Y)
+\operatorname{Dice}(p_S,B_Y)
\]

这样 SAM 的任务被严格限定为：

> **“变化边界在哪里？”**

而不是让 SAM 与 DINO 争夺同一个 semantic probability。

---

# 11. 教师—学生“互促”：只让 Student 决定 Teacher 该学什么，不让 Teacher 复制 Student

先训练/读取 clean student \(S_0\)。

固定 \(S_0\)，计算：

### 语义困难度

\[
d_{\rm sem}(x)
=
\operatorname{BCE}(p_{S_0}(x),Y(x))
\]

### 边界困难度

\[
d_{\rm bnd}(x)
=
\operatorname{BCE}(B(p_{S_0})(x),B_Y(x))
\]

然后 Teacher calibration 时：

### DINO

\[
\mathcal L_{T,D}
=
(1+\alpha d_{\rm sem})
\cdot
\mathcal L_{\rm seg}(p_D,Y)
\]

### SAM

\[
\mathcal L_{T,S}
=
(1+\alpha d_{\rm bnd})
\cdot
\mathcal L_{\rm bnd}(p_S,B_Y)
\]

关键约束：

- Student **detach**；
- Teacher 仍以 GT 为优化目标；
- Student 只决定 Teacher 的训练关注区域；
- 不做 `Teacher ← EMA(Student)`；
- 不做 residual teacher 初始化为 0；
- 不存在 RDT 的 teacher-collapse 路径。

这就是本项目最安全的“互促”。

---

# 12. Student failure profile 与软路由

当前 Student 在每个 region \(r\) 计算：

\[
e_{S,D}(r)=
\operatorname{BCE}(p_S,Y)
\]

\[
e_{D}(r)=
\operatorname{BCE}(p_D,Y)
\]

\[
e_{S,SAM}(r)=
\operatorname{BCE}(B(p_S),B_Y)
\]

\[
e_{SAM}(r)=
\operatorname{BCE}(p_{SAM},B_Y)
\]

定义 Teacher 相对 Student 的 advantage：

\[
g_D(r)
=
\frac{
e_{S,D}(r)-e_D(r)
}{\tau_r}
\]

\[
g_{SAM}(r)
=
\frac{
e_{S,SAM}(r)-e_{SAM}(r)
}{\tau_r}
\]

增加 Reject expert：

\[
g_R(r)=0
\]

路由：

\[
[w_D,w_{SAM},w_R]
=
\operatorname{softmax}
(
[g_D,g_{SAM},g_R]
)
\]

全部 detach。

### 这比历史 Brier hard gate 好在哪里

历史：

```text
Teacher gain <= threshold
    -> reject = 1
    -> KD = 0
```

导致 99.9% pixel 被硬拒绝。

新方案：

- 不做 0/1 gate；
- Teacher 稍好时得到较小但非零权重；
- Teacher 明显更好时权重自然增大；
- Teacher 都差时 Reject 胜出；
- router 本身不反传到 Student/Teacher，避免 gaming。

### Region 而不是逐像素

建议 8×8 或 16×16 region aggregate 后再上采样权重。

原因：

- 降低单像素 GT 噪声；
- 避免 CDD JPEG label 边缘噪声放大；
- Teacher 选择更接近“困难类型”而不是随机像素。

---

# 13. KD 路径：主 decoder 不再直接收到 Teacher target

现有 Student：

```text
T1/T2
 -> LWGANet
 -> NFA
 -> TFM
 -> Decoder
 -> main prediction
```

保持不变。

训练时另外读取：

```text
c2, c3, c4, c5 = TFM outputs
```

新增 **training-only**：

```text
c4 -> AuxSemanticHead -> z_s_sem
c2 -> AuxBoundaryHead -> z_s_bnd
```

推荐第一版极简：

```text
AuxSemanticHead: 1×1 Conv(64 -> 1)
AuxBoundaryHead: 3×3 depthwise/1×1 or 单 1×1 Conv(64 -> 1)
```

这些参数：

- 算训练参数；
- 不算部署参数；
- 部署 checkpoint 不包含；
- `switch_to_deploy()` 后主 Student 仍是 2,913,094。

---

# 14. Semantic KD

对 Teacher / Student auxiliary logit 做空间标准化：

\[
\tilde z
=
\frac{
z-\mu_{HW}(z)
}{
\sigma_{HW}(z)+\epsilon
}
\]

温度 \(T\) 后：

\[
q_D=\sigma(\tilde z_D/T)
\]

\[
q_S=\sigma(\tilde z_{S,sem}/T)
\]

用 soft BCE / Bernoulli KL：

\[
\mathcal L_D
=
\frac{
\sum_r
w_D(r)
D_{\rm Bern}
(q_D(r),q_S(r))
}{
\sum_r w_D(r)+\epsilon
}
\]

重点：

- 不再做 raw 768ch feature relation；
- 不要求绝对 logit magnitude 一致；
- 归一化分母由 active weight 决定，loss 尺度不会随 Teacher coverage 随机变化。

---

# 15. SAM structural KD

Student boundary auxiliary output：

\[
p_{S,b}=\sigma(z_{S,b})
\]

SAM structural target：

\[
p_{SAM}
\]

损失：

\[
\mathcal L_{SAM}
=
\frac{
\sum_r
w_{SAM}(r)
[
\operatorname{BCE}(p_{S,b},p_{SAM})
+\lambda_b\operatorname{Dice}
]
}{
\sum_r w_{SAM}(r)+\epsilon
}
\]

这样两个 Teacher 的知识目标始终分开：

- DINO 不管精细 boundary；
- SAM 不负责 semantic interior。

这就是“避免多教师冲突”的核心。

---

# 16. 防止 KD 再次压过主损失：Gradient Budget，而不是拍脑袋 lambda

总体：

\[
\mathcal L
=
\mathcal L_{main}
+
\lambda_D\mathcal L_D
+
\lambda_S\mathcal L_{SAM}
\]

但 \(\lambda_D,\lambda_S\) 不固定为 0.5。

在共享 Student feature 上计算梯度范数。

例如 DINO 对 c4：

\[
g_m=
\left\|
\nabla_{c4}\mathcal L_{main}
\right\|_2
\]

\[
g_D=
\left\|
\nabla_{c4}\mathcal L_D
\right\|_2
\]

设最大 KD / main gradient ratio 为 \(\rho\)：

\[
\lambda_D
=
\operatorname{clip}
\left(
\rho
\frac{g_m+\epsilon}{g_D+\epsilon},
0,
\lambda_{\max}
\right)
\]

SAM 对 c2 同理。

第一轮建议：

\[
\rho=0.25
\]

含义非常明确：

> **任何一个 Teacher 的梯度都不能超过对应主任务梯度的 25%。**

这不是把调参包装成创新，而是一个**训练安全约束**。

必须记录：

```text
main_loss
dino_kd_raw
sam_kd_raw
lambda_dino
lambda_sam
grad_ratio_dino
grad_ratio_sam
router_dino_ratio
router_sam_ratio
router_reject_ratio
teacher_advantage_dino
teacher_advantage_sam
```

### 失败保护

若观测到：

```text
grad_ratio > rho + tolerance
```

直接视为实现错误，不继续正式训练。

---

# 17. 为什么这个方案能正面突破前三次失败

| 历史失败 | 根因 | SAGE-CD 对应修复 |
|---|---|---|
| RDT-CD：Teacher 被 Student 拉平 | residual/EMA teacher 与 Student 数值耦合 | Teacher 只由 GT 训练；Student 仅提供 detached difficulty weights |
| RDT-CD：hard gate 饿死 KD | 99.9% pixel reject | softmax routing + explicit Reject，不做硬阈值 |
| RDT-CD：SAM/OV 混合仍冲突 | 不同 Teacher 被当成同一种知识 | DINO semantic 与 SAM boundary 分开定义 loss |
| SCTC：WHU-only | 固定 temporal transform 对所有数据统一作用 | Teacher 只在其真正优于 Student 的区域作用 |
| FA-SCRD：relation KD −0.33 | 异构 relation 强行对齐 | 不蒸馏 raw Foundation feature relation |
| FA-SCRD：KD 量级过大 | patch KL sum + mid/deep 累加 + fixed λ | active-weight mean + gradient budget |
| FA-SCRD：cache crop 错位 | 16×16 feature 无法正确重放小 crop | 只蒸馏 256×256 task target maps |
| FA-SCRD：一个 c4 同时追 mid/deep | 多个 teacher level 对同一 latent 施压 | semantic / structural 各走专用 train-only head |

---

# 18. T1/T2 交换对称性

必须逐项证明。

### DINO expert

\[
|F_1-F_2|
=
|F_2-F_1|
\]

成立。

### SAM expert

\[
|q_1B_1-q_2B_2|
=
|q_2B_2-q_1B_1|
\]

成立。

### Student features

现有 TFM：

\[
|x_1-x_2|
\]

成立。

### Router

依赖：

- Student change probability；
- GT；
- DINO change probability；
- SAM symmetric structure target；

均与 T1/T2 顺序无关。

### 必测

```text
max |P(T1,T2) - P(T2,T1)| < 1e-6
```

Teacher auxiliary branch也单独检查。

---

# 19. Cache 方案

## 19.1 不再缓存用于 relation KD 的 16×16 raw feature

新 cache 每样本仅需要：

```yaml
dino:
  change_logit_or_prob: [1,256,256]
  confidence:           [1,256,256]

sam2:
  structural_change:    [1,256,256]
  structural_quality:   [1,256,256]

meta:
  teacher_version
  checkpoint_hash
  source_resolution
  normalization
```

### 好处

- 几何 replay 精确；
- cache 更小；
- 不依赖 Foundation feature channel；
- DINO/SAM Teacher 可替换而不改 Student interface；
- 恢复训练只需检查 manifest/hash。

---

# 20. 训练图与部署图

## 20.1 训练图

```text
                      ┌─ DINOv3 semantic cache ─┐
T1,T2 ─ Student ──────┤                         ├─ soft router ─┐
  │                   └─ SAM2 structure cache ──┘               │
  │                                                              │
  ├─ c4 ─ AuxSemanticHead ─ semantic KD ─────────────────────────┤
  ├─ c2 ─ AuxBoundaryHead ─ structural KD ───────────────────────┤
  │                                                              │
  └─ original Decoder ─ GT BCE+Dice ─────────────────────────────┘
```

Teacher calibration 阶段在此之前单独完成。

## 20.2 部署图

```text
T1,T2
 -> shared LWGANet-L0
 -> NFA
 -> original TFM
 -> original Decoder
 -> binary change map
```

无：

- DINO；
- SAM2；
- cache；
- router；
- Aux heads；
- Teacher calibrator。

部署参数：

```text
2,913,094
```

FLOPs：

```text
2.75–2.77G @ 256×256
```

---

# 21. 逐文件修改清单（下一轮回来改代码时按这个顺序）

> 这部分是实施计划，不代表本次已经修改。

## 21.1 P0 修复

### `models/distill/cache_dataset.py`

- 禁止对 16×16 feature 使用当前整数 crop replay；
- 新 SAGE cache 仅重放 256×256 target maps；
- replay state 必须覆盖：
  - scale；
  - crop_resize；
  - flip_h；
  - flip_v；
  - temporal exchange。

### 新增单元测试

构造一个带坐标 ramp 的 256×256 map，确认：

```text
image transform == cache transform
```

在每种 augmentation 组合下最大位置误差为 0（离散 label）或仅为预期 bilinear 误差（soft map）。

---

## 21.2 Teacher calibration

建议新增：

```text
models/distill/sage_teacher.py
models/tools/calibrate_sage_teachers.py
models/tools/generate_sage_cache.py
```

DINO：

- backbone 冻结；
- LoRA Q/V + change head 可训练；
- 用 clean Student failure map 加权 GT loss。

SAM：

- SAM2 本体不在线训练；
- 读取现有 SAMStruct；
- 训练极小 structural calibrator；
- 输出 full-res boundary-change target。

---

## 21.3 Router

新增：

```text
models/distill/sage_router.py
```

要求：

- 完全 detached；
- region-level；
- `DINO / SAM / Reject` 三路 softmax；
- 不允许 hard threshold；
- 输出日志统计。

---

## 21.4 Student auxiliary heads

新增：

```text
models/distill/sage_heads.py
```

不要放进 `A2Net_LWGANet_L0` 的 deploy module tree。

Trainer 从：

```python
predictions, change = model(..., return_change_features=True)
```

取：

```text
change[0] -> SAM boundary head
change[2] -> DINO semantic head
```

---

## 21.5 Trainer

建议新增独立入口，而不是继续污染 FA-SCRD：

```text
models/scripts/train_sage.py
train_scripts/SAGE-CD/Run1/
```

新增：

- task-target KD；
- gradient budget；
- router statistics；
- auxiliary optimizer state；
- cache manifest hash；
- strict resume validation。

---

# 22. checkpoint / resume 设计

训练 checkpoint 至少包含：

```yaml
format_version
implementation_version

student
optimizer_student

aux_semantic_head
aux_boundary_head
optimizer_aux

epoch
global_step
best_val_f1

args
rng

teacher_cache_manifest_hash
teacher_calibration_hash
dataset_split_fingerprint
```

Teacher Foundation backbone 不进 Student checkpoint。

部署导出只保留：

```text
student.state_dict()
```

并测试：

```text
deploy params == 2,913,094
max prediction error before/after deploy < 1e-6
```

---

# 23. 最小实验矩阵

## 23.1 第一关必须先补：正面回答“Teacher 到底有没有用”

先只做 SYSU + WHU，原因：

- SYSU：过去 Teacher 最容易正增益；
- WHU：过去 Teacher 最容易负迁移；
- 这是最有辨别力的一对。

所有条件固定：

```text
seed=2333
batch=64
steps=40000
lr=5e-4
wd=1e-4
same pretrained student
same augmentation
same eval
```

### G0 — Clean

```text
A2Net
joint_temporal_bn=False
no teacher
```

### G1 — Joint-BN control

```text
A2Net
joint_temporal_bn=True
no teacher
```

### G2 — Safe-DINO

```text
G1
+ 修复后的 full-res DINO semantic target
+ train-only semantic head
+ logit standardization
+ gradient budget
无 SAM
无 multi-teacher router
```

### G2 的意义

这是整个下一方向的生死门。

它只回答：

> **在 P0 修复、KD 解耦、梯度受控之后，一个 DINO Teacher 能不能真正给 2.9M Student 带来正增益？**

### Gate

要求：

- SYSU：`G2 - G1 >= +0.15 F1`
- WHU：`G2 - G1 >= +0.15 F1`
- 任一数据集不得出现 Recall/Precision 单边崩塌；
- `grad_ratio_dino <= 0.25 + tolerance`；
- reject/active teacher 统计合理。

若 G2 仍在 SYSU 或 WHU ≤ G1：

> **停止 SAGE-CD，不继续 SAM2，不再做多教师。**

这是最重要的止损条件。

---

# 24. 第二关：只有 G2 通过才上双教师

### M0 — Dual Expert / no routing

```text
G2
+ SAM structural auxiliary KD
DINO/SAM 各做自己的 loss
但不按 Student failure 做 adaptive routing
```

目的：

> 检验“增加 SAM structural expert”本身是否有补充价值。

### M1 — Full SAGE-CD

```text
M0
+ student-conditioned teacher calibration
+ DINO / SAM / Reject soft routing
```

目的：

> 检验“按学生失败类型互选教师”是否真正减少多 Teacher 冲突。

### 必要诊断臂（仅当 M1 有明显提升时补）

`M1-no-calib`：

- 保留 router；
- 不做 student-conditioned teacher calibration。

只需先在 SYSU/WHU 跑。

它回答：

> 提升来自 routing，还是来自 Teacher 被 Student failure profile 定制？

---

# 25. 四数据集最终验证

通过两阶段 gate 后，再把：

```text
G1
G2
M0
M1
```

扩展到：

- CDD；
- LEVIR；
- SYSU；
- WHU。

不补多 seed。

## 主方法成功判据

M1 相对 G1：

1. 四数据集至少 3 个 ΔF1 > 0；
2. 任一数据集 ΔF1 不低于 −0.20；
3. 平均 ΔF1 至少 +0.30 才值得继续写论文主线；
4. CDD 不得用 Recall 换 Precision 造成伪增益；
5. WHU/LEVIR 应观察边界、小目标 FN 是否下降；
6. SYSU 应观察 semantic interior error 是否下降；
7. deploy 参数/FLOPs/交换误差必须完全过门。

如果仍呈：

```text
SYSU 明显升
WHU/LEVIR/CDD 不升或下降
```

就说明“Foundation Teacher transfer”在当前 2.9M Student 上仍未解决，应彻底停止这个方向，而不是继续加 Teacher。

---

# 26. 需要记录的诊断指标

除了正式：

- Recall
- Precision
- OA
- F1
- IoU
- Kappa

还必须记录训练诊断：

```text
main_loss

dino_teacher_bce
sam_teacher_boundary_loss

dino_kd_raw
sam_kd_raw

lambda_dino
lambda_sam

grad_norm_main_c4
grad_norm_dino_c4
grad_ratio_dino

grad_norm_main_c2
grad_norm_sam_c2
grad_ratio_sam

router_dino_ratio
router_sam_ratio
router_reject_ratio

dino_advantage_mean
sam_advantage_mean

dino_active_pos_ratio
sam_active_boundary_ratio
```

这些指标能直接回答：

- Teacher 有没有真实 advantage；
- Router 有没有再饿死；
- KD 有没有再压过主损失；
- 两个 Teacher 是否真的互补。

---

# 27. Smoke / real-cache dry run / deploy 测试

## 27.1 Synthetic smoke

必须验证：

1. Student params = `2,913,094`；
2. Student T1/T2 swap error < `1e-6`；
3. DINO teacher target swap error < `1e-6`；
4. SAM structural target swap error < `1e-6`；
5. router 权重逐 region：
   \[
   w_D+w_{SAM}+w_R=1
   \]
6. Teacher/Router tensors detached；
7. Aux heads 梯度能到 Student change features；
8. Teacher cache 不接收梯度；
9. gradient budget 实际满足上限。

---

## 27.2 Real-cache dry run

至少抽：

- CDD 8 样本；
- LEVIR 8；
- SYSU 8；
- WHU 8。

强制枚举：

```text
crop on/off
flip_h on/off
flip_v on/off
exchange on/off
```

可视化或数值检查：

- A/B；
- GT；
- DINO target；
- SAM structural target；

是否严格同几何位置。

这一项必须在任何正式训练之前过。

---

## 27.3 Deploy test

从完整 SAGE checkpoint：

1. 只加载 student；
2. 不安装 DINO/SAM/cache 模块；
3. `switch_to_deploy()`；
4. 前后预测误差 `<1e-6`；
5. 参数 `2,913,094`；
6. FLOPs `2.7475G ± 0.03G`；
7. T1/T2 swap `<1e-6`。

---

# 28. 论文叙事应怎么写

如果实验通过，最有价值的叙事不是：

> “我们把 DINOv3 和 SAM2 融合起来。”

而应该是：

> **Foundation Teacher 在极轻量 binary CD 中并非越强越好，也不是每个区域都可靠。我们把 Teacher transfer 重新表述为 student-conditioned expert selection：学生先暴露当前的语义与结构失败，再由异构 Foundation Teachers 分别提供 semantic / structural knowledge；通过 Reject-aware soft routing 与 gradient-bounded decoupled distillation，只传递当前 Student 可吸收且 Teacher 有实际优势的知识。**

这条叙事与现有工作的差异：

- **vs AM-RADIO / MoVE-KD**：不是通用视觉 encoder 聚合，而是二时相 binary CD 的 failure-type routing；
- **vs JL1-CD**：不是按 change-area ratio 划 specialist，而是按 Student 当前失败类型动态路由；
- **vs DIFO/DAMP**：不是 domain adaptation，而是 fully supervised CD 中 Student difficulty 反向决定 Teacher specialization；
- **vs CrossKD**：不是 detection head cross-head，而是 CD 中 semantic / boundary 两类异构 Teacher target 的解耦辅助头；
- **vs FA-SCRD**：不做 Foundation relation imitation，而是 task-target、region-routed、gradient-bounded transfer。

---

# 29. 当前不应做的 Student 结构改动

为避免下一轮再次“模块堆叠”，下面这些先全部冻结：

- 不给 LWGANet 加新 block；
- 不改 pretrained LWGA residual；
- 不把 ReLU 全换 GELU；
- 不给 decoder 额外 CBAM/SE；
- 不把 TFM 换 Transformer；
- 不加 flow alignment；
- 不引入 BiFA 的完整 ADFF/IND；
- 不把 Teacher feature 注入 deploy decoder；
- 不增加部署参数/FLOPs。

BiFA（TGRS 2024）说明 bitemporal alignment 可能有效，但 ADFF/IND 会改变主网络和计算图；在当前 Teacher-transfer 科学问题尚未解决前，不应和新 KD 同时做。

---

# 30. 立即执行顺序

按严格顺序：

### 1. 先修 P0，不训练
修复/替换低分辨率 DINO cache augmentation replay；新方案只用 full-res task target。

### 2. 写 cache-alignment 单元测试
确认 crop / resize / flip / exchange 完全同步。

### 3. 建立 clean Student failure profile
从当前 clean/joint-BN anchor 生成训练集难度图，detach 存储或在线计算。

### 4. 校准单 DINO Teacher
只用 GT + Student difficulty weighting；不引入 SAM。

### 5. 生成 full-res DINO cache
记录 manifest、checkpoint hash、split fingerprint。

### 6. 做 Safe-DINO 三臂门实验
`G0 clean / G1 joint-BN / G2 safe-DINO`，先 SYSU+WHU。

### 7. G2 不过门则立刻停止
不做 SAM，不继续多 Teacher。

### 8. G2 过门后构建 SAM structural calibrator
使用现有 SAM2 boundary/quality cache；不在线跑 SAM2 到 deploy graph。

### 9. 跑 M0
验证双专家“仅分工、不 routing”是否互补。

### 10. 跑 M1 Full SAGE
加入 Student-conditioned calibration + soft Router + Reject。

### 11. SYSU/WHU 通过后扩四数据集
CDD / LEVIR / SYSU / WHU，seed 2333。

### 12. 最后才考虑 Student deploy graph 结构消融
只有 Teacher transfer 已经成立，且误差诊断明确指向 decoder/TFM 时，才单独做 parameter-neutral Student structural modification。

---

# 31. 仍需补充的证据

在改代码前，当前还缺五类关键证据：

## 31.1 Teacher 自身到底比 Student 强多少

当前 repo 没有形成统一表格：

```text
DINO teacher standalone F1/IoU
SAM structural boundary quality
Student clean F1/IoU
```

四数据集同协议对比。

没有这一表，就不能直接说“强 Teacher”。

---

## 31.2 DINO 与 SAM 的 oracle complementarity

建议先离线统计：

```text
Student wrong & DINO right
Student wrong & SAM-structure right
Student wrong & both right
Student wrong & neither right
```

分：

- changed interior；
- change boundary；
- unchanged background。

如果 SAM 对 Student 错误几乎没有独立覆盖，则根本不值得进双 Teacher。

---

## 31.3 FA-SCRD P0 对历史负增益贡献有多大

当前不能知道：

- −0.33pp 主要来自 relation KD 本身；
- 还是一部分来自 crop target 错位。

不要求重跑完整 FA-SCRD，但至少要做一个 real-cache 对齐可视化/数值审计，并在论文中避免把旧实验解释得过度确定。

---

## 31.4 实际梯度比例

旧日志只有 loss 值，没有：

```text
||grad_main||
||grad_kd||
cos(grad_main, grad_kd)
```

下一轮必须记录，否则无法证明“KD 压过主损失”是梯度层面的事实，而只能说 loss scale 很大。

---

## 31.5 SAM2 structural target 的定义是否真有增量

第一版建议只用：

\[
|q_1B_1-q_2B_2|
\]

不要一开始恢复复杂 instance matching。

如果 boundary-only 已经没有独立 oracle coverage，则更复杂的 SAM instance routing 不值得继续。

---

# 32. 最终决策

**下一步不建议：**
- 继续 FA-SCRD relation KD；
- 直接 DINO+SAM feature fusion；
- 再做 EMA reciprocal teacher；
- 先改 A2Net deploy TFM/decoder；
- 继续尝试单一固定 teacher trick。

**唯一建议：**

> **SAGE-CD：先修 Teacher Cache 几何正确性，以 clean Student 的语义/边界失败画像校准 DINOv3 与 SAM2 两类异构专家；Student 训练时用 DINO/SAM/Reject 软路由，只把 Teacher 确实更可靠的 task-specific semantic/boundary target 送入独立训练期 auxiliary heads，并用 gradient budget 保证 KD 不超过主任务梯度。部署仍为原 2,913,094 参数 A2Net-LWGANet-L0。**

它最值得做的原因不是“机制最复杂”，而是它同时正面回答当前项目四个真正没有解决的问题：

1. **Teacher 是否真的有局部 advantage？**
2. **DINO 与 SAM 是否真的互补？**
3. **Student 是否能在不被 Teacher 压制的情况下吸收知识？**
4. **这种知识能否完全留在训练期，而部署仍保持 2.9M？**

只要第一关 Safe-DINO 仍过不了，就应该停止 Foundation Teacher 方向，而不是继续加模块。

---

# 33. 主要来源

## 项目代码与结果

- Repository: https://github.com/YuqiWang-code/LS-Rep_BCD
- Reviewed commit: https://github.com/YuqiWang-code/LS-Rep_BCD/commit/9d5096c7aa93b000e03c9e6dd3567504d17bd6eb
- `README.md` §6 历史归档
- `docs/experiment_metrics.xlsx`
- `train_scripts/SCTC/Run1/README.md`
- `models/a2net.py`
- `models/decoder/a2net_decoder.py`
- `models/backbone/lwganet.py`
- `models/distill/cache_dataset.py`
- `models/distill/relation_kd.py`
- `models/distill/failure_router.py`
- `models/distill/fa_scrd_teacher.py`
- `models/tools/train_teacher.py`
- `models/tools/generate_teacher_cache.py`
- `models/scripts/train.py`

## 2024–2026 文献

1. Ranzinger et al., **AM-RADIO**, CVPR 2024.  
   https://openaccess.thecvf.com/content/CVPR2024/html/Ranzinger_AM-RADIO_Agglomerative_Vision_Foundation_Model_Reduce_All_Domains_Into_One_CVPR_2024_paper.html  
   https://github.com/NVlabs/RADIO

2. Cao et al., **MoVE-KD**, CVPR 2025.  
   https://arxiv.org/abs/2501.01709  
   https://github.com/hey-cjj/MoVE-KD

3. Liu et al., **JL1-CD / MTKD**, IEEE TIM 2026.  
   https://arxiv.org/abs/2502.13407  
   https://github.com/circleLZY/JL1-CD

4. Tang et al., **Source-Free Domain Adaptation with Frozen Multimodal Foundation Model (DIFO)**, CVPR 2024.  
   https://openaccess.thecvf.com/content/CVPR2024/html/Tang_Source-Free_Domain_Adaptation_with_Frozen_Multimodal_Foundation_Model_CVPR_2024_paper.html  
   https://github.com/tntek/source-free-domain-adaptation

5. Du et al., **Domain-Agnostic Mutual Prompting (DAMP)**, CVPR 2024.  
   https://openaccess.thecvf.com/content/CVPR2024/html/Du_Domain-Agnostic_Mutual_Prompting_for_Unsupervised_Domain_Adaptation_CVPR_2024_paper.html  
   https://github.com/TL-UESTC/DAMP

6. Starodubcev et al., **Your Student is Better Than Expected**, CVPR 2024.  
   https://arxiv.org/abs/2312.10835  
   https://github.com/yandex-research/adaptive-diffusion

7. Wang et al., **CrossKD**, CVPR 2024.  
   https://openaccess.thecvf.com/content/CVPR2024/html/Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection_CVPR_2024_paper.html  
   https://github.com/jbwang1997/CrossKD

8. Sun et al., **Logit Standardization in Knowledge Distillation**, CVPR 2024 Highlight.  
   https://openaccess.thecvf.com/content/CVPR2024/html/Sun_Logit_Standardization_in_Knowledge_Distillation_CVPR_2024_paper.html  
   https://github.com/sunshangquan/logit-standardization-KD

9. Miles et al., **VkD: Improving Knowledge Distillation using Orthogonal Projections**, CVPR 2024.  
   https://openaccess.thecvf.com/content/CVPR2024/html/Miles_VkD_Improving_Knowledge_Distillation_using_Orthogonal_Projections_CVPR_2024_paper.html  
   https://github.com/roymiles/VkD

10. Zhang et al., **BiFA: Remote Sensing Image Change Detection With Bitemporal Feature Alignment**, IEEE TGRS 2024.  
    Official code: https://github.com/zmoka-zht/BiFA

---

**文档状态：方案评审版。未修改仓库代码、未启动实验、未生成或覆盖任何 checkpoint/cache。**
