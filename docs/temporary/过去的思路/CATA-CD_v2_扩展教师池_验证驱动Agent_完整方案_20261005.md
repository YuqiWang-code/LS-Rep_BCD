# CATA-CD v2：扩展教师池、验证驱动的教师选择 Agent 与可部署自适应模块

> 项目：LS-Rep_BCD / LS-Rep_BCD_RSML_3  
> 日期：2026-10-05  
> 任务：256×256 全监督遥感二值变化检测（SYSU-CD-256、LEVIR-CD-256、WHU-CD-256、CDD-CD-256）  
> 学生：A2Net-LWGANet-L0  
> 当前部署参数：2,913,094（约 2.91M）  
> 总部署参数硬约束：< 5M  
> 可新增部署参数理论上限：2,086,906（约 2.09M）

---

## 目录

1. [结论先行](#1-结论先行)
2. [本轮修正后的研究问题](#2-本轮修正后的研究问题)
3. [必须纠正的三点认识](#3-必须纠正的三点认识)
4. [网络调研：可扩展的教师基础模型与权重](#4-网络调研可扩展的教师基础模型与权重)
5. [最终建议的教师候选池分层](#5-最终建议的教师候选池分层)
6. [为什么不能先规定“某教师适合某数据集”](#6-为什么不能先规定某教师适合某数据集)
7. [总体技术路线：先验证，再建 Registry，再训练 Agent](#7-总体技术路线先验证再建-registry再训练-agent)
8. [Teacher Package：Agent 真正选择的不是裸教师](#8-teacher-packageagent-真正选择的不是裸教师)
9. [Teacher Cache v2：扩展教师池后的统一缓存协议](#9-teacher-cache-v2扩展教师池后的统一缓存协议)
10. [教师能力验证协议](#10-教师能力验证协议)
11. [数据集特征与教师能力特征](#11-数据集特征与教师能力特征)
12. [真正的 Agent：离线上下文 Bandit + None/拒绝动作](#12-真正的-agent离线上下文-bandit--none拒绝动作)
13. [可保留到推理的 Deployable Change Adapter（DCA）](#13-可保留到推理的-deployable-change-adapterdca)
14. [训练图与部署图](#14-训练图与部署图)
15. [参数量与 FLOPs 预算](#15-参数量与-flops-预算)
16. [完整实验矩阵与实验 ID](#16-完整实验矩阵与实验-id)
17. [成功判据、失败判据与可证伪假设](#17-成功判据失败判据与可证伪假设)
18. [逐文件实现方案](#18-逐文件实现方案)
19. [Smoke / Cache / Deploy / Resume 校验](#19-smoke--cache--deploy--resume-校验)
20. [推荐启动顺序](#20-推荐启动顺序)
21. [论文创新点如何组织](#21-论文创新点如何组织)
22. [主要风险与止损条件](#22-主要风险与止损条件)
23. [参考文献与官方资源](#23-参考文献与官方资源)
24. [立即执行顺序](#24-立即执行顺序)
25. [仍需补充的证据](#25-仍需补充的证据)

---

# 1. 结论先行

## 1.1 这条路线可以继续，但必须从“多教师可学习互选”改成“教师能力先验验证 + 选择 Agent”

本轮修正后，我建议将主线定义为：

> **CATA-CD v2：Capability-Validated Adaptive Teacher Agent for Lightweight Change Detection**  
> 先把每个候选基础模型做成独立 Teacher Package，在四数据集上用完全相同的学生、初始化、40K steps、seed=2333 逐一验证；只有被实验证明“至少在某些数据集确实产生正迁移”的教师包，才能进入 Teacher Capability Registry。之后 Agent 才有资格学习“数据集特征 → 教师包/None”的选择策略。

这与旧路线最关键的区别是：

- **旧路线**：先假设教师有用，再设计多教师权重、互选、互促；
- **新路线**：先用完整单教师实验把“教师有没有用、在哪里有用”变成实证事实，再让 Agent 学习这些事实背后的可解释映射。

因此，教师数量增加并不等于最终 Agent 同时使用更多教师。研究阶段可以广泛筛选，**最终 Agent 候选动作建议只保留 3–5 个互补且被证明有效的 Teacher Package + None**。

---

## 1.2 不再预设“某类数据集应该用某个教师”

以下都只能写为「待验证假设」：

- “变化像素多、场景复杂 → SAM2”；
- “变化稀疏、小目标 → DINOv3”；
- “语义类别复杂 → CLIP/RemoteCLIP”；
- “高分辨率建筑 → VHR 遥感基础模型”。

必须通过同协议实验得到最终教师—数据集适配矩阵。

项目实际 Train 变化像素比为：

| 数据集 | Train 变化像素比 |
|---|---:|
| SYSU-CD-256 | 21.1% |
| CDD-CD-256 | 11.9% |
| LEVIR-CD-256 | 4.1% |
| WHU-CD-256 | 3.4% |

因此 CDD 也不能再和 LEVIR/WHU 简单统一称作“极低变化占比”。

---

## 1.3 推理时可以保留一个小型可学习模块，但 Teacher/Cache 仍不得进入部署图

用户本轮修正是合理的：

- 当前学生部署参数：`2,913,094`；
- 硬上限：`5,000,000`；
- 理论剩余额度：`2,086,906`。

因此不必把所有可学习模块都删除。

本方案将 Agent 拆成两部分：

1. **Teacher Selector**：负责决定训练时使用哪个已验证 Teacher Package；Teacher/Cache 相关部分仍是训练期机制；
2. **Deployable Change Adapter（DCA）**：学生侧的小型、时相交换对称的可学习模块，训练时接受蒸馏塑形，推理时保留，但**只读取学生自身特征，不读取任何 Teacher/Cache/teacher map**。

推荐首版 DCA 仅约 **0.31M 参数**，这样总部署参数约：

`2.913M + 0.308M ≈ 3.221M`

远低于 5M，并为后续结构优化保留约 1.78M 安全余量。

不建议第一版直接吃满 2.09M，因为研究目标仍是“极轻量”，而不是“在 5M 上限附近堆模块”。

---

## 1.4 第一批建议真正生成 Cache 并跑完整 40K 的教师

### Wave-A：优先正式验证

1. SAM2.1 Hiera-Large（已有 Cache，重新纳入新协议）
2. DINOv2 ViT-B/14（已有 Cache，重新纳入新协议）
3. DINOv3 ViT-B/16 LVD-1689M **中立 frozen 版本**（需要重新生成，不能直接拿现有 SYSU 微调版当中立教师）
4. DINOv3 ViT-L/16 SAT-493M（新增，遥感卫星预训练）
5. RemoteCLIP ViT-L/14（新增，遥感视觉语言语义）
6. MaRS-Base RGB（新增，AAAI 2026，超高分辨率遥感）

### Wave-B：Wave-A 后按能力缺口扩展

7. AnySat（CVPR 2025）
8. UniverSat Base（NeurIPS 2026）
9. C-RADIOv3-B / RADIOv2.5 系（CVPR 2025 系列）
10. TerraMind-1.0-Small/Tiny（ICCV 2025）
11. SatMAE++ ViT-L FMoW-RGB（CVPR 2024）

### 暂缓

- SkySense（CVPR 2024）
- SkySense++（Nature Machine Intelligence 2025）

原因不是模型弱，而是依赖链偏旧、模型较重、原仓库推荐 PyTorch 1.13.1/MMCV 老栈，与当前 `lsrep`（PyTorch 2.14+cu132）直接集成风险高。可在后续单独隔离环境生成 Cache，**不能污染 cd_base/lsrep**。

---

# 2. 本轮修正后的研究问题

主问题不再是：

> “SAM2 和 DINOv3 怎么互选？”

而变为三个层级：

### RQ1：不同 Foundation Model 的知识在四个 CD 数据集上到底是否可蒸馏？

先得到实证 Teacher Capability Matrix。

### RQ2：能否根据数据集自身可测特征和低成本 Teacher Probe，预测哪一个 Teacher Package 会给轻量学生带来正迁移？

这是 Agent 的核心科学问题。

### RQ3：能否把教师指导形成的“适配能力”部分固化在一个 <2.09M 的学生侧可部署模块中，在推理不访问任何大模型的前提下继续受益？

这是本轮新增且很有论文价值的结构问题。

---

# 3. 必须纠正的三点认识

## 3.1 当前 FA-SCRD DINOv3 Cache 不是中立 DINOv3

项目现有 `FA_SCRD_DINOv3_CD`：

- DINOv3 ViT-B/16；
- frozen backbone + LoRA Q/V r=8 + 轻量 change head；
- **先在 SYSU 上微调 2000 步**；
- 再给 SYSU/WHU/CDD/LEVIR 四数据集生成 Cache。

因此它测到的是：

> `SYSU-adapted DINOv3 teacher package → 四数据集迁移`

而不是：

> `原生 frozen DINOv3 的跨数据集能力`。

所以新 Registry 中至少应区分：

- `D3-LVD-Neutral`：官方 frozen DINOv3 LVD 权重；
- `D3-SYSU-Adapted`：现有历史 Cache，只作为对照，不作为“原生 DINOv3”结论。

---

## 3.2 教师能力与数据集属性不能靠名称推断

例如：

- DINO 系模型语义强，但 ViT/16 在 256×256 输入下天然只有 16×16 patch 网格；
- SAM2 强边界/实例先验，但并不自动等于“复杂多类变化最强”；
- RemoteCLIP 是遥感视觉语言模型，但 binary CD 并没有直接语义类别监督；
- VHR 遥感 FM 与高分辨率建筑数据更匹配只是合理动机，不是结果。

因此所有“适配关系”在完整实验前均为「待验证假设」。

---

## 3.3 最终 Agent 的动作应该是 Teacher Package，而不是裸 backbone

不同 Foundation Model 的原生输出差异太大：

- SAM2：实例、边界、mask quality；
- DINO：patch dense features；
- CLIP/RemoteCLIP：视觉 patch + global semantic embedding；
- Swin/VHR FM：多尺度 feature pyramid；
- AnySat/UniverSat：传感器/分辨率条件化特征。

如果强行用完全相同的 KD loss，实际上会人为压制某些教师的强项。

所以 Agent 的动作定义为：

`a_j = {Teacher_j + CacheSchema_j + MinimalTranslator_j}`

即 **Teacher Package**。

这样实验结论应写成：

> “MaRS-package 在 WHU 上对当前轻量学生有效”

而不是夸大成：

> “MaRS 本身天然最适合 WHU”。

---

# 4. 网络调研：可扩展的教师基础模型与权重

以下只列可核验的官方仓库/官方模型仓库。

## 4.1 已有教师

### T-SAM2：SAM 2.1 Hiera Large

- 类型：通用可提示分割 Foundation Model
- 项目状态：现有 `SAMStruct` Cache 已覆盖四个 train split
- 官方 GitHub：<https://github.com/facebookresearch/sam2>
- 官方 SAM2.1 Large 权重：  
  <https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt>
- 官方仓库同时提供 tiny/small/base+/large 四种 2.1 checkpoint。

适合作为候选的机制动机：

- 对象结构、边界和实例先验强；
- 现有 Cache 已经把其结构知识显式保存为 `instance_id / boundary / quality`。

适配数据集：**待验证假设**。

---

### T-D2：DINOv2 ViT-B/14

- 类型：通用 self-supervised dense representation
- 项目状态：现有 `OVCDistill` Cache
- 官方 GitHub：<https://github.com/facebookresearch/dinov2>
- 官方权重/加载入口：<https://github.com/facebookresearch/dinov2#pretrained-models>
- ViT-B/14 官方直接 checkpoint：  
  <https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth>

适配数据集：**待验证假设**。

---

### T-D3-LVD：DINOv3 ViT-B/16，LVD-1689M

- 类型：通用 self-supervised dense feature
- 官方 GitHub：<https://github.com/facebookresearch/dinov3>
- 官方模型卡：<https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md>
- 官方权重申请入口：<https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>

注意：DINOv3 官方权重需要申请，批准后 Meta 通过临时下载 URL 发放。

新路线必须重新生成 **frozen neutral cache**，不能用现有 SYSU 2000-step 适配后的 Cache 代替。

适配数据集：**待验证假设**。

---

## 4.2 新增强候选 1：DINOv3 ViT-L/16 SAT-493M

### 为什么值得优先

DINOv3 官方额外发布了专门在卫星影像上训练的 SAT-493M 系列：

- SAT-493M：4.93 亿张 512×512 Maxar RGB 正射影像；
- 地面分辨率约 0.6m；
- 官方提供 ViT-L/16 distilled 与 ViT-7B/16。

这与 LEVIR/SYSU 等高分辨率 RGB 遥感影像在领域上明显比自然图像 DINO 更接近。

### 链接

- GitHub：<https://github.com/facebookresearch/dinov3>
- 官方模型卡：<https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md>
- 官方权重申请：<https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>

### 推荐权重

`DINOv3 ViT-L/16 SAT-493M`，不使用 7B。

理由：Teacher Cache 离线可以接受 300M 教师，但 7B 对 2×5090 的 Cache 生成吞吐、显存和工程复杂度没有必要。

### Cache 建议

- block 11 / block 23；
- 不存 raw 1024-channel A/B 全特征；
- 先形成对称时相证据，再压缩到 128 channels；
- FP16；
- 目标 Cache ≤12GB / teacher / 四数据集总 train。

### 适配数据集

**待验证假设。**

合理动机是 VHR/RS domain match，但不能提前断言 LEVIR/WHU/SYSU/CDD 中谁最佳。

---

## 4.3 新增强候选 2：RemoteCLIP ViT-L/14

### 论文层级

- `RemoteCLIP: A Vision Language Foundation Model for Remote Sensing`
- IEEE TGRS 2024
- SCI 权威期刊，不是 CCF-A 会议。

### 链接

- GitHub：<https://github.com/ChenDelong1999/RemoteCLIP>
- 官方 Hugging Face：<https://huggingface.co/chendelong/RemoteCLIP>
- ViT-L/14 权重：  
  <https://huggingface.co/chendelong/RemoteCLIP/blob/main/RemoteCLIP-ViT-L-14.pt>
- ViT-B/32 权重：  
  <https://huggingface.co/chendelong/RemoteCLIP/blob/main/RemoteCLIP-ViT-B-32.pt>

### 推荐

优先 ViT-L/14，不优先 ViT-B/32。

原因：B/32 的 patch 太粗，不适合拿来做 dense binary CD teacher；L/14 至少保留更细的 patch grid。

### Cache 建议

主实验先只缓存：

- 视觉 patch token 的对称差异/关系；
- global visual embedding；
- 不先引入人工类别文本 prompt。

文本 prompt 可作为后续独立消融，避免把人为 prompt 设计混入“基础模型教师是否有效”的首轮结论。

### 适配数据集

**待验证假设。**

语义复杂的 SYSU 可能是潜在受益对象，但绝不能在实验前写成结论。

---

## 4.4 新增强候选 3：MaRS-Base RGB

### 论文层级

- `MaRS: A Multi-Modality Very-High-Resolution Remote Sensing Foundation Model with Cross-Granularity Meta-Modality Learning`
- AAAI 2026
- CCF-A 会议。

### 为什么高度相关

MaRS 是专门面向 **Very-High-Resolution Remote Sensing** 的模型，支持 RGB 与 SAR，且官方下游列表直接包含 WHU-CD。

### 链接

- GitHub：<https://github.com/WanderRainy/MaRS>
- 官方权重 Zenodo：<https://zenodo.org/records/17800805>
- 推荐文件：`mars_base_rgb_encoder_only.pth`

官方 README 的 RGB 示例使用：

`SwinV2-Base / features_only=True / in_chans=3`

因此它天然适合生成多尺度 dense Cache。

### Cache 建议

优先取：

- 1/8 feature；
- 1/16 feature；
- 各压缩到 128 channels；
- 只保存双时相对称 evidence + confidence/statistics。

### 适配数据集

**待验证假设。**

其 VHR 领域匹配是强动机，但不能因为官方做过 WHU-CD 就把 WHU 正增益视为既定事实——我们的学生、KD 接口、256 切片、训练预算完全不同。

---

## 4.5 新增候选 4：AnySat

### 论文层级

- `AnySat: One Earth Observation Model for Many Resolutions, Scales, and Modalities`
- CVPR 2025 Highlight
- CCF-A。

### 特点

- 一个模型处理多分辨率、多尺度、多传感器；
- 官方输入注册里直接包含：
  - `spot`: 3-channel RGB, 1m；
  - `aerial`: RGB/NIR, 0.2m；
- 论文下游任务包括 change detection。

### 链接

- GitHub：<https://github.com/gastruc/AnySat>
- 官方 Hugging Face：<https://huggingface.co/g-astruc/AnySat>
- 标准权重 `AnySat.pth`：  
  <https://huggingface.co/g-astruc/AnySat/blob/main/models/AnySat.pth>
- Full 权重：  
  <https://huggingface.co/g-astruc/AnySat/blob/main/models/AnySat_full.pth>

### 推荐

先使用标准 `AnySat.pth`，输入按 `spot` 3-channel RGB 路径做 wrapper smoke。

由于项目数据没有可靠逐图 GSD 元数据，**不要伪造真实 GSD**。输入注册方式和尺度映射必须在 wrapper 中明确记录。

### 适配数据集

**待验证假设。**

---

## 4.6 新增候选 5：UniverSat Base

### 论文层级

- `UniverSat: Resolution- and Modality-Agnostic Transformers for Earth Observation`
- NeurIPS 2026（官方仓库标注 accepted）
- CCF-A。

### 特点

非常适合本项目调研的原因不是“它一定更准”，而是：

- 一个权重支持多传感器/多分辨率；
- `spot` 原生支持 1m VHR RGB；
- 输出空间分辨率可在 inference 时指定；
- sub-patch skip cross-attention 强调细粒度空间细节；
- 官方明确支持 frozen backbone feature extraction。

### 链接

- GitHub：<https://github.com/gastruc/UniverSat>
- Hugging Face：<https://huggingface.co/g-astruc/UniverSat>
- 权重文件：  
  <https://huggingface.co/g-astruc/UniverSat/blob/main/model.safetensors>

Base 约 201M。

### Cache 建议

利用其可指定输出网格的特点，优先测试：

- 1/8 spatial evidence；
- 1/16 spatial evidence；

但为了公平，首轮不要为 UniverSat 单独给更高分辨率 Cache 而让其他教师吃亏；应限定统一 Cache 字节预算。

### 适配数据集

**待验证假设。**

---

## 4.7 新增候选 6：C-RADIOv3-B / RADIOv2.5 系

### 论文层级

- AM-RADIO：CVPR 2024
- RADIOv2.5：CVPR 2025
- C-RADIOv3-B 是 NVIDIA 2025 后续公开权重版本。

### 为什么有价值

RADIO 本身是一个“被多教师蒸馏后的统一视觉基础模型”，其知识来源包含 DINO/CLIP/SAM 类能力。

它与直接同时使用 SAM+DINO+CLIP 的区别是：

- Cache 只需一个模型；
- 输出已被统一到一个 feature space；
- 可以测试“统一 agglomerated teacher 是否比独立 teacher routing 更稳”。

这对旧路线“多教师相互冲突”是一个很好的对照。

### 链接

- GitHub：<https://github.com/NVlabs/RADIO>
- C-RADIOv3-B 权重：<https://huggingface.co/nvidia/C-RADIOv3-B>
- RADIOv2.5 CVPR 2025 论文：  
  <https://openaccess.thecvf.com/content/CVPR2025/html/Heinrich_RADIOv2.5_Improved_Baselines_for_Agglomerative_Vision_Foundation_Models_CVPR_2025_paper.html>

### 推荐

使用 `C-RADIOv3-B`（约 90M 级）而不是 L/H/g。

### 适配数据集

**待验证假设。**

它可能成为“稳健通用教师”，也可能因为过度融合而失去某些 CD 特化细节，需要实验回答。

---

## 4.8 第二梯队：TerraMind 1.0 Small/Tiny

### 论文层级

- `TerraMind: Large-Scale Generative Multimodality for Earth Observation`
- ICCV 2025
- CCF-A。

### 链接

- GitHub：<https://github.com/IBM/terramind>
- Tiny：<https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-tiny>
- Small：<https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-small>
- Tiny 权重文件：  
  <https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-tiny/blob/main/TerraMind_v1_tiny.pt>
- Small 权重文件：  
  <https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-small/blob/main/TerraMind_v1_small.pt>

TerraMind 官方支持 `RGB` raw input modality。

### 为什么不是 Wave-A

- 多模态 generative FM 的接口更复杂；
- 本项目只有 RGB 双时相，无法发挥它全部跨模态能力；
- 先验证更直接匹配 VHR RGB 的 MaRS/DINOv3-SAT/AnySat 更划算。

适配数据集：**待验证假设。**

---

## 4.9 第二梯队：SatMAE++ ViT-L FMoW-RGB

### 论文层级

- `Rethinking Transformers Pre-training for Multi-Spectral Satellite Imagery`
- CVPR 2024
- CCF-A。

### 链接

- GitHub：<https://github.com/techmn/satmae_pp>
- FMoW-RGB ViT-L Pretrain 权重：  
  <https://huggingface.co/mubashir04/checkpoint_ViT-L_pretrain_fmow_rgb>

### 特殊注意

官方明确提醒 FMoW-RGB 预训练权重使用 **BGR** 图像顺序。

如果生成 Cache 时错误用 RGB 顺序，会人为损害教师能力，因此必须将颜色顺序写进 cache manifest/config hash。

### 为什么第二梯队

它对遥感 multi-scale 有明确设计，但原始任务更偏 scene representation，不如 MaRS/VHR 或 AnySat/UniverSat 对 dense change 更直接。

适配数据集：**待验证假设。**

---

## 4.10 暂缓：SkySense / SkySense++

### SkySense

- CVPR 2024，CCF-A
- GitHub：<https://github.com/Jack-bo1220/SkySense>
- 官方 README 的 pretrained weight 入口：该 GitHub README 中的官方 Notion 权重链接
- 高分辨率 RGB backbone：Swin Transformer v2 Huge

### SkySense++

- Nature Machine Intelligence 2025
- SCI 顶级综合 AI 期刊，不是 CCF-A 会议
- GitHub：<https://github.com/kang-wu/SkySensePlusPlus>
- 官方 README 提供模型权重入口

### 暂缓原因

官方下游栈依赖 PyTorch 1.13.1、旧 MMCV/MMDetection/MMSeg 等，与本项目 `lsrep` 不兼容。

若后续使用：

- 只允许新建隔离环境用于 **离线 Cache 生成**；
- 不修改 `lsrep` / `cd_base`；
- Cache 输出遵守项目统一 schema；
- 训练主工程仍回到 `lsrep`。

---

# 5. 最终建议的教师候选池分层

## 5.1 Wave-A：必须先做

| ID | Teacher Package | 知识类型 | 是否已有 Cache | 优先级 |
|---|---|---|---|---|
| T0 | None | 无教师 | — | 必须 |
| T1 | SAM2.1-H + Struct Translator | 实例/边界/结构 | 是 | S |
| T2 | DINOv2-B/14 + Relation Translator | 通用 dense SSL | 是 | A |
| T3 | DINOv3-B/16-LVD-Neutral + Relation Translator | 强 dense SSL | 否，需重建 | S |
| T4 | DINOv3-L/16-SAT493M + RS Dense Translator | 卫星 VHR dense SSL | 否 | S |
| T5 | RemoteCLIP-L/14 + Semantic Translator | 遥感视觉语言语义 | 否 | S |
| T6 | MaRS-Base-RGB + Multiscale Translator | VHR 遥感多尺度 | 否 | S |

Wave-A 共 6 个有教师动作 + None。

其中 T3 与现有 FA-SCRD DINOv3 Cache 必须区分。

---

## 5.2 Wave-B：仅在 Wave-A 后按缺口扩展

| ID | Teacher | 价值 | 进入条件 |
|---|---|---|---|
| T7 | AnySat | 多分辨率/多传感器，CVPR25 | Wave-A 对某数据集无正教师或 scale gap 明显 |
| T8 | UniverSat | 可控输出网格、细粒度，NeurIPS26 | 小目标/边界能力仍缺 |
| T9 | C-RADIOv3-B | SAM/DINO/CLIP 统一知识 | 独立教师冲突仍明显 |
| T10 | TerraMind-Small/Tiny | 多模态 EO generative | 语义/域泛化能力仍缺 |
| T11 | SatMAE++ RGB | 多尺度遥感预训练 | scale 特征不足时 |

---

## 5.3 Agent 最终动作池不等于调研池

最后进入 Agent 的教师应该满足：

1. 至少在 1 个数据集上有明确正增益；
2. 与已入池教师的优势不完全冗余；
3. Train-only capability probe 能解释其正/负迁移；
4. Cache 成本在允许范围内；
5. Teacher Package 的增益不是靠额外部署参数造成。

建议最终动作数：

`3–5 teacher packages + None`

而不是 10 个甚至更多。

原因与 DynaKD/多教师 KD 文献的共同观察一致：**教师数量更多并不保证更好**。

---

# 6. 为什么不能先规定“某教师适合某数据集”

正式路线中禁止写类似：

```text
SYSU -> SAM2
WHU -> DINOv3
LEVIR -> DINOv3
CDD -> RemoteCLIP
```

除非实验已经完成。

正确做法是先定义可能影响教师效用的 Dataset Signature：

### 标签几何

- change ratio；
- connected component 数量；
- component area P10/P25/P50/P75/P90；
- small/medium/large change area fraction；
- perimeter / area；
- boundary density；
- fragmentation；
- changed-region compactness。

### 双时相外观差异

- RGB mean/std shift；
- histogram distance；
- edge disagreement；
- illumination/contrast shift；
- low-frequency difference；
- high-frequency texture difference；
- approximate registration inconsistency。

### 学生可学习难度

从 clean anchor 训练过程统计：

- early BCE/Dice；
- positive pixel error；
- false-positive / false-negative imbalance；
- boundary error；
- component-size stratified error。

然后实验结束后分析：

> 什么 Signature 与什么 Teacher Package 的正迁移相关？

而不是先把经验猜测硬编码进 Agent。

---

# 7. 总体技术路线：先验证，再建 Registry，再训练 Agent

完整流程分 5 阶段。

---

## Stage 0：建立干净部署主图基线

### C0：Clean Student

`A2Net-LWGANet-L0`，2.913094M。

### C1：C0 + DCA，但没有任何 Teacher

目的：证明保留到推理的 DCA 本身是否有效，并把其影响从 Teacher 效用中剥离。

后续所有教师验证都以 **C1** 为 anchor：

`Teacher utility = T_j - C1`

这样 Teacher Package 的所有实验具有完全相同的部署图。

---

## Stage 1：Teacher Cache Compatibility Audit

每个候选教师先通过无 40K 训练的 P0 audit：

- 官方权重 hash；
- normalization；
- RGB/BGR；
- teacher native input resolution；
- A/B 顺序；
- output spatial size；
- crop/flip/resize replay；
- time exchange；
- NaN/Inf；
- Cache 读写速度；
- Cache 总容量估计；
- 8–32 sample 可视化。

不过 audit 只决定“能不能跑”，**不能决定教师是否有效**。

---

## Stage 2：Teacher Package Isolation Runs

每个教师在四数据集分别：

- 同一个 C1 学生；
- 同一个 ImageNet 初始化；
- seed 2333；
- batch 64；
- 40K steps；
- 同 augmentation；
- 同 main loss；
- Teacher/Cache/Translator 是唯一变量。

输出完整矩阵：

`U[D,T] = Metric(C1 + T) - Metric(C1)`

必须逐数据集报告：

- Precision；
- Recall；
- OA；
- F1；
- IoU；
- Kappa；
- deploy params；
- FLOPs。

正式结果只读 `train_log.txt` 最后完整 `=== TEST RESULTS ===` 区块。

---

## Stage 3：建立 Teacher Capability Registry

Registry 不只存“谁 F1 最高”，还存：

```yaml
teacher_id:
  teacher_meta:
    objective:
    pretraining_domain:
    native_stride:
    dense_or_global:
    multiscale:
  cache_meta:
    schema:
    bytes_per_sample:
    generation_time:
  dataset_probe:
    SYSU:
      train_only_probe: [...]
      val_utility: ...
      formal_test_delta: ...
    WHU: ...
```

### 两类数值严格区分

1. `val_utility`：允许 Agent 学习/选择；
2. `formal_test_delta`：只用于最终科研结论，禁止回流成 Agent 训练标签。

---

## Stage 4：训练 Dataset Teacher Agent

Agent 输入：

- Dataset Signature；
- 每个教师的 train-only capability probe；
- teacher metadata embedding。

输出：

`{T1,T2,...,Tk,None}` 中的一个动作。

不是同时融合多个教师。

---

## Stage 5：Agent 选择后重新从头训练 M1

Agent 得到某数据集选择，例如：

`SYSU -> T5`

正式 M1 必须：

- 清空实验目录；
- 从同一 ImageNet student 初始化；
- seed 2333；
- 40K；
- 使用 T5 Cache；
- 不加载 T5 isolation run 的 student checkpoint；
- 不续训。

这样证明的是选择策略 + 教师包本身，而不是 checkpoint 微调收益。

---

# 8. Teacher Package：Agent 真正选择的不是裸教师

## 8.1 SAM2 Package

输入 Cache：

- instance id；
- boundary；
- mask quality。

Minimal Translator：`Structural Translator`

目标：

- boundary consistency；
- instance-internal coherence；
- foreground/background structural separation。

不直接把 SAM mask 注入主预测。

---

## 8.2 DINOv2 / DINOv3 Package

输入：

- normalized dense temporal features / compact symmetric delta；
- local relation。

Translator：`Relation Translator`

建议避免再次复刻 FA-SCRD 中过强 relation KD。

第一版采用：

- feature relation cosine target；
- bounded/capped auxiliary loss；
- teacher reliability 仅做 loss weighting，不进入主图。

---

## 8.3 RemoteCLIP Package

输入：

- patch visual embedding；
- global visual embedding。

Translator：`Semantic-Local Translator`

核心：

- global semantic consistency；
- local patch relation；
- 不强迫学生逐通道 MSE 拟合 CLIP feature。

首轮不引入手写 text prompt。

---

## 8.4 MaRS Package

输入：

- Swin multi-scale features。

Translator：`VHR Multiscale Translator`

重点：

- 1/8 细节；
- 1/16 语义；
- teacher-student scale-to-scale matching。

---

## 8.5 AnySat / UniverSat Package

重点不是语义类别，而是：

- resolution-aware dense representation；
- fine-grained spatial structure。

Translator：`Resolution-aware Translator`

避免为了模型支持多模态而虚构不存在的 NIR/SAR 输入。

---

## 8.6 C-RADIO Package

使用单统一 spatial representation。

Translator：`Unified Dense Translator`

它是非常重要的对照：

> 一个已经把多教师知识聚合好的 FM，是否比本项目自己做教师选择更稳？

若 C-RADIO 单教师就全面优于 Agent，多教师选择主张会被削弱，这是必须接受的可证伪结果。

---

# 9. Teacher Cache v2：扩展教师池后的统一缓存协议

教师数增加后，继续保存完整 raw feature 会导致存储快速膨胀。

四数据集 train 样本总数：

`10000 + 7120 + 12000 + 5947 = 35067`

若每个新教师几十 GB，8 个教师就可能达到数百 GB。

因此建议新增 Compact Cache v2。

---

## 9.1 通用字段

```python
{
    "teacher_id": str,
    "sample_id": str,
    "local_change": Tensor[128, Ht, Wt],   # FP16
    "local_change_2": optional Tensor[128, H2, W2],
    "confidence": Tensor[1, Ht, Wt],
    "global_desc": Tensor[128],
    "aux": dict,
    "meta": {
        "weight_hash": ...,
        "normalization": ...,
        "color_order": ...,
        "teacher_input_size": ...,
        "cache_schema_version": 2,
    }
}
```

---

## 9.2 为什么优先缓存“对称变化证据”而非完整 A/B features

定义：

`E_T = |Norm(P(F_T^A)) - Norm(P(F_T^B))|`

其中 `P` 仅用于离线无标签降维，例如 frozen PCA / fixed projection。

优点：

- 天然 A/B 交换不变；
- 容量大幅下降；
- 降低教师维度差异；
- 仍保留局部 change evidence；
- 不需要在训练时加载 1024-channel × 双时相 × 多层 raw feature。

### 重要

首轮如果怀疑压缩影响教师能力，应在一个数据集做：

`raw cache vs compact cache`

最小验证；若 ΔF1 <0.05pp，则后续统一使用 compact。

---

## 9.3 Cache 容量上限

建议新增教师：

- 单 Teacher 全四 train Cache 目标 ≤12GB；
- 优先 ≤6GB；
- 单样本尽量 ≤384KB。

现有 SAMStruct 约 36GB 作为历史结构 Cache 可继续使用，不把它作为新 schema 的容量标准。

---

## 9.4 几何增强重放

A / B / Label / Teacher Cache 必须共享：

- crop；
- resize；
- horizontal flip；
- vertical flip；
- temporal exchange。

对称 delta 在 temporal exchange 下无需变化，但 spatial transform 必须严格同步。

禁止从 `os.listdir` 顺序猜 sample；必须由 `list/train.txt` sample name 定位 Cache。

---

# 10. 教师能力验证协议

## 10.1 不能用 zero-shot probe 直接宣布“适合”

Teacher Probe 只能做：

- 排查明显不匹配；
- 给 Agent 提供输入特征；
- 解释最终迁移结果。

教师是否真正适合某数据集，最终判据仍是完整学生训练：

`C1 + Teacher Package → 40K → formal test`

---

## 10.2 每个教师按相同顺序验证四数据集

推荐运行顺序：

1. SYSU：高变化比例、场景多样；
2. WHU：低变化比例、建筑为主；
3. CDD：季节/伪变化明显、变化占比中等；
4. LEVIR：稀疏建筑变化。

这个顺序只是为了尽快覆盖不同类型，不代表任何预设适配。

最终一个 Teacher Package 必须四个数据集都跑完才能进入完整 Registry。

---

## 10.3 教师适配等级

相对 C1：

### Strong Positive

- `ΔF1 >= +0.20pp`
- 且 `ΔIoU > 0`

### Weak / Ambiguous

- `0 < ΔF1 < +0.20pp`，或 F1/IoU 不同方向

不直接进入 Agent 主池。

### Negative

- `ΔF1 <= 0` 或明显 IoU 负迁移

该数据集上 Agent 应学习避开。

### Teacher Admission

一个 Teacher Package 进入最终候选池至少满足：

- 某一数据集 Strong Positive；
- 或两数据集均为稳定正向并存在独特能力；
- 并且不是另一教师完全冗余的弱版本。

注意：

**教师在其他数据集负迁移不是自动淘汰理由。**

如果它在某一类数据上明显专长，恰好为 Agent 的“选择”提供科学意义。

---

# 11. 数据集特征与教师能力特征

# 11.1 Dataset Signature

建议 24–40 维，不使用大模型生成。

### A. Change Geometry

- change pixel ratio；
- connected components per image；
- area quantiles；
- small-object fraction；
- large-region fraction；
- boundary / area；
- fragmentation；
- shape compactness。

### B. Bi-temporal Appearance

- RGB channel mean shift；
- RGB std shift；
- histogram JS / Wasserstein distance；
- edge disagreement；
- gradient magnitude difference；
- low-frequency difference；
- high-frequency difference。

### C. Registration / Pseudo-change Risk

- phase-correlation residual proxy；
- edge displacement proxy；
- unchanged-label area photometric difference（训练标签可用，因为任务全监督）。

### D. Clean Student Difficulty

由 C0/C1 训练日志得到：

- positive BCE；
- negative BCE；
- boundary error；
- FP/FN ratio；
- small-component recall proxy。

---

## 11.2 Teacher Capability Probe

在 train split 上，对每个 `(Dataset, Teacher)` 计算：

- change/non-change separability；
- boundary separability；
- small-component separability；
- large-region consistency；
- unchanged-region false response；
- teacher temporal symmetry；
- local feature entropy；
- confidence calibration；
- feature effective rank；
- spatial stride / token density。

这些 probe 不决定最终结论，但给 Agent 一个“为什么选它”的状态。

---

# 12. 真正的 Agent：离线上下文 Bandit + None/拒绝动作

## 12.1 为什么不推荐重型 RL

当前只有四个固定数据集。

如果直接上 PPO / DQN / Actor-Critic：

- state 数太少；
- reward 昂贵；
- 40K training 才产生真实 reward；
- policy variance 大；
- 在不做多 seed 的项目纪律下很难说服审稿人。

因此第一版选择 **Offline Contextual Bandit**。

它仍然是 Agent，因为具有：

- State；
- Action；
- Reward；
- Policy；
- Abstention（None）。

---

## 12.2 State

对教师 `T_j`：

`x(D,T_j) = [s_D, c(D,T_j), e_j]`

其中：

- `s_D`：Dataset Signature；
- `c(D,T_j)`：Teacher train-only probe；
- `e_j`：teacher metadata embedding。

---

## 12.3 Action

`A = {None, T1, T2, ... Tk}`

严格使用 hard selection。

首轮不做：

- per-pixel multi-teacher mixing；
- 同一步骤多教师加权 KD；
- teacher-to-teacher 互促；
- student 反向更新 teacher。

这是防止滑回旧“多教师互选”主线的关键限制。

---

## 12.4 Reward

Agent 训练只能使用 validation utility：

`r(D,T) = ΔF1_val + 0.5 * ΔIoU_val`

其中基准是 C1。

`None` 的 reward 定义为 0。

如某教师：

- F1 正但 IoU 负；
- 或明显破坏 Precision/Recall 平衡；

可增加约束 penalty，但不要一开始设计复杂 reward。

**test F1 禁止作为 policy 训练 reward。**

---

## 12.5 Policy

Tiny MLP 即可：

```text
pair feature x(D,T)
    -> Linear 96
    -> GELU
    -> Linear 32
    -> GELU
    -> utility score
```

对所有 Teacher Package 得到 utility score，和 None=0 比较：

`argmax_T u_hat(D,T)`

若最大 score ≤0：

`Action = None`

即 Agent 可以判断：

> “这个数据集不应该使用任何 Foundation Teacher。”

这个 None 动作对本项目尤其重要，因为历史证据已经证明“强教师也会负迁移”。

---

## 12.6 四数据集太少的问题：必须做 LODO

这是本路线最大的科学风险之一。

必须使用 Leave-One-Dataset-Out：

- Fold 1：train policy on WHU/CDD/LEVIR，test selection on SYSU；
- Fold 2：train SYSU/CDD/LEVIR，test WHU；
- Fold 3：train SYSU/WHU/LEVIR，test CDD；
- Fold 4：train SYSU/WHU/CDD，test LEVIR。

Agent 在目标数据集可以看到：

- 目标 train split 的 Dataset Signature；
- 目标 train split 的 teacher probe；

但不能看到：

- 目标数据集的 full Teacher Package val utility；
- target test metrics。

这样至少能证明：

> policy 不是只记住四个 dataset ID 的查表器。

---

# 13. 可保留到推理的 Deployable Change Adapter（DCA）

这是本轮修正后建议新增的主路径模块。

## 13.1 插入位置

当前 A2Net 主路径：

```text
Backbone
 -> NeighborFeatureAggregation
 -> TemporalFusionModule
 -> c2,c3,c4,c5 (全部 64 channels)
 -> Decoder
```

建议：

```text
Backbone
 -> NFA
 -> TFM
 -> c2,c3,c4,c5
 -> DCA
 -> c2',c3',c4',c5'
 -> Decoder
```

理由：

- TFM 后已经形成显式双时相 change features；
- 四尺度均 64 channel，模块容易统一设计；
- `|A-B|` 路径已经保证基础时间交换对称；
- 不需要改 LWGANet 主干。

---

## 13.2 DCA 结构

每个尺度使用一个 64→128 的 lightweight adapter。

### 三个专家

#### Expert-D：Detail / Boundary

`1×1 -> DWConv3×3 -> PWConv`

偏局部细节。

#### Expert-C：Context

`DWConv5×5(dilation) -> PWConv`

偏较大区域和语义上下文。

#### Expert-N：Noise/Pseudo-change Suppression

`DWConv3×3 -> PWConv + residual`

用于学习抑制伪变化。

### Gate

从四尺度 change features 做 GAP：

`[GAP(c2), GAP(c3), GAP(c4), GAP(c5)] ∈ R^256`

经过：

`256 -> 64 -> 12`

生成 `4 scales × 3 experts` 的 gate。

由于 gate 输入只依赖 TFM 后的对称 change feature，它对 A/B 交换天然不敏感。

---

## 13.3 参数估计

以内部宽度 128 计算：

- 四尺度 expert blocks：约 291K；
- global gate MLP：约 17K；
- 总计约 **308K**。

部署总参数约：

`2,913,094 + 308,300 = 3,221,394`

约 **3.22M**。

离 5M 仍有约 1.78M。

---

## 13.4 为什么不直接做 2.0M 模块

首轮 DCA 的任务是验证一个清晰假设：

> “由教师塑形的多尺度 change features，能否通过极小的可部署 mixture adapter 保留更多数据集适应能力？”

如果一开始加 2M 参数，即便涨点也很难判断：

- 是教师选择有效；
- 还是单纯模型容量翻大。

所以第一版必须控制在约 0.3M。

若 DCA 明确有效，第二轮再做：

- width=192：约 0.60M；
- width=256：约 0.99M。

仍远低于 2.0869M 上限。

---

## 13.5 DCA 与 Teacher 的关系

### Training

Teacher loss 同时更新：

- student backbone；
- NFA/TFM；
- DCA；
- decoder。

因此教师知识可以通过梯度被编码进 DCA 权重。

### Inference

DCA 输入只有：

`c2,c3,c4,c5`

没有：

- teacher feature；
- teacher map；
- cache；
- teacher ID 必需输入；
- Foundation Model forward。

因此 DCA 是普通学生网络结构，不违反轻量部署要求。

---

# 14. 训练图与部署图

## 14.1 Teacher Capability Run

```text
A,B -------------------------------> Student LWGANet
                                        |
                                        v
                                   NFA -> TFM
                                        |
                                  c2,c3,c4,c5
                                        |
                                       DCA
                                        |
                                     Decoder
                                        |
                                     L_main

A,B -> Offline Teacher Cache -> Teacher Translator
                                  |
                                  +---- L_teacher

L_total = L_main + capped(lambda_T * L_teacher)
```

Teacher Cache 不进入 forward 主特征路径。

---

## 14.2 Agent 阶段

```text
Dataset train statistics -----------+
Teacher train-only probes ----------+--> Agent --> {Teacher_j or None}
Teacher metadata -------------------+
```

Agent 决定下一次从头训练使用哪个 Teacher Package。

---

## 14.3 最终部署

```text
A,B
 -> shared LWGANet-L0
 -> NFA
 -> TFM
 -> DCA
 -> Decoder
 -> Change Map
```

不包含：

- Agent teacher scorer；
- Foundation Model；
- Teacher Cache；
- KD loss；
- teacher-specific Translator。

### 注意

本轮允许保留的是 **DCA 学生侧可学习模块**，不是 Teacher/Cache 本身。

若未来想让 tiny Agent gate 也留在部署，它也只能读取学生自身 feature statistics；绝对不能在线调用大教师。

---

# 15. 参数量与 FLOPs 预算

## 15.1 参数预算

| 组件 | Params |
|---|---:|
| 当前 A2Net-LWGANet-L0 | 2.913M |
| DCA-128（建议首版） | ~0.308M |
| 总部署 | ~3.221M |
| 距 5M 余量 | ~1.779M |

硬检查：

`effective_inference_params < 5,000,000`

而不是只报告 trainable params。

---

## 15.2 FLOPs

当前项目 256×256 主学生约 2.77G。

DCA 的高分辨率 c2 只有 64 channels，但 1×1 pointwise 仍会带来额外 FLOPs。

首版目标：

- DCA additional FLOPs ≤0.4G；
- 总 FLOPs 尽量 ≤3.2G。

若 DCA width=256 虽参数仍 <1M，但 FLOPs 会明显上升，因此不能只盯参数量。

---

# 16. 完整实验矩阵与实验 ID

# 16.1 基线

| ID | 唯一变量 | 数据集 | Steps | Seed | 目的 |
|---|---|---|---:|---:|---|
| C0 | Clean A2Net | 4DS | 40K | 2333 | 干净 anchor |
| C1 | C0 + DCA | 4DS | 40K | 2333 | DCA 本身是否有效 |

C1 是所有 Teacher Package 的直接 anchor。

---

## 16.2 Wave-A Teacher Capability Runs

每个 ID 实际拆为四数据集独立 run。

| ID | 相对 C1 的唯一变量 | Teacher |
|---|---|---|
| TV-SAM | + SAM2 Struct Package | SAM2.1-H |
| TV-D2 | + DINOv2 Relation Package | DINOv2-B/14 |
| TV-D3N | + Neutral DINOv3 Package | DINOv3-B/16 LVD |
| TV-D3S | + Satellite DINOv3 Package | DINOv3-L/16 SAT493M |
| TV-RCLIP | + RemoteCLIP Package | RemoteCLIP-L/14 |
| TV-MARS | + VHR Multiscale Package | MaRS-Base RGB |

共：

`6 teachers × 4 datasets = 24 full 40K runs`

加 C0/C1：8 runs。

Wave-A 总正式 run：32。

在 2×RTX5090 上建议两卡分别排队，不做跨卡 DDP，避免为了单个 256×256 模型增加同步复杂度。

---

## 16.3 Wave-B

仅在 Wave-A 后启动：

| ID | Teacher |
|---|---|
| TV-ANYSAT | AnySat |
| TV-UNISAT | UniverSat |
| TV-RADIO | C-RADIOv3-B |
| TV-TM | TerraMind Small/Tiny |
| TV-SATMAE | SatMAE++ RGB |

不是默认全部跑。

---

## 16.4 Agent 对照

| ID | 方法 | 是否学习 | 是否允许 None | 作用 |
|---|---|---|---|---|
| A-H | change-ratio heuristic | 否 | 是 | 证明简单规则不足 |
| A-F | best global fixed teacher | 否 | 否 | 固定教师基线 |
| A-REG | Registry lookup/oracle-val | 否 | 是 | 上界参考，不当主方法 |
| A-LODO | Offline contextual bandit LODO | 是 | 是 | 真正 Agent |

---

## 16.5 最终主实验

| ID | 唯一变量 | 说明 |
|---|---|---|
| M1 | C1 + Agent-selected Teacher Package | Agent 选择后从头重新 40K |

M1 不是复用 TV checkpoint。

---

# 17. 成功判据、失败判据与可证伪假设

## H1：确实存在跨数据集不同的最佳 Teacher Package

### 预测

Teacher Capability Matrix 中不同数据集的 top teacher 不完全相同。

### 成功

至少两个数据集的最优教师不同，并有 ≥0.20pp F1 级差异。

### 失败

同一个教师四数据集全部最好。

若失败：

Agent 没有必要，应该退化成“最佳单教师 + DCA”。

---

## H2：Dataset Signature + Probe 能预测教师效用

### 成功

LODO 四折中：

- 至少 3/4 fold 选择的 teacher 的 validation utility 与 oracle-best 差 ≤0.20pp F1；
- 或正确选择 None 避免负迁移。

### 失败

policy 与简单 heuristic / 固定 teacher 无区别。

若失败：

删除 Agent，保留 capability registry 作为分析，不包装成核心创新。

---

## H3：DCA 能在不依赖 Teacher 推理的情况下保留一部分教师收益

### 最小消融

- C0；
- C1（DCA only）；
- T_j without DCA；
- T_j with DCA。

只在最有代表性的两个数据集做这一组扩展消融即可，正式主表四数据集统一用 C1 anchor。

### 失败

DCA 本身无收益且与 Teacher 无协同。

若失败：

删除 DCA，恢复 2.91M 主路径；Agent 仍可只训练期使用。

---

## H4：最终方法跨四数据集避免负迁移

机制门：

- `M1 - C1`：四数据集 F1 不应出现明显负向；
- IoU 同方向；
- 至少 3/4 数据集有实质正提升；
- 若某数据集所有教师均负，Agent 应选择 None，此时 M1≈C1。

---

## 最终论文硬目标

必须同时：

- SYSU F1 ≥85；
- LEVIR F1 ≥92.5；
- WHU F1 ≥95；
- CDD F1 ≥98；
- IoU 同方向；
- effective inference params <5M。

---

# 18. 逐文件实现方案

建议新建：

```text
models/
  agent/
    __init__.py
    dataset_signature.py
    teacher_registry.py
    offline_bandit.py

  distill/
    cache_v2.py
    capability_probe.py
    teacher_package.py
    translators/
      sam_struct.py
      dense_relation.py
      semantic_local.py
      multiscale_vhr.py
      resolution_aware.py

    teachers/
      dinov3_sat.py
      remoteclip.py
      mars.py
      anysat.py
      universat.py
      radio.py
      terramind.py
      satmae_pp.py

  decoder/
    deployable_change_adapter.py

  tools/
    generate_teacher_cache_v2.py
    audit_teacher_cache.py
    build_dataset_signature.py
    probe_teacher_capability.py
    train_teacher_agent.py
    smoke_cata_v2.py
```

---

## 18.1 `models/a2net.py`

新增：

```python
self.dca = DeployableChangeAdapter(...)
```

放在 TFM 与 Decoder 之间。

`switch_to_deploy()`：

- DCA 必须保留；
- Teacher/Translator 不在 model main module 内注册，避免进入部署 state_dict。

---

## 18.2 `deployable_change_adapter.py`

要求：

- 输入四个 64ch change maps；
- temporal symmetric；
- 不接受 teacher map；
- 不接受 label；
- 参数统计独立函数；
- FLOPs 可被 THOP 捕获。

---

## 18.3 `teacher_registry.py`

只存 metadata 与结果索引，不存测试结果供 policy 训练。

建议区分：

```text
registry/research_report.json       # 可含 test delta
registry/agent_train_registry.json  # 只能含 train/val 信息
```

防止不小心 test leakage。

---

## 18.4 `cache_v2.py`

manifest 强制记录：

- teacher exact name；
- checkpoint SHA256；
- normalization；
- color order；
- source image size；
- cached feature size；
- projection hash；
- dataset list hash；
- schema version。

---

## 18.5 `train.py`

新增参数示例：

```text
--dca_mode none|moe128
--teacher_package none|sam2|dinov2|dinov3_lvd|dinov3_sat|remoteclip|mars|...
--teacher_cache_root ...
--teacher_loss_cap ...
--registry_version ...
```

不要用同一个模糊 `--teacher` 开关承载所有不同语义。

---

# 19. Smoke / Cache / Deploy / Resume 校验

## 19.1 Synthetic smoke

必须检查：

1. Forward shape；
2. backward；
3. DCA params；
4. total deploy params <5M；
5. A/B swap：

`max|Model(A,B)-Model(B,A)| < 1e-6`

6. teacher aux on/off 不改变 main forward 数值定义；
7. Teacher branch detach 正确。

---

## 19.2 Cache smoke

每个新教师：

- 8 samples；
- 读取 sample name；
- generate；
- save；
- reload；
- transform replay；
- temporal exchange；
- compare transformed cache alignment。

---

## 19.3 Real-cache dry run

每教师至少一个真实数据集：

- batch 2；
- 20–100 steps；
- 显存；
- throughput；
- KD loss scale；
- main loss；
- gradient norm；
- Cache hit rate=100%。

若 `L_teacher` 比 main loss 高一个数量级，先修归一化/cap，不要直接开 40K。

---

## 19.4 Deploy test

`switch_to_deploy()` 后：

- Teacher/Cache/Translator 不存在；
- DCA 保留；
- params <5M；
- deploy前后 main prediction max error <1e-6；
- FLOPs 重新实测；
- 输出尺寸一致。

---

## 19.5 Checkpoint v6 建议

新实验不要与 FA-SCRD v5 静默互换。

建议：

```yaml
format_version: 6
model:
optimizer:
rng:
global_step:
best_val_f1:
args:
dca_config:
selected_teacher_package:
teacher_cache_hash:
registry_version:
```

Resume 时核验：

- dataset；
- teacher package；
- DCA mode；
- cache hash；
- seed；
- step；
- RNG。

---

# 20. 推荐启动顺序

## 第 0 步：先别下载全部模型

先完成 DCA + C0/C1。

原因：

所有 Teacher utility 都要相对同一个部署 anchor C1。

---

## 第 1 步：整理现有 Cache

- SAM2.1；
- DINOv2；
- 历史 SYSU-adapted DINOv3。

将历史 DINOv3 标记为 `D3-SYSU-Adapted`，不能写成 neutral。

---

## 第 2 步：新增三套最高价值权重

优先下载/申请：

1. DINOv3 SAT-493M ViT-L/16；
2. RemoteCLIP ViT-L/14；
3. MaRS-Base RGB。

这是第一批最有“知识类型差异”的新增教师。

---

## 第 3 步：生成 Compact Cache v2

按：

`SYSU -> WHU -> CDD -> LEVIR`

逐教师完成。

先 smoke，再单数据集全量 Cache，再扩到四数据集。

---

## 第 4 步：跑 Wave-A Capability Matrix

GPU0 / GPU1 可以按 teacher 分队列：

### GPU0

```text
SAM -> D3-SAT -> MARS
```

### GPU1

```text
D2 -> D3-LVD -> RemoteCLIP
```

每个 teacher 内部四数据集串行。

如果服务器 I/O 成为瓶颈，不要两卡同时大量随机读 36GB SAM cache。

---

## 第 5 步：冻结 Registry v1

只有完整结果才写入。

没有完整 `TEST RESULTS` 的实验不得进入 research report。

---

## 第 6 步：决定 Wave-B 是否必要

问题驱动扩展：

- 小目标/空间细节仍无强教师 → UniverSat；
- resolution mismatch 明显 → AnySat；
- 多教师冲突严重 → C-RADIO；
- EO generalization 不足 → TerraMind；
- scale representation 不足 → SatMAE++。

---

## 第 7 步：训练 Agent + LODO

先做简单 MLP bandit。

不直接上 RL。

---

## 第 8 步：Agent 选完后跑全新 M1

禁止从 capability run checkpoint 续训。

---

# 21. 论文创新点如何组织

如果实验成功，建议论文贡献不是“我们用了很多大模型”，而是三点。

## Contribution 1：Capability-Validated Teacher Selection

核心问题：Foundation Model 对轻量 CD student 存在明显数据依赖的正/负迁移。

创新：

- 不盲目多教师融合；
- 先构建可证伪 Teacher Capability Registry；
- 再做 selection。

与 DynaKD / MTKD-RL 的实质区别：

- 它们主要在给定 teacher ensemble 上学习教师权重；
- 本项目先用遥感双时相数据特征和 teacher capability probe 建立“是否应使用该教师”的证据；
- 支持 `None`；
- 主动作是 dataset-level hard teacher package selection，而不是持续多教师 mixture。

---

## Contribution 2：Dataset–Teacher Capability Matching Agent

State：

- change geometry；
- pseudo-change risk；
- teacher probe；
- teacher metadata。

Action：Teacher / None。

Reward：student validation gain。

并用四数据集 LODO 防止变成查表。

---

## Contribution 3：Teacher-shaped Deployable Change Adapter

不是把 teacher feature 注入 inference。

而是：

> 教师只通过训练梯度塑形一个约 0.3M 的 DCA；推理只保留学生+DCA。

最终仍约 3.22M 参数。

这比“所有辅助全拆除”多了一条可发表的结构主张：

> **训练期 Foundation knowledge 可以被固化到轻量、时相对称、可部署的 change adapter 中。**

---

# 22. 主要风险与止损条件

## P0：公平性风险——不同教师 Translator 不同

处理：

论文主张必须称为 **Teacher Package** 效用，而非裸 backbone 公平排名。

另外在 1–2 个教师上增加统一 Relation Translator 对照，确认结果不是完全由 translator 容量决定。

---

## P0：test leakage

Agent 禁止读取 test gain。

- Registry research report 可包含 test；
- Agent train registry 只包含 train/val。

---

## P0：时间交换对称

任何保留在部署图的 DCA gate 都只能使用：

- `|A-B|`；
- `A+B`；
- symmetric pooled statistics。

禁止 `concat(A,B)` 后普通 MLP 产生顺序敏感 gate，除非显式对称化。

---

## P1：只有四个数据集，Agent 样本太少

这是最大审稿风险。

必须：

- LODO；
- tiny policy；
- 强正则；
- None；
- 不夸大“通用 agent”。

若 LODO 失败，立即降级为 capability-guided lookup，不继续堆 RL。

---

## P1：教师数太多导致实验爆炸

Wave-A 限 6 个教师。

Wave-B 只有明确能力缺口才开。

---

## P1：Cache 存储/I/O

Compact Cache v2；新教师目标 ≤12GB。

不能每个教师都存完整 A/B multi-layer raw tensor。

---

## P1：DCA 涨点来自容量，而不是教师

C0 vs C1 必须四数据集完整跑。

Teacher utility 一律相对 C1。

---

## P2：第三方模型环境冲突

MaRS/AnySat/UniverSat/C-RADIO 优先选择能在当前 Python/PyTorch 直接加载的路径。

SkySense 等老栈只能隔离 Cache 生成，不能污染 `lsrep`。

---

# 23. 参考文献与官方资源

## 23.1 与 Foundation Model / 遥感相关

1. **BAN — A New Learning Paradigm for Foundation Model-Based Remote-Sensing Change Detection**  
   IEEE TGRS, 2024.  
   GitHub: <https://github.com/likyoo/BAN>

2. **Burden-Free Distillation From Foundation Model for Efficient Remote Sensing Change Detection**  
   IEEE TGRS, 2025.  
   GitHub: <https://github.com/Younger-hua/Burden-Free-Distillation>

3. **Segment Any Change**  
   NeurIPS 2024, CCF-A.  
   GitHub: <https://github.com/Z-Zheng/pytorch-change-models>

4. **RemoteCLIP: A Vision Language Foundation Model for Remote Sensing**  
   IEEE TGRS, 2024.  
   GitHub: <https://github.com/ChenDelong1999/RemoteCLIP>

5. **SkySense: A Multi-Modal Remote Sensing Foundation Model Towards Universal Interpretation for Earth Observation Imagery**  
   CVPR 2024, CCF-A.  
   GitHub: <https://github.com/Jack-bo1220/SkySense>

6. **SatMAE++ / Rethinking Transformers Pre-training for Multi-Spectral Satellite Imagery**  
   CVPR 2024, CCF-A.  
   GitHub: <https://github.com/techmn/satmae_pp>

7. **AnySat: One Earth Observation Model for Many Resolutions, Scales, and Modalities**  
   CVPR 2025 Highlight, CCF-A.  
   GitHub: <https://github.com/gastruc/AnySat>

8. **TerraMind: Large-Scale Generative Multimodality for Earth Observation**  
   ICCV 2025, CCF-A.  
   GitHub: <https://github.com/IBM/terramind>

9. **MaRS: A Multi-Modality Very-High-Resolution Remote Sensing Foundation Model with Cross-Granularity Meta-Modality Learning**  
   AAAI 2026, CCF-A.  
   GitHub: <https://github.com/WanderRainy/MaRS>

10. **UniverSat: Resolution- and Modality-Agnostic Transformers for Earth Observation**  
    NeurIPS 2026, CCF-A.  
    GitHub: <https://github.com/gastruc/UniverSat>

11. **RADIOv2.5: Improved Baselines for Agglomerative Vision Foundation Models**  
    CVPR 2025, CCF-A.  
    GitHub: <https://github.com/NVlabs/RADIO>

12. **DINOv3**  
    Meta official 2025 technical release / preprint；本文不把它误标为已核验 CCF-A 论文。  
    GitHub: <https://github.com/facebookresearch/dinov3>

---

## 23.2 与多教师选择 / KD 相关

13. **How to Trade Off the Quantity and Capacity of Teacher Ensemble: Learning Categorical Distribution to Stochastically Employ A Teacher for Distillation (DynaKD)**  
    AAAI 2024, CCF-A.  
    核心启示：more teachers / stronger teacher 并不总是更好。

14. **Multi-Teacher Knowledge Distillation with Reinforcement Learning for Visual Recognition (MTKD-RL)**  
    AAAI 2025, CCF-A.  
    GitHub: <https://github.com/winycg/MTKD-RL>

15. **Reinforced Multi-teacher Knowledge Distillation for Efficient General Image Forgery Detection and Localization**  
    AAAI 2025, CCF-A.  
    动态教师选择参考，但任务不同。

16. **MoVE-KD: Knowledge Distillation for VLMs with Mixture of Visual Encoders**  
    CVPR 2025, CCF-A.  
    GitHub: <https://github.com/hey-cjj/MoVE-KD>

17. **AM-RADIO: Agglomerative Vision Foundation Model – Reduce All Domains Into One**  
    CVPR 2024, CCF-A.  
    GitHub: <https://github.com/NVlabs/RADIO>

18. **UNIC: Universal Classification Models via Multi-teacher Distillation**  
    ECCV 2024, CCF-A.  
    多教师平衡与 teacher dropping 参考。

---

# 24. 立即执行顺序

按优先级严格执行：

### 1. 锁定新方法协议

- 主学生 2.913094M；
- DCA 首版 width=128；
- Teacher Package 定义；
- Teacher utility anchor=C1；
- 40K/seed2333 不变。

### 2. 实现 DCA

先不碰新 Teacher。

### 3. 跑 C0 / C1 四数据集

建立新主线 anchor。

### 4. 建 Cache v2 + audit 工具

先用现有 DINOv3/SAM cache 做兼容测试。

### 5. 准备 Wave-A 权重

顺序：

1. DINOv3 SAT access；
2. RemoteCLIP L/14；
3. MaRS Base RGB；
4. neutral DINOv3 LVD；
5. 现有 SAM2；
6. 现有 DINOv2。

### 6. 每个教师先 8-sample smoke

通过后再全量 Cache。

### 7. 跑 24 个 Wave-A Teacher Capability 正式实验

所有 run 从头 40K。

### 8. 建 Teacher Capability Matrix

逐数据集算 `Δ(C1)`。

### 9. 决定是否启动 Wave-B

不要在 Wave-A 结果出来前一次性生成所有教师 Cache。

### 10. 冻结 Agent-train Registry

只放 train/val。

### 11. 做 LODO Agent

先 tiny offline contextual bandit。

### 12. Agent 选择完成后重新从头跑 M1

不复用 capability checkpoint。

### 13. 最终做 deploy audit

- Params；
- FLOPs；
- swap；
- deploy error；
- Teacher 彻底断开。

---

# 25. 仍需补充的证据

以下内容当前均不能写成论文结论：

1. **哪一个新教师适合 SYSU/WHU/CDD/LEVIR**：待完整 40K 实验；
2. DINOv3-SAT 是否一定优于 LVD-DINOv3：待验证；
3. RemoteCLIP 是否真的更适合语义复杂 SYSU：待验证；
4. MaRS 是否真的在 WHU/LEVIR 建筑变化更强：待验证；
5. AnySat/UniverSat 的多分辨率设计是否转化为轻量学生收益：待验证；
6. C-RADIO 是否会比独立 teacher routing 更稳定：待验证；
7. DCA 0.31M 是否能稳定提升四数据集：待验证；
8. Agent 的 LODO 是否优于固定 teacher / change-ratio heuristic：待验证；
9. Compact Cache v2 相对 raw features 是否无明显信息损失：待验证；
10. 新方法能否达到 SYSU≥85、LEVIR≥92.5、WHU≥95、CDD≥98：待验证。

---

# 最终建议一句话

**不要先训练一个 Agent 去“猜教师”；先把 SAM2、DINOv2、neutral DINOv3、DINOv3-SAT、RemoteCLIP、MaRS 等教师包在四数据集上逐一用同协议 40K 做成可审计的能力矩阵，再让带 None 动作的轻量 Agent 学习“数据集签名 + 教师 probe → 教师效用”；同时把一个约 0.31M 的对称 DCA 固化进学生部署图，使最终模型约 3.22M、推理仍完全不访问大模型。**

