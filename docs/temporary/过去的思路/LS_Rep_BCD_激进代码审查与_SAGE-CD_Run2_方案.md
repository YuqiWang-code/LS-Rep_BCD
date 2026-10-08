# LS-Rep_BCD 极轻量遥感变化检测：激进代码审查与 SAGE-CD/Run2 唯一方案

> 审查目标：解释“指标为什么上不来”，找到当前学生结构、Teacher→Student 迁移接口、缓存/训练工程中的关键问题，并给出一个 **只保留一条主线** 的 SAGE-CD/Run2 方案。  
> 结论优先级：**P0 正确性 > P1 方法瓶颈 > P2 结构/实验工程**。  
> 正式实验协议：SYSU/WHU 先，batch=64，40k steps，seed=2333；不做多 seed。  
> 部署硬约束：A2Net-LWGANet-L0，部署参数必须仍为 **2,913,094**，256×256 FLOPs 保持约 **2.75–2.77G**；Teacher/cache/router/aux-head 训练期拆除；T1/T2 交换误差 <1e-6。

---

## 0. 审查来源与版本边界

### 0.1 已实际核对

- GitHub `main`：`YuqiWang-code/LS-Rep_BCD`
- 固定审查快照：`main@9d5096c7aa93b000e03c9e6dd3567504d17bd6eb`
- `README.md`，特别是 §4、§6 历史归档。
- 当前 `models/` 主学生、decoder、LWGANet、FA-SCRD Teacher/cache/router/KD、trainer、DINOv3 third-party 实现。
- `docs/experiment_metrics.xlsx`：正式抽取表 75 行，重点核对 SAGE-CD/Run1 的 6 行。
- 你本轮提供的 **SAGE-CD RUN1 CODE + UNIFIED METRICS SNAPSHOT**，其中补齐了当前 GitHub `main` 尚未包含的：
  - `models/distill/sage_heads.py`
  - `models/distill/sage_kd.py`
  - `models/scripts/train_sage.py`
  - `models/tools/generate_sage_cache.py`
  - `models/tools/smoke_sage.py`

### 0.2 一个必须明确的版本事实

当前 GitHub `main@9d5096c7...` **还没有 SAGE-CD 文件，也没有 `docs/temporary/LS_Rep_BCD_下一步唯一改进方案_SAGE-CD.md`**。因此：

- 当前主学生/FA-SCRD 文件的行号：来自 GitHub 固定 commit，可复核。
- `sage_heads.py / sage_kd.py / train_sage.py`：依据你本轮给出的 Run1 snapshot；其中两个小文件的行号可精确定位，`train_sage.py` 只做函数级/近似行区间定位，因为它尚未进入 current main。
- 原始 SAGE 方案文档在 current main 与当前 Project 文件检索中均未找到，因此本文**不伪称已经读过该文件**；SAGE 原始意图只依据你给出的代码、xlsx 说明与 Run1 描述还原。

---

# 1. 先给结论

## 1.1 最核心结论

**SAGE-CD Run1 的失败已经不是“KD 太大”，而是“错误/不匹配的知识被安全地传进了学生”。**

FA-SCRD 已经证明过 `kd≈12–15` 会压过 `main≈3.6`；SAGE Run1 把 KD 降到约 0.6，并用 gradient budget 控制局部有效 KD 梯度后，SYSU 仍然负增益。这说明继续调 `rho / temperature / lambda` 不会解决根因。

当前最可能的瓶颈按重要性排序是：

1. **Teacher target 的任务/尺度方向不对**：DINO change head 本质只在 16×16 token 网格上产生语义变化，再 bilinear 放大到 256×256；Run1 却把它当作 full-resolution 像素教师。
2. **per-image logit z-score 丢掉了变化比例与绝对置信度**：对变化率 21.1% 的 SYSU 和 3.4% 的 WHU，用同一种“每图强行零均值/单位方差”的 target 变换，天然容易产生相反的 Recall/Precision 推动。
3. **Student 的 Teacher 落点太窄**：只把一个 `1×1 Conv` aux head 挂在 c4(1/16)，而学生真正的细粒度变化判别还依赖 c2/c3、Decoder 逐级恢复；DINO 梯度没有足够好的“落地通道”。
4. **学生主路径先做独立 SWA，再直接 `|F1-F2|`，把共同语义上下文直接丢掉**。Teacher 想传“这是什么对象/区域”，但 TFM 接收到的是“两个向量差多少”，语义落点本身受限。
5. **Teacher 没有按学生失败类型被选择**：Run1 是全图统一 DINO。SYSU 的错误类型与 WHU 不同，统一语义教师自然会出现“一个数据集补漏检、另一个数据集造成漏检”。
6. 还有若干 **P0 工程正确性问题** 会污染 Teacher/cache 结论，必须在 Run2 前修。

因此下一步不应该继续“DINO-only Safe-KD + 调权重”，也不应该回到旧 hard gate。

---

# 2. SAGE-CD/Run1：指标拆解说明了什么

xlsx 正式结果：

| Dataset | Arm | Recall | Precision | OA | F1 | IoU | Kappa |
|---|---|---:|---:|---:|---:|---:|---:|
| SYSU | G0 clean | 79.4944 | 85.6911 | 92.0338 | 82.4765 | 70.1788 | 77.3311 |
| SYSU | G1 joint BN | 81.3508 | 84.1303 | 91.9831 | 82.7172 | 70.5280 | 77.5003 |
| SYSU | G2 Safe-DINO | 80.4331 | 84.6131 | 91.9362 | 82.4702 | 70.1696 | 77.2383 |
| WHU | G0 clean | 90.9233 | 94.7227 | 99.4389 | 92.7841 | 86.5395 | 92.4923 |
| WHU | G1 joint BN | 91.2881 | 96.3824 | 99.5184 | 93.7661 | 88.2638 | 93.5158 |
| WHU | G2 Safe-DINO | 92.1232 | 96.4167 | 99.5516 | 94.2211 | 89.0736 | 93.9880 |

### G2 − G1

| Dataset | ΔRecall | ΔPrecision | ΔF1 | ΔIoU | 解释 |
|---|---:|---:|---:|---:|---|
| SYSU | **−0.9177** | **+0.4828** | **−0.2470** | −0.3584 | DINO 把学生推得更保守：FP 少一点，但 FN 明显增加 |
| WHU | **+0.8351** | +0.0343 | **+0.4550** | +0.8098 | 主要是补 Recall，Precision 几乎不动 |

这不是“随机地一正一负”，而是**非常明确的决策边界方向差异**。

### 与历史 Teacher 失败对照

RDT-CD Run3 Full 对 SYSU：

- Recall：+3.2498
- Precision：−0.7887
- F1：+1.3812

RDT-CD Run3 Full 对 WHU：

- Recall：+0.2486
- Precision：−0.9857
- F1：−0.3557

FA-SCRD DINO relation KD（A1−C1）对 SYSU：

- Recall：+1.7343
- Precision：−2.7048
- F1：−0.3312

这组历史非常关键：**同样是 Foundation Teacher，迁移接口改变后，Recall/Precision 的推动方向会反过来。**

所以不能简单说：

> “DINO 天生只适合 WHU / SAM 天生只适合 SYSU。”

更准确的证据结论是：

> **Teacher 知识与 Student 表征/蒸馏接口存在强耦合；当前最失败的不是某个 Teacher 名字，而是“一刀切地把一种知识作用到所有样本、所有错误类型、所有尺度”。**

---

# 3. P0：正确性问题——Run2 前必须先修

## P0-1：Teacher 微调阶段存在二时相随机位置编码不一致

**文件：**

- `models/distill/fa_scrd_teacher.py:99–118`
- `models/distill/fa_scrd_teacher.py:152–166`
- `models/tools/train_teacher.py:105–126`
- `models/thirdparty/dinov3/layers/rope_position_encoding.py:93`

### 代码事实

Teacher 构造 DINOv3 时：

```python
pos_embed_rope_rescale_coords=2
```

而 DINOv3 RoPE：

```python
if self.training and self.rescale_coords is not None:
    ...
```

`train_teacher.py` 又直接：

```python
teacher.train()
```

然后 `forward_change_logit()` 先后独立执行：

```python
mid_a, deep_a = self.extract_features(xA)
mid_b, deep_b = self.extract_features(xB)
```

### 后果

训练模式下，T1/T2 两次独立 backbone forward 会各自抽取随机 RoPE coordinate rescale。

即：

\[
R_A \neq R_B
\]

Teacher 学到的差分中混入了**人为的位置编码扰动**：

\[
|F_A(x_A;R_A)-F_B(x_B;R_B)|
\]

而不是严格同坐标系：

\[
|F(x_A;R)-F(x_B;R)|
\]

这与 binary CD 的二时相对称/共坐标假设不一致。

### 怎么改

Teacher 微调时：

- `teacher.backbone.eval()`：关闭 RoPE 训练期随机 rescale。
- LoRA Linear 在 eval 下仍然可反向传播，不影响参数学习。
- `teacher.head.train()`。
- 或显式把 DINO RoPE 的训练期 `rescale_coords` 关闭。

**必须增加 Teacher swap smoke：**

\[
\max |T(A,B)-T(B,A)| < 10^{-6}
\]

在 cache 生成前验证。

---

## P0-2：SAGE aux optimizer 没有跟随 poly scheduler

**文件：**

- Run1 snapshot `models/scripts/train_sage.py`
- 定位：`train_epoch()` 中 `adjust_learning_rate(args, optimizer, ...)` 到 `aux_optimizer.step()` 段（snapshot 约 290–335 行）

### 代码事实

只有主 optimizer 被：

```python
adjust_learning_rate(args, optimizer, ...)
```

但：

```python
aux_optimizer = build_optimizer(args, aux_head.parameters())
```

后续只有：

```python
aux_optimizer.step()
```

没有 scheduler。

### 后果

40k 后半程：

- Student 主模型 LR 已接近 0；
- Aux head 仍保持初始 `5e-4`。

这会使“Teacher target → aux head → c4”的梯度接口随训练阶段发生额外漂移。  
即使 gradient budget 控制了 c4 局部梯度比例，也不能把这个训练动态当作“完全同协议”。

### 怎么改

Run2 不再使用两个独立 optimizer。

建议：

```text
一个 Adam
├── student params
├── semantic aux params
└── boundary aux params
```

同一个 poly scheduler；训练参数量单独统计，部署参数量只统计 student。

---

## P0-3：Cache provenance 当前几乎没有真正校验

**文件：**

- `models/distill/cache_dataset.py:85`
- Run1 snapshot `models/tools/generate_sage_cache.py`
- 当前 `models/tools/generate_teacher_cache.py`

### 问题 A：Reader 丢弃 meta

```python
return {k: v for k, v in data.items() if isinstance(v, torch.Tensor)}
```

`meta` 被直接扔掉。

所以 trainer 无法确认：

- Teacher checkpoint
- normalization
- resolution
- cache schema/version
- dataset
- list fingerprint

### 问题 B：SAGE cache 的 checkpoint hash 只是“路径字符串 hash”

snapshot 中：

```python
hashlib.sha256(str(args.teacher_ckpt).encode()).hexdigest()
```

如果同一个文件路径被覆盖成新 checkpoint，hash **完全不变**。

### 怎么改

Cache meta 至少保存并在训练启动时强校验：

```text
schema_version
teacher_model
actual_checkpoint_sha256(bytes)
teacher_training_config_hash
dataset
split
list_file_sha256
resolution
normalization
cache_kind
```

`TeacherCacheReader` 必须返回/验证 meta，而不是丢弃。

---

## P0-4：现有 cache replay 不能直接拿来接 SAMStruct

**文件：**

- `models/distill/cache_dataset.py:19–35`
- `models/distill/cache_dataset.py:45–65`

### 代码事实

crop/resize 对所有 Tensor 一律：

```python
mode="bilinear"
```

### 对 DINO feature/logit

基本合理。

### 对 SAMStruct

如果包含：

- `instance_id`：必须 nearest；bilinear 会直接破坏离散 ID。
- binary boundary：优先 nearest 或专门的 soft-boundary replay。
- quality：如果是 region scalar/map，需要按 schema 处理，不能无脑当连续图。

### 怎么改

建立 key/schema-aware replay：

```text
continuous_feature/logit -> bilinear
probability              -> bilinear
binary_boundary          -> nearest / dedicated soft replay
instance_id              -> nearest
region_quality           -> region-aware
```

且 temporal exchange 必须按具体 cache 字段定义交换。

---

## P0-5：SAGE resume 没有做协议一致性检查

**文件：**

- Run1 snapshot `models/scripts/train_sage.py`
- `main()` 的 resume block（snapshot 后半部分）

FA-SCRD `train.py` 有 `validate_resume()`，会检查：

- implementation_version
- experiment
- dataset
- batch
- max_steps
- seed
- lr
- wd
- fingerprint

SAGE Run1 只检查：

```python
format_version == 6
```

### 怎么改

把 FA-SCRD 的 strict resume 契约迁移到 SAGE v2，并额外检查：

```text
teacher cache fingerprint
SAM cache fingerprint
student structure mode
router mode
aux-head version
```

---

## P0-6：LWGANet 对 antialiased operator 的 silent fallback 风险

**文件：**

- `models/backbone/lwganet.py:31–43`

```python
try:
    import antialiased_cnns
except ImportError:
    ...
    return F.avg_pool2d(...)
```

### 问题

如果原 LWGANet 预训练权重来自真正 `antialiased_cnns.BlurPool`，而当前服务器缺包，则运行时网络算子已经变了，但：

- 参数量一样；
- checkpoint 仍能加载；
- 不会报错。

这类 silent operator drift 对小模型尤其危险。

### 怎么改

Run2 启动前直接打印：

```text
BlurPool implementation = antialiased_cnns / fallback
```

正式实验禁止 silent fallback：

- 要么依赖存在；
- 要么明确把 fallback 固化为实验定义并重新做 anchor。

---

## P0-7：Teacher/backbone `strict=False` 只打印，不阻止错误权重

**文件：**

- `models/distill/fa_scrd_teacher.py:120–133`
- `models/backbone/lwganet.py:528` 附近

Teacher/Foundation 权重是方法核心，不能只：

```python
strict=False
print(missing, unexpected)
```

### 怎么改

建立 whitelist：

- 允许明确的 classifier/head 差异；
- backbone 核心层、RoPE、register/storage token、attention 层有异常时直接 fail。

---

## P0-8：formal trainer 只“计算” swap/deploy error，没有强制 hard contract

Run1 trainer 定义：

```python
SWAP_ATOL = 1e-6
DEPLOY_ATOL = 1e-6
```

但最终只是 log，没有 `raise`。

### 怎么改

正式 TEST 前：

```python
if swap_error >= 1e-6: raise RuntimeError(...)
if deploy_error >= 1e-6: raise RuntimeError(...)
```

不能依赖 smoke 代替 formal run 的最后一道检查。

---

# 4. P1：真正限制指标的方法瓶颈

## P1-1：当前 TFM 的根本信息瓶颈是纯绝对差分

**文件：`models/decoder/a2net_decoder.py:237–253`**

关键一行：

```python
x = torch.abs(x1 - x2)
```

之后所有 dilation branch 都只看 `x`。

### 丢失了什么

假设两个位置的差值幅度相同：

```text
A: building → building-like texture shift
B: vegetation → bare land
```

只看：

\[
|F_1-F_2|
\]

无法知道“发生差异的共同语义背景是什么”。

更合理的交换对称变化表示至少应该包含：

\[
D=|F_1-F_2|
\]

和某种共同上下文：

\[
M=\frac{F_1+F_2}{2}
\]

但不能简单 concat 后破坏预算。

### 为什么这会影响 Teacher transfer

DINO 给的是高层语义变化，但 Student c4 已经是纯差分编码后的结果。

Teacher 想告诉 Student：

> “这里是某类对象/区域，当前差异应解释为真实变化。”

Student 的可塑表示却主要保留：

> “两个时相在这里差了多少。”

这是 Teacher 知识无法稳定落地的结构性原因之一。

---

## P1-2：SWA 在两个时相完全独立执行，直到 TFM 才发生第一次真正交互

**文件：`models/a2net.py:114–124`**

```python
aggregated1 = self.swa(*features1)
aggregated2 = self.swa(*features2)
change = self.tfm(*aggregated1, *aggregated2)
```

SWA 做了很强的多尺度重编码，但整个过程不知道另一个时相。

因此：

- 多尺度融合先把各时相信息单独压成 64ch；
- 之后才比较；
- 如果某些对齐/对象上下文在 SWA 压缩阶段被损失，TFM 已无法恢复。

SCTC 已经证明“简单再做一个全局统计校准”不是答案，所以 Run2 不建议再加 generic alignment。

---

## P1-3：SAGE Run1 的 per-image z-score 对二值变化比例不友好

**文件：`models/distill/sage_kd.py:20–24, 44–53`**

```python
mu = z.mean(HW)
sigma = z.std(HW)
(z - mu) / sigma
```

然后：

```python
sigmoid(z/T)
```

### 数学后果

它把每张图 Teacher/Student 的绝对 logit 基线都抹掉。

对于 binary CD：

- WHU change ratio ≈3.4%
- SYSU ≈21.1%

变化先验差异很大。

但 per-image z-score 强行让每张图的 logit：

\[
E[z']=0,\quad Var[z']=1
\]

因此 KD 更像学习“空间排序/相对峰值”，而不是教师的绝对 change confidence。

### 与 Run1 结果吻合

SYSU：

- Recall −0.92
- Precision +0.48

说明低排名的真实变化更容易被抑制。

WHU 稀疏变化里，只要 DINO 对少量 building-change region 排序较准，就可能补 Recall。

### 结论

**Run2 删除 per-image logit standardization。**

不是把 standardization 换个温度，而是换掉这条知识定义。

---

## P1-4：AuxSemanticHead 太浅，而且把 1/16 假装成 full-res

**文件：`models/distill/sage_heads.py:19–29`**

```python
c4 -> 1x1 Conv -> bilinear 256
```

### 问题 1：容量太弱

单个 `1×1 Conv` 只能在每个 16×16 location 做通道线性组合。

### 问题 2：假的 full-resolution supervision

Teacher 自己的 change head 也是：

**文件：`models/distill/fa_scrd_teacher.py:64–80`**

```python
16x16 diff
-> conv
-> 1 channel
-> bilinear 256
```

所以 SAGE Run1 的 full-res semantic KD，本质是：

> Student 16×16 → 放大 256  
> 对齐 Teacher 16×16 → 放大 256

它没有真正新增 256×256 边界知识。

### 怎么改

DINO 语义就在 **native semantic scale** 蒸馏，不再假装提供 full-resolution structure。

---

## P1-5：Run1 只是“独立 aux head”，并不是真正的 CrossKD

CrossKD 的关键不是“另建一个 head”本身，而是：

> 学生中间特征进入 teacher head / teacher task prediction pathway，从而让 Student 自己的主 head 保持 GT supervision，缓解 Teacher 与 GT 的直接冲突。

当前 SAGE：

```text
student c4
-> 新建 1x1 aux
-> teacher logit
```

并没有复用 Teacher task head，也没有更强的跨层任务投影。

因此可以说：

> **借鉴了监督隔离思想，但不是 CrossKD 式 cross-head。**

Run2 应保留“GT 主头隔离”这个优点，但把 aux 改为多尺度 ladder，而不是继续单 c4 线性头。

---

## P1-6：Gradient budget 修对了“幅值”，没修“方向”

**文件：`models/distill/sage_kd.py:64–78`**

Run1 保证：

\[
\lambda_{KD}
=
\rho
\frac{\|\nabla_{c4}L_{main}\|}
{\|\nabla_{c4}L_{KD}\|}
\]

这解决了：

> KD 不能比 main 大很多。

但没有解决：

\[
\langle \nabla L_{KD},\nabla L_{main}\rangle < 0
\]

即 Teacher 梯度方向可能仍然与 GT 主任务冲突。

这正是：

> **“安全地学错”。**

另外它只在 `c4` 量梯度，不代表整个 `(c2,c3,c4,c5)` 或 Student 参数空间都满足 25%。

### Run2 改法

对完整 change feature tuple：

\[
C=(c2,c3,c4,c5)
\]

计算 multi-feature gradient norm，再把 **所有 Teacher loss 的总预算** 控制在 `rho=0.25`：

\[
\sum_k \lambda_k\|\nabla_C L_k\|
\le
0.25\|\nabla_C L_{main}\|
\]

不增加新超参搜索。

---

## P1-7：Decoder 的 SupervisedAttention 是“破坏性乘法门控”

**文件：`models/decoder/a2net_decoder.py:304–328`**

```python
mask_f = sigmoid(mask)
mask_b = 1 - mask_f
context = ConvBNReLU([mask_f, mask_b])
x = x.mul(context)
```

### 问题

1. `[p,1-p]` 本质高度冗余。
2. `Conv+BN+ReLU` 后的 context：
   - 不在 [0,1]
   - 可以为 0
   - 可以 >1
3. 原始 feature 没有 identity bypass。

如果 coarse mask 早期判断错，错误会直接乘到 feature 上并传到下一层。

这很容易形成：

> coarse prediction → feature suppression → 更难纠正 FN

对 SYSU 这种变化密度高、类型复杂的数据尤其不理想。

---

## P1-8：TFM residual branch 的统计尺度不统一

**文件：`models/decoder/a2net_decoder.py:185–253`**

例：

```python
conv_branch1: Conv3x3 -> BN
conv_branch2: Conv1x1                # 无 BN
x_branch2 = ReLU(conv_branch2(x) + x_branch1)
```

branch3/4/5 同样存在“裸 1×1 projection + BN 后分支相加”。

这不是代码 bug，但会让 shortcut/projection 与主支统计尺度依赖训练自行适应。

如果 Run2 重构 TFM，可以顺便消除；但**不建议仅为它单独跑实验**。

---

## P1-9：FeatureFusionModule 的 identity projection 也没有 normalization

**文件：`models/decoder/a2net_decoder.py:29–41`**

```python
conv_fuse = Conv-BN-ReLU-Conv-BN
conv_identity = Conv1x1
return ReLU(c_fuse + conv_identity(c))
```

与典型 projection residual 的 `Conv+Norm` 不同。

同样属于结构质量问题，但优先级低于绝对差分瓶颈。

---

## P1-10：LWGANet 内部 residual 设计不统一，但当前不建议先动 backbone

**文件：`models/backbone/lwganet.py:306–321`**

```python
x1 = x1 + PA(x1)
x2 = LA(x2)
x3 = MRA(x3)
x4 = x4 + GA(...)
...
x = shortcut + norm1(mlp(x_att))
```

局部分支中：

- PA / GA 有 residual；
- LA / MRA 无 branch-local residual。

但整个 block 最外层仍有 `shortcut`。

### 判断

这不是当前最值得动刀的位置，因为：

- backbone 有预训练权重；
- 改它会把“Teacher/interaction 问题”和“pretrain architecture drift”混在一起；
- 当前最明显的信息损失发生在 SWA→TFM。

**Run2 主实验先不改 LWGANet block。**

---

# 5. P2：结构/实验工程问题

## P2-1：`models/__init__.py` 顶部仍写 Run3 BT-SAM-RDT

当前 package docstring 已经和 README/现状不一致。

影响不在指标，但非常容易让后续 Agent/人误读“当前活动实现”。

---

## P2-2：`get_loader(return_meta=True, return_state=False)` 实际不会返回 meta

**文件：`models/datasets/cd_dataset.py:245`**

```python
return_meta=bool(return_state)
```

忽略了调用参数 `return_meta`。

当前 cache path 因为 `return_state=True` 没踩雷，但 API 行为不对。

---

## P2-3：FA cache 里 `teacher_change_logit` 实际存的是 sigmoid probability

旧 `generate_teacher_cache.py`：

```python
change_prob = sigmoid(logit)
...
"teacher_change_logit": change_prob
```

后续代码按 probability 用，因此数值上没有错，但命名会制造未来错误。

---

## P2-4：Teacher training 无 shuffle / 无 val-selected best

**文件：`models/tools/train_teacher.py:105–126`**

训练按 list 顺序循环：

```python
sample_ids[sample_idx % len(sample_ids)]
```

没有 DataLoader shuffle，也没有 validation selection。

这会增加 Teacher 自身的 dataset/order bias。

Run2 若重新生成 DINO cache，应修；但不要再引入大规模 Teacher tuning。

---

## P2-5：当前 full-res DINO cache 浪费了 Teacher 原生尺度语义

生成 256×256 再训练时拿来 full-res KD，看起来精细，但本质信息仍来自 16×16。

Run2 应“按知识原生尺度使用”，而不是继续提高插值后的视觉分辨率。

---

# 6. LoRA / Change Head / Router：哪些公式是真的错，哪些只是设计不足

## 6.1 LoRA Q/V

**文件：`models/distill/fa_scrd_teacher.py:33–61`**

当前：

\[
h=A(x)
\]

\[
\Delta Q=B_Q h,\quad \Delta V=B_V h
\]

即 Q/V 共用一个 `A`，各自有独立 `B_Q/B_V`。

### 判断

- **不是公式错误。**
- 相比 Q/V 各自一套 A/B，容量更受约束。
- 但 Teacher 是训练期，容量不是当前最强证据根因。

**Run2 不建议把“拆 A_Q/A_V”作为主变量。**

先修 RoPE paired symmetry 和知识路由。

---

## 6.2 Change Head

**文件：`models/distill/fa_scrd_teacher.py:64–80`**

优点：

- mid/deep 都用；
- 时间交换对称；
- 轻量。

真正问题是：

> **原生空间分辨率只有 16×16。**

所以它适合：

- semantic region
- coarse object/change presence

不适合被当作：

- 精细 boundary teacher
- full-resolution pixel teacher

这也是引入 SAM2 structural expert 的真正理由。

---

## 6.3 Run1 region weight / gradient budget

FA-SCRD 的 `difficulty × teacher advantage` 思路方向没错，但旧 relation KD 的 target 本身错位且 loss 过大。

Run1 SAGE 又走到了另一个极端：

- 先不做 router；
- uniform DINO；
- 只靠 gradient cap 保安全。

结果已经说明：

> “安全上限”不能替代“Teacher 选择”。

---

# 7. 唯一建议的 SAGE-CD/Run2 主方案

# **Failure-Type Routed Native-Scale Dual-Expert Distillation + Symmetric Context Fusion**

中文概括：

> **学生先保住“差异 + 共同语义上下文”；训练时再由学生自己的失败类型决定谁来教：DINO 只负责可靠的语义区域，SAM2 只负责可靠的结构/边界区域，不可靠就 Reject；两教师共享一个总 KD 梯度预算。**

不做：

- 再调 Safe-DINO 的 rho/temperature；
- raw DINO relation KD；
- hard teacher gate；
- Teacher EMA 跟学生跑；
- SAM/DINO map 注入部署主路径；
- 再做一版 SCTC；
- backbone 大改。

---

# 8. Run2 学生结构：不加参数的 Symmetric Context Fusion

## 8.1 替换 TFM 的输入定义，但保留现有卷积参数

当前：

\[
D_s=|F^1_s-F^2_s|
\]

Run2：

\[
D_s=|F^1_s-F^2_s|
\]

\[
M_s=\frac{F^1_s+F^2_s}{2}
\]

\[
A_s=\frac{1-\cos(F^1_s,F^2_s)}{2}
\]

其中：

\[
A_s\in[0,1]
\]

新输入：

\[
X_s=D_s+A_s\odot |M_s|
\]

然后：

```text
X_s
-> 原 TemporalFeatureFusionModule dilation chain
-> C_s
```

### 性质

#### 交换对称

交换 T1/T2：

- `D` 不变；
- `M` 不变；
- cosine 不变。

所以：

\[
SCF(F_1,F_2)=SCF(F_2,F_1)
\]

#### 参数量

**0 新参数。**

仍然使用原 TFM 的所有 Conv/BN 权重。

#### 额外 FLOPs

只有：

- channel norm
- dot
- add/mul

4 个尺度合计远小于 0.01G，仍应落在 2.75–2.77G 范围。

### 为什么比单纯 concat 更合适

它在不扩通道的情况下补回了：

> “发生差异的位置，本来处于什么语义响应强度。”

而 unchanged 高相似位置：

\[
A_s\approx0
\]

不会大规模注入 common context。

---

# 9. Decoder 同时做一个零参数量的 gate 修正

现有 SupervisedAttention：

```python
context = ConvBNReLU([p, 1-p])
x = x * context
```

改成 identity-centered gate：

```python
g = sigmoid(BN(Conv([p, 1-p])))
x = x * (0.5 + g)
x = conv2(x)
```

于是缩放范围：

\[
0.5 \le 0.5+g \le 1.5
\]

而 BN 输出接近 0 时：

\[
g\approx0.5\Rightarrow scale\approx1
\]

### 好处

- 保留近 identity 初始行为；
- 可以抑制，也可以增强；
- 不会像 ReLU context 那样把 feature 直接乘到 0；
- 参数量完全不变。

这与 SCF 一起作为 **Student structure reform** 一个整体消融臂，不拆成两个微小实验。

---

# 10. Teacher 结构：DINO + SAM2，但不做“两个 Teacher 同图竞争”

## 10.1 DINO = semantic expert

只负责：

- change interior
- semantic region completeness
- student FN/semantic ambiguity

**蒸馏尺度：1/16。**

不再把 bilinear 256×256 当成精细 target。

为了保证 crop replay 精度，可以：

1. cache 仍保存 full-res raw logit；
2. 先按 student augmentation 在 256×256 replay；
3. 再 area/bilinear downsample 到 16×16 形成 semantic target。

这样比直接对 16×16 cache 做 crop round 更稳定。

---

## 10.2 SAM2 = structural expert

已有 cache：

```text
instance_id
boundary
quality
```

Run2 第一版只把核心贡献控制在：

- boundary
- quality
- instance_id 用于 region aggregation / routing

而不是再造一个 SAM full change mask。

**结构监督尺度：1/4。**

SAM 的优势是 object boundary / instance structure，不让它去做 DINO 的语义工作。

---

# 11. Student failure profile：二值变化检测特有的 Teacher Agent

主预测：

\[
p_S
\]

GT：

\[
Y
\]

全部 router 输入 `detach()`。

---

## 11.1 Semantic failure

GT boundary band 之外的 interior：

\[
I=1-\text{band}(B_Y)
\]

Student semantic difficulty：

\[
d_D=\operatorname{Pool}_{16}(|p_S-Y|\odot I)
\]

DINO reliability：

\[
r_D=1-\operatorname{Pool}_{16}(|p_D-Y|)
\]

最终：

\[
w_D=\operatorname{stopgrad}(d_D\cdot r_D)
\]

含义：

> Student 当前错，而且 DINO 在这里确实比“瞎教”更可靠，才让 DINO 发声。

---

## 11.2 Structural failure

从主预测得到 differentiable/detached boundary estimate：

\[
B_S
\]

GT boundary：

\[
B_Y
\]

Student boundary difficulty：

\[
d_S=|B_S-B_Y|
\]

SAM reliability：

\[
r_S=Q_{SAM}\cdot (1-|B_{SAM}-B_Y|)
\]

最终：

\[
w_S=\operatorname{stopgrad}(d_S\cdot r_S)
\]

`instance_id` 用于把局部 difficulty / quality 在同一结构实例内做 region aggregation，避免 boundary pixel 独立抖动。

---

# 12. 不再使用 per-image z-score

DINO loss：

\[
L_D=
\frac{
\sum w_D\cdot BCEWithLogits(z_D^S,\sigma(z_D^T/T))
}{
\sum w_D+\epsilon
}
\]

其中：

- Student `z_D^S` 来自 train-only SemanticLadderHead；
- Teacher 用 raw task logit；
- 不做每图 z-score。

Teacher 校准差异由 `r_D` / reject 解决，而不是破坏 class prior。

---

# 13. Aux head：从单 c4 改成 train-only ladder

## SemanticLadderHead

输入：

```text
c3 1/8
c4 1/16
c5 1/32
```

统一到 1/16：

```text
c3 -> down
c4 -> identity
c5 -> up
-> 各自轻投影
-> sum
-> GELU
-> 1×1 semantic logit
```

### 为什么

这借鉴 UNIC 的核心原则：

> intermediate features 应该有更直接的信息高速路到蒸馏目标。

但全部 projector 都是 train-only。

---

## BoundaryHead

输入：

```text
c2 1/4
```

建议：

```text
DW3x3 -> GELU -> 1x1
```

输出 1/4 boundary logit。

不需要 upsample 256 再算 Teacher KD。

---

# 14. 两个 Teacher 如何不冲突：共享一个“总梯度预算”

定义所有变化特征：

\[
C=(c2,c3,c4,c5)
\]

多 feature gradient norm：

\[
g(L)=
\sqrt{
\sum_i
\left\|
\frac{\partial L}{\partial c_i}
\right\|_2^2
}
\]

Teacher utility：

\[
u_D=\operatorname{mean}(w_D)
\]

\[
u_S=\operatorname{mean}(w_S)
\]

加入 Reject prior：

\[
\pi_D=\frac{u_D}{u_D+u_S+\epsilon_r}
\]

\[
\pi_S=\frac{u_S}{u_D+u_S+\epsilon_r}
\]

剩余就是 reject。

固定 Run1 的：

\[
\rho=0.25
\]

每个 Teacher：

\[
\lambda_k=
\min
\left(
1,
\frac{
\rho\pi_k g(L_{main})
}{
g(L_k)+\epsilon
}
\right)
\]

最终：

\[
L=
L_{main}
+\lambda_D L_D
+\lambda_S L_S
\]

则近似满足：

\[
\lambda_Dg_D+\lambda_Sg_S
\le
0.25g_{main}
\]

### 这比 Run1 强在哪里

Run1：

> 只有一个 Teacher；保证“别太大”。

Run2：

> 先判断“该不该教、该由谁教”，最后才保证“别太大”。

---

# 15. 为什么这套方案有机会突破“只帮一个数据集”

## SYSU

Run1 DINO：

- Recall −0.92
- Precision +0.48

说明它在一些真实 change 上把 Student 往负类推。

Run2 中，如果某个 SYSU positive region：

- Student 是 FN；
- DINO 也是 FN；

则：

\[
r_D \downarrow
\]

DINO 自动接近 Reject，不能继续压 Recall。

如果 SAM2 对该 region 的 object boundary 有可靠结构：

\[
w_S \uparrow
\]

让 structural expert 接管。

---

## WHU

WHU Run1 的收益几乎完全来自：

\[
Recall +0.84
\]

说明 DINO 对 sparse building-change 的一部分漏检确实有增益。

Run2 不会删除 DINO，而是：

> 在它真的可靠、Student 真的困难时保留它。

因此理论上应保留 WHU 的正增益，同时避免 SYSU 的错误语义扩散。

---

# 16. 逐文件修改清单

## `models/decoder/a2net_decoder.py`

### 改

1. `TemporalFeatureFusionModule.forward()`：
   - `abs_diff` → SCF：
     - abs diff
     - symmetric mean
     - cosine discrepancy
     - context injection
2. `SupervisedAttentionModule`：
   - `ConvBNReLU context`
   - 改为 `ConvBN + sigmoid`
   - identity-centered scale `0.5 + gate`
3. 保持所有已有 Conv/BN 参数尺寸不变。

### 为什么

直接解决 Student 表征落点问题，同时保证 deploy params 精确不变。

---

## `models/a2net.py`

### 改

- 主模型接口基本不动。
- 保持 `return_change_features=True` 返回 `(c2,c3,c4,c5)`。
- 增加可选 diagnostics，不把 Teacher 加进 model state。
- formal test 强制 swap/deploy assertion。

### 不改

- Teacher/cache 不进入 `A2Net_LWGANet_L0`。
- `switch_to_deploy()` 仍只作用学生。

---

## `models/distill/sage_heads.py`

### 删除

```text
单 c4 1×1 full-res semantic head
```

### 增加

- `SemanticLadderHead(c3,c4,c5) -> 1/16`
- `AuxBoundaryHead(c2) -> 1/4`

全部 training-only。

---

## `models/distill/sage_kd.py`

### 删除

- `standardize_logit()`
- full-res uniform semantic BCE

### 增加

- native-scale semantic KD
- boundary KD
- multi-feature gradient norm
- total gradient budget allocation

---

## 新增 `models/distill/sage_router.py`

职责：

- student FN/FP/boundary difficulty
- DINO semantic reliability
- SAM boundary/quality reliability
- region aggregation
- reject
- detached routing diagnostics

必须 log：

```text
DINO usage
SAM usage
Reject usage
DINO utility
SAM utility
effective gradient ratio
FN-region coverage
boundary-region coverage
```

---

## `models/distill/fa_scrd_teacher.py`

### 修

- Teacher paired symmetry / RoPE train-mode问题。
- ChangeHead 提供 native semantic representation 的接口。

### 暂时不改

- 不把 LoRA rank 当创新。
- 不先拆 Q/V shared A。

---

## `models/distill/cache_dataset.py`

### 重写为 schema-aware replay

不同 key 不同 interpolation。

同时：

- 保留 meta；
- 检查 actual checkpoint hash；
- 检查 list fingerprint；
- 检查 schema version。

---

## `models/tools/train_teacher.py`

### 修

- `backbone.eval()` + LoRA/head 可训练。
- shuffle。
- paired augmentation 必须共享。
- validation/audit 至少报告 Teacher Recall/Precision/F1，不盲目只保存最后 step。

---

## `models/tools/generate_sage_cache.py`

### 修

- actual checkpoint bytes SHA256。
- 保存 schema/version。
- raw semantic logit。
- list fingerprint。
- 不把路径 hash 当 checkpoint hash。

---

## `models/scripts/train_sage.py`

### 改成 SAGE v2 trainer

- 单 optimizer/统一 scheduler。
- strict resume。
- 训练参数 = student + train-only aux/router 明确统计。
- infer params = student only。
- swap/deploy formal assertion。
- 记录 router usage 与 effective KD gradient ratio。
- cache fingerprint 写入 checkpoint/log。

---

## `models/tools/smoke_sage.py`

新增检查：

1. student params = 2,913,094
2. SCF swap invariance
3. aux toggle 不改变 main prediction
4. DINO/SAM cache replay 同步
5. `instance_id` nearest replay
6. Teacher A/B swap
7. semantic/boundary loss 有梯度
8. total KD gradient budget ≤ 0.25 main
9. deploy before/after <1e-6

---

# 17. 部署参数/FLOPs保证

## 参数

SCF：

- 只做 tensor operation；
- 0 参数。

Decoder gate：

- 复用原 Conv/BN；
- 只换 activation/data flow；
- 0 新参数。

Teacher/router/aux：

- 全部在 Student model 外；
- deploy 时不保存/不加载。

因此部署参数应严格保持：

\[
\boxed{2,913,094}
\]

---

## FLOPs

新增 deploy functional ops 主要是 4-scale cosine/mean/multiply。

数量相对原 2.75G 卷积非常小，预计：

\[
<0.01G
\]

但正式 Run2 必须同时报告：

1. THOP；
2. functional-op 手工审计。

目标仍保持：

\[
2.75\text{–}2.77G
\]

若手工审计超过上限，SCF 直接失败，不做“忽略不计”。

---

# 18. SAGE-CD/Run2 最小消融

我建议最终论文级消融保留 **5 臂**。因为这次有两个真正变量：

1. Student interaction reform；
2. failure-type dual-teacher routing。

少于 5 臂很难证明到底是谁有效。

| ID | 配置 | 相对上一臂唯一变量 | 目的 |
|---|---|---|---|
| R0 | Clean A2Net | — | clean anchor |
| R1 | R0 + joint temporal BN | joint BN | 与 Run1 normalization control 对齐 |
| R2 | R1 + SCF + identity-centered decoder gate | Student structure | Student 本体是否真的变强 |
| R3 | R2 + DINO semantic + SAM boundary，uniform/no failure routing | Dual Teacher knowledge | “只是加两个 Teacher”是否仍冲突 |
| R4 | R3 + failure-type reliability routing + Reject | Router | **完整主方法** |

全部：

```text
SYSU + WHU
batch 64
40k
seed 2333
lr / wd / dice / main loss 与 Run1 相同
rho = 0.25 固定，不调
```

---

# 19. 成功阈值与失败判据

## 主成功门

Teacher 的真实贡献必须看：

\[
R4-R2
\]

因为 R2 已包含 Student structure。

要求：

### SYSU

\[
\Delta F1(R4-R2)\ge +0.30
\]

### WHU

\[
\Delta F1(R4-R2)\ge +0.30
\]

**两者同时满足。**

否则不能说 Teacher routing 有效。

---

## 总方法门

\[
R4-R1 > 0
\]

在 SYSU、WHU 都必须正。

建议论文进入下一阶段的较强门槛：

\[
R4-R1 \ge +0.40\text{ pp}
\]

两数据集均满足。

---

## Router 有效性

如果 R3 已经两数据集都正，R4 至少不能退化。

更理想：

\[
R4 > R3
\]

且能看到：

- SYSU FN region 的 DINO usage 自动下降/Reject 或 SAM 上升；
- WHU reliable semantic region 的 DINO usage 保留。

---

## 直接失败

任何一个成立就停止：

1. `R2 < R1` 在 SYSU 或 WHU 明显下降（例如 <−0.30）  
   → Student SCF 设计失败。
2. `R4 <= R2` 任一数据集  
   → Teacher 没提供净增益。
3. R4 又变成 SYSU-only 或 WHU-only  
   → Teacher routing 叙事失败。
4. Reject >95%  
   → 重演旧 hard-gate starving。
5. DINO 或 SAM usage >95%  
   → Router 退化成单 Teacher。
6. `R4-R2` 只体现 Recall/Precision 极端互换而 F1 无增益  
   → 不是有效知识迁移。
7. deploy params/FLOPs/swap 破硬约束  
   → 方法直接否决。

---

# 20. 实际启动顺序：为了少浪费 GPU

## Step 0：不训练，先做 Teacher utility audit

用 **validation split**，不要用 test 做设计。

对 G1/clean checkpoint 统计：

### DINO

- Teacher positive ratio
- Recall / Precision / F1
- FN region coverage
- Student FN 中：
  - DINO 正确比例
  - DINO 同样错误比例

SYSU、WHU 分开。

### SAM

- boundary precision/recall/F1（固定 boundary band）
- quality 与 boundary correctness 的相关性
- student boundary failure 中 SAM 可修复比例

如果两个 Teacher 在学生真实 failure 上都没有可利用 advantage，**Run2 不启动 40k**。

---

## Step 1：P0 smoke

依次：

1. `antialiased_cnns` implementation audit
2. Teacher swap
3. cache fingerprint
4. cache replay
5. instance ID nearest
6. student swap
7. deploy consistency
8. total gradient budget

---

## Step 2：R0/R1 anchor

如果新 Run2 merge 后：

- Student 原始 `none` path code hash；
- 数据 pipeline；
- loss；
- optimizer；
- scheduler

与 SAGE Run1 G0/G1 完全一致，可把 Run1 G0/G1 作为历史 anchor 展示。

**更稳妥做法仍是 Run2 重跑 R0/R1**，因为正式消融最好来自同一 implementation version。

---

## Step 3：先跑 R2

SYSU + WHU。

目的：

> 不借 Teacher，先确认 Student interaction reform 不是另一个 SCTC 式单数据集 trick。

R2 任一数据集明显负，停止。

---

## Step 4：直接跑 R4 主方法

先验证真正目标：

> Dual experts + failure routing 是否能在 SYSU/WHU 同时超过 R2。

如果 R4 失败，停止，不需要补漂亮消融。

---

## Step 5：只有 R4 过门，再跑 R3

R3 是为了论文证明：

> “不是 DINO+SAM 堆叠本身有效，而是 student-failure routing 解决 Teacher conflict。”

这是最节省算力的顺序。

---

# 21. 为什么不建议的几个方向

## 不建议 1：只把 AuxSemanticHead 从 c4 挪到 c2

会让 DINO coarse semantic target 强行约束高分辨率 feature，更容易把插值伪边界当知识。

---

## 不建议 2：继续 tuning `rho`

Run1 已证明 magnitude safe 但 direction wrong。

把 0.25 调 0.1，只会“少学错一点”。

---

## 不建议 3：DINO + SAM logits 直接加权求和

两个 Teacher 输出空间不同：

- DINO：semantic change confidence
- SAM：structure/boundary/instance

直接 fusion 是把互补知识硬塞成一个伪概率。

---

## 不建议 4：再做一个 learnable heavy router

当前只有 seed 2333 且两个数据集做 gate。

复杂 router 很容易自己成为第三个模型并过拟合。

第一版用：

> deterministic detached failure profile + reliability + Reject

论文逻辑反而更清楚。

---

## 不建议 5：先重写 LWGANet

当前 backbone 有预训练，最强证据瓶颈位于：

```text
SWA -> |F1-F2| -> TFM
```

先改 backbone 会让变量爆炸。

---

# 22. 与 2024–2026 文献的实质关系

## CrossKD — CVPR 2024

可借：

> Teacher supervision 与 Student 主 GT head 解耦，减少矛盾监督；task prediction knowledge 比盲目 feature imitation 更直接。

本项目保留：

- main decoder 只吃 GT；
- Teacher 只走 expendable aux。

但不照搬 object detection teacher head。

---

## UNIC — ECCV 2024

可借：

1. ladder projectors，让中间层更直接收到 Teacher signal；
2. 多 Teacher 影响必须平衡。

本项目不采用 random teacher dropping，而是利用 fully-supervised BCD 的优势：

> **student failure + teacher reliability 做有语义的 teacher selection。**

---

## Mask-CDKD — ISPRS JPRS 2026

最重要的启示不是抄 MMoA，而是：

> Foundation Model → VHR-RS 存在跨域知识偏差，不能默认 Teacher 输出就是正确知识；需要 adaptive suppression / selection。

这与本项目连续负迁移证据高度一致。

---

## BiFA — TGRS 2024

支持：

> CD 的双时相表示不能只依赖最终差分，temporal interaction/alignment 是核心问题。

但本项目 SCTC 已证明 generic temporal calibration 不够，因此 Run2 选择：

> **差异条件化的共同语义上下文注入**

而不是再做统计对齐。

---

# 23. 最终根因判断

按我目前的证据强度：

## 根因 A — 最高概率

**Teacher target 的尺度/任务定义与 Student failure 类型不匹配。**

证据：

- DINO target 原生仅 16×16；
- full-res 只是 bilinear；
- SYSU/WHU Recall 方向相反；
- 不同 KD 接口会把 Recall/Precision 推向不同方向。

---

## 根因 B — 很高概率

**纯 `|F1-F2|` 让 Student 变化表示缺乏共同语义上下文，Foundation semantic knowledge 没有理想落点。**

---

## 根因 C — 很高概率

**统一 Teacher 缺少 Reject 与 failure-type specialization。**

过去：

- hard gate 太硬 → KD 饿死；
- uniform Safe-DINO → 安全负迁移。

正确中间态应是：

> continuous reliability + explicit reject + specialized experts。

---

## 根因 D — 必须排除的工程混杂

- Teacher RoPE paired stochasticity
- cache provenance
- aux scheduler
- resume validation
- SAM cache replay type

这些不修，Run2 即使跑出 ±0.2–0.5，也很难确定是在验证方法还是验证工程偶然性。

---

# 24. 立即执行顺序

你现在说“先看方案和文献，回来再改代码”，所以**此时先不要修改代码**。

回来准备实现时建议严格按：

1. **Teacher/cache audit**  
   DINO/SAM 对 SYSU/WHU val 的 failure coverage。
2. **修 P0**  
   RoPE、scheduler、cache schema/hash、resume、typed replay、formal assertion。
3. **实现 Student SCF + identity-centered gate**  
   不加任何 deploy 参数。
4. **实现 native-scale aux heads**。
5. **实现 failure-type router + Reject + total grad budget**。
6. synthetic smoke。
7. real-cache 10–50 step dry run。
8. deploy params/FLOPs/swap。
9. R2。
10. R4。
11. R4 过门后再补 R0/R1/R3 所需正式臂。

---

# 25. 仍需补充的证据

在真正改代码前，最值得补的不是更多论文，而是下面 6 个本项目数据：

1. **DINO Teacher 自身在 SYSU/WHU val 的 Recall/Precision/F1。**
2. DINO prediction positive ratio 与 GT positive ratio。
3. G1 Student FN 中 DINO 能修复的比例。
4. SAM boundary 在真实 GT change boundary 上的 precision/recall 与 quality calibration。
5. `antialiased_cnns` 在 RSML-3 的实际 import 状态。
6. 当前 SAGE_DINO cache 对应的真实 checkpoint SHA256，以及 Teacher 是在哪个数据集、多少 step 适配得到的。

如果第 1–4 项显示：

> Teacher 在 Student failure 上没有互补 advantage，

那么应直接停止 Teacher 路线，而不是继续设计 router。

---

# 26. 一句话版本

**Run2 不再问“怎么把 DINO 蒸馏得更安全”，而是问“Student 当前错在哪里、哪个 Teacher 在这个错误类型上真的比 Student 更可靠；可靠才教，不可靠就拒绝”，同时把 A2Net 的纯绝对差分改成零参数、严格对称的差异条件语义融合，让 Teacher 知识终于有合适的 Student 表征落点。**

---

## 参考依据

- Wang et al., **CrossKD: Cross-Head Knowledge Distillation for Object Detection**, CVPR 2024.
  - https://openaccess.thecvf.com/content/CVPR2024/html/Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection_CVPR_2024_paper.html
- Sarıyıldız et al., **UNIC: Universal Classification Models via Multi-teacher Distillation**, ECCV 2024.
  - https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/581_ECCV_2024_paper.php
- Shu et al., **Mask-CDKD: A source-free and label-free cross-domain knowledge distillation framework from SAM for satellite onboard VHR land-cover mapping**, ISPRS JPRS 2026.
  - https://www.sciencedirect.com/science/article/pii/S0924271626001486
- Zhang et al., **BiFA: Remote Sensing Image Change Detection With Bitemporal Feature Alignment**, IEEE TGRS 2024.
- 项目提供的 `参考文献_TeacherRouting.txt`、`参考文献_FA-SCRD.txt`、`参考文献_文献调研0920.txt`。
- 项目 `experiment_metrics.xlsx`，抽取规则：同一 `train_log.txt` 最后一个完整 `=== TEST RESULTS === ... === END TEST RESULTS ===` 区块。
