# 方向C 架构图

## 1. 架构概述

方向 C 在全监督二值变化检测的 A2Net-LWGANet-L0 主路径之外，增加仅用于训练的“困难诊断—可靠性评估—教师路由—Reject—独立蒸馏损失”分支：学生预测 P 与 GT 产生六维困难画像 h，两类离线教师的自身质量 q 与困难适配度 d 共同形成可靠性 r，小型两层 MLP 接收 [h,r]，输出两位教师与一个 Reject 的权重。h、r、w 对 P 的梯度路径全部截断，学生仍通过 GT 损失和被接受的教师损失学习；Reject 时所有教师损失贡献为零，GT 监督照常。部署只保留原有主网络，参数 2,913,094、256×256 FLOPs 约 2.75G。本图是下一阶段设计图：最新 AGENTS.md 明确方向 C 尚未实现，具体指标、适配计算、损失头和 Router 学习目标未定部分均标为 `? TBD`。

## 2. 训练图

图内只使用 ASCII 字符，中文释义见第 4 节。`(P)`、`(H)` 等同名端口表示同一张量导线，用于减少交叉线；主路径只出现一个占位盒子。实线表示方向已确定，虚线框或 `? TBD` 表示待定设计。

```text
TRAINING GRAPH   |   K = 2   |   Same (port) = same tensor / wire
          +------------------------------------+
A, B ---->| STUDENT: A2Net-LWGANet-L0          |----> (P) ----------------------> change prediction
          | Existing main-path figure          |
          | No internal structure redrawn      |. . > (F?)  existing-feature tap ? TBD; training only
          +------------------------------------+

(P), GT ----> [ L_GT: existing BCE + Dice supervision ] ----> (G)

(P) --------> [ sg(P) ] --------> (P0)

All h / r / w routes below use (P0), never an undetached (P).


OFFLINE TEACHER POOL                                          (P0), GT
                                                                             |
+----------------------------------------------------+                       v
| K1: SAMStruct; T1 + T2                             |        +------------------------------------------------+
| instance_id / boundary / quality                   |        | ERROR ANALYZER                                 |
| K2: OVCDistill; DINOv2-derived                     |        | build_cd_difficulty(P0, GT)                    |
| soft_change / confidence / relation                |        | h = [boundary_miss, fragmentation,             |
| Fixed cache; no online teacher forward             |        |      small_change_miss, bg_false_alarm,        |
+----------------------------------------------------+        |      pred_uncertainty, change_ratio]           |
                         |                                    | Output: h [sg_P] + region masks                |
                         v                                    | ? TBD: metrics / thresholds / empty sets       |
+----------------------------------------------------+        +------------------------------------------------+
| Shared geometry replay with A / B / GT             |
| Targets -> (T); quality/confidence -> (Q)          |             h [sg_P] ----> (H)
| q pooling / calibration ? TBD                      |
+----------------------------------------------------+
                                                              (H), (T), GT; (P0) if needed
                                                                             |
                                                                             v
                                                              +- - - - - - - - - - - - - - - - - - - - - - - - +
                                                              | DIFFICULTY FIT ? TBD                           |
                                                              | d_k = fit(teacher k, current errors)           |
                                                              | estimate_teacher_fit(); no grad to P           |
                                                              +- - - - - - - - - - - - - - - - - - - - - - - - +
                                                                      d [sg_P] ----> (D)

                      (H), (Q), (D)
                                      |
                                      v
                      +----------------------------------------------------------------------+
                      | RELIABILITY EVALUATOR                                                |
                      | r_k = psi(h, q_k, d_k); r = [r_1, r_2] [sg_P]                        |
                      | q = teacher self-quality; d = difficulty fit                         |
                      | ? TBD: psi / calibration / supervision                               |
                      +----------------------------------------------------------------------+
                                                       |
                                                       v
                      +----------------------------------------------------------------------+
                      | TRAIN ROUTER: phi                                                    |
                      | Input [h, r] -> Linear -> ReLU -> Linear -> softmax                  |
                      | Output (W) = [w_1, w_2, w_Reject]; K+1 = 3; [sg_P]                   |
                      +----------------------------------------------------------------------+
                                                       |
                                                       v
                      +----------------------------------------------------------------------+
                      | REJECT GATE + WEIGHT FREEZE FOR STUDENT UPDATE                       |
                      | Reject: g = 0; accept: g = 1                                         |
                      | w_eff[k] = g * sg(w_k), k = 1..K -> (WE)                             |
                      | ? TBD: Reject decision rule (argmax / threshold / ...)               |
                      +----------------------------------------------------------------------+

(P), (F?) retain student gradients on loss branches; (T) is fixed.

+- - - - - - - - - - - - - - - - - - - - - - - - - - +        +- - - - - - - - - - - - - - - - - - - - - - - - +
| TEACHER HEAD 1 ? TBD (student-side)                |        | TEACHER HEAD 2 ? TBD (student-side)            |
| (P)/(F?) -> SAM target-space predictions           |        | (P)/(F?) -> OV target-space predictions        |
| (T:SAM) -> boundary / instance loss                |        | (T:OV) -> soft-change / relation loss          |
| Output (L1) = ell_1; no cross-teacher mean         |        | Output (L2) = ell_2; loss forms ? TBD          |
+- - - - - - - - - - - - - - - - - - - - - - - - - - +        +- - - - - - - - - - - - - - - - - - - - - - - - +

                      (L1), (L2), (WE)
                                        |
                                        v
                      +----------------------------------------------------------------------+
                      | ALL-TEACHER LOSS REDUCER: reduce_teacher_losses()                    |
                      | L_teacher = sum_k w_eff[k] * ell_k                                   |
                      | Reject: ALL teacher loss contributions = 0                           |
                      | Includes every response / boundary / region / relation term          |
                      +----------------------------------------------------------------------+
                                                       |
                                                       v
                      +----------------------------------------------------------------------+
                      | TOTAL STUDENT LOSS: L = (G) + lambda * L_teacher                     |
                      | Reject -> L = L_GT only; GT branch always active                     |
                      | Backprop -> student + active training heads; NOT via h/r/w           |
                      +----------------------------------------------------------------------+

ROUTER LEARNING PATH (separate from the student-loss update above)

+- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
| (W), detached training evidence . . > L_router ? TBD . . > update Router phi                                 |
| ? TBD: fit/utility target and non-RL optimizer schedule; no gradient into student P/F                        |
| An independent learning signal is required; sg(w) does NOT train the Router by itself.                       |
+- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
```

## 3. 部署图

```text
DEPLOYMENT GRAPH

          +------------------------------------+
A, B ---->| A2Net-LWGANet-L0                   |----> P ----> binary change map
          | Existing main-path figure          |
          | All training auxiliaries removed   |
          +------------------------------------+

Parameters: 2,913,094     FLOPs: approximately 2.75G at 256 x 256

No Cache / Error Analyzer / Reliability / Router / Teacher Head / auxiliary loss

No GT input. No Reject decision. No added inference branch.
```

## 4. 图例说明

### 4.1 符号、端口与四个接口

- `A / B`：T1 / T2 图像；`GT`：训练集二值变化标签；`P`：主路径变化预测。现有代码产生多尺度 sigmoid 概率图，图中以 P 统称；诊断使用的具体尺度仍待定，原有多尺度 GT 监督不因图中简写而删除。
- `(P0) = sg(P)`：仅供诊断、可靠性与路由决策使用的预测副本；`(F?)` 是现有学生特征的可选训练期读取口，取哪一层、是否需要读取尚待定，不代表新增主路径模块。
- `(T)`：完成同步几何增强的固定 teacher targets；`(Q)`：SAM quality / OV confidence 原始可靠性信息；`(H)`：困难画像及区域掩码；`(D)`：教师对当前困难的适配度；`(W)`：Router 原始 softmax 权重；`(WE)`：Reject 后用于学生更新的有效权重；`(G)`：GT 主损失；`(L1)/(L2)`：分别属于两位教师的原始损失。
- `build_cd_difficulty()` 对应 Error Analyzer；`estimate_teacher_fit()` 对应困难适配与可靠性评估；`route_teachers()` 对应 MLP Router；`reduce_teacher_losses()` 对应逐教师聚合及 Reject 后的损失归零。这四个名称来自设计清单，不是当前仓库已经实现的新模块。
- `-->` / `|` / `v` 表示确定的数据传递；`. . >`、`:`、间隔短横线的框和 `? TBD` 表示待定路径或待定实现；它们不表示概率大小或梯度大小。

### 4.2 六维困难画像与教师可靠性

h 的顺序固定为：**[边界漏检、内部碎裂、小变化漏检、背景误报、预测不确定性、变化面积比]**，分别对应图中 `boundary_miss`、`fragmentation`、`small_change_miss`、`bg_false_alarm`、`pred_uncertainty`、`change_ratio`。内部碎裂需要连通性/碎片层面的定义，不能默认为内部漏检率；变化面积比取 GT 还是预测统计、边界带宽、小目标阈值、归一化与空区域规则均待定。以每个样本一组 h 和一组路由权重表示，不擅自扩成逐像素 Router。

`q_k` 是教师自身的质量信号，`d_k` 是其证据对当前错误类型的适用性，`r_k = psi(h,q_k,d_k)` 才是 Router 使用的可靠性。q 不能直接等同于 d：例如高质量 SAM 分割并不意味着该区域真的发生变化。`psi`、d 的具体公式及其校准/监督仍待定，图中没有将它们画成已完成的算法。

第一阶段 **K=2**：SAMStruct 与 OVCDistill 各为一位教师，Router 输出维度为 **K+1=3**。boundary、instance、soft_change、relation 是教师内部字段或监督分项；quality/confidence 主要作为可靠性证据，不自动成为额外教师或必须模仿的目标。SAMStruct 含独立 T1/T2 结构，OVCDistill 为 DINOv2 派生变化/关系缓存；不新增在线 SAM/DINO 前向，也不加入 CLIP/VLM。

### 4.3 stop-gradient 的准确含义

**主预测 P 分成两路：决策路先 sg，损失路保留梯度。** 图中的 `sg_P` 表示该量对学生预测 P 不存在反传路径：h 由 `sg(P)` 构造，r 与 w 沿这条已截断的路径计算；d 若额外读取学生预测/特征，也必须先停止该输入的梯度。因而学生不能通过修改诊断统计来降低路由分支的目标、主动“制造困难”。

`sg_P` **不等于冻结 Router 参数**。图中为使学生更新语义明确，在教师损失聚合前进一步使用 `sg(w)`，将本步路由权重视为常量；Router 参数 phi 通过另行定义的 `L_router` 学习，其目标与非 RL 更新安排用虚线标为待定。这是清单中“分开训练 Router”的接口表达，不能声称仅执行总学生损失的 backward 就能训练整个 Router。若可靠性评估器也含可学习参数，其学习目标与更新路径同样待定。

(P)/(F?) 到学生侧 Teacher Head、再到教师损失的路径**不做 sg**，否则该蒸馏项无法训练学生。Teacher Head 指训练专用的学生投影/预测头，不是可更新的 SAM/DINO 教师网络；具体输入、结构、目标构造和损失形式尚未确定。

### 4.4 Reject 与损失的严格语义

令原始教师损失为 `ell_k`，Router 输出为 `w=[w_1,w_2,w_Reject]`；接受样本时 `g=1`，Reject 时 `g=0`，学生更新采用 `w_eff[k]=g*sg(w_k)`：

$$L = L_{GT} + \lambda\sum_{k=1}^{K} w^{eff}_k\,\ell_k.$$

这就是你要求的 `L_GT + lambda * sum_k w_k * L_k` 的展开写法：用于实际加权的 w 已经过 Reject 门控。**Reject 时所有有效教师权重为零，全部 teacher loss 贡献归零，L 严格退化为 L_GT。** 原始 `ell_k` 的数值可以用于日志，不必伪造为零；关键是它们不能再贡献本步学生梯度。边界、区域/实例、变化响应、关系及任何其他教师诱导项均受同一 Reject 控制。

softmax 本身不会在有限 logits 下产生严格的全零教师权重，因此图中显式保留 Reject Gate；由 argmax 还是阈值触发、阈值多少等材料未定，不在图里替你决定。接受时图示沿用原教师权重，不把 Reject 概率重新分摊给教师；后续若采用其他归一化需另作定义。

`L_router` 不能仅等于当前加权的非负教师损失，否则“全部 Reject”就能降低该目标；所需的适配/效用监督尚待确定，图中保留虚线入口而不臆造已完成的奖励或元学习算法。没有引入 RL。

### 4.5 训练与部署边界

所有 Cache 字段与 A/B/GT 按同一 crop、resize、flip、时相交换规则对齐；实例 ID 使用最近邻插值。SAM 两时相 ID 不能按数值相等直接匹配；OV relation 的通道含义及变换规则需向缓存生成实现核验。现有 Cache 不含 AnyChange 所需的原始 SAM dense embeddings，因此本图没有擅自加入它的 latent matching 模块。

Teacher Head 和评估/路由分支只读取学生信息、计算辅助监督，不将教师 map、r 或 w 注入学生主预测。部署删除 Cache 读取与 replay、Error Analyzer、适配/可靠性评估器、Router、Reject Gate、Teacher Head、辅助 loss 和 Router 训练目标；部署不需要 GT。图中参数/FLOPs 为既定部署约束，没有把尚未实现的方向 C 当作已测量结果。新辅助开关也不能继承旧 `z2_srd` 模式造成的 joint-BN 训练差异。

### 4.6 本图依据与未定项来源

已核对材料：本对话的 `源码调研清单.md`；GitHub 最新 [others/teacher_routing_code.txt](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/main/others/teacher_routing_code.txt)（与上传副本一致，含 13 个源码文件）；[AGENTS.md](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/main/AGENTS.md)、[models/a2net.py](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/main/models/a2net.py)、[models/scripts/train.py](https://github.com/YuqiWang-code/LS-Rep_BCD/blob/main/models/scripts/train.py) 及 Cache replay；上传的服务器/缓存说明与 `参考文献_TeacherRouting.txt` 正文。

其中，UNIC 的 `loss_per_teacher()` / `aggregate_losses()` 支撑“逐教师保留损失、再聚合”的结构，但原实现强制至少一位教师，不能直接当作 Reject；U-Know-DiffPAN 的 `kd_loss()` 和离线保存脚本支撑“固定 Cache 与独立损失分项”，其 feature loss 并不会随 soft-response 权重自动归零；Segment Any Change 的 `bitemporal_match()` 支撑“分割质量不等于变化证据”的区分。**六维诊断、困难适配 psi/d、K+1 Router 和全教师 Reject 的组合来自当前方向 C 设计，不是三篇论文已提供的完整架构。**
