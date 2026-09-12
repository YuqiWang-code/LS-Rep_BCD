# DART-R 失败审阅与 DART-R-TS 实现交付

**结论：现有证据支持停止把“独立辅助头在 p2 的梯度余弦”作为教师准入的唯一依据，但不能证明梯度余弦一般无效，也不能排除 Cache 语义与重放问题。已实现 C1：DART-R-TS（Task-Space DART-R，任务空间困难感知教师路由与拒绝）。这是可检验的新训练机制，尚无真实数据集精度结果。**

交付基准：2026-09-12 读取的 [GitHub 提交 98c58266a99e73e340ae844a81aca366613fda4b](https://github.com/YuqiWang-code/LS-Rep_BCD/tree/98c58266a99e73e340ae844a81aca366613fda4b)。附件 `models_and_metrics_DART-R_Run1.txt` 抽取出全部 65 个 Python 文件，与仓库逐文件 AST 比较全部一致；差异仅空文件/末尾空行，仓库另含 SAM 配置与 CUDA 源文件等非 Python 文件，已随完整 models 保留。`SOURCE_AUDIT.json` 给出原始文件 SHA256。

代码在单独交付副本中修改，未向 GitHub 提交。模型源码、附件两篇基础论文、16 篇文献合集、13 个外部代码片段及环境文档用于此次审阅；未展开另一轮文献搜索。原第三方代码保留，不在新训练路径导入 SAM/DINO 教师推理工程。

## 1. 证据边界与结果核对

附件第 12201 行声明按每个 train_log 最后一个完整正式 test block 抽取；12213–12220 行是本次 B0/C0 八行汇总。以下是**该汇总的转录与差值计算**，不是本次从服务器原始日志重新认证的结果。GitHub 没有对应的完整 train_log；末 epoch 的 d/q/reject 数值由用户说明与 AGENTS.md 提供，不能当作已核对的逐步日志。

全部使用 batch64、40k steps、seed2333、lr=5e-4、Adam、poly、wd=1e-4、backbone lr multiplier=1、batch Dice。B0/C0 训练参数分别 2,913,094 / 2,914,459；部署均 2,913,094，汇总 FLOPs=2.7676G，部署与辅助开关误差均记录为 0。

| 数据集 | ID | Recall | Precision | OA | F1 | IoU | Kappa |
|---|---|---:|---:|---:|---:|---:|---:|
| SYSU | B0 | 79.7853 | 84.9447 | 91.8980 | 82.2842 | 69.9007 | 77.0389 |
| SYSU | C0 | 77.4590 | 87.1364 | 91.9875 | 82.0132 | 69.5105 | 76.8818 |
| SYSU | ΔC0−B0 | -2.3263 | +2.1917 | +0.0895 | -0.2710 | -0.3902 | -0.1571 |
| WHU | B0 | 92.7962 | 95.6359 | 99.5462 | 94.1946 | 89.0264 | 93.9586 |
| WHU | C0 | 92.3374 | 96.2882 | 99.5548 | 94.2714 | 89.1636 | 94.0399 |
| WHU | ΔC0−B0 | -0.4588 | +0.6523 | +0.0086 | +0.0768 | +0.1372 | +0.0813 |
| CDD | B0 | 97.7124 | 97.8732 | 99.4555 | 97.7927 | 95.6808 | 97.4822 |
| CDD | C0 | 97.6815 | 97.8797 | 99.4526 | 97.7805 | 95.6574 | 97.4683 |
| CDD | ΔC0−B0 | -0.0309 | +0.0065 | -0.0029 | -0.0122 | -0.0234 | -0.0139 |
| LEVIR | B0 | 90.6627 | 91.9230 | 99.1185 | 91.2885 | 83.9732 | 90.8243 |
| LEVIR | C0 | 89.4077 | 92.3684 | 99.0841 | 90.8639 | 83.2575 | 90.3820 |
| LEVIR | ΔC0−B0 | -1.2550 | +0.4454 | -0.0344 | -0.4246 | -0.7157 | -0.4423 |


平均 ΔF1 = **−0.15775 个百分点**。四数据集均出现 Recall 下降、Precision 上升，SYSU 和 LEVIR 尤其明显。这支持“存在更保守预测的可能”，但没有样本预测/阈值曲线，尚不能认定具体因果。

单 seed 没有方差估计，不能说“全部在噪声内”；约 ±0.1 的历史两次运行差值不是置信区间，尤其不能直接覆盖 −0.2710 或 −0.4246。“当前没有稳定正收益证据”比“已证明机制普遍无效”更准确。

## 2. 分级诊断

| 级别 | 发现及直接依据 | 对失败的解释与证据强度 |
|---|---|---|
| P0 | `cache_transforms._replay_tensor` 用 grid_sample nearest；GT 用 OpenCV INTER_NEAREST | 两者裁剪后最近邻坐标规则不同，已用唯一像素 ID 栅格复现。属于可证实的同步实现问题；实际数据受影响区域还未测量。 |
| P0 | 旧重放先 value.float，再在大 ID 时转 double | 大于 2^24 的实例 ID 精度已经丢失。通常小 ID 不受此项影响；新实现直接整数索引，不经浮点。 |
| P1 | `TeacherHeads` 的四个 1×1 头独立初始化；GT 来自主 decoder.cls | d 同时受任务语义、头方向、损失归一化和空间抵消影响；它并非独立于辅助头训练状态的教师质量。 |
| P1 | Router 标签只由 q·d 与 margin 决定；router CE 仅更新 Router | 高 accuracy 说明复制标签成功，不能验证标签代表真实教师收益。 |
| P1 | `reduce_teacher_losses` 在 Reject 样本把全部教师 loss 置零；无独立头预训练目标 | 被拒绝同时失去辅助头的数据梯度，造成冷启动反馈；可能仍有优化器动量/weight decay，不能把“数据梯度为零”说成参数绝对不动。 |
| P1 | SAM target=max(boundary_t1,boundary_t2)，同实例邻接逐时相计算 | 监督所有物体结构，而学生 p2 表达变化。稳定建筑、道路等边界未必是变化边界；“不是变化检测标签”是确定事实，是否导致指标退化待消融。 |
| P1 | OV relation 的8通道做空间 Gram | 通道置换不变只证明数学性质，不证明这些缓存通道在历史生成器下适合该关系监督。新主方案暂不使用该项。 |
| P1 | GT 代理是单输出逐样本 BCE+Dice，实际默认是四尺度 batch Dice | d 不是真实 Adam 参数更新收益；特别是 Dice reduction 不一致。p2 处本就看不到更高尺度损失的独立路径。 |
| P2 | 每批三次特征 autograd.grad、CPU 连通域及大量标量同步 | 额外训练成本没有得到已证明的精度回报。新方案去掉用于路由的多次梯度探针，仍保留困难诊断成本。 |

**对原判断的修正：**

1. 在同一 p2 上定义的两个梯度，其内积仍有局部特征下降的数学意义；“两个 loss 不同”本身不等于余弦无效。独立头高维方向、任务冲突与冷启动使它不适合作为当前实现的唯一收益代理，这是更有依据的表述。
2. 末 epoch 的均值 d≈0 不代表每个样本 d≈0；负值和正值可以抵消。需要分位数、正比例、按困难区域分组的分布。
3. 原代码没有把 r 直接乘到 KD loss；r 进入 Router 输入，q·d 产生标签，最终损失权重来自 Router softmax 与硬 Reject。因此 r 小不能直接推出学生 KD 梯度为零，84%–90% Reject 也仍保留其余样本的 KD。
4. q 是缓存 quality/confidence 数值，不能仅凭范围“正常”就认定它已校准为二值变化正确率。
5. 历史与重建 Cache 上签名一致，提高“公共训练机制有问题”的可信度，但公共重放错误、同样的任务语义偏差仍可能共同影响四数据集，不能据此排除 Cache 相关问题。

## 3. 候选比较与本次选择

| 候选 | 改变什么 | 不作为本次主方案的原因/选择理由 |
|---|---|---|
| 独立头先学习、再用匹配代理路由 | 用 detached 特征校准辅助头，解除 Reject 对头的冷启动影响 | 能测试冷启动，但 SAM 目标仍不是变化语义，余弦仍不是实际优化收益；作为后续因果定位可用。 |
| 虚拟 Adam 参数更新，测 GT loss 前后差 | 更接近实际优化的效用评估 | 需复制优化器状态、处理 BN/RNG、额外前后向；训练成本与比较预算明显增加。单步训练集收益仍不保证泛化。 |
| **统一任务空间提案 + GT 相对收益准入（选择）** | 将 SAM 结构转译成变化概率；与 OV 在相同概率空间比较，逐像素拒绝 | 不需要学习一个复制已知训练标签的 MLP；同时处理监督语义、冷启动和整图拒绝。直接输出梯度可检验。 |

与已有材料的实质区别：AnyChange（NeurIPS 2024）用 SAM mask 与双时相潜特征匹配作变化判断，本方案没有所需潜特征，使用已缓存实例对学生变化概率做训练期传输，不能称 AnyChange 复现。UNIC（ECCV 2024）的 teacher dropping 基于各教师蒸馏损失并保证至少保留一位；本方案按二值变化任务收益检查每个像素，可以全部拒绝。U-Know-DiffPAN（CVPR 2025）用预测方差分配 PAN-sharpening 的硬/软监督，任务和缓存内容不同；这里没有将它的重建损失照搬成二值变化检测。GoodSAM（CVPR 2024）也涉及将 SAM 结构转为目标任务知识，因此“利用 SAM 边界”本身不是本方案新颖性主张。

本次代码的机制假设是：**教师信息经变化语义转译后，在困难位置只接受确有当前任务修正价值的提案，比独立头梯度门控更容易形成有用的学生梯度。** 这不是宣称学术首创；Brier 准入、实例平均和软目标损失各自都不是新发明，组合的新颖性及论文价值仍需文献定位与实证。

## 4. C1 的数学定义与数据流

设主输出为 p，训练标签 y∈{0,1}，sg 表示 detach。网络主图始终为：A/B → 同一 LWGANet 分别前向 → SWA → |时相差| TFM → Decoder → 四尺度预测。保持原 BN 调用顺序，不引入 joint-BN。

### 4.1 SAM 结构引导的自蒸馏提案

对时相 t 内同一非零实例 I，定义源权重 a_j=q_j(1−boundary_j)。对目标像素 i 进行 leave-one-out 传输：

\[
 s_i^t=\frac{\sum_{j\in I,j\ne i}a_j\,\mathrm{sg}(p_j)}
                  {\sum_{j\in I,j\ne i}a_j}.
\]

排除自己防止大部分信号退化为恒等自监督；边界像素只作为源时降权，仍可以被内部提案纠正。若实例未覆盖、singleton 或源质量总和不足，则回退为 sg(p_i) 且 confidence=0。按目标 quality 对两个时相的提案加权平均，q_SAM 为两时相有效目标 quality 的均值；始终独立处理两个时相实例编号，不比较 t1 与 t2 的 ID 数值。

这是 **SAM 结构引导的学生自蒸馏**，不是声称 SAM 能直接输出二值变化标签。若某一实例包含多种变化状态，传输可能错误；后续 GT 准入会拒绝当前像素上较差的提案，但并不能保证实例划分对泛化有益。

### 4.2 OV 变化提案

将 l1/l2 的 soft_change 与 confidence 按同步增强结果插值到 p 的尺寸。用 confidence 加权平均两个层级得到 s_OV，q_OV 为两层 confidence 均值。无 confidence 支持时回退 sg(p) 并拒绝。proposal 构造函数不接收 GT；不会用 GT 伪造教师输出。

未确认语义的 OV relation 在 C1 不进入 loss；loader 仍按原 schema 校验它，以保持缓存契约。后续如证明 relation 的确切含义，可以单独研究，不能默认为当前 Gram 正确。

### 4.3 相对收益、路由和拒绝

\[
 e_i=(\mathrm{sg}(p_i)-y_i)^2,\quad
 u_{ki}=e_i-(s_{ki}-y_i)^2,\quad
 v_{ki}=\operatorname{clip}(u_{ki}/\max(e_i,10^{-8}),-1,1).
\]

只有 q_ki>0 且 u_ki>8·eps_FP32 的教师准入。后一个阈值只是数值相等容差（约9.54e−7），不是把旧 utility_margin 调小；旧 q·d 已完全退出 C1。对准入教师令 b_ki=q_ki·max(v_ki,0)，α_ki=b_ki/Σ_k b_ki。没有准入教师的像素为 Reject；其所有教师权重精确为0。有效强度 c_i=Σ_k α_ki q_ki，最终 w_ki=c_i α_ki。

相对收益只决定教师比例，质量强度只衰减一次，避免原代理数值尺度作为多重衰减。不会强行维持固定非 Reject 比例；学生已超过教师时，拒绝增加是合理现象。

存在有效教师时允许二者按比例混合；`action` 记录最大 b 的教师索引，仅用于展示，**不会再以 action 对整幅图清零**。`weights` 记录 α 与硬 Reject 占比；`effective_weights` 记录真正进入 loss 的 c·α。两者不能混读。

### 4.4 学生目标

\[
 L_{KD}=\frac1B\sum_n\frac{\sum_i I_{ni}\sum_k w_{nki}
 D_{KL}(\mathrm{Ber}(s_{nki})\Vert\mathrm{Ber}(p_{ni}))}
 {\sum_i I_{ni}},\qquad
 L=L_{GT}^{\text{原四尺度 batch BCE+Dice}}+0.06L_{KD}.
\]

I 沿用原六维困难诊断产生的空间 importance；拒绝像素保留在分母，不对剩余很小区域重归一化为整图强度。KL 等于 soft BCE 减去目标熵，去掉不可消除的目标熵后 loss 更易解释；梯度仍是 soft-target BCE 梯度。没有修改主损失权重、学习率或新增 loss cap；保留原 kd_lambda=0.06。KL 理论非负，浮点舍入可能出现极微小负值；不据此推断模型异常。

所有提案、GT 收益、路由、importance 都 detach；只通过现有预测 p 向学生反传。没有教师 map 加入 backbone/SWA/TFM/decoder 的特征。

**可证明的有限性质：** 对二值 GT，u>0 意味着 s 比 p 更接近 y，soft BCE 在该像素对 logit 的导数 p−s 与 GT BCE 的 p−y 同号。因此在把输出位置视为独立变量时，接受的 KD 修正朝向 GT。**这不等于** 全网共享参数/Adam 更新必定降低 GT loss，更不等于 test F1 必定提高。

训练辅助新增参数 **0**；C1 与 B0 训练、部署均为 **2,913,094**。switch_to_deploy 删除整个辅助对象；类内没有持久缓存、教师权重、头或优化器。中间诊断张量仅属于当前 forward 返回值，不注册到模型。部署 FLOPs 不变，实测 2.7676336G。训练计算/显存仍增加，未测 RSML-3 batch64 吞吐，不能用部署 FLOPs 推断训练成本。

## 5. 数据重放修复与对照隔离

新 `aligned` 依次重放 Scale → 相同像素 margin 的 crop/resize → vertical/horizontal flip → temporal exchange。SAM ID 使用整型索引的 floor 最近邻，与 GT OpenCV INTER_NEAREST 一致，支持大 int64 ID。连续字段使用 bilinear；OV 先在学生栅格重建，做相同几何变换，再返回原生 l1/l2 尺寸。

低分辨率 OV 的上采样重建不等于恢复教师未保存的高频信息；这里只保证已缓存连续场使用一致几何，没有声称它与“重新跑教师后再增强”完全等价。OV 的 soft_change/confidence 按缓存的无序时相对描述使用；真实历史生成器的交换不变性仍需外部证据。C1 不使用 relation，减少了这部分未知通道变换风险。

唯一像素 ID 的合成测试（256×256，top=7，left=8）显示旧普通 ID 坐标对应不一致约70.12%；大 ID 另叠加浮点精度问题。**这不是实际 SAM Cache 有70%语义错误**：真实大块实例内部相邻像素往往 ID 相同，实际影响主要由区域边缘和实例形状决定，必须在真实 Cache 上检查。新版与 OpenCV ID 重放逐像素完全一致。

C0 保留 legacy 重放用于复现；C0F 仅把 C0 改成 aligned；C1–C5 统一 aligned。图像/GT 的原始增强代码不变，所以 B0 主训练流不受 Cache 修复影响。**C0F 是旧路径诊断对照，不是满足新重放要求的推荐旧默认。**

## 6. 最小实验与失败判据

所有正式运行固定四数据集、batch64、40k、相同预训练/优化器/增强协议；主要比较 seeds=2333/3407/5871。预算不同的筛选不能混入正式结果。若资源有限，先在 SYSU、LEVIR 完成 seed2333 的 B0/C0F/C1/C2/C3，再决定是否扩展，不把两数据集筛选当普适结果。

| ID | 唯一变化/作用 | 检验问题 |
|---|---|---|
| B0 | clean，原主图和 GT 协议 | 总效果 anchor |
| C0 | 保留旧完整机制及 legacy 重放 | 旧代码复现，仅作历史诊断 |
| C0F | 相对 C0 只修复 Cache 重放 | 有多少变化来自几何正确性 |
| **C1** | task-space 双提案、收益准入、困难加权 | 唯一新主方案 |
| C2 | 相对 C1 仅改 quality 路由，不检查 GT 收益 | 收益检查是否优于一般置信度加权 |
| C3 | 相对 C1 仅禁用 SAM 提案 | SAM 结构传输是否提供增益 |
| C4 | 相对 C1 仅禁用 OV 提案 | 单独的结构自蒸馏能否成立 |
| C5 | 相对 C1 仅 importance=1，保留诊断记录 | 困难加权是否确有贡献 |

C0F→C1 同时改变了监督语义和路由，**不能用这一个差值单独归因到 Brier 信号**；C1/C2 用来隔离任务空间收益准入，C1/C3/C4 用来检查教师贡献。C2 是可能伤害学生的负对照，不应改名为推荐方法。

预注册建议（这些阈值是本次实验决策规则，不是统计事实）：

- 机制成立最低要求：真实 cache dry run 梯度有限；训练早期有有效覆盖时 `effective_mass` 与 KD 梯度非零；若所有像素均无正收益，应先分析提案质量而不是降低准入条件。
- 值得扩展的单 seed 筛选：SYSU/LEVIR 平均 ΔF1≥+0.20 pp，且任一不低于−0.15 pp；只决定是否投入更多实验，不构成论文结论。
- 正式支持条件：四数据集三 seed 的平均配对 ΔF1≥+0.20 pp，至少三数据集均值为正，任一数据集均值不低于−0.15 pp；报告每 seed、均值、标准差及配对差值。三 seed 仍是有限证据，不夸大显著性。
- 路由贡献若 C1 相对 C2 无一致收益，不能声称收益检查有效；SAM 贡献若 C1 不优于 C3，停止双教师叙事，保留 OV-only 事实。
- 即便 rejection 降低，若 Recall/IoU/F1 未改善或比 C2 更差，新方法仍判失败；降低 Reject 本身不是成功指标。

## 7. 日志、保存与恢复

保留 `train_log.txt`、`last_checkpoint.pth`、`best_model_F1={F1:.6f}.pth`、验证集 F1 严格变好才更新 best，以及完整 `=== TEST RESULTS ===` 至 `=== END TEST RESULTS ===` 格式。六指标与训练/部署参数、FLOPs、开关和部署误差继续记录。

C1 日志新增：pixel/image reject、变化/背景接受率、每教师 available/eligible 比例、proposal Brier、原始/相对 proposal gain、effective mass、SAM/OV 实际 loss。旧 d/r/router accuracy 字段在 C1 epoch 记录中省略，防止把“不使用”误读成“为0”。配置中的 router_lr/hidden/utility_margin 为旧入口兼容参数，在 C1 不参与算法；`routing_signal=relative_brier_gain` 明确区分。

`probe_cls_kd_gt_ratio/cosine` 是每 epoch 首批、最终分类器权重处的实际 λ·KD 梯度与实际 GT 梯度诊断，**仅记录，不控制路由**。它不是全网梯度或实测泛化效用；旧 C0 的辅助头损失不经过主 cls，其该项可为0，而 backbone 仍可收到教师梯度，不能误判。

checkpoint 保留 format_version=2；`implementation_version=dart_r_ts_v2`，mechanism、cache_replay 和原超参/数据指纹参加恢复校验。C1 没有头/Router 参数，`router_optimizer=None` 正确表示不存在；主 Adam 和 RNG 正常保存。C0/C0F 的辅助头、Router、两个优化器仍完整保存。

**只允许同版本、同实验、同预算和同配置从 last_checkpoint 精确恢复。** 旧 v1、C0→C1 或改变 max_steps 均拒绝。旧 checkpoint 仍可通过 export_deploy 导出学生，但不能把它接续为公平的新 C1 实验。恢复粒度是保存的 epoch 边界；中途崩溃会重放未保存的 epoch，未实现任意 batch 中断瞬间恢复。终止预算在最后一个 epoch 中间时，恢复完成运行仅重新加载 best 进行 test，不再接续其它预算。

## 8. 逐文件修改与交付

| 文件 | 修改 |
|---|---|
| models/distill/task_space.py | 新建任务提案、Brier 收益、解析式路由和 loss |
| models/a2net.py | 训练辅助类选择；主计算路径不变 |
| models/datasets/cache_transforms.py | aligned 重放，保留 legacy 显式分支 |
| models/datasets/cd_dataset.py | 传递 cache_replay，不改变图像/GT 增强 |
| models/scripts/train.py | C0F/C1–C5 配方、无 Router 优化器支持、新诊断和恢复版本 |
| models/tools/smoke_task_space.py | 梯度、对称性、拒绝、增强、恢复及部署测试 |
| models/tools/dry_run_task_space.py | 实际数据/Cache 单批只读检查与梯度诊断 |
| models/tools/integration_task_space.py | 临时磁盘合成数据，完整 trainer/日志/checkpoint/export 集成测试 |
| models/scripts/run_task_space.sh | 新目录启动与同配置恢复；已 bash -n |
| models/scripts/TASK_SPACE_SHELLS.txt | 新 shell 的自动汇总 |
| models/README_DIRECTION_C.md | 新入口说明；旧文档改名 V1 归档 |

旧核心 distill/routing.py、losses.py、diagnostics.py 保留用于 C0/C0F；并未把新机制伪装成旧 C0。backbone、decoder、主 loss、指标和 checkpoint helper 未修改。包内的最小 patch 相对上述 GitHub HEAD 生成。

## 9. 验证结果与限制

本地独立 CPU 测试环境为 PyTorch 2.14.0+cpu；不涉及服务器 lsrep/cd_base，也未替换其 PyTorch/CUDA。

- 新机制与旧 C0 smoke 均通过：真实 A2Net 前后向、两教师梯度、全 Reject 与 clean baseline 学生梯度及 BN 一致、detach、空/满 GT、singleton/未覆盖教师、ID 重命名与时相交换。
- 新 C1 辅助开关误差=0、部署误差=0、模型交换误差=0；训练/部署参数2,913,094，256×256 FLOPs2.7676336G。
- checkpoint/Adam/RNG 恢复后相同下一步权重最大误差=0（本次CPU）；新机制拒绝旧 implementation_version。
- 使用256×256临时磁盘数据与两套模拟 Cache，B0/C1 各完成2步训练、验证选best、正式结果块格式、完成运行恢复和部署导出；数据和Cache前后SHA256未变化。这些合成数值不是遥感正式指标。
- aligned SAM ID 与 OpenCV 连续/离散变换核对通过，OV 多分辨率连续场的重放核对通过。
- 具体机器可读记录见 `VALIDATION.json`，没有用旧仓库 validation_results 冒充新测试。

**尚未执行：RSML-3 的真实数据/真实 Cache dry run、现有预训练加载、RTX5090 CUDA 数值/显存/batch64 吞吐、真实多worker和长程恢复、四数据集40k训练。** 新增工具可以在原环境执行这些检查；合成 dry run 不能替代真实检查。没有承诺或伪造精度提升。

## 10. 立即执行顺序

1. 解压到新目录；将现有 models 做单独备份，再把本包完整 models 放到项目根目录。也可以在原 HEAD 上先 `git apply --check models_DART_R_TS.patch`，再应用 patch。保留原数据、Cache、checkpoint 与预训练文件。
2. 激活原 `lsrep`，在项目根目录执行 `python -m models.tools.smoke_task_space --device cuda:0`。
3. 按 `models/README_DIRECTION_C.md` 的命令，对四个数据集依次执行 `validate_teacher_cache`（已有可靠全量校验可复用）和 `dry_run_task_space`。若 Cache 不匹配，修复读取配置或补充证据，不重建缓存。
4. 使用 `bash models/scripts/run_task_space.sh <ID> <dataset> <physical_gpu> <seed>`。先 B0、C0F、C1，再 C2/C3，最后视证据追加 C4/C5 与其它 seeds。可将 SYSU/CDD 放 GPU0 队列，LEVIR/WHU 放 GPU1 队列；各卡内部串行，不叠加同卡训练。
5. 新日志根为 `/home/yqwang/outputs/LS-Rep_BCD_RSML_3/DART-R-TS`，新 checkpoint 根为 `/home/yqwang/checkpoints/LS-Rep_BCD_RSML_3/DART-R-TS`。二者下均为 `<ID>/<dataset>/steps_40000/seed_2333` 等对应目录。
6. 精确恢复例：`bash models/scripts/run_task_space.sh C1 SYSU 0 2333 resume`，即同目录 `last_checkpoint.pth`。最终成绩只读取该运行 train_log 最后一个完整正式 test block。现有仓库汇总脚本未随仓库提供，可能不扫描新实验根；不要默认会自动收集 C1–C5，需将新日志明确纳入汇总。

仍需补充证据：八份原始 B0/C0 train_log 与缓存指纹；旧 d/q/有效权重逐样本分布；真实 Cache 的 soft_change 与 GT 在困难区域的误差/覆盖；历史 OV 生成语义及交换不变性；C1 与必要对照的同协议多 seed 正式结果。
