结论先给

我建议下一步只做一个方向：固定、独立、任务适配的 Foundation Teacher + Failure-Aware Symmetric Change-Relation Distillation（暂称 FA-SCRD）。

核心不是再造一个“会跟着学生变化的动态教师”，而是：

先把一个强 Foundation Model 适配成真正会做 binary CD 的固定教师，再把它的双时相“变化关系知识”离线缓存；训练 A2Net 时，只在学生真正困难、且教师确实可靠的局部区域做软选择式 relation KD。教师永远不被学生反向更新，不用像素级 hard reject，不把 teacher map 注入主路径。

我不建议现在直接上 SAM2+DINO+CLIP 多教师 Agent。原因不是这个方向不好，而是你现在连“一个独立、真正强、不会塌到学生上的教师能否稳定帮助 2.9M 学生”都还没有被公平验证。直接上三教师 routing，会把“teacher 是否有效”“routing 是否有效”“哪种知识有效”三个问题重新搅在一起。

这条路线同时修掉你前两次实验最致命的两个问题：

教师惰性彻底消失：历史 RDT 的 teacher 是围绕 p_student 做 residual correction，且末层零初始化，初始天然满足 q_teacher=p_student；随后又通过 EMA 跟随，导致 dynamic_teacher_gain≈0。新方案 teacher 完全固定、离线，学生没有能力把教师“拉平”。
不再 99% 拒绝 KD：旧实现要求逐像素 Brier gain 严格为正才放行；当 teacher≈student 时几乎全被 gate 掉。新方案改成区域级、连续软权重，没有 0/1 hard audit。
一、我对仓库代码和历史结果的结论

我实际通过 GitHub connector 检查了当前仓库和历史 commit。当前部署主图很干净：

shared LWGANet-L0 → SWA / NeighborFeatureAggregation → TemporalFusionModule → Decoder

当前 TFM 内部的真正二时相变化形成点，是 abs(x1-x2) 后再做多膨胀卷积；Decoder 有 p2/p3/p4/p5 四级 64C 特征及四级监督。因此非常适合在 SWA→TFM 或 TFM→Decoder 之间只挂训练损失，不改 forward 主路径。当前 A2Net 主模型 A2Net decoder/TFM

历史 RDT-CD Run3 的正式结果也和你的描述一致：SYSU 有正收益，而 WHU/CDD/LEVIR 没形成稳定收益。更关键的是历史源码明确写出了动态教师的形式：

$$ q_T=\sigma\left(\operatorname{logit}(p_S) +r_f\Delta_{\max}\tanh(\delta)\right), $$

且 residual teacher 最后一层是 zero initialization，也就是起点 q_T=p_S。再叠加逐像素 gain=(p-y)^2-(q-y)^2>0 的硬 audit，这个设计从机制上就非常容易进入“teacher≈student → gain≈0 → KD 被拒绝”的死区。历史 Run3 结果说明 历史动态教师实现 历史 Brier audit 实现

当前 DataLoader 的几何增强已经具备 replay 状态基础，且有 T1/T2 RandomExchange；因此重新做 feature cache 并不需要重写整个数据管线，关键只是保证 crop/resize/flip/exchange 同步作用到 teacher cache，尤其 exchange 时要把 teacher T1/T2 feature 一起交换。现有服务器上四数据集及 SAMStruct、OVCDistill cache 的覆盖和字段也已经明确。

所以我把问题归为：

**P0 正确性：**没有发现当前 deploy graph 本身需要先修的错误。
**P1 方法瓶颈：**旧教师不是独立知识源；pixel hard-gate 造成蒸馏信号饥饿；foundation prior 被过早压成“变化概率”，丢掉了 FM 真正有价值的中间表示。
**P2 工程：**新 cache 必须重新支持 synchronized replay，但当前 transform 结构已经给了基础。
二、A. 真正有效的 KD：我最建议看这 3 篇
文献	层级 / 官方代码	核心机制	对本项目的直接启示
CrossKD: Cross-Head Knowledge Distillation for Object Detection — Jiabao Wang, Yuming Chen, Zhaohui Zheng, Xiang Li, Ming-Ming Cheng, Qibin Hou, 2024	CVPR 2024，CCF-A。 官方代码 CrossKD	把 student head 的中间特征送入 teacher head，避免 student 自己的 head 同时被 GT 和 teacher 两套可能矛盾的监督拉扯；对异构 backbone 也有效。	**不要让教师监督直接与 GT 在同一预测头打架。**如果以后加 logit/head KD，应该采用 task-head/cross-head 思路，而不是直接对最终 p_student 加一个 teacher BCE/KL。
Scaled Decoupled Distillation — Shicai Wei, Chunbo Luo, Yang Luo, 2024	CVPR 2024，CCF-A。 官方代码 SDD	将全局 logit 拆成局部知识，进一步区分一致知识和互补/困难知识，提高 ambiguous sample 的权重。	binary CD 不应该整幅图平均 KD。局部 patch/region 是更合理的传输单位，尤其 WHU/LEVIR 前景只有约 3–4%。
VkD: Improving Knowledge Distillation using Orthogonal Projections — Roy Miles, Ismail Elezi, Jiankang Deng, 2024	CVPR 2024，CCF-A。 官方代码 VkD	用受约束的 feature projection 与 task-specific normalization 解决 teacher/student 架构、任务和表示空间差异。	对“2.9M CNN-like 学生 ← 大 ViT/Foundation Teacher”很关键：不要直接 MSE 两边 raw feature，更应该蒸馏关系/归一化结构。

因此 A 的结论非常明确：

优先级：relation/feature KD > 局部解耦 logit KD > 全局 vanilla logit KD。

ReviewKD、经典 DKD 本身虽然重要，但不在你规定的 2024–2026 检索窗口，因此我没有拿它们充当下载清单里的核心论文。

另外，ICCV 2025 的 A Good Teacher Adapts Their Knowledge for Distillation 直接研究了“大 teacher 与小 student 容量差距过大”的问题，并指出输出分布失配会降低 KD 效果；但我没有找到可核验的官方代码，所以不列入你要求的“有源码论文”清单。

三、B. Foundation Teacher Cache：真正应该缓存什么
1. PeftCD —— 我认为是你现在最该下载的教师候选

PeftCD: Leveraging Vision Foundation Models With Parameter-Efficient Fine-Tuning for Remote Sensing Change Detection
Sijun Dong, Yuxuan Hu, Libo Wang, Geng Chen, Xiaoliang Meng，2026，IEEE JSTARS，SCI 期刊，非 CCF-A。
官方代码 PeftCD

它用权重共享 Siamese VFM encoder，把 SAM2 和 DINOv3 通过 Adapter/LoRA 真正适配到 CD，而不是直接把原生 FM 当 CD teacher；公开结果覆盖你关心的 SYSU、WHU、CDD、LEVIR 等数据集。

这是对你当前 cache 最大的启示：

你以前存的是 SAM instance/boundary、DINOv2 派生的 soft-change/confidence/relation；下一版应该存：

“经过 binary-CD task adaptation 之后的 FM 双时相 feature”，而不是未经适配的 FM 伪变化概率。

这两个东西本质不同。

2. Segment Any Change / AnyChange

Segment Any Change — Zhuo Zheng, Yanfei Zhong, Liangpei Zhang, Stefano Ermon，2024，NeurIPS 2024，CCF-A。
官方代码 AnyChange

它最重要的不是 zero-shot 成绩，而是 bitemporal latent matching：利用 SAM latent 内部的 intra-/inter-image semantic similarities 建模变化，而不是简单将 T1/T2 的分割 mask 相减。

这恰好支持你下一版 cache 应保存：

$$ (F_T^{t1},F_T^{t2}) $$

然后在训练阶段形成对称 temporal relation，而不是提前把 FM 压成一个 soft_change。

3. BAN

A New Learning Paradigm for Foundation Model-Based Remote-Sensing Change Detection — Kaiyu Li, Xiangyong Cao, Deyu Meng，2024，IEEE TGRS，SCI 权威期刊。
官方代码 BAN

BAN 用 frozen foundation model + Bi-TAB，在多个 hierarchy 上选择、对齐并注入 foundation features。它本身因为 inference 仍依赖 FM，不能直接作为你的最终结构，但很好地证明“多层 FM feature 而非最终 mask”才是值得利用的知识。

我建议的新 cache 格式

第一版不要再把 SAM_boundary + OV soft_change 作为主 target。重新生成一套比如：

CDTaskFMFeat-v1/<dataset>/train/<sample>.pt

内部只存：

t1/mid_feature
t2/mid_feature
t1/deep_feature
t2/deep_feature
teacher_change_logit：仅用于评价教师可靠度，不作为主要 KD target
teacher checkpoint hash / backbone / normalization / input resolution / split hash

特征存 FP16。具体 DINOv3 哪两个 block 等你下载 PeftCD 后我们再按源码定，不应现在凭空指定 block 编号。

几何 replay 必须满足：

$$ T(A),T(B),T(Y),T(F_{T1}),T(F_{T2}) $$

使用同一 crop/resize/flip；发生 temporal exchange 时同步交换 teacher t1/t2。这是硬正确性条件。

四、C. 教师与学生互选：值得做，但不是第一枪

这里有三篇很重要。

JL1-CD / MTKD

JL1-CD: A New Benchmark and a Robust Multi-Teacher Knowledge Distillation Framework for Remote Sensing Change Detection
Ziyuan Liu, Ruifei Zhu, Long Gao, Yuanxiu Zhou, Jingyu Ma, Yuantao Gu，2026，IEEE Transactions on Instrumentation and Measurement，SCI，非 CCF-A。
官方代码 JL1-CD/MTKD

它是这一组里与本项目任务最直接的：按 Change Area Ratio 把数据划成 small/medium/large change，训练 specialist teachers，再做 MTKD。官方代码完整给出了三个 specialist teacher 的训练过程。

对你的意义不是照搬 CAR，而是：

不同样本确实可能需要不同 teacher specialization。

但 CAR 是一个比较粗的静态分区，你的论文若做这个方向，更好的创新应该是“按学生当前失败类型选择教师”。

MoVE-KD

MoVE-KD: Knowledge Distillation for VLMs with Mixture of Visual Encoders
Jiajun Cao, Yuan Zhang, Tao Huang, Ming Lu, Qizhe Zhang, Ruichuan An, Ningning Ma, Shanghang Zhang，2025，CVPR 2025，CCF-A。
官方代码 MoVE-KD

多种 visual foundation encoders 各有所长；它不是简单平均 teacher，而是对不同 teacher 和不同 visual token 动态加权，避免互相冲突。

这基本就是你未来 Teacher Agent 的最强上位参考：

$$ \text{sample/region} \rightarrow \text{选择哪一个专家知识} $$

而不是 SAM + DINO + CLIP 全部平均。

Mask-CDKD

Mask-CDKD: A Source-Free and Label-Free Cross-Domain Knowledge Distillation Framework from SAM for Satellite Onboard VHR Land-Cover Mapping
Daoyu Shu, Zhan Zhang, Xiao Huang, Ru Wang, Nan Jia, Xinzhe Fu, Bingnan Yang, Fang Wan, Jianzhong Lu, Jianya Gong，2026，ISPRS Journal of Photogrammetry and Remote Sensing，SCI 权威期刊。
官方代码 Mask-CDKD

它让 frozen SAM backbone 配合可训练 adapter 做目标域校准，并做 single-stage bidirectional KD。说明“teacher 并非绝对不能适应 student/target domain”。但它做的是 VHR land-cover mapping，不是 binary CD，而且你的历史已经出现 teacher 被 student 拉平，所以本项目第一阶段不应采用双向更新 teacher。

C 的结论

Teacher Agent 是后续最有论文潜力的扩展，但不是现在最小可信实验。

先证明：

固定、独立、task-aligned teacher → 2.9M student

确实有正迁移。

否则 Agent 只是在“选择几个没有被证明有用的 teacher”。

五、D. 蒸馏应该放哪里？我选“形成变化表示之后”，不选 raw backbone

这里我认为答案比较明确。

BAN 告诉我们多层 FM feature 有价值；CrossKD 说明越靠任务 head 的监督越 task-oriented；而 CD 本身的 BiFA 则说明二时相 interaction/alignment 与后续 difference 的位置会实质影响变化表示。BiFA 是 Haotian Zhang、Hao Chen、Chenyao Zhou、Keyan Chen、Chenyang Liu、Zhengxia Zou、Zhenwei Shi，TGRS 2024，并有完整官方实现。 BiFA 官方代码

所以对你这个 A2Net，我不推荐：

LWGANet stage1 raw feature ↔ DINO feature MSE

这种蒸馏太早，容易把 illumination/texture/land-cover semantic 等与变化无关的单时相信息硬塞进只有 2.9M 的学生。

我推荐 teacher 先形成：

$$ C_T^l = \left| \operatorname{Norm}(F_{T1}^l) - \operatorname{Norm}(F_{T2}^l) \right| $$

它天然满足

$$ C_T(T_1,T_2)=C_T(T_2,T_1). $$

学生直接利用当前 TFM 产生的 change feature：

$$ C_S^l=\mathrm{TFM}(F_{S1}^l,F_{S2}^l). $$

蒸馏的是二者的局部空间关系，而不是 raw channel values。

这样 teacher/student 即使一个是 ViT、一个是 LWGANet，通道数不同也没关系。

六、BFD 很关键，但我要特别提醒你源码状态

你附件里的：

Burden-Free Distillation From Foundation Model for Efficient Remote Sensing Change Detection
Shuang Wang, Chonghua Lv, Dou Quan, Ning Huyan, Xianwei Cao, Jingxi Sun, Licheng Jiao，TGRS 2025，

其实是与我们目标最直接的一篇：DFM 做 dual-temporal feature matching，PCD 做 patch contrastive distillation，并且 inference 只保留轻量 CD model。

但是我刚刚专门检查了论文给出的官方 GitHub：

BFD 官方仓库

截至 2026-09-21，该仓库实际上只有一个 26-byte 的 README，没有公开实现文件。

所以：

论文：强烈建议读，借鉴价值极高。
代码：目前不能算“有可下载源码”。

我没有为了凑你的要求把它伪装成“代码已公开”。

七、按“对本项目下一步的借鉴价值”排序

我的排序不是按论文档次，而是按对你当前失败原因的针对性：

BFD — TGRS 2025：最直接证明 FM→CD 的 feature/patch KD + 无推理负担；遗憾是官方代码目前空。
PeftCD — JSTARS 2026：决定“什么才应该作为教师”——先把 SAM2/DINOv3 适配成真正 CD teacher。
AnyChange — NeurIPS 2024：决定“cache 应该存什么”——双时相 latent relation，而非两个 mask 简单相减。
CrossKD — CVPR 2024：决定“如何避免 GT 与 teacher 打架”。
JL1-CD MTKD — TIM 2026：直接证明 CD 中 specialist multi-teacher 有价值。
MoVE-KD — CVPR 2025：决定未来多 foundation teacher 应该“选择”，不是平均。
BAN — TGRS 2024：决定 FM 知识应该是多层 feature，而不是最终 mask。
SDD / VkD — CVPR 2024：分别支持局部困难知识和异构 feature space 的稳健对齐。
Mask-CDKD — ISPRS JPRS 2026：证明 teacher domain adaptation 有潜力，但暂时不要照搬其 teacher↔student 双向更新。
BiFA — TGRS 2024：不是 KD 论文，但对“二时相 interaction/difference 应放哪”很有价值。

它们共同指向一个趋势：

现代 KD 已经不是“大教师输出一个 soft label，小学生照着学”。真正有效的方向正在转向：task-aligned teacher、intermediate/relational knowledge、local/region selection、heterogeneous expert specialization，以及避免不可靠 teacher knowledge 的无差别转移。

八、唯一主方向：FA-SCRD
一句话核心思想

用一个固定的、经过 CD 任务适配的 Foundation Teacher 生成双时相 feature cache，将 teacher knowledge 转换为 T1/T2 交换不变的“变化关系”，再由 student difficulty × teacher reliability 对局部区域软加权，只蒸馏可靠且学生真正缺失的 change relation。

1. Teacher：第一版只要一个，不上多骨干

我首选：

PeftCD-DINOv3 + Adapter/LoRA，按你自己的 train split 重新训练。

不是因为我要照搬 PeftCD，而是把它当作任务适配后的离线 teacher generator。

暂时不要同时用 SAM2+DINOv3。

原因：

两教师 routing 会增加一个尚未必要的变量；
旧实验首先缺的是“独立且真正强的 teacher”；
单教师先通过，后续才有资格研究 Agent；
32GB 单卡下 teacher 可以单独 PEFT/生成 cache，student 40k 训练时完全不用加载大 teacher。

PeftCD 公布的 SAM2 与 DINOv3 在不同 CD 数据集上表现并非完全同序，这反而说明未来做 heterogeneous routing 是有理由的；但这些公开结果不能直接替代你自己的 split/protocol 验证。

2. KD 位置：SWA → TFM → Decoder 中的 TFM change feature

我第一版只监督两级中高分辨率 change representation，例如 p2/p3 对应尺度；具体实际 tensor shape 等下载 PeftCD 后再核准。

主路径仍然：

$$ A,B \rightarrow LWGANet \rightarrow SWA \rightarrow TFM \rightarrow Decoder \rightarrow P_S $$

Teacher cache：

$$ A,B \rightarrow T_{\mathrm{PeftCD}} \rightarrow F_{T1}^{l},F_{T2}^{l} \quad \text{(offline)} $$

训练时不把 teacher feature喂给 TFM，只计算 loss。

3. KD 目标：Local Symmetric Change Relation

对 teacher：

$$ C_T^l = \left| \hat F_{T1}^l-\hat F_{T2}^l \right|. $$

对 student 使用 TFM 已形成的：

$$ C_S^l. $$

在每个局部 patch \(r\) 内，不要求 channel 对齐，而计算空间 affinity：

$$ A_T^{l,r} = \mathrm{softmax} \left( \frac{C_T^{l,r}(C_T^{l,r})^\top}{\tau} \right), $$ $$ A_S^{l,r} = \mathrm{softmax} \left( \frac{C_S^{l,r}(C_S^{l,r})^\top}{\tau} \right). $$

然后：

$$ L_{\mathrm{rel}} = \sum_{l,r} w_r D_{\mathrm{KL}} (A_T^{l,r}\Vert A_S^{l,r}). $$

这比 raw feature MSE 更适合“大 ViT teacher → 64C tiny student”，因为只要求学生学会“哪些位置的变化关系类似”，不要求复制 teacher 的高维特征本身。

4. Student 与 teacher 怎么“互相选择”

这里我只保留一个很轻的 C 思路，不更新 teacher。

区域 \(r\) 的 student difficulty：

$$ d_r= \operatorname{mean}_{i\in r}|p_S(i)-y(i)|. $$

固定 teacher 的区域任务误差：

$$ e_T^r=\operatorname{BCE}(p_T^r,y^r). $$

学生误差 \(e_S^r\) 类似。

不用历史的：

$$ \mathbf 1[e_S-e_T>0], $$

而采用连续 soft advantage：

$$ a_r= \sigma \left( \frac{e_S^r-e_T^r}{\tau_a} \right). $$

最终：

$$ w_r= \operatorname{stopgrad}(d_r a_r). $$

于是：

学生已经会的区域 → 小权重；
学生不会、teacher 会 → 最大权重；
teacher 明显错误 → 权重自然下降；
teacher≈student → 权重约 0.5，不会像旧 Brier hard gate 一样直接变成零。

这正面针对 pixel_reject_ratio≈0.999。

九、为什么它比此前两次更有机会
旧 BT-SAM-RDT	FA-SCRD
Teacher 以 student prediction 为中心生成	Teacher 在 student 训练前已完全固定
residual zero-init，起点 teacher=student	teacher 与 student 独立
EMA teacher 会持续跟随	无 EMA
raw SAM/OV 被压成 change probability	task-adapted FM intermediate features
pixel Brier hard accept/reject	patch/region continuous weighting
logit/pseudo-label 为核心	symmetric change relation 为核心
99%+ KD 信号可能消失	relation loss持续存在
teacher/student 容量差直接做输出拟合	蒸馏 architecture-independent spatial relations
dataset class ratio容易影响像素 gate	patch relation 对 3.4%～21.1% foreground 比例更不敏感

但注意最后一条目前是可证伪假设，不是已经成立的结论。

十、部署约束完全满足

训练图：

$$ Student + TeacherCache + L_{\mathrm{rel}} + w_r. $$

部署图：

$$ LWGANet-L0 \rightarrow SWA \rightarrow TFM \rightarrow Decoder. $$

所有：

teacher
cache reader
difficulty router
teacher reliability
relation loss
任何 auxiliary buffer

全部删除。

因此目标仍然必须严格保持：

Deploy Params = 2,913,094；FLOPs ≈2.75–2.77G；auxiliary ON/OFF 主预测一致；switch_to_deploy() 前后误差 <1e-6。

十一、与现有工作的实质区别

这点决定它有没有论文空间。

相对 BFD：BFD 核心是 FM 与 CD model 的 feature matching + patch contrastive；你的方案不是逐时相匹配 foundation feature，而是先形成显式 T1/T2 交换不变的 change relation，再在 student 的 CD change space 蒸馏，而且由 student failure × teacher task reliability 做区域选择。BFD 本身并没有你的旧实验暴露出来的“teacher knowledge 何时应该拒绝”的问题建模。

相对 JL1-CD：JL1-CD 根据 CAR 预先训练 small/medium/large specialist teacher；FA-SCRD 的选择变量不是全图 change ratio，而是学生当前在哪个局部区域失败，以及固定 teacher 在该区域是否可靠。

**相对 AnyChange：**AnyChange直接利用 SAM latent matching 做 zero-shot CD；你只借鉴 latent relation 的思想，Foundation Model 在你的系统里永远只是 training teacher。

**相对旧 BT-SAM-RDT：**最根本的区别是 teacher 不再由 student 构造，也没有 EMA teacher，更没有 hard pixel Brier acceptance。

十二、必要实验，不扩张

正式跑之前我仍建议按照项目既定证据纪律，把 joint-BN 混杂先隔离掉。第一轮只跑 seed 2333，不做多 seed：

ID	唯一变量	目的
C0	clean A2Net	clean anchor
C1	C0 + joint temporal BN	排除历史 normalization/BN 混杂
A1	C1 + fixed teacher、但不做 failure weighting，均匀 relation KD	正面回答“独立 teacher 本身是否有效”
M1	A1 + failure-aware soft region weighting	主方法 FA-SCRD，唯一新增 router 权重

四组全部保持：

seed=2333 / batch=64 / 40k steps / 同增强 / 同 optimizer / 同 split。

先 SYSU + WHU 做机制闸门，因为二者变化像素比例分别约 21.1% 和 3.4%，正好覆盖一高一低；通过以后才补 CDD + LEVIR。数据与现有 cache 情况见服务器说明。

我会预先把失败判据写死，避免结果出来以后解释：

A1 ≤ C1 on SYSU 和 WHU：“固定 task-aligned teacher”前提失败，直接停止 Teacher Agent，不再上多 teacher。
**A1 只提升一个数据集：**仍视为单数据集 teacher trick，不进入主论文。
**M1 ≤ A1：**failure-aware routing 无贡献，删掉 router，不包装创新。
最终至少需要不同类别比例的数据集上有一致正趋势；单 seed 只能证明该协议下的消融有效，不能称统计普适。
十三、多 backbone / heterogeneous teacher 的最终判断

理论上值得做，当前不做。

JL1-CD 已经说明 CD specialist multi-teacher 可以形成互补，MoVE-KD 则进一步表明异构 foundation encoders 应该动态选择而非简单融合。

如果 FA-SCRD 单 teacher 通过，第二阶段最自然的是：

DINOv3-CD teacher：语义/变化判别专家
SAM2-CD teacher：instance/boundary 结构专家

然后把 \(a_r\) 从一个 teacher reliability 扩成：

$$ a_r\in \{ T_{\rm DINO}, T_{\rm SAM}, Fusion, Reject \}. $$

这时才真正进入你的“教师智能体”论文故事。

但现在上两到三个 backbone 反而会降低实验解释力。32GB 也不是核心限制，因为我建议 teacher 全部离线生成 cache；真正限制是你还没有证明单教师 transfer channel 是有效的。

你现在下载文献/源码的顺序

第一批只下载 PeftCD → AnyChange → CrossKD → JL1-CD → MoVE-KD。这五个已经足够让你理解我推荐方案的 teacher、cache、KD、routing 四个组成部分。然后下载 BAN、SDD、VkD、Mask-CDKD、BiFA 做补充；BFD 论文必须读，但目前官方仓库没有可用实现。

立即执行顺序
下载并重点看 PeftCD：确认 SAM2/DINOv3 feature extraction 与 decoder 接口。
看 AnyChange：重点读 bitemporal latent matching。
看 CrossKD：重点理解为什么不直接让 student head 同时吃 teacher+GT。
看 JL1-CD + MoVE-KD：为后续 multi-teacher agent 储备，不急着实现。
回来后我先根据这些源码和你当前 A2Net 画出 FA-SCRD 精确训练图、数学定义、cache schema 和逐文件修改表，然后再动代码。