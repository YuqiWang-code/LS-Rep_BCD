# RDT-CD Run3 — BT-SAM-RDT 双时相结构教师有效性验证

Run3 将方向 C 收敛为 **BT-SAM-RDT（Bi-Temporal Structural SAM Teacher for RDT-CD）**。学生模型仍为 A2Net-LWGANet-L0，部署主图保持不变：部署参数 **2,913,094**，256×256 FLOPs 约 **2.75G**；SAM/OV Cache、Fast Teacher、EMA Target Teacher 与全部诊断仅在训练期存在，`switch_to_deploy()` 后必须全部移除。

Run3 的核心目标不是继续增加路由复杂度，而是先回答一个更基础的问题：

> **经过双时相实例对应与结构变化重构后，Foundation Teacher 先验是否能稳定帮助固定的极轻量变化检测学生？**

---

## 与 Run2 的差异

Run2 使用 SCGR 解决稀疏变化条件下的 KD 信号饥饿与梯度冲突问题，但其核心 Teacher 输入仍没有正面解决 **“SAM 单时相实例如何形成可靠的双时相变化知识”** 这一前置问题。

Run3 因此取消旧的区域路由、梯度一致门、多教师路由等机制，改为一条更直接、可证伪的训练链路：

1. **BT-SAM 双时相实例对应**
   - T1/T2 SAM instance ID 只视为不透明标签，不直接做 ID 相减或 XOR。
   - 对跨时相实例计算 intersection / IoU / directional coverage。
   - 以最大 IoU 选择对应实例。
   - 匹配重叠区域视为结构持续区，结构变化 prior 约为 0。
   - T1-only 区域对应 shrink/disappearance，T2-only 区域对应 expansion/appearance，结构变化 prior 约为 1。
   - directional coverage、SAM quality 与 boundary 主要用于 **可靠性估计**，不再把 coverage 缺失直接当成重叠区域的变化标签。

2. **SAM Structural + OV Semantic Foundation Prior**
   - SAM 提供双时相结构变化 prior `S(x)` 与可靠性 `R_S(x)`。
   - OVCDistill 提供语义变化 prior `O(x)` 与可靠性 `R_O(x)`。
   - 两者在 task space 中按可靠性融合：
     \[
     q_{prior}(x)=\frac{R_S(x)S(x)+R_O(x)O(x)}
     {R_S(x)+R_O(x)+\epsilon}
     \]
   - 当 SAM 与 OV 同时可用时，用二者 disagreement 抑制融合可靠性。
   - R3A 与 R3 使用完全相同的 Teacher 容量；R3A 仅关闭 OV 信息。

3. **One Fast Residual Teacher + EMA Target Teacher**
   - 不再使用多教师路由。
   - Fast Teacher 输入为 Student feature / Student probability / Foundation prior 等 task-space 信息。
   - Fast Teacher 在 Student logit space 中学习一个有界 residual correction。
   - EMA Target Teacher 用于 Student KD。
   - Student optimizer 与 Fast Teacher optimizer 严格隔离；EMA Target Teacher 不参与反向传播。

4. **GT Positive-Brier-Gain Audit**
   - GT 只作为训练期安全审计，不用于构造 Foundation prior。
   - 对每个像素比较 Student proposal 与 Target Teacher proposal 的 Brier error。
   - 仅在 Teacher proposal 相对 Student 有正 Brier gain 且 Foundation reliability 有效时准入 KD。
   - 当前固定：
     - `routing_signal=pixel_positive_brier_gain`
     - `reject_unit=pixel`
     - `policy=advantage`

核心代码：

```text
models/distill/task_space.py
models/distill/dynamic_teacher.py
models/distill/diagnostics.py
models/distill/teacher_cache.py
models/datasets/cache_transforms.py
models/scripts/train.py
```

当前实现版本：

```text
implementation_version=bt_sam_rdt_run3_v1
cache_replay=aligned
```

---

## 消融设计（3 组）

| 组 | 实验 ID | 语义 | 证明的问题 |
|---|---|---|---|
| G1 | B0 | clean A2Net-LWGANet-L0，无 Teacher、无 Cache | 同协议 clean anchor |
| G2 | R3A | BT-SAM structural prior + one Fast Teacher + EMA Target，**关闭 OV 信息** | 双时相 SAM 结构先验是否有独立增量 |
| G3 | R3 | BT-SAM structural prior + OV semantic prior + reliability fusion + one Fast Teacher + EMA Target | 完整 BT-SAM-RDT 的有效性；R3 vs R3A 检验 OV 语义增量 |

注意：

- **R3A 虽然不使用 OV prior，但当前 `train.py` 为保持完全一致的数据输入链，仍要求同时提供 `--sam_cache_root` 和 `--ov_cache_root`。**
- B0 不需要任何 Teacher Cache。
- 当前 Run3 不包含旧 Run2 的 D1NG / D1 / D2，也不包含 Direction-C / SCGR / gradient gate / region router。
- 当前代码尚未实现额外的 no-prior capacity-matched control，因此不要在本 Run3 README 中把该对照当作已完成实验。

---

## 正式协议

默认正式协议由当前 `models/scripts/train.py` 固定/默认：

```text
input              256 × 256
batch_size         64
max_steps          40000
seed               2333
student lr         5e-4
student wd         1e-4
backbone lr mult   1.0
lr mode            poly
dice reduction     batch
main loss weights  1,1,1,1
kd_lambda          0.06
```

R3A / R3 Teacher 默认参数：

```text
teacher_lr            5e-4
teacher_hidden        24
teacher_ema           0.99
max_logit_delta       2.0
teacher_weight_decay  1e-4
teacher_grad_clip     5.0
boundary_radius       2
small_area            64
```

Student 主监督仍为四尺度 **batch BCE + Dice GT loss**。

正式比较必须保证：

```text
同 dataset
同 batch
同 max_steps
同 seed
同 Student 初始化
同主损失
同数据增强
同评估协议
```

单数据集、单 seed、微小差异不得表述为普适提升。若 Run3 主机制通过第一轮有效性门槛，再补论文级多 seed：

```text
2333 / 3407 / 5871
```

---

## 数据、Cache 与预训练权重

RSML-3 当前统一位置：

```text
项目：
/home/yqwang/projects/LS-Rep_BCD_RSML_3

数据集根：
/share_datasets/CD

Teacher Cache 总根：
/share_datasets/CD_teacher_cache

预训练权重目录：
/home/yqwang/projects/LS-Rep_BCD_RSML_3/pre-trained_weights
```

四个正式数据集：

```text
CDD-CD-256
LEVIR-CD-256
SYSU-CD-256
WHU-CD-256
```

SAM / OV 的具体 dataset cache 子目录必须以当前服务器实际 `manifest.json` 为准，不要根据旧 Run1/Run2 路径猜测。

正式训练开启预训练时，`train.py` 要求显式提供：

```text
--pretrained
--pretrained_path <实际预训练权重文件>
```

README 不固定具体权重文件名，避免与服务器实际文件不一致。

---

## 目录与保存

沿用 Run2 开始采用的简化目录约定，不再增加 `steps_40000/seed_2333` 中间目录。

checkpoint 根：

```text
/share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/RDT-CD/Run3
```

日志根：

```text
/home/yqwang/outputs/LS-Rep_BCD_RSML_3/RDT-CD/Run3
```

建议相对叶目录：

```text
<实验>/<数据集>/
```

例如：

```text
RDT-CD/Run3/B0/SYSU/
RDT-CD/Run3/R3A/SYSU/
RDT-CD/Run3/R3/SYSU/
```

每个叶目录中：

```text
train_log.txt
last_checkpoint.pth
best_model_F1=*.pth
```

`max_steps=40000`、`seed=2333` 等协议通过日志配置头记录，不再写入路径。

Checkpoint 为当前 **format v3**：

```text
model
optimizer
teacher_optimizer   # R3A/R3；B0 为 None
epoch
global_step
best_val_f1
args
rng
```

R3A/R3 的 Fast Teacher 与 EMA Target Teacher 均注册在 `model.training_auxiliary` 中，因此权重直接随 `model.state_dict()` 保存。

---

## Run3 文件夹建议结构

本目录建议最终整理为：

```text
train_scripts/RDT-CD/Run3/
├── README.md
├── run_gpu0_sysu_whu.sh
└── run_gpu1_cdd_levir.sh
```

如需额外准备单 run 调试脚本，可以增加：

```text
run_single.sh
```

但正式矩阵的唯一变量必须仍由 `--experiment B0/R3A/R3` 控制。

---

## GPU 与执行顺序

### Phase 1：先做 SYSU + WHU 有效性验证

优先在 GPU 0 串行：

```text
SYSU: B0 → R3A → R3
WHU : B0 → R3A → R3
```

共 6 个正式 run。

原因：当前 Run3 首要问题仍是 **Teacher 是否真正有效**。只有 SYSU / WHU 同协议结果支持机制后，再扩展到 CDD / LEVIR。

### Phase 2：通过有效性门槛后扩展 CDD + LEVIR

GPU 1：

```text
CDD  : B0 → R3A → R3
LEVIR: B0 → R3A → R3
```

共 6 个 run。

若希望两卡并行全矩阵，也可以：

```text
GPU 0：SYSU + WHU
GPU 1：CDD + LEVIR
```

但论文判断仍应先单独检查 SYSU / WHU 的有效性结论，不要因为其它数据集偶然提升跳过机制审查。

---

## 启动前必须完成

服务器环境：

```bash
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate lsrep

cd /home/yqwang/projects/LS-Rep_BCD_RSML_3
```

### 1. 清理并确认 Python 缓存不进入 Git

仓库应忽略：

```gitignore
__pycache__/
*.py[cod]
```

不要提交：

```text
数据集
Teacher Cache
预训练权重
checkpoint
训练日志
.pyc / __pycache__
```

---

### 2. Run3 synthetic smoke

```bash
python models/tools/smoke_dynamic_teacher.py \
    --device cuda \
    --gpu_id 0
```

必须通过的关键契约包括：

```text
opaque SAM ID 不影响匹配
disappearance / appearance 正确
expansion overlap core ≈ 0
expansion new ring ≈ 1
shrinkage remaining core ≈ 0
shrinkage disappeared ring ≈ 1
SAM quality 只调 reliability，不改结构标签
SAM/OV conflict 会降低 fused reliability
positive-Brier-gain audit 正确准入/拒绝
Student / Fast Teacher 梯度隔离
EMA Target 只通过 update_ema() 更新
auxiliary ON/OFF 不改变主预测
switch_to_deploy() 删除训练辅助
deploy prediction max error < 1e-6
deploy params == 2,913,094
```

synthetic smoke 通过 **不等于** real-cache 链路通过。

---

### 3. Teacher Cache 全量只读校验

对每个将要正式训练的数据集分别执行：

```bash
python models/tools/validate_teacher_cache.py \
    --data_root <DATASET_ROOT> \
    --dataset_name <CDD|LEVIR|SYSU|WHU> \
    --sam_cache_root <SAM_CACHE_ROOT> \
    --ov_cache_root <OV_CACHE_ROOT>
```

示例中的 `<SAM_CACHE_ROOT>` / `<OV_CACHE_ROOT>` 必须替换为服务器当前实际 manifest 所在目录。

校验内容包括：

```text
train.txt 无重复
SAM / OV manifest 合法
source_dataset 一致
config_hash 一致
train split 100% 覆盖
sample_id 一致
文件存在
tensor shape 合法
tensor finite
boundary / quality / soft_change / confidence ∈ [0,1]
SAM / OV paired load 正常
fingerprint 可记录
```

该工具为 **只读**，不得重建或覆盖 Teacher Cache。

---

### 4. real-cache dry run

在正式目录之外建立临时输出目录，用最复杂的 **R3 / SYSU** 跑少量 step。

示意：

```bash
python models/scripts/train.py \
    --experiment R3 \
    --dataset_name SYSU \
    --data_root /share_datasets/CD/SYSU-CD-256 \
    --sam_cache_root <SYSU_SAM_CACHE_ROOT> \
    --ov_cache_root <SYSU_OV_CACHE_ROOT> \
    --pretrained \
    --pretrained_path <PRETRAINED_WEIGHT> \
    --batch_size 64 \
    --max_steps 20 \
    --seed 2333 \
    --gpu_id 0 \
    --save_dir /home/yqwang/outputs/LS-Rep_BCD_RSML_3/_dryrun/Run3_R3_SYSU
```

dry run 重点验证：

```text
A/B/label/SAM/OV 同步 crop/resize/flip/time-swap
真实 Cache 能完整进入 BT-SAM task-space
Fast Teacher loss 非 NaN
Student loss 非 NaN
Teacher optimizer 正常更新
EMA 正常更新
checkpoint v3 正常保存
resume 能恢复
step_time / data_time 可接受
```

特别关注 `models/distill/diagnostics.py` 中 CPU connected-component 诊断是否造成明显吞吐瓶颈。

---

### 5. deploy/export 验证

至少对一个 R3A/R3 checkpoint 执行：

```bash
python models/tools/export_deploy.py \
    --checkpoint <RUN3_V3_CHECKPOINT> \
    --output <NEW_DEPLOY_CHECKPOINT>
```

必须满足：

```text
只导出 Student
training_auxiliary.* 不进入部署权重
switch_to_deploy() 成功
deploy params = 2,913,094
部署主图不包含 Teacher
```

正式 FLOPs 契约：

```text
256×256 ≈ 2.7475G
允许工程验证误差约 ±0.03G
```

---

## 正式启动脚本建议

### GPU 0：SYSU + WHU

建议文件：

```text
train_scripts/RDT-CD/Run3/run_gpu0_sysu_whu.sh
```

内部顺序：

```text
B0 / SYSU
R3A / SYSU
R3 / SYSU

B0 / WHU
R3A / WHU
R3 / WHU
```

### GPU 1：CDD + LEVIR

建议文件：

```text
train_scripts/RDT-CD/Run3/run_gpu1_cdd_levir.sh
```

内部顺序：

```text
B0 / CDD
R3A / CDD
R3 / CDD

B0 / LEVIR
R3A / LEVIR
R3 / LEVIR
```

每个 shell 建议实现：

```text
1. set -euo pipefail
2. 检查 project / data / cache / pretrained 文件存在
3. 已存在完整正式 TEST RESULTS 时跳过
4. 存在 last_checkpoint.pth 且未完成时自动 --resume
5. 新 run 不覆盖已有正式目录
6. 每次调用明确写出 experiment / dataset / seed / max_steps
```

修改 shell 后必须：

```bash
bash -n train_scripts/RDT-CD/Run3/run_gpu0_sysu_whu.sh
bash -n train_scripts/RDT-CD/Run3/run_gpu1_cdd_levir.sh
```

---

## tmux 启动示例

Phase 1：

```bash
tmux new -s rdtcd3_gpu0 -d \
'cd /home/yqwang/projects/LS-Rep_BCD_RSML_3 && \
bash train_scripts/RDT-CD/Run3/run_gpu0_sysu_whu.sh'
```

确认 SYSU / WHU 结果与健康信号后，再启动 Phase 2：

```bash
tmux new -s rdtcd3_gpu1 -d \
'cd /home/yqwang/projects/LS-Rep_BCD_RSML_3 && \
bash train_scripts/RDT-CD/Run3/run_gpu1_cdd_levir.sh'
```

---

## 日志重点检查

R3A / R3 重点关注以下健康信号。

### BT-SAM correspondence

```text
pair_match_ratio
pair_match_iou
pair_match_cov12
pair_match_cov21
```

解释：

- `pair_match_ratio`：SAM 双时相对象对应覆盖情况。
- `pair_match_iou`：对应实例的空间一致程度。
- `pair_match_cov12 / pair_match_cov21`：方向覆盖；用于分析 expansion / shrinkage 与匹配可靠性。
- directional coverage 不应再直接把 matched overlap core 标成变化。

### Foundation prior reliability / fusion

```text
sam_reliability_mean
ov_reliability_mean
fused_reliability_mean
sam_ov_conflict
foundation_support_ratio
```

R3 中若 `sam_ov_conflict` 长期很高，需检查两类 Foundation prior 是否在真实变化区域系统性冲突。

### GT safety audit

```text
pixel_reject_ratio
image_reject_ratio
accepted_change_ratio
accepted_bg_ratio
available_ratio
eligible_ratio
effective_mass
effective_per_error_mass
```

需要避免两种失败：

```text
几乎全部拒绝 → KD 信号饿死
几乎全部接受 → audit 失去筛选作用
```

### Prior / Teacher 质量

```text
student_brier
sam_prior_brier
ov_prior_brier
fusion_prior_brier
dynamic_teacher_brier
fast_teacher_brier

sam_prior_gain
ov_prior_gain
fusion_prior_gain
dynamic_teacher_gain
```

正式 40k 前应优先确认：

```text
BT-SAM prior 是否优于旧的单时相 SAM seed
fused prior 是否优于单独 OV prior
Target Teacher 是否真正优于 Student，而不是简单复制 Student
```

### Teacher 演化

```text
teacher_lr
teacher_grad_norm
teacher_ema_update_norm
teacher_target_gap
target_residual_magnitude
fast_residual_magnitude
dynamic_shift
```

若：

```text
teacher_grad_norm ≈ 0
teacher_ema_update_norm ≈ 0
teacher_target_gap 长期 ≈ 0
dynamic_shift 长期 ≈ 0
```

则说明 Teacher 可能未真正学习或退化为 Student identity，需要停止正式矩阵并排查。

---

## 正式结果读取规则

正式结果 **只能** 来自对应 `train_log.txt` 中最后一个完整：

```text
=== TEST RESULTS ===
...
=== END TEST RESULTS ===
```

区块。

正式报告：

```text
Recall
Precision
OA
F1
IoU
Kappa
训练参数量
部署参数量
FLOPs
```

禁止：

```text
拿 validation best 当 test
拿中途 epoch 当正式结果
混用不同 seed / batch / steps / protocol
只报 F1 不报其它正式指标
```

---

## Run3 判定

### B0 vs R3A

回答：

> **经过双时相实例匹配与空间结构变化重构后的 SAM structural prior，是否能给极轻量 Student 带来独立增量？**

重点比较：

```text
F1
IoU
Kappa
sam_prior_brier
dynamic_teacher_gain
effective_mass
```

如果 R3A 在 SYSU / WHU 均无法超过同协议 B0，则不能声称 BT-SAM structural Teacher 已被支持。

---

### R3A vs R3

回答：

> **在相同 Teacher 容量、相同训练协议下，OV semantic prior 是否给 BT-SAM structural prior 带来额外信息？**

唯一核心变量：

```text
use_ov=False  → R3A
use_ov=True   → R3
```

若 R3 不优于 R3A，说明 SAM + OV fusion 的语义补充价值没有得到证据支持。

---

### B0 vs R3

回答：

> **完整 BT-SAM-RDT 是否在同协议下改善固定的 A2Net-LWGANet-L0 Student？**

第一轮建议成功阈值：

```text
SYSU：R3 相对 B0 F1 ≥ +0.4
WHU ：R3 相对 B0 F1 ≥ +0.2
且任一数据集不出现 >0.3 F1 的明显回退
```

这是当前有效性探索阶段的工程 gate，不是论文最终统计结论。

---

## 失败判据

出现以下任一情况时，不应直接继续扩大正式矩阵：

1. BT-SAM structural prior 的 Brier / 分层质量没有明显优于旧 SAM seed。
2. R3 fused prior 不优于 OV prior，或 SAM/OV conflict 集中在真实变化区域。
3. R3A 在 SYSU / WHU 均不优于 B0。
4. R3 不优于 R3A。
5. WHU 出现 >0.3 F1 的明显回退。
6. `effective_mass` / `eligible_ratio` 长期坍缩，Teacher 信号被拒绝殆尽。
7. `dynamic_shift`、`teacher_target_gap` 长期接近 0，Target Teacher 实际复制 Student。
8. deploy 前后主预测最大误差 ≥ 1e-6。
9. deploy 参数量不等于 2,913,094。
10. auxiliary ON/OFF 改变 Student 主预测。
11. Cache replay 不能证明 A/B/label/SAM/OV 同步变换。

---

## Run3 当前能够支持与不能支持的论文主张

### 若实验成功，可以支持

```text
双时相 SAM 实例对应形成的结构变化先验
在训练期能够为固定的极轻量 CD Student 提供有效结构知识。

SAM structural 与 OV semantic foundation priors
在可靠性约束下具有互补性。

训练期 Foundation Teacher / residual teacher
可以在零部署额外开销下改善轻量变化检测 Student。
```

### 当前不能直接支持

```text
所有 Foundation Teacher 都有效
所有数据集都普适提升
SAM 奇偶分解有效
Teacher Agent / learned router 有效
提升一定来自 Foundation prior 而非 residual teacher 容量
```

最后一点若后续需要做论文级严格归因，应再增加 **capacity-matched no-prior control**；该对照当前不属于 `train.py` 已实现 Run3 三组矩阵，因此本目录暂不启动。

---

## 立即执行顺序

```text
1. 完成 Run3 shell 文件
2. bash -n 两个 shell
3. synthetic smoke
4. 四数据集 Teacher Cache 只读验证
5. R3/SYSU real-cache dry run
6. 短程吞吐检查
7. checkpoint v3 / resume 检查
8. deploy/export 检查
9. 原始 prior Brier 审计
10. 正式跑 SYSU：B0 → R3A → R3
11. 正式跑 WHU ：B0 → R3A → R3
12. 判断 Run3 是否通过有效性 gate
13. 通过后再跑 CDD / LEVIR
14. 主结论稳定后再补 3407 / 5871 seed
```

---

## 仍需补充的证据

Run3 代码通过 smoke/dry run 只证明工程正确性，不证明方法有效。正式投稿前至少仍需要：

```text
同协议 B0/R3A/R3 正式 test 结果
SYSU / WHU 主有效性
CDD / LEVIR 外部数据集验证
多 seed 稳定性
原始 SAM / OV / fused prior 的 Brier 与分层诊断
训练参数 / deploy 参数 / FLOPs
deploy 前后 prediction consistency
必要时 capacity-matched no-prior attribution control
```

Run3 的判断原则保持不变：

> **先证明 Foundation knowledge 本身有用，再讨论更复杂的 Teacher Agent 或路由机制。**
