# README.md — LS-Rep_BCD_RSML_3

> Last updated: 2026-09-23
> 本文件是当前项目中 AI/Coding Agent 的工作约束。当前唯一活动方向为 **SAGE-CD（Student-Aware Gated Expert Distillation）**：Run1「Safe-DINO」门（G0/G1/G2）已完成——WHU +0.46 / SYSU −0.25，门未过（SYSU 负），但 KD 机制已修对（kd 12–15→0.6、gradient budget 有效 KD 梯度 ~25% 主损失）；正在调研 Run2 的激进改进（SAM2 结构专家协同 / 教师-学生互促 / 模块与 KD 形式优化 / 学生模型结构深挖）。学生模型恒为 A2Net-LWGANet-L0，部署参数 2,913,094 不变。RDT-CD、SCGR、SAM-HSD、DART-R、SCTC、FA-SCRD 均作为历史实验归档（见 §6）。项目长期处于「方法有效性探寻阶段」，主线结论随时可能被新实验推翻；不得把本文档中的任何结果当作已定稿的论文结论。
>
> **实验约定（固定）：永远不做多 seed。** 每个机制只跑一组同协议（seed 2333、batch 64、40000 steps）的消融表实验，用 S 臂对 clean anchor 的逐数据集差值证明「机制是否有效」即可；不补 3407/5871、不做多 seed 复现、不做显著性检验。

## GitHub 提交流程

完成 `models/`、`README.md`、`others/`（参考代码）或当前脚本/说明的修改并通过必要检查后，按以下顺序提交到 GitHub：

```bash
git add models README.md train_scripts/SAGE-CD/Run1 docs/experiment_metrics.xlsx "docs/RSML-3_服务器环境与变化检测数据统一说明.md" others
git status
git commit -m "Update code"
git push
```

`others/` 只保留精简后的参考代码（供网页 GPT 阅读），不提交其中的权重、`__pycache__/`、`*.pyc`、`work_dirs/`、数据集或完整框架副本。

执行 `git commit` 前必须先检查 `git status`，确认暂存区只包含本次准备提交的文件；不要把权重、缓存、数据集、日志、`__pycache__/`、`*.pyc` 或无关改动加入提交。

## 研究定位与优先级

本项目服务于 211 高校硕士研究生的学术论文研究与实验验证，不以通用软件工程建设为主要目标。Agent 应将具有论文价值的**模型创新和方法性改进**置于最高优先级，重点关注网络结构、特征表征与交互、训练期辅助机制及其可解释性。

1. 优先提出机制明确、区别于现有模块且能形成论文贡献点的方法创新；每项方案都要说明创新动机、作用机制、与已有方法的差异、可证伪假设及最小消融实验。
2. 损失函数和超参数仅作为公平训练、稳定收敛与验证机制有效性的配套手段；除非有明确的新理论或新机制，否则不得把调参结果包装成核心创新。
3. 工程修改只需服务于实验正确性、可复现性和必要的运行效率。
4. 评价方案优先考虑学术新颖性、机制合理性、实验可验证性和论文叙事完整性，警惕无机理的模块堆叠与只追求单次指标的改动。

## 1. 作用域与权威来源

当前唯一活动实验族：

```text
train_scripts/SAGE-CD/Run1/      # SAGE-CD（G0 / G1 / G2，教师+蒸馏）
```

（`train_scripts/RDT-CD/`、`train_scripts/SCTC/`、`train_scripts/FA-SCRD/`、`train_scripts/SAM-HSD/`、`train_scripts/DART-R/` 等均为历史实验，已归档，不再活动。）

按以下优先级判断事实：

1. 当前 `models/` 代码与 shell 参数；
2. `train_scripts/SCTC/Run1/README.md`（上一轮 SCTC 实验协议，作历史参考）；
3. `docs/RSML-3_服务器环境与变化检测数据统一说明.md`（环境/数据/路径）；
4. 本文件中的摘要。

不得从旧项目复制实验结果、GPU/存储配置、完成状态或结论。需要报告指标时，必须重新读取当前服务器上对应运行的正式 test block。

## 2. RSML-3 项目环境

```yaml
project:
  name: LS-Rep_BCD_RSML_3
  server: RSML-3
  user: yqwang
  path: /home/yqwang/projects/LS-Rep_BCD_RSML_3

environment:
  name: lsrep          # clone 自 cd_base
  path: /home/yqwang/miniforge3/envs/lsrep
  python: 3.10.21
  pytorch: 2.14.0+cu132
  cuda_runtime: "13.2"

gpu:
  count: 2
  model: NVIDIA GeForce RTX 5090   # Blackwell, sm_120, 每卡 32GB
```

环境使用：

```bash
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate lsrep
cd /home/yqwang/projects/LS-Rep_BCD_RSML_3
```

环境规则：

- 训练、验证、推理和复杂度统计固定使用 `lsrep`，不要直接使用或污染 `cd_base`。
- 不要自行降级、替换或无约束升级 PyTorch/CUDA。
- 缺失的普通 Python 依赖只安装到 `lsrep`；CUDA Extension 必须先确认兼容 PyTorch 2.14.0、CUDA 13.2 与 `sm_120`。

## 3. 服务器目录约定

| 用途 | 路径 |
|---|---|
| 项目代码 | `/home/yqwang/projects/LS-Rep_BCD_RSML_3` |
| 数据集 | `/share_datasets/CD` |
| SAGE-CD checkpoint | `/share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/SAGE-CD/Run1/<实验>/<数据集>/` |
| SAGE-CD 训练/测试日志 | `/home/yqwang/outputs/LS-Rep_BCD_RSML_3/SAGE-CD/Run1/<实验>/<数据集>/train_log.txt` |
| 预训练权重 | `/home/yqwang/projects/LS-Rep_BCD_RSML_3/pre-trained_weights/lwganet_l0_e299.pth` |

（旧 RDT-CD/SCTC/FA-SCRD/SAM-HSD/DART-R 的 checkpoint/log 仅作归档，不再活动。当前 SAGE-CD 实验目录不使用 `steps_40000/seed_2333` 中间目录，这两项写进训练日志配置头。SAGE-CD 的教师 cache 在 `/share_datasets/CD_teacher_cache/SAGE_DINO3_CD/`。）

`.vscode/sftp.json` 用于手动上传代码，自动上传保持关闭；该配置忽略 `pre-trained_weights/` 和常见权重文件。

## 4. SAGE-CD（当前活动实验）

方向 C 经历「教师/蒸馏（RDT-CD，SYSU-only）→ 机制级校准（SCTC，WHU-only）→ 固定教师 relation KD（FA-SCRD，负迁移）」三次失败后，根因收敛为：**Teacher 监督没有被限制在「Teacher 真正比 Student 可靠且 Student 有容量吸收」的知识子空间**，且 FA-SCRD 的 KD 尺度失衡 + 16×16 relation feature 与增强视图错位。SAGE-CD 据此换掉 relation imitation，第一阶段只做 G0/G1/G2「Safe-DINO」。

> **Run1 结果与结论（seed 2333，6 run 完成，2026-09-22）**：G0/G1/G2 × SYSU/WHU 全跑完——SYSU 82.48/82.72/**82.47**、WHU 92.78/93.77/**94.22**。**G2−G1 = WHU +0.46 / SYSU −0.25** → Safe-DINO 门未过（SYSU 负）。但关键：**KD 机制修对了**——kd loss 从 FA-SCRD 的 12–15 降到 **0.6**（logit standardization 生效），gradient budget 把有效 KD 梯度钳在 ~25% 主损失（`lambda_kd × grad_ratio ≈ 0.25`）。DINO 语义教师仍只帮低变化 WHU（Recall +0.83）、伤高变化 SYSU。据此 Run2 转向：SAM2 结构专家协同 + 教师-学生互促 + 模块/KD 激进优化。

### 4.1 核心思想

DINOv3 语义专家只蒸馏 **full-res 变化 target**（256×256 change logit），不蒸馏 raw Foundation feature relation；KD 走训练期 auxiliary head（CrossKD 式解耦），并做 logit standardization + gradient budget，从机制上防止 KD 压过主损失。第二阶段（G2 过门后）才加入 SAM2 结构专家 + DINO/SAM/Reject 软路由。

```text
训练图：T1,T2 → Student → TFM → c4 → AuxSemanticHead → z_S
        T1,T2 → DINOv3(离线) → 256×256 change logit z_D（cache）
        z_S, z_D → logit standardization + soft BCE + gradient budget → 只更新 student

部署图：T1,T2 → LWGANet-L0 → SWA → TFM → Decoder → Change map（2,913,094 参数）
```

### 4.2 教师（离线，复用 FA-SCRD 教师）

- DINOv3 ViT-B/16（frozen + LoRA Q/V + change head），SYSU 微调一次，产出 full-res change logit。
- SAGE cache 只存 `dino_change_logit` + `dino_change_prob`（均 [1,256,256]，FP16），**不存 16×16 feature**（修掉 FA-SCRD 的 crop 错位 P0）。

### 4.3 KD 与梯度预算

- 训练期 `AuxSemanticHead`：1×1 Conv(64→1) 于 c4（1/16）→ 上采样 256×256。
- Logit standardization：对 teacher/student logit 各做逐样本空间 z-score，再做温度 + soft BCE，解耦容量差。
- Gradient budget：`λ_D = clip(ρ·‖∇_c4 L_main‖ / ‖∇_c4 L_KD‖, 0, λ_max)`，`ρ=0.25`，保证任一 KD 梯度 ≤ 主损失 25%。

### 4.4 实验矩阵（第一阶段门，seed 2333）

| ID | 唯一变量 | 目的 |
|---|---|---|
| G0 | 无（clean A2Net） | clean anchor |
| G1 | G0 + joint temporal BN | 归一化/BN 控制 |
| G2 | G1 + full-res DINO 语义 target + aux head + logit standardization + gradient budget | Safe-DINO 生死门 |

### 4.5 训练协议（固定）

```text
input 256×256 / batch 64 / max_steps 40000 / seed 2333
student lr 5e-4 / wd 1e-4 / dice batch / main loss BCE+Dice（1,1,1,1）
KD: temperature=1.0, rho=0.25, lambda_max=1.0
```

### 4.6 门判据

`G2 − G1 ≥ +0.15 F1` 于 SYSU 与 WHU，且无 Recall/Precision 单边崩塌、`grad_ratio_dino ≤ 0.25`。否则停止 SAGE-CD，不上 SAM2、不做多教师。

（FA-SCRD 已归档，机制与结果见 §6.3。）

## 5. 模型与部署约束

- 主学生模型是 A2Net-LWGANet-L0；教师/cache/aux-head/KD 全是训练期辅助，**不进部署主图、不被 `switch_to_deploy()` 带出**。
- 部署参数量固定为 `2,913,094`。
- 256×256 部署 FLOPs 实测约 `2.7676G`（`train_sage.py` 契约 2.7475G ± 0.03G）。正式报告需同时给 THOP 实测与 functional-op 手工审计。
- 时间交换对称是硬性质：`max |Model(A,B) − Model(B,A)| < 1e-6`（smoke 强制验证）。
- `switch_to_deploy()` 为 no-op（返回 self），部署前后主预测最大误差 `< 1e-6`。
- 无 Teacher map / Foundation prior / Cache 注入主特征路径；G0/G1/G2 都不改部署 forward。
- 训练、验证、测试、部署使用同一 A2Net 部署图。
- checkpoint 仅支持 **SAGE-CD format v6**；不得静默恢复旧 FA-SCRD format-v5 / SCTC format-v4 checkpoint。
- 教师/cache/aux-head 不进部署 checkpoint；部署导出只保留 `student.state_dict()`。

当前核心代码：

```text
models/__init__.py
models/a2net.py
models/backbone/lwganet.py
models/decoder/a2net_decoder.py
models/decoder/temporal_calibration.py      # SCTC 遗留（已归档，SAGE-CD 恒用 none）
models/datasets/cd_dataset.py
models/datasets/transforms.py
models/losses/combined_loss.py
models/distill/fa_scrd_teacher.py           # DINOv3 教师（SAGE-CD 复用）
models/distill/cache_dataset.py
models/distill/sage_heads.py
models/distill/sage_kd.py
models/thirdparty/dinov3/                   # DINOv3 ViT-B/16 最小抽取
models/scripts/train_sage.py
models/tools/generate_sage_cache.py
models/tools/smoke_sage.py
models/utils/metrics.py
models/utils/scheduler.py
```

`models/thirdparty/dinov3/` 是从 Meta DINOv3 抽取的最小 teacher backbone（只读参考，不改）。`models/distill/relation_kd.py`、`failure_router.py`（FA-SCRD）、`models/tools/smoke_temporal_calibration.py`（SCTC）、`models/scripts/train.py`（FA-SCRD）随旧方向归档不再使用。教师/cache/aux-head/KD 模块（`models/distill/`）只训练期使用、不进部署图；`models/scripts/train_sage.py` 中 `implementation_version=sage_cd_v1`、`checkpoint format_version=6`。

## 6. 历史实验归档（不再活动）

以下实验仅作为论文叙事的历史证据，不再活动、不得重新混入当前代码路径。

### 6.1 RDT-CD（方向 C 教师/蒸馏，Run1→Run3）

**Run3 BT-SAM-RDT**（Bi-Temporal Structural SAM Reciprocal Dynamic Teacher）：BT-SAM 双时相实例对应构造结构先验 + OVCDistill 语义先验 → reliability fusion → 单个 Fast/EMA residual teacher → GT positive-Brier-gain audit。学生仍为 A2Net-LWGANet-L0。全部 Foundation Cache / Fast Teacher / EMA / audit 都是纯训练期机制，不进部署主预测路径。

Run3 首轮（seed 2333，同协议 40000 steps / batch 64）正式 test F1（%）：

| Dataset | B0 | R3A (SAM) | R3O (OV) | R3 (SAM+OV) |
| --- | ---: | ---: | ---: | ---: |
| SYSU | 81.75 | 82.29 | 82.76 | 83.13 |
| WHU | 94.06 | 93.94 | 93.33 | 93.70 |
| CDD | 97.79 | 97.78 | — | 97.79 |
| LEVIR | 91.13 | 91.03 | — | 90.98 |

**结论（单 seed，不构成定稿）**：所有教师臂在 WHU/LEVIR/CDD 均 ≤ 干净 B0，只有变化像素比最高（21.1%）的 SYSU 有正增益；R3O(OV-only) 在 WHU 最差（−0.73），推翻「SAM 污染 OV」。日志诊断 `aux_raw≈0`、`dynamic_teacher_gain≈0`、`pixel_reject_ratio≈0.999`，说明 Fast Teacher 经「初始 residual=0 → 接近 Student → gain≈0 → 99.6%~99.9% 像素被 audit 拒绝」后近乎惰性。据此判定 Teacher/蒸馏方向不可继续堆叠，转向 §4 的机制级校准。

**Run1/Run2**：Run1 双教师竞争、Run2/SCGR（regional routing + gradient-concordance gate）均无可信正增益，已停止；对应 `train_scripts/RDT-CD/Run1/`、`Run2/`、`Run3/` 与 `models/distill/` 均已归档。

### 6.2 SCTC（方向 C 换机制的第二次尝试）

`train_scripts/SCTC/Run1/`：S0/S1/S2 三臂（none / symmetric / sctc）。结果 WHU-only 正增益（+0.43），SYSU/CDD/LEVIR 中性到负（−0.25 / −0.20 / −0.10），H4 失败 → 归档为单数据集 trick（同 Run3 教师，方向镜像）。机制细节见 `train_scripts/SCTC/Run1/README.md`。

### 6.3 FA-SCRD（方向 C 第三次尝试）

`train_scripts/FA-SCRD/Run1/`：固定 DINOv3 ViT-B/16 教师（LoRA Q/V + change head）+ 对称变化关系 KD + failure-aware 软加权。结果 A1−C1 = SYSU −0.33 / CDD −0.33（教师迁移为负），M1 未救回 → 归档为「固定教师迁移无效」。机制细节见 `docs/temporary/FA-SCRD_Implementation_Specification.md`。

### 6.4 其它历史

`train_scripts/SAM-HSD/`（CR-SRD）、`train_scripts/DART-R/`（DART-R-TS 固定教师）等亦为历史实验，仅作对照证据，不再活动。

## 7. 启动、恢复与结果纪律

推荐顺序：

1. 激活 `lsrep` 并进入项目目录；
2. 跑 `python -m models.tools.smoke_sage --device cuda --gpu_id 0`，验证学生参数 2,913,094、swap/deploy 不变性、aux head / semantic KD / gradient budget；
3. 准备教师 cache：`bash train_scripts/SAGE-CD/Run1/prepare_sage_cache.sh`（复用 FA-SCRD 教师，生成 SYSU/WHU full-res DINO 语义 cache）；
4. 启动队列（SYSU/WHU 并行）：`tmux new -s sage1_gpu0_sysu -d 'bash train_scripts/SAGE-CD/Run1/run_gpu0_sysu.sh'` + `sage1_gpu0_whu`（`run_gpu0_whu.sh`）；
5. 从 `train_log.txt` 收集正式 test 指标。

每个实验目录使用：

- `last_checkpoint.pth`：SAGE-CD format-v6 精确恢复；
- `best_model_F1=*.pth`：验证集选择出的最佳模型；
- `train_log.txt`：训练、恢复和最终测试记录。

SAGE-CD v6 checkpoint 至少包含：`format_version=6`、`model`、`aux_head`、`optimizer`、`aux_optimizer`、`epoch`、`global_step`、`best_val_f1`、`args`、`rng`。教师/cache/aux-head 不进部署 checkpoint。

只有 `train_log.txt` 中最后一个完整：

```text
=== TEST RESULTS ===
...
=== END TEST RESULTS ===
```

区块才是正式结果。

结果规则：

- 正式指标只能来自最后一个完整 test block，不能使用验证集最佳行或 checkpoint 文件名替代。
- 报告 Recall、Precision、OA、F1、IoU、Kappa，以及训练/部署参数量和 FLOPs。
- 比较前检查 dataset、batch、max_steps、seed、Student 初始化、参数量、数据增强与评估协议是否一致。
- 单数据集、单种子、微小差异不能被表述为普适提升。
- 永远不做多 seed：每个机制只跑一组同协议（seed 2333）消融表，用各臂对 clean anchor 的逐数据集差值证明机制有效性即可；不补 seed、不做显著性检验。
- 消融表对照范式：`clean anchor` vs `主方法` vs 必要对照臂（单一变量），逐数据集算 ΔF1。

## 8. Agent 工作规则

1. 修改前读取当前代码、SAGE-CD Run1 README 与实际日志，不依赖旧项目记忆；若文档与代码冲突，以当前代码/shell 为准。
2. 方法创新优先，但每项主张必须对应明确机制、与已有工作的实质区别、可证伪假设、最小消融和失败判据。
3. 保持部署主图、时间交换一致性 `<1e-6`、`switch_to_deploy()` no-op、部署参数 2,913,094 / FLOPs ~2.77G 等约束不变。
4. 不得重新引入旧 Direction-C、SCGR、Run1 双教师竞争、SAM transport 或历史 router，除非用户明确要求重新开启该方向（SAGE-CD 是当前唯一活动的教师方向）。
5. 修改 shell 后执行 `bash -n` 语法检查并确认 LF 换行（不是 CRLF）；修改后重新检查 shell 中 dataset/pretrained/checkpoint/log 路径。
6. 修改模型后至少跑 synthetic smoke（`smoke_sage.py`）；涉及真实数据/Cache 时再跑短 dry run。
7. 修改 checkpoint/resume 逻辑时保持 SAGE-CD format-v6、optimizer、aux optimizer、model state、训练进度和 RNG 的精确恢复能力。
8. `models/scripts/train_sage.py` 中旧 best checkpoint 的主动删除/替换策略是当前有意保留的行为；除非用户明确要求，不要擅自移除。
9. 不恢复已删除的历史 routing 文件或任何 `__pycache__/*.pyc`；`models/thirdparty/dinov3/` 只读参考、不改。
10. 不在 `cd_base` 中安装项目专属依赖，不擅自更换 PyTorch/CUDA 栈。
11. 任何删除数据、checkpoint 或预训练权重的操作都需要用户明确授权和精确路径校验。
12. Git 提交前必须检查暂存区；不得提交权重、Cache、数据集、checkpoint、训练日志、临时 dry-run 产物或 Python 字节码。
13. 当前仍处于有效性探寻阶段：不得因为模块来自强论文或单次实验有提升，就默认该机制适用于本项目。
14. 若某机制不能在同协议消融表下证明稳定正增益（多数据集、多臂对照），应停止扩展、归档并回到证据诊断，而不是继续加模块或换 loss。
15. 永远不做多 seed：每个机制只跑一组 seed 2333 消融表证明有效性；不补多 seed、不做显著性检验。
