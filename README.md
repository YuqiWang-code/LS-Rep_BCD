# README.md — LS-Rep_BCD_RSML_3 / SCTC

> Last updated: 2026-09-20
> 本文件是当前项目中 AI/Coding Agent 的工作约束。当前唯一活动实验为 `train_scripts/SCTC/Run1/`（方向 C 换机制），主方法为 **SCTC（Unchanged-Aware Symmetric Cross-Temporal Calibration，不变区域感知的对称跨时相校准）**；学生模型 A2Net-LWGANet-L0，部署参数 2,913,094 不变。原方向 C 的 RDT-CD（BT-SAM-RDT 教师/蒸馏）系列与更早的 Run1/Run2/SCGR/SAM-HSD/DART-R 均作为历史实验归档（见 §6），不再作为当前活动实现继续迭代。项目长期处于「方法有效性探寻阶段」，主线结论随时可能被新实验推翻；不得把本文档中的任何结果当作已定稿的论文结论。

## GitHub 提交流程

完成 `models/`、`README.md` 或当前 SCTC Run1 脚本/说明的修改并通过必要检查后，按以下顺序提交到 GitHub：

```bash
git add models README.md train_scripts/SCTC/Run1 docs/experiment_metrics.xlsx "docs/RSML-3_服务器环境与变化检测数据统一说明.md"
git status
git commit -m "Update code"
git push
```

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
train_scripts/SCTC/Run1/      # 方向 C 换机制：SCTC（S0 / S1 / S2，Student-only）
```

（`train_scripts/RDT-CD/`、`train_scripts/SAM-HSD/`、`train_scripts/DART-R/` 等均为历史实验，已归档，不再活动。）

按以下优先级判断事实：

1. 当前 `models/` 代码与当前 SCTC shell 参数；
2. `train_scripts/SCTC/Run1/README.md`（当前实验协议）；
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
| SCTC checkpoint | `/share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/SCTC/Run1/<实验>/<数据集>/` |
| SCTC 训练/测试日志 | `/home/yqwang/outputs/LS-Rep_BCD_RSML_3/SCTC/Run1/<实验>/<数据集>/train_log.txt` |
| 预训练权重 | `/home/yqwang/projects/LS-Rep_BCD_RSML_3/pre-trained_weights/lwganet_l0_e299.pth` |

（旧 RDT-CD/SAM-HSD/DART-R 的 checkpoint/log 仅作归档，不再活动。当前 SCTC 实验目录不使用 `steps_40000/seed_2333` 中间目录，这两项写进训练日志配置头。SCTC 无 Teacher/Cache，因此不使用 `/share_datasets/CD_teacher_cache`。）

`.vscode/sftp.json` 用于手动上传代码，自动上传保持关闭；该配置忽略 `pre-trained_weights/` 和常见权重文件。

## 4. SCTC（当前活动实验，方向 C 换机制）

方向 C 经历了「教师/蒸馏」到「机制级校准」的转变：C0（旧梯度路由）、C1（DART-R-TS 固定教师）、RDT-CD Run1/Run2/Run3（BT-SAM-RDT 动态教师）均未能在同协议下证明稳定、可归因的正增益（详见 §6）。据此停止继续堆叠 Teacher/router/loss，转向诊断定位到的 Student 主路径瓶颈。

### 4.1 诊断与动机

A2Net 当前主图在 SWA 输出后直接做 `|F1 - F2|` 绝对差分，此前**没有任何显式跨时相分布校准**。一旦光照/季节/纹理/成像条件差异进入 `F1/F2`，就会和真实变化一起被编码成「差分能量」，后续 TFM 只能学「这个差值像不像变化」，无法区分跨时相域偏移与语义变化。对于 WHU/LEVIR 这类变化像素仅 3%–4% 的数据，少量伪变化 FP 就足以抵消大量 TP。

SCTC 的解法只有一句：**先利用双时相自身的高一致区域估计「成像域偏移」，把 T1/T2 对称映射到同一个公共特征域，再交给原有 TFM 做差分。**

### 4.2 机制（零参数、交换对称）

对 SWA 输出的第 s 个尺度（64 通道）：

1. **不变可信度**：每个空间位置的跨时相 channel vector 的 cosine agreement
   `a = (1 + cos(F1, F2)) / 2 ∈ [0,1]`，高值=稳定区域；权重 `w = stopgrad(a)`（detach，防止 backbone 钻空子）。
2. **稳定区域统计量**：用 `w` 加权估计两时相每通道的 `μ_t, σ_t²`（区别于整图等权的 InstanceNorm）。
3. **公共中点域**：`μ_* = (μ1+μ2)/2`，`σ_* = sqrt((σ1²+σ2²)/2 + ε)`，然后
   `F̃_t = σ_* · (F_t − μ_t)/σ_t + μ_*`。
4. 只把 `F̃1, F̃2` 送入**原样不改的 TFM**：`D = |F̃1 − F̃2|`；后续 dilation=7/5/3/1 分支与 Decoder 全不变。

SCTC 只有 cosine、加权 mean/std 与 affine 运算，**ΔParams=0**；严格时间交换对称（`swap(F1,F2) → swap(F̃1,F̃2)`，因此 `|F̃1−F̃2|` 不变）。SCTC 是部署主图的一部分，训练/验证/测试/部署同一图，**不是训练辅助、不进 `switch_to_deploy()` 删除**。

### 4.3 实验矩阵（Student-only，无 Teacher/Cache/Foundation Model）

| ID | Difference 前操作 | temporal_calibration_mode | 目的 |
|---|---|---|---|
| S0 | 无（原始 A2Net 直接 `abs(F1-F2)`） | `none` | clean anchor |
| S1 | 全像素对称统计校准 | `symmetric` | 排除「普通 normalization 就够了」 |
| S2 | **Unchanged-Aware SCTC**（agreement 加权统计） | `sctc` | 完整主方法 |

三臂只差 `--experiment S0|S1|S2`，其余参数一致。`models/scripts/train.py` 中 `implementation_version=sctc_v1`，`checkpoint format_version=4`。

### 4.4 训练协议（固定）

```text
input           256×256
batch_size      64
max_steps       40000
seed            2333
student lr      5e-4
student wd      1e-4
backbone lr mult 1.0
dice reduction  batch
main loss       BCE + Dice（四尺度权重 1,1,1,1）
optimizer       Adam
Teacher         OFF
Cache           OFF
```

### 4.5 可证伪假设与失败判据

- **H1 伪变化抑制**：S2 应主要降背景 FP（Precision↑），Recall 下降 ≤ 0.20；WHU 上要求 ΔF1 ≥ +0.30。
- **H2 稀疏变化特异性**：低变化比 WHU 应比高变化比 SYSU 获益更明显（若再现「SYSU 明显升、WHU 明显降」则停止）。
- **H3 unchanged-aware 必要性**：`|F1(S2) − F1(S1)| < 0.10` 则 unchanged-aware 无独立贡献。
- **H4 多数据集一致性**：4 数据集 ≥3 个正增益，且任一 ΔF1 ≥ −0.20。

失败判据（任一成立即停止扩展，不再加模块）：WHU `S2 ≤ S0`；SYSU 正 / WHU/LEVIR 负（重现教师方向性）；Precision↑ 但 Recall 明显↓ 致 F1 无提升；`S2≈S1`；部署参数 ≠ 2,913,094；256×256 FLOPs 超出契约。

### 4.6 启动顺序

```text
第一阶段：S0/S1/S2 × WHU + SYSU（机制门）
        ↓ 通过
第二阶段：S0/S1/S2 × LEVIR + CDD
        ↓ 四数据集通过
多 seed：2333 / 3407 / 5871
```

当前（2026-09-20）第一阶段与第二阶段已按用户指示**并行**跑在 GPU 0（两个 tmux 队列：`sctc1_gpu0`=WHU/SYSU、`sctc1_phase2_gpu0`=LEVIR/CDD，每队列串行 S0→S1→S2）。「门」不再是启动门槛，而是解释已跑结果的判据。

## 5. 模型与部署约束

- 主学生模型是 A2Net-LWGANet-L0；SCTC 是零参数 temporal 操作，属于部署主图，**不是训练辅助、不被 `switch_to_deploy()` 移除**。
- 部署参数量固定为 `2,913,094`。
- 256×256 部署 FLOPs 实测约 `2.7676G`（`train.py` 契约 2.7475G ± 0.03G；SCTC 新增的是 element-wise reduction/affine，可忽略）。正式报告需同时给 THOP 实测与 functional-op 手工审计。
- SCTC 三种模式 `none/symmetric/sctc` 下部署参数量恒等；`mode=none` 是精确恒等路径。
- 时间交换对称是硬性质：`max |Model(A,B) − Model(B,A)| < 1e-6`（smoke 强制验证）。
- `switch_to_deploy()` 为 no-op（返回 self），部署前后主预测最大误差 `< 1e-6`。
- 无 Teacher map / Foundation prior / Cache 注入主特征路径；S0/S1/S2 都不用 SAM/OV Cache。
- 训练、验证、测试、部署使用同一 SCTC 图。
- checkpoint 仅支持 **SCTC format v4**；不得静默恢复旧 Run3 format-v3 的 Teacher checkpoint 到 SCTC 实验（`validate_resume` 会拒绝）。
- SCTC 无 state_dict 参数，旧 B0 权重理论上可无新增 missing/unexpected 加载，但实验模式必须写入 checkpoint/log，恢复时核验 `temporal_calibration_mode`，禁止 B0 与 SCTC checkpoint 静默互换实验语义。

当前核心代码：

```text
models/__init__.py
models/a2net.py
models/backbone/lwganet.py
models/decoder/a2net_decoder.py
models/decoder/temporal_calibration.py
models/datasets/cd_dataset.py
models/datasets/transforms.py
models/losses/combined_loss.py
models/scripts/train.py
models/utils/metrics.py
models/utils/scheduler.py
models/tools/smoke_temporal_calibration.py
```

`models/distill/`（Teacher/KD）、`cache_transforms.py`、旧 `checkpoint.py`/`logger.py`（已内联进 `train.py`）及旧 `smoke_dynamic_teacher.py`/`validate_teacher_cache.py`/`export_deploy.py` 已从当前 `models/` 清理，不得因历史引用重新恢复，除非用户明确改变研究方向。

## 6. 历史实验归档（不再活动）

以下实验仅作为论文叙事的历史证据，不再活动、不得重新混入当前 SCTC 代码路径。

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

### 6.2 其它历史

`train_scripts/SAM-HSD/`（CR-SRD）、`train_scripts/DART-R/`（DART-R-TS 固定教师）等亦为历史实验，仅作对照证据，不再活动。

## 7. 启动、恢复与结果纪律

推荐顺序：

1. 激活 `lsrep` 并进入项目目录；
2. 跑 `python -m models.tools.smoke_temporal_calibration --device cuda --gpu_id 0`，验证 S0 恒等、S1/S2 零参数、时间交换对称、梯度隔离、部署参数 2,913,094 与 `switch_to_deploy()` 不变性；
3. 对 S2 做短 dry run（`--max_steps 5`，真实数据）核验数据 pipeline、checkpoint 与 256×256 FLOPs 契约；
4. 启动队列：`tmux new -s sctc1_gpu0 -d 'bash train_scripts/SCTC/Run1/run_gpu0_whu_sysu.sh'`（Phase 2 同理 `run_gpu0_levir_cdd.sh`）；
5. 从 `train_log.txt` 收集正式 test 指标。

每个实验目录使用：

- `last_checkpoint.pth`：SCTC format-v4 精确恢复；
- `best_model_F1=*.pth`：验证集选择出的最佳模型；
- `train_log.txt`：训练、恢复和最终测试记录。

SCTC v4 checkpoint 至少包含：`format_version=4`、`model`、`optimizer`、`epoch`、`global_step`、`best_val_f1`、`args`、`rng`。

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
- seed 2333 只用于首轮有效性；论文级主比较需补多 seed（`2333/3407/5871`）。
- 当前 SCTC 首要比较：`S0 vs S2`（SCTC 是否改善固定 Student）、`S0 vs S1`（基础 pair calibration 是否足够）、`S1 vs S2`（unchanged-aware weighting 是否有独立贡献）。

## 8. Agent 工作规则

1. 修改前读取当前代码、SCTC Run1 README 与实际日志，不依赖旧项目记忆；若文档与代码冲突，以当前代码/shell 为准。
2. 方法创新优先，但每项主张必须对应明确机制、与已有工作的实质区别、可证伪假设、最小消融和失败判据。
3. 保持部署主图、SCTC 零参数/交换对称、`switch_to_deploy()` no-op、时间交换一致性 `<1e-6` 等约束不变。
4. 不得重新引入旧 Teacher/蒸馏、Direction-C、SCGR、Run1 双教师竞争、SAM transport 或历史 router，除非用户明确要求重新开启该方向。
5. 修改 shell 后执行 `bash -n` 语法检查并确认 LF 换行（不是 CRLF）；修改后重新检查 shell 中 dataset/pretrained/checkpoint/log 路径。
6. 修改模型后至少跑 synthetic smoke（`smoke_temporal_calibration.py`）；涉及真实数据时再跑短 dry run。
7. 修改 checkpoint/resume 逻辑时保持 SCTC format-v4、optimizer、model state、训练进度和 RNG 的精确恢复能力。
8. `models/scripts/train.py` 中旧 best checkpoint 的主动删除/替换策略是当前有意保留的行为；除非用户明确要求，不要擅自移除。
9. 不恢复已删除的 `models/distill/`、历史 routing 文件或任何 `__pycache__/*.pyc`；这些不属于当前 SCTC 源码。
10. 不在 `cd_base` 中安装项目专属依赖，不擅自更换 PyTorch/CUDA 栈。
11. 任何删除数据、checkpoint 或预训练权重的操作都需要用户明确授权和精确路径校验。
12. Git 提交前必须检查暂存区；不得提交权重、Cache、数据集、checkpoint、训练日志、临时 dry-run 产物或 Python 字节码。
13. 当前仍处于有效性探寻阶段：不得因为模块来自强论文或单次实验有提升，就默认该机制适用于本项目。
14. 若 SCTC 在 WHU/SYSU 上不能通过 §4.5 的机制门，应停止扩展、回到证据诊断，而不是继续加模块或换 loss。
