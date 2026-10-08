# README.md — LS-Rep_BCD_RSML_3

> Last updated: 2026-10-07
> 当前主线：**CATA-CD v2 —— Capability-Validated Adaptive Teacher Agent for Lightweight Change Detection**。
> 完整方案见 `docs/temporary/CATA-CD_v2_扩展教师池_验证驱动Agent_完整方案_20261005.md`。
> 项目长期处于「方法有效性探寻阶段」，主线结论随时可能被新实验推翻；不得把本文档中的任何结果当作已定稿的论文结论。

## 任务与硬目标

- 任务：推理时**极轻量化**的遥感图像**全监督二值变化检测**（不涉及半监督/多类/语义分割）。
- 数据集（固定 4 个，256×256）：SYSU-CD、LEVIR-CD、WHU-CD、CDD-CD。
- **最终硬目标（必须同时全部达到）**：
  - 四数据集 F1：SYSU ≥ 85、LEVIR ≥ 92.5、WHU ≥ 95、CDD ≥ 98（IoU 与 F1 同方向）；
  - **有效推理参数 < 5M**。

## 主线（CATA-CD v2）

1. 先把每个候选基础模型做成独立 **Teacher Package**，在四数据集上用完全相同的学生、初始化、40K steps、seed=2333 逐一验证（`train_scripts/CATA-CD/teacher_adaptation/`）。
2. 只有被实验证明「至少在部分数据集产生正迁移」的教师包才进入 Teacher Capability Registry。
3. 用带 **None/拒绝动作**的离线上下文 Bandit 学习「数据集签名 + 教师 train-only probe → 教师包/None」的选择策略（LODO 防止变成 4 行查表）。
4. 把一个约 **0.37M 的对称 Deployable Change Adapter（DCA）**固化进学生部署图，使最终模型约 3.28M，推理完全不访问任何大模型。

**教师—数据集适配关系全部由实验决定，不预设结论。** 最终 Agent 候选动作只保留 3–5 个互补且被证明有效的 Teacher Package + None。

## 实验约定（固定）

- **永远不做多 seed**：每个机制只跑一组同协议（seed 2333、batch 64、**40000 steps**）消融表；不补 seed、不做显著性检验。
- **从头训练纪律**：主实验与全部消融一律从头训练（同一 ImageNet 预训练 `lwganet_l0_e299.pth` + 固定 seed），禁止用任何已有 checkpoint 微调/续训作为实验组。
- 唯一变量：C0（clean A2Net）/ C1（C0+DCA）对照 + TV-*（C1+教师包）/ M1 主实验。
- 教师效用 = `TV-* − C1`，逐数据集、逐指标报告；正式结果只读 `train_log.txt` 最后一个完整 `=== TEST RESULTS ===` 区块。

## Run1 结果（teacher_adaptation，44/44 完成）

教师能力矩阵 `U[D,T] = TV-* − C1`（F1 百分点，seed 2333 / 40K / batch 64）：

| 数据集 | C0 F1 | C1 F1 | SAM | D2 | D3N | D3S | RCLIP | MARS | ANYSAT | UNISAT | RADIO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SYSU | 82.13 | **82.57** | −0.28 | −0.12 | −0.24 | −0.67 | −0.81 | −0.03 | −0.66 | −0.27 | **+0.68** |
| WHU | 92.66 | **94.04** | −1.15 | −0.36 | −0.21 | +0.08 | +0.08 | **+0.56** | −0.13 | **+0.40** | −0.32 |
| CDD | 97.79 | **97.75** | −0.01 | −0.02 | −0.05 | −0.04 | −0.01 | −0.02 | −0.01 | +0.00 | −0.01 |
| LEVIR | 90.92 | **91.04** | +0.06 | −0.22 | +0.08 | +0.14 | −0.07 | −0.06 | +0.06 | −0.18 | −0.00 |

部署参数：C0 = 2,913,094（2.7676G FLOPs）；C1/TV-* = 3,280,274（3.2432G FLOPs）；swap/deploy 误差 = 0。

**结论（单 seed，不构成定稿结论）**

1. **DCA 是主增益来源**（C1−C0）：WHU +1.38 / SYSU +0.44 / LEVIR +0.12 / CDD −0.04（H3 初步成立）。
2. **H1 成立**：最优教师随数据集变化（SYSU→RADIO，WHU→MARS/UNISAT，LEVIR→D3S，CDD→无）。
3. **Strong Positive（ΔF1 ≥ +0.20 且 ΔIoU > 0）仅 3 组**：RADIO→SYSU (+0.68/+0.99)、MARS→WHU (+0.56/+1.01)、UNISAT→WHU (+0.40/+0.71)。
4. **教师效用强数据集依赖、多数为负迁移**（36 组中 30 组非强正）→ 实证支持「带 None/拒绝动作的选择 Agent」这一核心论点。
5. CDD 已饱和（97.75），所有教师均无增益。
6. **硬目标未达**：SYSU 83.25/85、LEVIR 91.17/92.5、WHU 94.60/95、CDD 97.76/98。

## Stage 3/4（Registry + LODO Agent + M1）

- **Registry**（`outputs/CATA-CD/registry/`）：`research_report.json`（test 效用，仅论文证据）与 `agent_train_registry.json`（**仅 val 效用**，Agent 输入，无 test 泄漏）严格分离；另有 4 个 dataset signature + 36 个 teacher train-only probe。
- **Agent（A-LODO：tiny MLP + LODO，state = dataset signature + teacher probe）选择**：SYSU→None、WHU→None、CDD→dinov3_lvd、LEVIR→anysat。
- **H2 基本成立**：2/4 fold 与 oracle-val 精确一致，3/4 fold gap ≤0.20pp（WHU 0.221pp 略超）；明显优于基线 A-F（固定教师，gap 最高 +1.84pp）与 A-H（启发式，+0.86pp）。
- **M1 结果**（按 Agent 选择从头训练，2-way）：SYSU 82.76 / WHU 93.01 / CDD 97.74 / LEVIR 90.99 → `M1 − C1` = [+0.19, −1.03, −0.01, −0.05] → **H4 不成立**。

## ⚠️ 关键方法论发现：单 seed 结果不可复现（噪声 > 效应）

M1/SYSU 与 M1/WHU 的配置与 C1 **逐字段完全相同**（`dca_mode=moe128`、`teacher_package=none`、`seed=2333`、`batch=64`、`40K`、同一预训练权重），但结果分别为 **82.76 / 93.01**，而 C1 为 **82.57 / 94.04** → **同配置差异达 +0.19 / −1.03pp F1**。

- 训练曲线**从第 0 个 epoch 即分叉**（SYSU epoch0 loss 3.5051 vs 3.4897；WHU 5.2305 vs 5.2067）。
- 原始 C1 以 **3-way 并发**运行、M1 以 **2-way** 运行；并发度不同（`nn.SyncBatchNorm` backward 归约 + cuBLAS/cuDNN 算法选择 + best-val checkpoint 选择放大）是主要嫌疑。
- WHU 上"同配置"run 的 test F1 散布达 **1.9pp**（C0 92.66 … TV-MARS 94.60）。

**后果**：单 seed 的 run-to-run 波动与教师效应**同量级**，因此上面的能力矩阵、Agent 选择与 M1 差异**不能直接当作机制有效性的证据**，必须先测出每个数据集的 σ。

### 噪声底结果（已完成：C1 配置 × R1/R2/R3，12 runs，均 3-way，与 C1 同并发，test F1）

| 数据集 | C1 | R1 | R2 | R3 | **σ_3way** | 极差 |
|---|---:|---:|---:|---:|---:|---:|
| SYSU | 82.57 | 82.22 | 82.91 | 82.42 | **0.289** | 0.684 |
| WHU | 94.04 | 94.64 | 93.42 | 93.63 | **0.537** | 1.220 |
| CDD | 97.75 | 97.75 | 97.76 | 97.73 | **0.013** | 0.031 |
| LEVIR | 91.04 | 90.69 | 91.13 | 91.07 | **0.195** | 0.432 |

平均 σ = **0.259pp**，**随数据集强烈变化**（CDD 0.013 ↔ WHU 0.537）；并发度只在 WHU 上再加一点（σ_mixed 0.623）→ 主因是**内在 run-to-run 噪声**，不是并发。

**效应/σ 重新判读**：DCA→WHU **2.6σ**、RADIO→SYSU **2.4σ**（接近显著）；DCA→SYSU 1.5σ、MARS→WHU 1.0σ、UNISAT→WHU 0.7σ、D3S→LEVIR 0.7σ（均在噪声内）；CDD 全部 ~0（确认真无效应）。

→ **结论修正**：并非"一切皆噪声"——SYSU 的 RADIO 效应与 DCA 的效应超出噪声，而 **WHU 上的教师效应全部不可测**。"教师是否有效"是**部分可测、部分不可测，且依数据集而定**。

下一步：把这份 σ 证据交给网页 GPT 改进 Agent 结构（prompt 见 `docs/temporary/网页GPT_CATA-CD_Agent改进_prompt.md`）。

## 环境（RSML-3）

```yaml
project: LS-Rep_BCD_RSML_3
server: RSML-3
user: yqwang
path: /home/yqwang/projects/LS-Rep_BCD_RSML_3
env: lsrep            # PyTorch 2.14.0+cu132, CUDA 13.2, Python 3.10
gpu: RTX 5090 ×2      # 每卡 32GB, Blackwell/sm_120
```

```bash
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate lsrep
cd /home/yqwang/projects/LS-Rep_BCD_RSML_3
```

## 教师池

| ID | Teacher Package | 权重文件（pre-trained_weights/） | 波次 |
|---|---|---|---|
| T1 | SAM2.1 Hiera-Large | `sam2.1_hiera_large.pt` | Wave-A |
| T2 | DINOv2 ViT-B/14 | `dinov2_vitb14_pretrain.pth` | Wave-A |
| T3 | DINOv3 ViT-B/16 LVD-1689M（neutral） | `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` | Wave-A |
| T4 | DINOv3 ViT-L/16 SAT-493M | `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` | Wave-A |
| T5 | RemoteCLIP ViT-L/14 | `RemoteCLIP-ViT-L-14.pt` | Wave-A |
| T6 | MaRS-Base RGB | `mars_base_rgb_encoder_only.pth` | Wave-A |
| T7 | AnySat | `AnySat.pth` | Wave-B |
| T8 | UniverSat Base | `universat_base.safetensors` | Wave-B |
| T9 | C-RADIOv3-B | `c-radio_v3-b_half.pth` | Wave-B |

教师编码器的最小可加载实现位于 `models/thirdparty/<name>/`；`others/<name>-main/` 仅保留 LICENSE + README（参考）。

## 目录结构

```text
models/
  a2net.py                            A2Net-LWGANet-L0（+ 可选 DCA）
  decoder/deployable_change_adapter.py   DCA（3 专家 × 4 尺度，对称 gate，~0.37M）
  distill/                             cache_v2 / teacher_package / kd / capability_probe
  agent/                               dataset_signature / teacher_registry / offline_bandit
  thirdparty/                          8 个教师 backbone 最小加载
  tools/                               generate_teacher_cache_v2 / audit / smoke / agent 工具
  scripts/train.py                     CATA-CD 训练器（C0/C1/TV-*）
train_scripts/CATA-CD/
  teacher_adaptation/                  数据集适配性验证（C0/C1 + Wave-A 能力矩阵）
  Run1/                                Agent 选教师后从头重跑 M1
analyse/                               extract_metrics.py / models_to_txt.py
others/                                <name>-main/ 精简参考代码
docs/temporary/CATA-CD_v2_...md       完整方案
```

## 部署约束

- 学生 = A2Net-LWGANet-L0；DCA 首版 width=128。
- 部署参数：C0 = 2,913,094；C1 = 3,280,274；**有效推理参数 < 5M**。
- 教师 / Cache / Translator / Agent 全部训练期使用，**不进部署图**。
- `switch_to_deploy()` 为 no-op；时间交换对称 `< 1e-6`；部署前后主预测误差 `< 1e-6`。
- checkpoint 仅支持 **format_version=8**（CATA-CD v2）。

## 训练与结果

```bash
# smoke
bash train_scripts/CATA-CD/teacher_adaptation/smoke_test.sh

# 生成教师 Cache（逐教师，四数据集）
bash train_scripts/CATA-CD/teacher_adaptation/prepare_teacher_cache.sh <teacher_id> <gpu_id>

# 训练（两卡排队；GPU 空闲后）
tmux new -s cata_gpu0 -d 'bash train_scripts/CATA-CD/teacher_adaptation/run_gpu0.sh 0'
tmux new -s cata_gpu1 -d 'bash train_scripts/CATA-CD/teacher_adaptation/run_gpu1.sh 1'

# 结果统计（实验完成后）
python -B analyse/extract_metrics.py          # 写入 docs/experiment_metrics.xlsx
python analyse/models_to_txt.py --cata        # 快照 → docs/temporary/models_and_metrics_CATA-CD_Run1.txt
```

## GitHub 提交流程

```bash
git add models README.md train_scripts/CATA-CD docs analyse others
git status
git commit -m "CATA-CD v2: adaptive teacher agent + DCA"
git push
```

`others/` 只保留精简后的参考代码（供网页 GPT 阅读），不提交其中的权重、`__pycache__/`、`*.pyc`、`work_dirs/`、数据集或完整框架副本。执行 `git commit` 前必须先检查 `git status`；不得提交权重、缓存、数据集、日志或无关改动。
