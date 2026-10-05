# README.md — LS-Rep_BCD_RSML_3

> Last updated: 2026-10-05
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
