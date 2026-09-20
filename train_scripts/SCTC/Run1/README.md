# SCTC Run1

方向 C 换机制后的当前方法 **SCTC**（Unchanged-Aware Symmetric Cross-Temporal Calibration）。学生模型 A2Net-LWGANet-L0，部署参数 2,913,094 不变；**无 Teacher、无 Cache、无 Foundation Model**，SCTC 是学生主图的零参数 temporal 操作，训练/验证/测试/部署使用同一图。

## 实验矩阵（Student-only）

| ID | Difference 前操作 | temporal_calibration_mode | 目的 |
|---|---|---|---|
| S0 | 无（原始 A2Net 直接 `abs(F1-F2)`） | `none` | clean anchor |
| S1 | 全像素对称统计校准 | `symmetric` | 排除「普通 normalization 就够了」 |
| S2 | **Unchanged-Aware SCTC**（agreement 加权统计） | `sctc` | 完整主方法 |

三个臂的区别只在于 `--experiment S0|S1|S2`（对应 `temporal_calibration_mode=none|symmetric|sctc`），其它参数完全一致。

## 训练脚本构成

```text
run_gpu0_whu_sysu.sh    Phase 1：WHU → SYSU（每个数据集 S0 → S1 → S2）
run_gpu0_levir_cdd.sh   Phase 2：LEVIR → CDD（Phase 1 机制门通过后启动）
```

每个脚本内部结构相同：

1. 顶部定义路径变量（checkpoint / log / data / pretrained 根）；
2. `run_one EXP DS DS_FOLDER` 函数负责单个 run：
   - 若 `train_log.txt` 已含 `=== END TEST RESULTS ===` 则跳过；
   - 若存在 `last_checkpoint.pth` 则自动加 `--resume`；
   - S0/S1/S2 都不带任何 cache 参数（SCTC 无教师/Cache）。
3. 底部按 `S0 → S1 → S2` 顺序依次调用 `run_one`。

脚本第一参数可覆盖 GPU，默认 `GPU_ID=0`（当前 GPU 0 空闲，SCTC 极轻量、batch 64 峰值约 9GB）。

## 路径约定

```text
checkpoint: /share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/SCTC/Run1/<EXP>/<DS>/
日志:       /home/yqwang/outputs/LS-Rep_BCD_RSML_3/SCTC/Run1/<EXP>/<DS>/train_log.txt
```

不使用 `steps_40000/seed_2333` 中间目录，`--max_steps 40000 --seed 2333` 写进日志配置头。

## 启动

```bash
source /home/yqwang/miniforge3/etc/profile.d/conda.sh && conda activate lsrep
cd /home/yqwang/projects/LS-Rep_BCD_RSML_3

# 先跑 smoke + dry（部署前检查）
python -m models.tools.smoke_temporal_calibration --device cuda --gpu_id 0

# 正式训练（Phase 1）
tmux new -s sctc1_gpu0 -d 'bash train_scripts/SCTC/Run1/run_gpu0_whu_sysu.sh'
```

## 协议（固定）

batch 64、40000 steps、seed 2333、lr 5e-4、wd 1e-4、backbone lr mult 1.0、四尺度 batch BCE+Dice 主损失。正式指标只读 `train_log.txt` 最后一个完整 `=== TEST RESULTS ===` 块。

## 可证伪假设（第一阶段门）

- **H1 伪变化抑制**：S2 应主要降低背景 FP（Precision↑），Recall 下降 ≤ 0.20；WHU 上要求 ΔF1 ≥ +0.30。
- **H2 稀疏变化特异性**：低变化比 WHU 应比高变化比 SYSU 获益更明显（若再现「SYSU 明显升、WHU 明显降」则停止）。
- **H3 unchanged-aware 必要性**：`|F1(S2)-F1(S1)| < 0.10` 则 unchanged-aware 无独立贡献。
- **H4 多数据集一致性**：4 数据集 ≥3 个正增益，且任一 ΔF1 ≥ -0.20。

## 失败判据

WHU `S2 ≤ S0`；或 SYSU 正 / WHU/LEVIR 负（重现教师方向性）；或 Precision↑ 但 Recall 明显↓ 致 F1 无提升；或 `S2≈S1`；或部署参数 ≠ 2,913,094；或 256×256 FLOPs 超出 2.7475G ± 0.03G。任一成立即停止扩展，不再加模块。
