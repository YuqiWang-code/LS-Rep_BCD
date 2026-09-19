# RDT-CD Run3

方向 C 当前方法 **BT-SAM-RDT**（Bi-Temporal Structural SAM Reciprocal Dynamic Teacher）。学生模型 A2Net-LWGANet-L0，部署参数 2,913,094 不变，教师/Cache 仅为训练期辅助。

## 实验矩阵

| ID | 说明 |
|---|---|
| B0 | clean baseline，无 Teacher、无 Cache |
| R3A | BT-SAM structural prior + 单个 Fast/EMA Teacher，关闭 OV 信息 |
| R3 | BT-SAM + OV semantic prior + reliability fusion + 单个 Fast/EMA Teacher（主方法） |

## 训练脚本构成

本目录只有两个队列脚本，每个数据集内部按 `B0 → R3A → R3` 串行：

```text
run_gpu0_sysu_whu.sh    SYSU → WHU（每个数据集 B0 → R3A → R3）
run_gpu1_cdd_levir.sh   CDD → LEVIR（每个数据集 B0 → R3A → R3）
```

每个脚本内部结构相同：

1. 顶部定义路径变量（checkpoint / log / data / SAM+OV cache / pretrained 根）；
2. `run_one EXP DS DS_FOLDER` 函数负责单个 run：
   - 若 `train_log.txt` 已含 `=== END TEST RESULTS ===` 则跳过；
   - 若存在 `last_checkpoint.pth` 则自动加 `--resume`；
   - `B0` 不带 cache 参数，`R3A`/`R3` 带 `--sam_cache_root` 和 `--ov_cache_root`；
3. 底部按 `B0 → R3A → R3` 顺序依次调用 `run_one`。

脚本第一参数可覆盖 GPU，默认 `GPU_ID=1`（当前 GPU 0 显存不足，两队列都跑在 GPU 1）。

## 路径约定

```text
checkpoint: /share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/RDT-CD/Run3/<EXP>/<DS>/
日志:       /home/yqwang/outputs/LS-Rep_BCD_RSML_3/RDT-CD/Run3/<EXP>/<DS>/train_log.txt
```

不使用 `steps_40000/seed_2333` 中间目录，`--max_steps 40000 --seed 2333` 写进日志配置头。

## 启动

```bash
source /home/yqwang/miniforge3/etc/profile.d/conda.sh && conda activate lsrep
cd /home/yqwang/projects/LS-Rep_BCD_RSML_3

tmux new -s rdtcd3_gpu0 -d 'bash train_scripts/RDT-CD/Run3/run_gpu0_sysu_whu.sh'
tmux new -s rdtcd3_gpu1 -d 'bash train_scripts/RDT-CD/Run3/run_gpu1_cdd_levir.sh'
```

## 协议（固定）

batch 64、40000 steps、seed 2333、lr 5e-4、wd 1e-4、四尺度 batch BCE+Dice 主损失。正式指标只读 `train_log.txt` 最后一个完整 `=== TEST RESULTS ===` 块。

## 首轮结果与诊断快照（seed 2333，供排查用）

首轮 12 个 run 已全部完成，正式 test F1（%）：

| Dataset | B0 | R3A | R3 | R3−B0 |
| --- | ---: | ---: | ---: | ---: |
| SYSU | 81.75 | 82.29 | 83.13 | +1.38 |
| WHU | 94.06 | 93.94 | 93.70 | -0.36 |
| CDD | 97.79 | 97.78 | 97.79 | 0.00 |
| LEVIR | 91.13 | 91.03 | 90.98 | -0.15 |

核心矛盾：Teacher 只在变化像素最多的 SYSU 有正增益，在建筑小目标、极不平衡的 WHU/LEVIR 上反而略负。

下面是各 `train_log.txt` **末 epoch（最终状态）** 抽取的诊断量，用于定位该矛盾。字段含义见文末。注：R3A 中 OV 不参与融合（`ov_rel=0`、`fusion=sam`、`conflict=0`），其 `ov_*` 值仅作对照、不代表实际生效先验。

### 先验 / 教师 / 学生质量

| run | aux_raw | sam_brier | ov_brier | fusion_brier | sam_gain | ov_gain | fusion_gain | dyn_gain | dyn_brier | student_abs_err | error_mass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R3/SYSU | 0.0009 | 0.3765 | 0.1485 | 0.1559 | -0.3737 | -0.1456 | -0.1531 | -0.0003 | 0.0031 | 0.0048 | 0.0108 |
| R3A/SYSU | 0.0003 | 0.3765 | 0.2133 | 0.3765 | -0.3736 | -0.2104 | -0.3736 | -0.0001 | 0.0030 | 0.0049 | 0.0109 |
| R3/WHU | 0.0000 | 0.3788 | 0.0455 | 0.0829 | -0.3781 | -0.0448 | -0.0822 | 0.0000 | 0.0007 | 0.0010 | 0.0023 |
| R3A/WHU | 0.0000 | 0.3788 | 0.0420 | 0.3788 | -0.3781 | -0.0413 | -0.3781 | 0.0000 | 0.0007 | 0.0010 | 0.0023 |

### 可靠性 / 支撑 / audit / 实例对应

| run | sam_rel | ov_rel | fused_rel | conflict | support_ratio | reject_ratio | accepted_change | accepted_bg | match_ratio | match_iou |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R3/SYSU | 0.2540 | 0.8540 | 0.5342 | 0.5894 | 1.0000 | 0.9961 | 0.0219 | 0.0054 | 0.6648 | 0.1663 |
| R3A/SYSU | 0.2540 | 0.0000 | 0.2540 | 0.0000 | 0.5672 | 0.9978 | 0.0134 | 0.0038 | 0.6648 | 0.1663 |
| R3/WHU | 0.3615 | 0.8212 | 0.4688 | 0.4723 | 1.0000 | 0.9991 | 0.0123 | 0.0009 | 0.8093 | 0.1885 |
| R3A/WHU | 0.3615 | 0.0000 | 0.3615 | 0.0000 | 0.7812 | 0.9993 | 0.0113 | 0.0007 | 0.8093 | 0.1885 |

### 字段含义

- `aux_raw`：教师 KD 损失原始值（学生接受教师 proposal 的蒸馏损失，乘 `kd_lambda` 之前）。
- `sam_brier` / `ov_brier` / `fusion_brier`：SAM 结构先验 / OV 语义先验 / 融合先验相对 GT 的 Brier 误差（越小越准）。
- `sam_gain` / `ov_gain` / `fusion_gain`：`student_brier − prior_brier`（正值=先验比学生准，负值=先验比学生差）。
- `dyn_gain`：`student_brier − teacher_brier`（≈0 表示教师相对学生无增益）。
- `dyn_brier`：EMA Target Teacher 的 Brier 误差。
- `student_abs_err`：学生 `|p−y|` 均值。
- `error_mass`：`mean(|p−y| × importance)`，`importance` 为 detach 的 difficulty 权重。
- `sam_rel` / `ov_rel` / `fused_rel`：各先验及融合后的 reliability 均值。
- `conflict`：`sam_ov_conflict`，SAM 与 OV 先验的不一致度 `|S−O|`。
- `support_ratio`：`foundation_support_ratio`，融合先验有支撑（`fused_reliability > 0`）的像素占比。
- `reject_ratio`：`pixel_reject_ratio`，GT positive-Brier-gain audit 拒收的像素占比。
- `accepted_change` / `accepted_bg`：被 audit 接受的像素中「变化 / 背景」占比。
- `match_ratio` / `match_iou`：T1/T2 SAM 实例匹配率 / 匹配 IoU。

（字段精确定义见 `models/distill/dynamic_teacher.py` 与 `models/distill/task_space.py`。）
