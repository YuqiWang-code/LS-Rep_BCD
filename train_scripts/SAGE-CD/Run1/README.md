# SAGE-CD Run1

当前方法 **SAGE-CD**（Student-Aware Gated Expert Distillation）第一阶段门实验：**G0/G1/G2 Safe-DINO**。先只做 SYSU + WHU，正面回答「修复 P0、KD 解耦、梯度受控后，单个 DINOv3 教师能否真正给 2.9M 学生正增益」。

## 实验矩阵（seed 2333，batch 64，40000 steps）

| ID | 唯一变量 | 目的 |
|---|---|---|
| G0 | 无（clean A2Net） | clean anchor |
| G1 | G0 + joint temporal BN | 归一化/BN 控制 |
| G2 | G1 + full-res DINO 语义 target + train-only semantic head + logit standardization + gradient budget | Safe-DINO 生死门 |

**Gate**：`G2 − G1 ≥ +0.15 F1` 于 SYSU 与 WHU，且无 Recall/Precision 单边崩塌、`grad_ratio_dino ≤ 0.25`。否则停止 SAGE-CD，不上 SAM2、不做多教师。

## 脚本构成

```text
prepare_sage_cache.sh  生成 SAGE DINO 语义 cache（复用 FA-SCRD 教师，SYSU/WHU）
run_gpu0_sysu_whu.sh   G0/G1/G2 × SYSU + WHU（串行）
```

## 路径约定

```text
SAGE cache：  /share_datasets/CD_teacher_cache/SAGE_DINO3_CD/<数据集>/train/<sample>.pt
学生 checkpoint：/share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/SAGE-CD/Run1/<EXP>/<DS>/
学生日志：    /home/yqwang/outputs/LS-Rep_BCD_RSML_3/SAGE-CD/Run1/<EXP>/<DS>/train_log.txt
DINO 教师 ckpt：/share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/FA-SCRD/teacher/SYSU/teacher_checkpoint.pth
```

## 协议（固定）

batch 64、40000 steps、seed 2333、lr 5e-4、wd 1e-4、四尺度 batch BCE+Dice 主损失。KD：`temperature=1.0`、`rho=0.25`、`lambda_max=1.0`。教师/cache/aux-head/KD 全训练期，部署仍 2,913,094 参数。正式指标只读 `train_log.txt` 最后一个 `=== TEST RESULTS ===` 块。
