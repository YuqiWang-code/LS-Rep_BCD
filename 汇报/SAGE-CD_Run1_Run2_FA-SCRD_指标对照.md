# SAGE-CD（Run1 / Run2）与 FA-SCRD 指标对照

> 数据来源：`docs/experiment_metrics.xlsx`（每个 `train_log.txt` 的最后一个完整 TEST RESULTS 块）。
> 均为 seed 2333 / batch 64 / 40000 steps 单次实验，非定稿论文结论；指标已四舍五入到两位小数。
> 参数量 = 部署参数量（Infer Params，单位 M）；FLOPs = 256×256 部署 FLOPs（单位 G）。

## 行说明

| 行 | 实验 ID | 开关（config 摘要） |
|---|---|---|
| clean anchor | SAGE-CD Run2 R0（无教师/无 cache） | joint_bn=False, teacher=False |
| Run1 最完整 | SAGE-CD Run1 G2（Safe-DINO） | joint_bn=True, teacher=True |
| Run2 最完整 | SAGE-CD Run2 R4（主方法） | joint_bn=True, scf=True, gate=True, teacher=True, routing=True |
| FA-SCRD 最完整 | FA-SCRD Run1 M1（FASCRD） | joint_bn=True, teacher=True（+ failure-aware 加权） |

## 1. SYSU-CD-256

| 实验 | 教师 cache 类型 | 参数量 (M) | FLOPs (G) | Recall | Precision | OA | F1 | IoU | Kappa |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean anchor（Run2 R0） | 无 | 2.9131 | 2.7676 | 80.18 | 86.27 | 92.32 | 83.11 | 71.11 | 78.15 |
| Run1 最完整（G2 Safe-DINO） | DINOv3 语义 full-res change logit（256×256） | 2.9131 | 2.7676 | 80.43 | 84.61 | 91.94 | 82.47 | 70.17 | 77.24 |
| Run2 最完整（R4） | DINO 语义（1/16）+ SAM 结构/边界（1/4）+ 失败类型路由/Reject | 2.9131 | 2.7676 | 77.66 | 86.67 | 91.91 | 81.91 | 69.37 | 76.73 |
| FA-SCRD 最完整（M1） | DINOv3 ViT-B/16 关系（LoRA Q/V + change head） | 2.9131 | 2.7676 | 79.80 | 83.82 | 91.61 | 81.76 | 69.15 | 76.32 |

## 2. WHU-CD-256

| 实验 | 教师 cache 类型 | 参数量 (M) | FLOPs (G) | Recall | Precision | OA | F1 | IoU | Kappa |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean anchor（Run2 R0） | 无 | 2.9131 | 2.7676 | 90.22 | 95.86 | 99.46 | 92.96 | 86.84 | 92.67 |
| Run1 最完整（G2 Safe-DINO） | DINOv3 语义 full-res change logit（256×256） | 2.9131 | 2.7676 | 92.12 | 96.42 | 99.55 | 94.22 | 89.07 | 93.99 |
| Run2 最完整（R4） | DINO 语义（1/16）+ SAM 结构/边界（1/4）+ 失败类型路由/Reject | 2.9131 | 2.7676 | 91.97 | 95.82 | 99.52 | 93.86 | 88.42 | 93.61 |
| FA-SCRD 最完整（M1） | —（未跑 WHU） | — | — | — | — | — | — | — | — |

## 3. CDD-CD-256

| 实验 | 教师 cache 类型 | 参数量 (M) | FLOPs (G) | Recall | Precision | OA | F1 | IoU | Kappa |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean anchor（FA-SCRD C0） | 无 | 2.9131 | 2.7676 | 97.67 | 97.91 | 99.45 | 97.79 | 95.67 | 97.48 |
| Run1 最完整（G2） | —（SAGE-CD 未跑 CDD） | — | — | — | — | — | — | — | — |
| Run2 最完整（R4） | —（SAGE-CD 未跑 CDD） | — | — | — | — | — | — | — | — |
| FA-SCRD 最完整（M1） | DINOv3 ViT-B/16 关系（LoRA Q/V + change head） | 2.9131 | 2.7676 | 97.67 | 97.91 | 99.46 | 97.79 | 95.68 | 97.48 |

## 4. LEVIR-CD-256

| 实验 | 教师 cache 类型 | 参数量 (M) | FLOPs (G) | Recall | Precision | OA | F1 | IoU | Kappa |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean anchor（SCTC S0） | 无 | 2.9131 | 2.7676 | 89.40 | 92.45 | 99.09 | 90.90 | 83.32 | 90.42 |
| Run1 最完整（G2） | —（SAGE-CD 未跑 LEVIR） | — | — | — | — | — | — | — | — |
| Run2 最完整（R4） | —（SAGE-CD 未跑 LEVIR） | — | — | — | — | — | — | — | — |
| FA-SCRD 最完整（M1） | —（未跑 LEVIR） | — | — | — | — | — | — | — | — |

## 覆盖说明

- SAGE-CD（Run1 G0/G1/G2、Run2 R0–R4）只跑了 **SYSU、WHU** 两个数据集。
- FA-SCRD（Run1 C0/C1/A1/M1）只跑了 **SYSU、CDD** 两个数据集（WHU/LEVIR 早停）。
- 因此：SYSU 表 4 行齐全；WHU 表缺 FA-SCRD；CDD 表只有 clean anchor + FA-SCRD；LEVIR 表只有 clean anchor（取 SCTC S0 作为 clean A2Net 代表）。
- clean anchor 在 SYSU/WHU 用 SAGE-CD Run2 R0；CDD 用 FA-SCRD C0；LEVIR 用 SCTC S0（均为 clean A2Net-LWGANet-L0，seed 2333 / batch 64 / 40000）。
