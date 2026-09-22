# FA-SCRD Run1

当前方法 **FA-SCRD**（Failure-Aware Symmetric Change-Relation Distillation）：固定、独立、CD 适配的 DINOv3 ViT-B/16 教师（LoRA Q/V + 轻量 change head）离线生成双时相特征 cache；训练 A2Net-LWGANet-L0 时只做「对称变化关系 KD」+「failure-aware 软加权」。教师/cache/router/KD 全部训练期，推理时拆除，部署仍是 2,913,094 参数学生。

## 实验矩阵（最小消融，seed 2333）

| ID | 唯一变量 | 目的 |
|---|---|---|
| C0 | 无（clean A2Net） | clean anchor |
| C1 | C0 + joint temporal BN | 排除归一化/BN 混杂 |
| A1 | C1 + 固定教师 relation KD（均匀权重，无 failure weighting） | 独立教师本身是否有效 |
| M1 | A1 + failure-aware 软加权 | 完整 FA-SCRD |

成功判据：`M1 > C1` 于 SYSU 和 WHU。失败判据：A1 失败=教师迁移无效；A1 成但 M1 败=删 router；只单数据集提升=数据集特化 trick。

## 脚本构成

```text
prepare_teacher_cache.sh    教师微调（SYSU）+ 生成 SYSU/WHU 教师 cache（训练前跑一次）
run_gpu0_sysu_whu.sh        Phase-1 训练队列：SYSU→WHU（各 C0→C1→A1→M1）
```

## 路径约定

```text
教师 cache：   /share_datasets/CD_teacher_cache/FA_SCRD_DINOv3_CD/<数据集>/train/<sample>.pt
教师 checkpoint：/share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/FA-SCRD/teacher/<数据集>/teacher_checkpoint.pth
学生 checkpoint：/share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/FA-SCRD/Run1/<EXP>/<DS>/
学生日志：      /home/yqwang/outputs/LS-Rep_BCD_RSML_3/FA-SCRD/Run1/<EXP>/<DS>/train_log.txt
DINOv3 权重：   /home/yqwang/projects/LS-Rep_BCD_RSML_3/pre-trained_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

## 协议（固定）

batch 64、40000 steps、seed 2333、lr 5e-4、wd 1e-4、四尺度 batch BCE+Dice 主损失。KD：`lambda_KD=0.5`、`lambda_mid=lambda_deep=1.0`、`tau=0.07`、`tau_a=0.5`。正式指标只读 `train_log.txt` 最后一个完整 `=== TEST RESULTS ===` 块。
