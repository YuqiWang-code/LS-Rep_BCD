# CATA-CD v2 — teacher_adaptation（数据集适配性验证）

本目录验证「不同 Foundation Model 教师包在四数据集上的正/负迁移」——即方案的 Stage 2
Teacher Capability Matrix。**只跑实验、不预设结论**：哪个教师适合哪个数据集完全由 40K
正式 test 结果决定。

## 实验矩阵（唯一变量，seed 2333 / batch 64 / 40K steps）

| ID | dca_mode | teacher_package | 唯一变量（相对 C1） |
|---|---|---|---|
| C0 | none | none | 干净 A2Net anchor |
| C1 | moe128 | none | C0 + DCA（教师效用的 anchor） |
| TV-SAM | moe128 | sam2 | + SAM2.1-H dense package |
| TV-D2 | moe128 | dinov2 | + DINOv2-B/14 relation package |
| TV-D3N | moe128 | dinov3_lvd | + neutral DINOv3-B/16 LVD |
| TV-D3S | moe128 | dinov3_sat | + satellite DINOv3-L/16 SAT-493M |
| TV-RCLIP | moe128 | remoteclip | + RemoteCLIP-L/14 semantic package |
| TV-MARS | moe128 | mars | + MaRS-Base RGB multiscale package |

Wave-B（仅按能力缺口扩展）：`TV-ANYSAT`(anysat) / `TV-UNISAT`(universat) / `TV-RADIO`(radio)。

**Teacher utility = TV-* − C1**（逐数据集，正式 test block）。

## 执行顺序

```bash
# 1) smoke（可选单教师冒烟）
bash train_scripts/CATA-CD/teacher_adaptation/smoke_test.sh

# 2) 生成教师 Cache（每个教师四数据集；GPU 空闲后）
bash train_scripts/CATA-CD/teacher_adaptation/prepare_teacher_cache.sh sam2 0
bash train_scripts/CATA-CD/teacher_adaptation/prepare_teacher_cache.sh dinov2 0
bash train_scripts/CATA-CD/teacher_adaptation/prepare_teacher_cache.sh dinov3_lvd 0
bash train_scripts/CATA-CD/teacher_adaptation/prepare_teacher_cache.sh dinov3_sat 0
bash train_scripts/CATA-CD/teacher_adaptation/prepare_teacher_cache.sh remoteclip 1
bash train_scripts/CATA-CD/teacher_adaptation/prepare_teacher_cache.sh mars 1

# 3) 训练（两卡排队）
tmux new -s cata_gpu0 -d 'bash train_scripts/CATA-CD/teacher_adaptation/run_gpu0.sh 0'
tmux new -s cata_gpu1 -d 'bash train_scripts/CATA-CD/teacher_adaptation/run_gpu1.sh 1'
```

## 路径约定

```text
教师权重： /home/yqwang/projects/LS-Rep_BCD_RSML_3/pre-trained_weights/
教师 Cache：/share_datasets/CD_teacher_cache/CATA_CD_v2/<teacher_id>/<DS>/train/<sample>.pt
checkpoint：/share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/CATA-CD/teacher_adaptation/<EXP>/<DS>/
日志：      /home/yqwang/outputs/LS-Rep_BCD_RSML_3/CATA-CD/teacher_adaptation/<EXP>/<DS>/train_log.txt
```

## 结果统计

全部完成后：`python -B analyse/extract_metrics.py`（把结果写入 `docs/experiment_metrics.xlsx`），
再 `python analyse/models_to_txt.py --cata`（快照 txt），并更新 README。
