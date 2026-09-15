# RDT-CD Run2 — SCGR 有效性最小消融

方向 C 主方法在 Run1 之后迭代为 **RDT-CD + SCGR（Sparse-Change Gradient-Concordant Routing，稀疏变化梯度一致路由）**。学生模型仍是 A2Net-LWGANet-L0（部署参数 2,913,094 不变），SCGR 只是训练期机制，`switch_to_deploy()` 删除全部辅助。

## 与 Run1 的差异

Run1 的纯逐像素 Brier 增益路由在稀疏变化数据集（WHU/LEVIR）上 image 级整图拒绝率 55%~75%、KD 信号被饿死；且 KD 梯度与 GT 梯度频繁反相。SCGR 针对性地做了三处机制改动：

1. **区域路由**（region routing）：路由粒度从逐像素改为固定 16×16 区域，避免稀疏有用变化区域被 H×W 图像面积淹没。
2. **误差质量归一化**（error-mass KD）：按区域误差质量而非图像面积加权蒸馏。
3. **梯度一致门**（gradient-concordance gate）：教师提案只有在区域级分析分类器梯度与 GT 一致（正 concordance）时才被准入。

核心代码：`models/distill/dynamic_teacher.py` + `models/distill/task_space.py`（`implementation_version=rdt_scgr_v2`，`scgr_region_size=16` 固定，不暴露为 CLI 超参）。

## 消融设计（4 组）

| 组 | 实验 ID | 语义 | 证明的问题 |
|---|---|---|---|
| G1 | B0 | clean baseline，无教师 | anchor（下界） |
| G2 | D1NG | RDT + Cache + 区域/误差质量路由，**关闭梯度一致门** | 与 D1 对照 → 梯度一致门的贡献 |
| G3 | D1 | RDT + Cache + 区域/误差质量路由 + 梯度一致门（**完整 SCGR**） | D1 vs B0 = 整体有效性 |
| G4 | D2 | RDT + **No Cache** + 完整 SCGR（容量匹配负对照） | D1 vs D2 = Cache 先验的增量 |

## 协议

batch 64、40000 steps、seed 2333、lr 5e-4、wd 1e-4、backbone lr 1.0、四尺度 batch BCE+Dice GT 主损失。D1NG/D1/D2 额外 `kd_lambda=0.06`；教师超参用 train.py 默认（teacher_lr=5e-4、teacher_hidden=24、teacher_ema=0.99、max_logit_delta=2.0）。`implementation_version=rdt_scgr_v2`。

## 目录与保存（本 Run 起简化，不再有 steps/seed 子目录）

- checkpoint 根：`/share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/RDT-CD/Run2`
- 日志根：`/home/yqwang/outputs/LS-Rep_BCD_RSML_3/RDT-CD/Run2`
- 相对叶目录：`<实验>/<数据集>/`（`train_log.txt` 直接落在这里；`last_checkpoint.pth` / `best_model_F1=*.pth` 也在这里）

> `steps_40000` / `seed_2333` **不再作为中间目录**，它们通过 `--max_steps 40000 --seed 2333` 写入训练日志的配置头。后续实验沿用此约定（仅做有效性证明，不需要在路径里再细分 steps/seed）。

## GPU 与执行顺序（双卡并行）

- **GPU 0**：SYSU + WHU（8 个 run，`run_gpu0_sysu_whu.sh`）
- **GPU 1**：CDD + LEVIR（8 个 run，`run_gpu1_cdd_levir.sh`）
- 每卡内部串行 **B0 → D1NG → D1 → D2**；两卡可同时启动。

启动前必须完成（服务器 `lsrep` 环境）：

```bash
source /home/yqwang/miniforge3/etc/profile.d/conda.sh && conda activate lsrep
cd /home/yqwang/projects/LS-Rep_BCD_RSML_3

# 1) smoke（合成契约测试，覆盖 D1 默认 / D1NG / D2 三条路径）
python -m models.tools.smoke_dynamic_teacher --device cuda --gpu_id 0
python -m models.tools.smoke_dynamic_teacher --device cuda --gpu_id 0 --no-gradient-gate
python -m models.tools.smoke_dynamic_teacher --device cuda --gpu_id 0 --no-cache-conditioning

# 2) dry run（真实数据 + 真实 Cache 的短训练，验证完整数据/缓存/SCGR/checkpoint 链路）
#    用少量 max_steps 跑 D1/SYSU（最复杂路径），结果落临时目录，不进正式 Run2 目录。
```

启动队列（各自一个 tmux）：

```bash
tmux new -s rdtcd2_gpu0 -d 'cd /home/yqwang/projects/LS-Rep_BCD_RSML_3 && bash train_scripts/RDT-CD/Run2/run_gpu0_sysu_whu.sh'
tmux new -s rdtcd2_gpu1 -d 'cd /home/yqwang/projects/LS-Rep_BCD_RSML_3 && bash train_scripts/RDT-CD/Run2/run_gpu1_cdd_levir.sh'
```

脚本自带「跳过已完成 + 自动 `--resume`」，中断后重新执行队列即可续跑。batch 64 若 OOM 可同比例下调并在新目录重跑。

## 判定

- **D1 vs B0**：完整 SCGR 的整体有效性（F1/IoU/Kappa 提升）。
- **D1 vs D1NG**：梯度一致门的净贡献（唯一变量是门开关）。
- **D1 vs D2**：Cache 先验是否有增量（唯一变量是 Cache 条件）。
- 关键健康信号（看 D1 日志）：`scgr_region_accept_ratio` / `scgr_region_reject_ratio`（区域级准入是否被喂饱）、`scgr_positive_concordance_ratio` / `scgr_gradient_conflict_ratio`（梯度一致门是否在起作用）、`scgr_effective_per_error_mass`（教师贡献/学生剩余误差，不应过早坍缩）、`teacher_ema_update_norm` / `teacher_target_gap`（教师是否持续演化）。
- 单 seed（2333）不得称普适；论文级主比较 seed `2333/3407/5871`。
