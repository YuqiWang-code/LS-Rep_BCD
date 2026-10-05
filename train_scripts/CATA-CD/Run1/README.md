# CATA-CD v2 — Run1（第一次迭代：M1）

在 `teacher_adaptation/` 得到 Teacher Capability Matrix 并完成 LODO Agent 选择后，在本目录
**从头重跑 M1**（不复用任何 capability run 的 checkpoint）。

## M1 定义

- 学生 = A2Net-LWGANet-L0 + DCA（`--dca_mode moe128`）。
- 每个数据集使用 Agent 选择的 Teacher Package（或 None）。
- 清空实验目录 → 同一 ImageNet 初始化 → seed 2333 → 40K → 正式 test。

## 使用

先运行 Agent：

```bash
python -m models.tools.train_teacher_agent \
  --signatures_dir /share_datasets/CD_teacher_cache/CATA_CD_v2/signatures \
  --probes_dir /share_datasets/CD_teacher_cache/CATA_CD_v2/probes \
  --registry_dir /share_datasets/CD_teacher_cache/CATA_CD_v2/registry \
  --output_dir /share_datasets/CD_teacher_cache/CATA_CD_v2/agent
```

再按 Agent 选择，用 `teacher_adaptation/common.sh` 的 `run_one` 重新跑 M1（新 `--experiment_id M1`）。
本目录的 run 脚本在 Agent 结果出来后再补（每次迭代使用 Run1、Run2… 编号）。
