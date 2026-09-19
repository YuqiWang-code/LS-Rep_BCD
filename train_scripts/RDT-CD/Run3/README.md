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
