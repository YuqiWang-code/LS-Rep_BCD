# CATA-CD v2 — Run3（第三轮：CTP-Critic 知识通道）

对应 `docs/temporary/CATA-CD v2 第三轮研究分析报告.md`（唯一基线 commit `e6e9827`）。
**创新点保持不变**：九个 foundation 教师 + 按「教师—数据集适配性」选师的 Agent。
本轮只改**训练期的知识通道**，部署图完全不动。

---

## 0. 本轮要回答的唯一问题

> 在**同容量**的辅助结构、**同损失**下，把教师的真实配对证据换成
> 零（`zero`）或错配（`shuffle`），学生主预测会不会变差？

- 只有 `real` 明显胜过 **两个**反事实分支，才可以写「教师特异知识有贡献」。
- `real` ≈ `shuffle` ⇒ 教师的作用更像数据正则，不能写知识迁移创新。
- `real` 胜 `zero` 但不胜 `shuffle` ⇒ 配对信息不必要。
- 辅助 critic 变好但学生 test 不动 ⇒ 教师分支没有形成有效学生梯度。

**判据（机制定位的高价值阈值，不是"不达即取消"的门）**：
LEVIR 上 `real − zero` 优先观察 **≥ +1.0pp F1 且 ΔIoU 同向**，且 `real − shuffle > 0`。

**噪声尺度参考**（历史同配置重复）：SYSU σ=0.478 / WHU 0.561 / LEVIR 0.193 / CDD 0.011 pp
⇒ 单次可分辨下限 1.35 / 1.59 / 0.55 / 0.20 pp。**低 σ 域（LEVIR/CDD）先筛，高 σ 域后确认。**

---

## 1. 机制（训练图 / 部署图）

```
训练图（teacher 模块绝不在 student.forward 内）
 A,B ---> [学生 LWGANet/SWA/TFM/DCA/Decoder] ---> 主 4-scale 预测 ---> L_main(GT)
                   |
                   +-- c2(Z) ----> h(Z + g(Z, 0))  ------------> L_task(GT)   [teacher-free view]
                          |
 teacher v3 Cache(E,M) --> P --> U --> h(Z + g(Z, U)) ---------> L_pri(GT)    [privileged view]
                                                      |
                                    stopgrad --> L_transfer(KL) ---> teacher-free view

部署图：A,B ---> 原封不动 C1 student ---> 主变化 mask
         无 P / g / h / teacher / Cache / Agent
```

- `E = |LN(F_A) − LN(F_B)|`（变化证据）、`M = ½(LN(F_A) + LN(F_B))`（对称场景共性），
  两者都满足 `A↔B` 交换不变 ⇒ 几何重放只需 crop/flip。
- `Z = ψ(c2)`（stride 4 的学生变化特征，**DCA 之后**，即 decoder 实际看到的特征）。
- `L = L_main + λ_a (L_task + L_pri) + λ_t L_transfer`。
- **部署不变**：学生仍 **3,280,274** 参数、**3.2432G** FLOPs（沿用 C1 复测值）；
  critic 参数（本轮 `d=32, z_dim=64` ⇒ **76,033**）只出现在训练账里。

## 2. 反事实模式（本轮的核心对照）

| 模式 | `U` 的取值 | 唯一变量 |
|---|---|---|
| `zero` | 在使用处替换为全零；适配器 `P` **照常前向**、cache 照常读取 | 是否有教师证据 |
| `shuffle` | 取**同 batch 内其它样本**的 `U`（`roll` 一位，无 RNG） | 配对关系（幅值分布不变） |
| `real` | 同图配对的真实 `U` | — |

`zero` 与 `shuffle` 与 `real` **参数量、batch 顺序、损失权重、cache I/O 完全相同**。

## 3. 实验 ID 与排期（方案 §7.1）

| 阶段 | ID | 数据集/教师 | 40K 数 |
|---|---|---|---:|
| 低 σ 筛 | `C3L-Z` / `C3L-R-D3S` / `C3L-S-D3S` / `C3L-R-D2` | LEVIR（D3-SAT、DINOv2） | 4 |
| 低 σ 筛 | `C3C-Z` / `C3C-R-D3S` / `C3C-R-D2` | CDD | 3 |
| 高 σ 确认 | `C3S-Z` / `C3S-R-D3S` / `C3S-R-RADIO` | SYSU | 3 |
| 高 σ 确认 | `C3W-Z` / `C3W-R-MARS` / `C3W-R-SAM2` | WHU | 3 |

两卡排期（每卡单进程，**不因为两卡把 batch64 变成全局 batch128**）：
GPU0 `LEVIR 筛 → SYSU 确认`；GPU1 `CDD 筛 → WHU 确认`。

## 4. 使用

```bash
source /home/yqwang/miniforge3/etc/profile.d/conda.sh && conda activate lsrep
cd /home/yqwang/projects/LS-Rep_BCD_RSML_3

# 0) 单测（不需要 GPU）
python tests/test_cache_v3.py
python tests/test_privileged_critic.py

# 1) 增量生成 v3 cache（绝不写 v2 树）
bash train_scripts/CATA-CD/Run3/build_cache.sh 1 dinov3_sat LEVIR,CDD
bash train_scripts/CATA-CD/Run3/build_cache.sh 0 dinov2      LEVIR,CDD

# 2) 低 σ 筛选（两卡各一个 40K 队列）
bash train_scripts/CATA-CD/Run3/run_screen.sh 0 C3L-Z,C3L-R-D3S,C3L-S-D3S,C3L-R-D2 LEVIR
bash train_scripts/CATA-CD/Run3/run_screen.sh 1 C3C-Z,C3C-R-D3S,C3C-R-D2          CDD

# 3) 高 σ 确认（仅在筛选出现正向信号后）
bash train_scripts/CATA-CD/Run3/run_confirm.sh 0 C3S-Z,C3S-R-D3S,C3S-R-RADIO SYSU
bash train_scripts/CATA-CD/Run3/run_confirm.sh 1 C3W-Z,C3W-R-MARS,C3W-R-SAM2   WHU

# 监控
bash train_scripts/CATA-CD/Run3/monitor.sh LEVIR
```

产物：

```text
/share_datasets/CD_teacher_cache/CATA_CD_v3/<teacher>/<DS>/train/{manifest.json,*.pt}
/share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/CATA-CD/Run3/<EXP>/<DS>/
/home/yqwang/outputs/LS-Rep_BCD_RSML_3/CATA-CD/Run3/<EXP>/<DS>/train_log.txt
```

## 5. P0 检查清单（方案 §2.3 / §7.3）

| 项 | 要求 | 状态 |
|---|---|---|
| P0-A | v3 cache 的 `sample_id`、crop/flip 重放与 GT 网格一致；若存原始 A/B 特征，交换必须同时互换 | `apply_cache_state_v3` **拒绝**原始时序 taps；本 schema 只存对称 E/M（单测覆盖） |
| P0-A | 数据路径必须与 v2 一致（直接读文件、BGR→RGB、/255、教师 mean/std） | 已镜像；`--verify_against` 实测 `max\|diff\|=2.7e-4`（fp16 误差） |
| P0-B | teacher/critic 不挂到 deployable student；`aux on/off` 与 `switch_to_deploy` 前后主预测 `<1e-6` | critic 在学生之外构造，不在 `state_dict` 内；主前向按构造不受影响 |
| P0-B | 训练参数分账（student / aux / total） | 日志输出 `Student/Aux/Train/Infer Params`；CTP 的 76,033 计入 Aux |
| P0-C | 新 CTRL/EXP **同 data iterator / 同 cache 读取 / 同 batch / 同 40K**；CTRL 用 `teacher_input_mode=zero` | `zero` 与 `real` 使用完全相同的 loader 与 cache 路径 |
| P0-C | 不能用"无需 cache 的旧 C1"充当新 EXP 的同期对照 | Run3 的 `zero` 臂即同期 CTRL |
| P1-B | 教师 native grid 不一致（14²/16²/25²/64²） | 已实测；同屏比较时需注意空间信息量差异 |
| P2-A | 全新 EXP 必须写新目录、不借旧 checkpoint | `run_one` 仅在 `TAG=""` 且**同目录**存在 `last_checkpoint.pth` 时 resume；Run3 与 Run2 路径完全隔离 |
| P2-A | `CHECKPOINT_FORMAT_VERSION` 升级 | 8 → **9**（新增 critic 参数），拒绝旧格式 resume |
| 环境 | 本机非 git 仓库；代码以 SFTP 同步 | 已同步 |

## 5.1 并发条件记录（必须随结果一起报告）

- 训练**严格每卡单进程**（方案 §7.2），两卡各跑一条串行队列；
  **不因为两卡就把 batch64 变成全局 batch128**，也**不在一张卡上叠第二个训练进程**
  （否则 CTRL 与 EXP 会处于不同并发条件下，与本项目已记录的并发噪声混淆）。
- 本轮的实际情况：两个 `zero` CTRL 臂（`C3L-Z` / `C3C-Z`）起跑时 GPU **基本独占**；
  之后为了不浪费空闲显存，同一张卡上并行跑了 **cache 生成**（`build_cache.sh`），
  因此**后续臂是在"训练 + cache 生成"的背景下跑的**。
- **后果与处置**：若某个 `REAL − ZERO` 差值落在阈值附近，
  **必须**在相同背景下重测该 CTRL 之后再下结论；不得直接把
  "独占起跑的 CTRL" 与 "带背景负载的 EXP" 相减当作干净对照。
  这一点与方案 P0-C（同期、同迭代器、同 batch）的精神一致。

## 6. 禁止事项

- 不覆盖/删除任何 v2 cache、Run1/Run2 日志或 checkpoint。
- 不把 Run2 的 C1 checkpoint 拿来微调；每个 40K 从同一 ImageNet 权重**从头训练**。
- 不读 val/test 的教师 cache（`TeacherCacheV3Reader` 对非 `train` split 直接 `PermissionError`）。
- 本轮**不重新拟合 Agent**：先把通道问题解决，再谈选师。
- 不用「未达硬目标」终止机制线（硬目标是努力方向，不是必达门）。
- 不把 smoke / 短跑当作机制结论。
