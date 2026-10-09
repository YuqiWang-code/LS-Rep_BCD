# CATA-CD v2 — Run2（J0 可重复性 + J1 机制判决）

对应 `docs/temporary/CATA-CD_v2_负结果严格复盘_下一轮可证伪迭代方案_GitHub最新_20261008.md`
的 §4（阶段 J0/J1）与 §11.3（最终 Gate）。

**Run2 与 Run1/teacher_adaptation 完全隔离**：使用自己的 checkpoint 与 log 根目录，
不覆盖任何既有 run 的产物。

---

## Gate（预注册，先写在这里再看结果）

- **Gate R0（正确性）**：同配置同 seed 的运行必须能复现**初始化权重**与**数据序列**，
  且每份 40K 都要有最后完整 `=== TEST RESULTS ===` 块。达不到 → **不得解释 40K 差值**。
- **Gate R1（teacher knowledge）**：`J1-TASKKD` 相对 `J1-GATE`（同辅助头容量、同训练预算、
  同区域 G、同数据流，仅软目标不同）的 SYSU test **ΔF1 必须 > 0.82pp** 且 ΔIoU 同向，
  并有 train-only 探针证据一致；否则**停止把 teacher selector 作为论文主线**。
  （0.82pp = `max(0.20, 2√2·σ_SYSU)`，σ_SYSU=0.289pp，是工程阈值而非 p<0.05。）
- **Gate R2（structure）**：S-PCG-S2 同协议在四域优于 S1；SYSU/LEVIR/WHU 提升、CDD 不降；
  部署 <5M、swap/deploy <1e−6；未达则拒绝该创新，不新增第四专家。
  **阈值按实测 σ 判读**：Run2 的 `S0-C1` 相对 `J1-C1` 出现 1.33pp 同配置散布，
  使 SYSU 的 σ 由 0.291（n=4）上调至 **0.478（n=6）**，对应 **SYSU 阈值 ≈1.35pp、WHU ≈1.59pp**。
  未超阈值只能说"未检出"，不得声称有效或无效应。
- **Gate R3（硬目标）**：单次同协议公开 test 必须 SYSU≥85、LEVIR≥92.5、WHU≥95、CDD≥98
  （IoU 同向）。未全部达到即仍属**方法探寻阶段**。

---

## J0 结果（已执行，SYSU / WHU，同卡背靠背 2 次）

工具：`models/tools/audit_reproducibility.py` + `run_j0.sh`（200 步；实际每 epoch 187 步）。

### 结论一：初始化与数据流**完全可复现**

| 检查项 | 结果 |
|---|---|
| 初始权重（整模型 + 每个顶层模块组）SHA | **两次运行逐位一致** |
| step-0 batch 指纹（image/label/逗号统计量） | **逐位一致** |
| step-0 RNG 状态 SHA | **逐位一致** |
| step-0 `loss_total` | **逐位一致** |

`CDDataset.__getitem__` 用 `random.seed(seed + epoch*1000003 + idx)` 固定每个样本的增强，
DataLoader 也由固定 generator 驱动，因此 **Gate R0 的"初始化 + 数据序列"两项通过**。

### 结论二：反向传播数值**不可复现**，且**训练是混沌的**

step-0 `grad_norm`（同一步、同输入、同权重）：

| 配置 | run0 | run1 | 绝对差 | 相对差 |
|---|---|---|---|---|
| **A** baseline（TF32 开，`CUBLAS_WORKSPACE_CONFIG` 未设） | 8.4467656 | 8.4483382 | 1.57e−3 | 1.9e−4 |
| **B** A + `CUBLAS_WORKSPACE_CONFIG=:4096:8` | 8.4527022 | 8.4488505 | 3.85e−3 | 4.6e−4 |
| **C** B + 关闭 cudnn/matmul TF32 | 8.4270704 | 8.4270565 | **1.39e−5** | **1.6e−6** |
| **D** C + `use_deterministic_algorithms(True)` | — | — | **运行失败** | — |

- **B 无效** → cuBLAS workspace 不是原因。
- **C 把 step-0 分歧压小 113 倍** → **cuDNN TF32 是主因**（TF32 尾数 ~10 bit，
  与观测到的 ~2e−4 相对差量级一致）。
- **D 直接抛错**：`max_unpooling2d_forward_out does not have a deterministic implementation`。
  LWGANet-L0 的 decoder 使用 `MaxUnpool2d`，**PyTorch 无确定性实现** →
  **本架构无法获得逐位确定性**（`warn_only=True` 只会降级为警告，不解决问题）。

**最关键的一条**：把 step-0 扰动压小 113 倍后，**187 步（1 epoch）后的 loss 差反而更大**：

| 配置 | 187 步后 loss 差 |
|---|---|
| A baseline（TF32 开） | −1.04e−2 |
| C（TF32 关） | **+4.64e−2** |

即 **扰动一旦进入训练就会被指数放大**：起点差 6.6e−7（约 5–6 个 float32 ULP）也能在
一个 epoch 内长成 2% 的 loss 差。WHU 同样从 step-0 的 5.0e−4 长到 187 步的 −7.6e−2。

> **对 Gate R0 的判定**：R0 字面要求的"初始化 + 数据序列可复现"**通过**；
> 但"同配置两次 40K 可复现"**不通过**，且**没有任何可用开关能修复**
> （完全确定性被 `MaxUnpool2d` 阻断）。
> 因此 Run2 的 40K 结果**只能作为单次抽样**解读，**不得**把 <约 1–2pp 的差值
> 归因于方法差异；跨臂比较必须依赖预注册阈值 + 配对/同协议设计，而非"同 seed 即相同"。

---

## J1 设计（唯一变量对照，review §4.3）

诊断教师 **dinov3_lvd** 按 train-only probe 预冻结（SYSU train AUROC≈0.870），
**不是**按 test F1 选的。

| 臂 | dca_mode | teacher | aux_task | 唯一变量 |
|---|---|---|---|---|
| `J1-C1` | moe128 | none | none | clean anchor，无辅助头 |
| `J1-GT` | moe128 | dinov3_lvd | gt | +1ch GT 辅助头（全像素、正负均衡） |
| `J1-GATE` | moe128 | dinov3_lvd | gate | 只在 teacher∩GT 一致区域 `G` 监督，target 仍为 GT |
| `J1-TASKKD` | moe128 | dinov3_lvd | taskkd | 同区域 `G`，target 换成冻结的 teacher 软响应 `q_T` |

`GATE` vs `TASKKD` 是**最干净的对照**（同像素、同容量、仅 target 不同）；
`GT` vs `GATE` 改变了有效样本量，故训练日志每 epoch 额外记录区域占比与正负像素数。

四臂都使用**同一个** teacher cache 以保证**数据流逐位一致**（含 `J1-GT`：
它不使用 `q_T`，但共享同一条增强/批序流，否则"同像素"控制会被增强随机性污染）。

### 部署/容量不变式

- 推理参数恒为 **C1 = 3,280,274**（1ch 辅助头 `Conv2d(64,1,1)` = **65** 参数，训练期独有）。
- `switch_to_deploy()` 后部署 state_dict 不含 aux；`Temporal swap max error` 与
  `Deploy max error` 必须 `<1e−6`（`train.py` 会强制校验）。
- 教师编码器全冻结 + offline cache，`q_T` 是 `detach` 的常数目标（单测已覆盖）。

---

## 使用

```bash
source /home/yqwang/miniforge3/etc/profile.d/conda.sh && conda activate lsrep
cd /home/yqwang/projects/LS-Rep_BCD_RSML_3

# 1) J0 可重复性取证（无 40K，不写 checkpoint）
bash train_scripts/CATA-CD/Run2/run_j0.sh 0 200 2

# 2) J0 开关隔离（定位不确定来源；诊断用，不是生产协议）
bash train_scripts/CATA-CD/Run2/run_j0_switches.sh 0 30 SYSU

# 3) J1 train-only 冻结校准（只读 train，绝不打开 val/test）
bash train_scripts/CATA-CD/Run2/run_calibration.sh 512 SYSU

# 4) J1 四臂 40K（SYSU）
bash train_scripts/CATA-CD/Run2/run_j1_sysu.sh 0
```

产物：

```text
outputs/CATA-CD/Run2/j0/<DS>-<i>.json               # J0 逐 step 取证
outputs/CATA-CD/Run2/j0_switches/<NAME>-<i>.json     # 开关隔离
outputs/CATA-CD/Run2/calibration/<DS>_<T>.json       # 冻结校准（+ .diagnostics.json）
outputs/CATA-CD/Run2/<EXP>/<DS>/train_log.txt        # 正式 40K
/share_datasets/yqwang/checkpoints/.../Run2/<EXP>/<DS>/
```

---

## SYSU/dinov3_lvd 校准诊断（train-only，512 样本）

| 量 | 值 | 含义 |
|---|---|---|
| `native_hw` | 16×16 | teacher cache 空间分辨率 |
| `m_T`, `s_T` | 0.0864, 0.0619 | `q_T = sigmoid((z_T−m_T)/s_T)` |
| `t_plus` | 0.6412 | 变化像素上 `q_T` 的中位数 |
| `t_minus` | 0.6462 | 未变化像素上 `q_T` 的 Q0.90 |
| balanced AUROC(q_T, GT) | 0.8523 | 教师响应对 GT 的整体可分性 |
| changed recall @`t_plus` | 0.5173 | 可靠变化区占真变化的比例 |
| **unchanged leakage @`t_plus`** | **0.0962** | **9.6% 未变化像素被教师判为"可信变化"**（伪变化，支持 H3） |
| AUROC by size | small 0.748 / medium 0.810 / large 0.848 | 小目标更差（支持 H4） |
| **AUROC boundary vs interior** | **0.533 vs 0.861** | **边界≈随机**（强支持 H4） |
| ECE raw → affine refit | 0.3423 → 0.0640 | `q_T` 不是校准概率，但单调重标定后可大幅改善 |

> `t_plus (0.6412) ≤ t_minus (0.6462)`：教师响应在两类之间的分离非常弱，
> "可信变化"的门槛并不比"可信未变化"的门槛高。该量被显式记录，不隐藏。

---

## 禁止事项（与 review §8.1 一致）

- 不覆盖既有 `registry/`、`teacher_adaptation/`、`Run1/` 的日志或 checkpoint。
- 不用 `--resume` 从旧 arm 的 checkpoint 续训当作新 arm；`--resume` 仅用于
  **同实验 ID、同配置**的真正中断恢复。
- 不把 teacher / cache / aux head / agent 接入部署图。
- 不改 seed 2333、不做多 seed 取平均、不把重复均值包装成主结果。
- 不用"2σ/2.4σ 接近显著"这类措辞；σ 是跨 run 的 F1 标准差，不是 `TV−C1` 的 SE。
