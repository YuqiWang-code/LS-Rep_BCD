# CATA-CD v2 — Run4（第四轮：PSFB 训练期特权空间特征桥）

对应 `docs/temporary/CATA-CD v2 第四轮完整中文研究报告.md`。
**创新点不变**：九个 foundation 教师 + 按「教师—数据集适配性」选师的 Agent。
本轮只改**训练期知识通道**：把教师证据接到**真实 Decoder 的训练期影子前向**上，
而不是 Run3 那种只作用于辅助 1 通道 critic 的间接监督。

---

## 0. 机制（一句话）

在 DCA 后、Decoder 前的 **c3（stride 8，64×32×32）**插入训练期残差旁路：

```
Z = GN(C3);  U = Resize32(P([E, M]));  Ũ ∈ {U, 0, RollBatch(U,1)}
Δ = R([Z, Ũ]);  H = C3 + α·Δ;   α = 1 固定
L = L_main(C2,C3,C4,C5) + λ_s·L_shadow(Decoder_θD,tempBN(C2, H_T, C4, C5))
                        + λ_f·L_feat(H_0, sg(H_T))
```

- **主预测路径完全不接教师**；影子前向共享 Decoder **权重**、但不写主路径的 BatchNorm buffers。
- 部署仍 **3,280,274 参数 / 3.2432G FLOPs@256²**；bridge 约 **0.0556M** 参数只在训练期。

## 1. 反事实模式（唯一变量）

| 模式 | `Ũ` | 唯一变量 |
|---|---|---|
| `zero` | 0（在**使用处**置零；`P`、`R`、影子前向照常执行） | 有无教师内容 |
| `shuffle` | 同 batch 内别的样本（`roll` 1 位，无 RNG） | 配对关系（幅值分布不变） |
| `real` | 同图配对 | — |
| `NOFEAT` | = `real` 但 `λ_f = 0` | 64 通道特征直传是否有贡献 |

## 2. 相对 Run3 CTP-Critic 的实质区别

| | Run3 CTP | Run4 PSFB |
|---|---|---|
| 注入位置 | c2（stride 4） | **c3（stride 8）** |
| 教师去向 | 辅助 1ch critic 的 logits | **真实 Decoder 的影子前向输入** |
| 监督 | `L_task + L_pri`（1ch BCE） | `L_shadow`（四尺度 BCE+Dice，与主损失同尺度） |
| 迁移项 | `KL(sg σ(l_pri/τ) ‖ σ(l_nul/τ))·τ²`（1ch） | **`L_feat`：64 通道逐像素余弦** |

## 3. 相对方案的**两处有意偏离**（已在代码 docstring 写明）

1. **教师无关的初始化**。方案说"某教师的 ZERO 不能当另一教师的 CTRL，因为 `P` 参数量与 RNG 流不同"——判断对，但**根因可消除**：`P` 首层形状随教师通道数变化，构造它消耗的全局 RNG 数不同，于是**之后构造的共享 `R` 在不同教师间初始化不同**。本实现把 `P`/`R` 用**独立、与教师无关的 generator** 取权重，并在构造期间**保存/恢复全局 RNG 状态**，使 `R` 与后续一切在教师间逐位一致（`bridge_init_fingerprint` 可验证）。各教师仍配自己的 ZERO 作为保险。
2. **塌缩判据**。`L_feat` 的梯度**只经 `H_0` 回到 `R`**，因此 `U` 那部分权重从这条损失收不到梯度（单测 `test_feature_loss_leaves_the_U_slice_of_R_ungradiented` 固定了这个事实）。日志每 epoch 输出 `||HT-H0||`：**若它早期塌到 ~0，说明 `R` 已学会忽略 `U`，`L_feat` 无物可传 ⇒ 判失败**，不得当成收敛。

## 4. 判据（报告 §6.2，**是工程判读尺度，不是显著性检验**）

| 数据集 | σ（历史同配置） | 1 次/臂可分辨尺度 | 本轮主判据 |
|---|---:|---:|---|
| **LEVIR** | 0.193 | 0.546pp | `REAL−ZERO ≥ +1.00pp` 且 IoU 同向；配对特异性另需 `REAL−SHUFFLE > +0.546pp` |
| CDD | 0.011 | 0.20pp（约定地板） | `REAL−ZERO ≥ +0.20pp` 且 IoU 同向（近饱和域不要求 1pp） |
| SYSU | 0.478 | 1.353pp | 单次 `REAL−ZERO ≥ +1.35pp` |
| WHU | 0.561 | 1.587pp | 单次 `REAL−ZERO ≥ +1.59pp` |

**教师"可区辨"**的额外要求：至少两名教师**各自的** `REAL−ZERO` 之差超过相应尺度。

## 5. 使用

```bash
source /home/yqwang/miniforge3/etc/profile.d/conda.sh && conda activate lsrep
cd /home/yqwang/projects/LS-Rep_BCD_RSML_3

bash train_scripts/CATA-CD/Run4/selftest_ids.sh          # ID 解析 + cache 就绪检查（无需 GPU）
python tests/test_privileged_spatial_bridge.py           # 15 个单测
bash train_scripts/CATA-CD/Run4/smoke_psfb.sh 0 dinov2 LEVIR 6 64   # 真实 cache 冒烟
bash train_scripts/CATA-CD/Run4/launch_2x.sh             # 低 σ 筛选排期（每卡 2 进程）
bash train_scripts/CATA-CD/Run4/launch_2x.sh single      # 显存不足时回退每卡 1 进程
bash train_scripts/CATA-CD/Run4/monitor.sh
```

实验 ID：`D4-<DS>-<TEACHER>-<MODE>`，`DS∈{L,C,S,W}`、`TEACHER∈{D3S,D2,D3L,RAD,MARS,SAM2}`、`MODE∈{Z,R,S,NF}`。
例：`D4-L-D3S-Z`、`D4-L-D3S-NF`、`D4-S-RAD-S`。

## 6. 低 σ 筛选排期（11 run，每卡 2 进程 ⇒ 4 槽）

同域的 CTRL 与 EXP **混排、同步启动**（报告 §6.4 要求），避免"CTRL 独占跑、EXP 挤在拥堵阶段"。

| 队列 | GPU | 臂 |
|---|---|---|
| r4g0a | 0 | `D4-L-D3S-Z` → `D4-L-D3S-S` → `D4-C-D3S-Z` |
| r4g0b | 0 | `D4-L-D3S-R` → `D4-L-D3S-NF` → `D4-C-D3S-R` |
| r4g1a | 1 | `D4-L-D2-Z` → `D4-L-D2-R` → `D4-C-D3S-S` |
| r4g1b | 1 | `D4-L-RAD-Z` → `D4-L-RAD-R` |

## 7. P0 检查清单

| 项 | 要求 | 状态 |
|---|---|---|
| cache 只读 train、manifest 完整 | `TeacherCacheV3Reader` 对非 train split 直接 `PermissionError` | 单测覆盖 |
| E/M 交换不变、replay 同步 | `apply_cache_state_v3` 拒绝原始时序 taps | 单测覆盖 |
| **影子 Decoder 不污染 BN buffers** | `functional_call` + 克隆 buffers；前后逐位比较 | 单测 `test_shadow_forward_does_not_touch_main_bn_buffers` |
| 影子前向仍给共享权重梯度 | 否则不存在"迁移" | 单测 `test_shadow_forward_still_gives_gradients_to_shared_weights` |
| 主预测 on/off 不变、swap/deploy `<1e-6` | 冒烟已验（四模式全为 0） | 冒烟通过 |
| bridge 不进 `model.state_dict()` | 独立于学生构造 | 冒烟 Infer Params 3.2803M |
| 教师无关初始化 | `bridge_init_fingerprint` 跨教师一致 | 单测 `test_R_initialisation_is_teacher_independent` |
| **禁止静默 resume** | 输出目录非空即拒绝（报告 §7.4） | `run_one` 已实现 |
| checkpoint 格式 | 9 → **10**（新增 bridge），拒绝旧格式 | 已实现 |

## 8. 论文口径（必须写清，否则会被当成训练期泄漏）

> 共享 Decoder 的梯度**会**改变学生 Decoder 训练后的权重——这正是要检验的知识迁移。
> "辅助开关不改变主预测"指的是：**固定同一组学生权重、同一 eval 输入、同一次主预测前向，不依赖教师**；
> **不是**说"有/无辅助训练的两组模型应训练到相同权重"。

## 9. 禁止事项

- 不覆盖 Run1/Run2/Run3 任何产物；`launch_2x.sh` **不提供 `rm -rf` 式 fresh**。
- 不用旧臂 checkpoint 续训当作新臂；`--resume` 仅限**同 ID 同配置**真中断。
- 不读 val/test 教师 cache；不以 test 结果选教师。
- 不把 `L_feat` 的 cosine 包装成"全新相似度创新"；创新只能来自"任务解码器内部的特权反事实旁路及直达表征传递"。
- 不因未过 1pp 就否定机制线；也不因单次正向就宣称成功。
