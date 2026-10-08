# LS-Rep_BCD_RSML_3 / SAM-HSD  
## Run3 证据驱动复盘、下一代方法选择与 Run4 可实施实验设计

> **审计日期：2026-09-08**  
> **研究对象：A2Net-LWGANet-L0 + SAM2 Teacher Cache，Run1–Run3 → Run4**  
> **结论口径：只把同一 `train_log.txt` 最后一个完整 `=== TEST RESULTS === ... === END TEST RESULTS ===` 作为正式测试结果。**  
> 文中严格区分 **[直接事实] / [合理推断] / [待验证假设] / [当前无法确认]**。

---

# 1. 执行摘要

1. **Run3 当前最强配方不是 N0 Full，而是 N2 Mixed Signed Control，而且这个结论在 SYSU 与 WHU 两个 Stage1 数据集上一致。**  
   N2 相对 N0：SYSU `F1 +0.9782 / IoU +1.4181 / Kappa +1.1173`，WHU `F1 +0.5797 / IoU +1.0287 / Kappa +0.6032`。因此，当前证据**不支持“严格 even+odd 群分解优于 mixed signed representation”**。

2. **N0 相对 N1 的结果直接说明：把 odd 分支加入 even-only 后，两数据集 F1 都下降。**  
   SYSU：`-0.4867 F1`，核心表现为 Recall `-3.3920`、Precision `+2.7601`；WHU：`-0.2127 F1`。这表明 odd 分支至少在当前 12k、seed 2333 下没有提供互补增益，并可能对召回或优化方向造成干扰。

3. **Run3 最大的实验归因问题不是一个“小细节”，而是 P0 级混杂：`z2_srd` 同时改变了主模型的训练态 BN 前向方式。**  
   当前 `A2Net_LWGANet_L0` 在 `z2_srd` 训练时，将 T1/T2 合并成 `2B` 一次通过 backbone，并再次合并后一次通过 SWA，以使 BN 统计对时相顺序不敏感；`auxiliary_mode=none` 的 clean anchor 没有这条训练路径。因此 Run3 相对 Run1/Run2 clean anchor 的变化，**不能纯归因于 Z2-SRD 辅助监督**。上传快照与当前 GitHub `a2net.py` 在这一点上一致。

4. **现有 auxiliary-toggle=0 并不能排除上述混杂。**  
   当前 toggle test 只比较同一个 `z2_srd` 模型下 `compute_auxiliary=False/True`；两次仍走相同 joint-BN 主路径。因此它证明“计算辅助 loss 本身不改当前 forward prediction”，但没有证明“Z2 模式和 clean 模式的主训练图相同”。

5. **Z2 数学实现本身基本成立。**  
   `Z2PairProjector` 对 ordered pair 做 forward/reverse 后形成 `(f+f_swap)/2` 与 `(f-f_swap)/2`；odd head 无 bias 且接 `tanh`；teacher signed code 在时相交换下反号，mixed target 在 `[0,1]` 中互补。当前 smoke 也覆盖 N0–N4 的交换性质、辅助梯度、主输出交换一致性和部署检查。问题主要不是“群性质写错”，而是**严格群分支是否是正确的 teacher–student 监督语义**。

6. **我判断 Run3 失败模式最值得继续研究的不是“even/odd 权重怎么调”，而是：teacher 的“变化强度可信”并不等于“方向符号可信”。**  
   当前 `trust_change` 根据 SAM quality、coverage 和三个对称变化通道的一致性建立，但 signed local/geometry 的正负方向没有独立可靠度。也就是说，一个像素可以被判定“这里确实有结构变化”，却仍可能无法可靠回答“究竟是 T1→T2 的正方向还是负方向”。这是解释 N0 odd 受损、N2 却更强的一条可证伪机制假设。

7. **Run4 唯一推荐主方法：CR-SRD（Coherence-Routed Structural Residual Distillation，一致性路由结构残差蒸馏）。**  
   不再增加一个新分支，而是保留当前最强 N2 的单共享 mixed head：  
   **方向可靠 → 学 signed direction；方向模糊 → 不强迫错误符号，退化为 sign-free magnitude supervision。**  
   路由依据完全由当前 SAM2 cache 动态派生，不增加 cache、不增加部署结构，理论上也不增加 N2 的训练参数。

8. **CR-SRD 与 Mask-CDKD 的关系应该是“借鉴原则而不是复制机制”。**  
   Mask-CDKD 是 source-free / label-free 单时相 VHR 语义分割蒸馏，使用在线 SAM ViT-L + trainable MMoA/MAE、75% masking、多层 KD 和动态 loss schedule；它不是二时相变化检测，而且学生约 29M / 120G@1024，不能直接迁移到本项目。可借鉴的只有“对于不稳定 teacher supervision，不应一刀切强制对齐”的思想。上传的 16 个 Python 文件已经逐类审查，无需再次补传。

9. **12k 可以继续作为 Stage1 筛选预算，但当前证据不足以称其“训练充分”。**  
   SYSU 12k 对应名义 `64.00` effective epochs；WHU 已是 `129.14`。同样 steps 在两个数据集代表完全不同的数据曝光量。由于没有 Run3 完整训练曲线、最佳 validation step、cap 命中率和末段趋势，当前不能断言欠拟合或应直接上 40k。

10. **Run4 不应完全扁平化目录。**  
    因为正式方案明确包含多预算和 3 seeds。推荐仅保留**一层叶目录**：
    `Run4/<experiment>/<dataset>/s40000_seed2333/`。  
    这比 Run3 的 `steps_*/seed_*` 少一层，同时不会牺牲恢复、并发、幂等和追溯。

---

# 2. 材料完整性、来源等级与版本冲突

## 2.1 已实际读取材料

| 等级 | 材料 | 状态 | 本报告用途 |
|---|---|---|---|
| A1 | `models_and_metrics_SAM-HSD_Run3.txt` | 已完整检索/代码级审查 | 当前 `models/` 权威快照、Run3 实现、统一结果 |
| A1 | `experiment_metrics.xlsx` | 已读取 3 sheets / 44 条完整结果 | Run1–Run3 证据矩阵 |
| A2 | `RSML-3_服务器环境与变化检测数据统一说明.md` | 已审查 | 数据量、GPU、Cache、环境 |
| A3 | GitHub `YuqiWang-code/LS-Rep_BCD` | 已联网核验 | 对当前公开代码做 spot-check |
| B | `models_and_metrics_SAM-HSD_Run2.txt` | 已读取 | 旧服务器趋势与失败模式，不覆盖 Run3 |
| A/B 文献 | `参考文献_all.txt` | 已全文检索 | 本地论文证据 |
| 文献源码 | `mask_cdkd_all_python_code.txt` | 16 个 Python 文件已审查 | Mask-CDKD 实现迁移判断 |
| A1 shell | `Run3_all_shell_scripts.txt` | **本对话附件中不存在** | **无法完成 shell 级精确审计** |

当前 Run3 快照是模型代码与统一指标的最高权威来源；Run2 明确是 archival snapshot，不参与覆盖当前实现。

## 2.2 发现的版本/材料问题

### 冲突 C1：项目实际环境与环境说明的 `cd_base`

**[直接事实]** 环境说明记录的是验证过的通用 `cd_base`：PyTorch 2.14.0+cu132、CUDA 13.2、RTX 5090/sm_120；但当前项目最新约束明确指定运行环境：

```text
/home/yqwang/miniforge3/envs/lsrep
```

因此本报告采用：

```text
硬件、数据集、Cache：以环境说明为依据
Run4 实际 Python 环境：以最新项目约束 lsrep 为依据
```

不得为了 Run4 修改或污染 `cd_base`。

### 冲突 C2：Run3 FLOPs 与 Run1/Run2 不同

Run1/Run2 记录部署 FLOPs `2.7475G`，Run3 记录 `2.7676G`；但三者 infer params 都是约 `2.9131M`，当前部署图也始终是：

```text
Backbone → SWA → TFM → Decoder
```

训练辅助在 `switch_to_deploy()` 中删除。当前 `train.py` 对部署 FLOPs 还允许约 `±0.03G` 容差。

**结论：当前无法确认 0.0201G 差异来自 profiler/环境/算子统计差异还是其他因素；不能据此声称 Run3 增加了部署计算。**

### 冲突 C3：Shell 材料缺失

用户要求按：

```text
current Run3 model snapshot + shell > Excel > env > GitHub > Run2
```

审计，但实际 `/mnt/data` 中没有 `Run3_all_shell_scripts.txt`，当前公开 GitHub 根目录也只有 `models/` 和 `AGENTS.md`。

因此：

- Python 内部 checkpoint/resume/best-test 行为可以确认；
- **shell 的完成跳过、目录构造、并发锁、launcher GPU 映射、shell 恢复参数拼接，目前不能声称已经验证。**

---

# 3. Run1–Run3 完整证据矩阵

Excel 共确认 **44 条正式完整 test 结果**：

```text
Run1 = 18
Run2 = 18
Run3 = 8
```

Run3 只有 N0–N3 × SYSU/WHU，全部 `Stage1 / 12k / seed=2333`；没有 N4 或 40k 正式结果证据。统一快照也明确标注 Run3 当前下载结果为 Stage1 12k。

## 3.1 Run1

|Stage|ID|实验名|数据集|steps|seed|Train P(M)|Infer P(M)|FLOPs(G)|Recall|Precision|OA|F1|IoU|Kappa|
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Full|H0|H0_Clean_Anchor|SYSU|40000|2333|2.9131|2.9131|2.7475|77.8818|86.3832|91.8887|81.9125|69.3660|76.7027|
|Full|H0|H0_Clean_Anchor|WHU|40000|2333|2.9131|2.9131|2.7475|92.0067|96.1101|99.5351|94.0137|88.7036|93.7720|
|Full|H1|H1_Legacy_SAMStruct|SYSU|40000|2333|2.9131|2.9131|2.7475|78.7398|85.5930|91.8607|82.0235|69.5253|76.7747|
|Full|H1|H1_Legacy_SAMStruct|WHU|40000|2333|2.9131|2.9131|2.7475|92.5579|95.3232|99.5246|93.9202|88.5374|93.6729|
|Full|H2|H2_Encoder_Directional|SYSU|40000|2333|2.9208|2.9131|2.7475|80.4645|84.6615|91.9551|82.5097|70.2268|77.2902|
|Full|H2|H2_Encoder_Directional|WHU|40000|2333|2.9208|2.9131|2.7475|91.6686|95.9332|99.5153|93.7524|88.2396|93.5004|
|Full|H3|H3_Decoder_SCGR|SYSU|40000|2333|2.9131|2.9131|2.7475|83.1638|83.7282|92.2181|**83.4450**|**71.5929**|**78.3587**|
|Full|H3|H3_Decoder_SCGR|WHU|40000|2333|2.9131|2.9131|2.7475|93.1796|92.6482|99.4360|92.9131|86.7643|92.6195|
|Full|H4|H4_Full_SAM_HSD|SYSU|40000|2333|2.9208|2.9131|2.7475|79.0507|85.3798|91.8673|82.0934|69.6258|76.8424|
|Full|H4|H4_Full_SAM_HSD|WHU|40000|2333|2.9208|2.9131|2.7475|91.6034|96.2725|99.5261|93.8800|88.4658|93.6337|
|Full|H5|H5_Boundary_Scalar|SYSU|40000|2333|2.9208|2.9131|2.7475|79.9723|85.7890|92.1528|82.7786|70.6173|77.7054|
|Full|H5|H5_Boundary_Scalar|WHU|40000|2333|2.9208|2.9131|2.7475|91.8784|94.5719|99.4685|93.2057|87.2759|92.9292|
|Full|H6|H6_Unsigned_Evidence|SYSU|40000|2333|2.9208|2.9131|2.7475|81.4440|84.5518|92.1148|82.9688|70.8946|77.8409|
|Full|H6|H6_Unsigned_Evidence|WHU|40000|2333|2.9208|2.9131|2.7475|91.9860|96.3783|99.5449|**94.1309**|**88.9126**|**93.8943**|
|Full|H7|H7_No_SCGR|SYSU|40000|2333|2.9208|2.9131|2.7475|80.6592|82.2965|91.3470|81.4697|68.7332|75.8260|
|Full|H7|H7_No_SCGR|WHU|40000|2333|2.9208|2.9131|2.7475|92.8701|94.9294|99.5203|93.8884|88.4808|93.6388|
|Full|H8|H8_No_Robust_Filter|SYSU|40000|2333|2.9208|2.9131|2.7475|80.0090|85.7685|92.1548|82.7887|70.6320|77.7161|
|Full|H8|H8_No_Robust_Filter|WHU|40000|2333|2.9208|2.9131|2.7475|92.3736|94.1398|99.4693|93.2483|87.3507|92.9722|

**Run1 数据集内最强：**

- SYSU：H3，F1 `83.4450`
- WHU：H6，F1 `94.1309`

注意这已经表明早期历史实验不存在“一个 Full 配方在两数据集都最强”的稳定现象。

---

## 3.2 Run2

|Stage|ID|实验名|数据集|steps|seed|Train P(M)|Infer P(M)|FLOPs(G)|Recall|Precision|OA|F1|IoU|Kappa|
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Full|R0|R0_Clean_Anchor|SYSU|40000|2333|2.9131|2.9131|2.7475|78.4562|85.9354|91.8912|82.0257|69.5284|76.8044|
|Full|R0|R0_Clean_Anchor|WHU|40000|2333|2.9131|2.9131|2.7475|92.2900|95.6469|99.5275|93.9385|88.5698|93.6927|
|Full|R1|R1_Run1_Unsigned_Reference|SYSU|40000|2333|2.9208|2.9131|2.7475|80.1666|85.7918|92.1917|**82.8838**|**70.7706**|**77.8336**|
|Full|R1|R1_Run1_Unsigned_Reference|WHU|40000|2333|2.9208|2.9131|2.7475|91.4822|96.6042|99.5345|93.9735|88.6321|93.7316|
|Full|R2|R2_Exchange_Invariant_Code|SYSU|40000|2333|2.9214|2.9131|2.7475|79.5999|85.2575|91.9431|82.3316|69.9692|77.1213|
|Full|R2|R2_Exchange_Invariant_Code|WHU|40000|2333|2.9214|2.9131|2.7475|93.3980|95.1066|99.5474|**94.2446**|**89.1156**|**94.0091**|
|Full|R3|R3_Spatial_Residual_Encoder|SYSU|40000|2333|2.9230|2.9131|2.7475|78.9978|85.4711|91.8803|82.1071|69.6455|76.8662|
|Full|R3|R3_Spatial_Residual_Encoder|WHU|40000|2333|2.9230|2.9131|2.7475|92.2615|96.2347|99.5497|94.2062|89.0470|93.9721|
|Full|R4|R4_Residual_Decoder_Head|SYSU|40000|2333|2.9143|2.9131|2.7475|80.8404|84.8582|92.0798|82.8006|70.6493|77.6599|
|Full|R4|R4_Residual_Decoder_Head|WHU|40000|2333|2.9143|2.9131|2.7475|91.9909|93.8861|99.4446|92.9288|86.7916|92.6398|
|Full|R5|R5_Full_EIR_HSD|SYSU|40000|2333|2.9242|2.9131|2.7475|81.5131|83.9851|91.9746|82.7306|70.5475|77.5048|
|Full|R5|R5_Full_EIR_HSD|WHU|40000|2333|2.9242|2.9131|2.7475|91.6992|96.6565|99.5448|94.1126|88.8800|93.8761|
|Full|R6|R6_Fixed_Fusion|SYSU|40000|2333|2.9242|2.9131|2.7475|77.8548|85.6822|91.7094|81.5811|68.8920|76.2478|
|Full|R6|R6_Fixed_Fusion|WHU|40000|2333|2.9242|2.9131|2.7475|91.9688|95.8800|99.5246|93.8837|88.4724|93.6364|
|Full|R7|R7_No_Residual_Correction|SYSU|40000|2333|2.9242|2.9131|2.7475|80.0754|85.6317|92.1326|82.7604|70.5908|77.6712|
|Full|R7|R7_No_Residual_Correction|WHU|40000|2333|2.9242|2.9131|2.7475|91.8783|96.1406|99.5314|93.9612|88.6101|93.7176|
|Full|R8|R8_Directional_Restore|SYSU|40000|2333|2.9242|2.9131|2.7475|78.5457|85.7336|91.8581|81.9824|69.4662|76.7362|
|Full|R8|R8_Directional_Restore|WHU|40000|2333|2.9242|2.9131|2.7475|91.2018|95.6134|99.4849|93.3555|87.5390|93.0878|

**Run2 数据集内最强：**

- SYSU：R1，F1 `82.8838`
- WHU：R2，F1 `94.2446`

再次说明：不存在统一强配方。

---

## 3.3 Run3 Stage1

|Stage|ID|实验名|数据集|steps|seed|Train P(M)|Infer P(M)|FLOPs(G)|Recall|Precision|OA|F1|IoU|Kappa|
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Stage1|N0|N0_Z2_SRD_Full|SYSU|12000|2333|2.9219|2.9131|2.7676|78.4104|86.0520|91.9113|82.0537|69.5687|76.8474|
|Stage1|N0|N0_Z2_SRD_Full|WHU|12000|2333|2.9219|2.9131|2.7676|91.9874|95.1676|99.4968|93.5505|87.8825|93.2887|
|Stage1|N1|N1_Z2_Even_Only|SYSU|12000|2333|2.9219|2.9131|2.7676|81.8024|83.2919|91.8387|82.5404|70.2714|77.2158|
|Stage1|N1|N1_Z2_Even_Only|WHU|12000|2333|2.9219|2.9131|2.7676|92.0743|95.5153|99.5140|93.7632|88.2587|93.5105|
|Stage1|N2|N2_Mixed_Signed_Control|SYSU|12000|2333|2.9214|2.9131|2.7676|81.0155|85.1512|92.1912|**83.0319**|**70.9868**|**77.9647**|
|Stage1|N2|N2_Mixed_Signed_Control|WHU|12000|2333|2.9214|2.9131|2.7676|92.5799|95.7332|99.5419|**94.1302**|**88.9112**|**93.8919**|
|Stage1|N3|N3_No_Group_Projection|SYSU|12000|2333|2.9214|2.9131|2.7676|78.0448|87.1413|92.1065|82.3426|69.9851|77.2796|
|Stage1|N3|N3_No_Group_Projection|WHU|12000|2333|2.9214|2.9131|2.7676|91.9427|94.9603|99.4867|93.4271|87.6651|93.1602|



---

## 3.4 Run3 核心 absolute delta

### N0 Full − N1 Even Only

|Dataset|ΔRecall|ΔPrecision|ΔOA|ΔF1|ΔIoU|ΔKappa|
|---|---:|---:|---:|---:|---:|---:|
|SYSU|-3.3920|+2.7601|+0.0726|**-0.4867**|-0.7027|-0.3684|
|WHU|-0.0869|-0.3477|-0.0172|**-0.2127**|-0.3762|-0.2218|

**[直接事实]** N0 与 N1 在 group topology 下主要差异就是启用 odd head。smoke 中精确训练参数仅从 `2,921,882 → 2,921,930`，即 +48 参数，不足以用容量解释退化。

**结论：当前没有证据证明 odd branch 带来有效互补，反而两数据集都降低 F1。**

---

### N2 Mixed Signed − N0 Full

|Dataset|ΔRecall|ΔPrecision|ΔOA|ΔF1|ΔIoU|ΔKappa|
|---|---:|---:|---:|---:|---:|---:|
|SYSU|+2.6051|-0.9008|+0.2799|**+0.9782**|+1.4181|+1.1173|
|WHU|+0.5925|+0.5656|+0.0451|**+0.5797**|+1.0287|+0.6032|

这是 Run3 最重要的结果。

SYSU 上 N2 相对 N0 的主要变化不是通过“提高 Precision”获得，而是：

```text
Recall +2.6051
Precision -0.9008
F1 +0.9782
```

说明 N0 Full 很可能过度抑制了一部分 positive change；N2 恢复了召回，并在整体上得到更好的 Precision–Recall 平衡。

WHU 则 Recall/Precision 同时上升，因此不是单一 threshold trade-off。

---

### N2 Mixed Signed − N1 Even Only

|Dataset|ΔRecall|ΔPrecision|ΔOA|ΔF1|ΔIoU|ΔKappa|
|---|---:|---:|---:|---:|---:|---:|
|SYSU|-0.7869|+1.8593|+0.3525|**+0.4915**|+0.7154|+0.7489|
|WHU|+0.5056|+0.2179|+0.0279|**+0.3670**|+0.6525|+0.3814|

相对 even-only，N2 在 SYSU 用一点 recall 换取大量 precision，WHU 则两者同时提升。

因此 N2 的收益不能简单概括为“signed 信息提高召回”，而更像是**signed + shared representation 产生了更合适的判别约束**。

---

### N2 Mixed Signed − N3 Invariant Control

|Dataset|ΔRecall|ΔPrecision|ΔOA|ΔF1|ΔIoU|ΔKappa|
|---|---:|---:|---:|---:|---:|---:|
|SYSU|+2.9707|-1.9901|+0.0847|**+0.6893**|+1.0017|+0.6851|
|WHU|+0.6372|+0.7729|+0.0552|**+0.7031**|+1.2461|+0.7317|

N2/N3 训练参数相同，均约 `2.9214M`，且同用 control-head topology。

因此这是当前对 **signed residual vs invariant absolute residual** 最干净的一组证据之一：

> **在两个数据集上，保留 signed temporal information 都优于完全 abs/invariant control。**

但这并不推出“signed 信息所有位置都可靠”，这正是 Run4 要继续问的问题。

---

## 3.5 与 Run1/Run2 的比较：只能做趋势，不做因果

|比较|SYSU ΔF1|WHU ΔF1|是否可直接归因|
|---|---:|---:|---|
|N2 − Run1 clean H0|+1.1194|+0.1165|**否**|
|N2 − Run2 clean R0|+1.0062|+0.1917|**否**|
|N2 − Run1 数据集最强|−0.4131|−0.0007|**否**|
|N2 − Run2 数据集最强|+0.1481|−0.1144|**否**|

原因至少有两个：

1. Run3 = 12k；Run1/2 = 40k；
2. Run3 `z2_srd` 改了 training-time joint-BN 主路径。

还有一个很有价值的历史噪声参考：

```text
Run2 clean R0 − Run1 clean H0:
SYSU F1 = +0.1132
WHU  F1 = -0.0752
```

即使名字都是 clean、40k、seed 2333，也已有约 `0.08–0.11 pp` 浮动。  
因此论文里不能把 `0.05–0.10 pp` 级别的单 seed 差值包装成稳定方法收益。

---

# 4. Run3 薄弱点：事实、推断、假设、缺失证据

## 4.1 直接事实

### F1. strict even+odd 并没有胜过 simpler controls

结果排序：

```text
SYSU: N2 83.0319 > N1 82.5404 > N3 82.3426 > N0 82.0537
WHU : N2 94.1302 > N1 93.7632 > N0 93.5505 > N3 93.4271
```

N2 唯一实现了**两个数据集同时第一**。

### F2. signed information 有价值，但“严格 odd branch”没有显示价值

这两件事必须区分：

```text
N2 > N3  → signed information 有价值
N0 < N1  → 加 strict odd head 没有价值
```

所以失败的可能不是“方向信息本身”，而是：

> **把方向信息硬拆成独立 odd irreducible branch 并对全部可信 change 像素强监督。**

### F3. N4 缺失使 even/odd 冲突无法闭环

当前无法区分：

```text
假设 A：odd 本身就是弱监督
假设 B：odd 单独有效，但 even+odd 联合训练发生竞争
```

N4 正是区分这两者的最小实验。

---

## 4.2 合理推断

### I1. N0 的 SYSU Recall 损失说明 odd 可能造成 positive suppression

N0 比 N1：

```text
Recall    -3.3920
Precision +2.7601
F1        -0.4867
```

这种形态与“额外 teacher constraint 把一部分困难变化压回 negative”相容。

这不是证明，但比“训练不够”解释更直接，因为 N1 与 N0 使用相同预算。

### I2. N2 的 shared mixed head 可能比显式群分支更容易形成任务相关表示

`N2` 并没有要求学生先学习一个严格 odd latent，再通过独立 bias-free head 输出方向；它直接使用：

```text
signed residual + product
        ↓
shared control head
        ↓
4-channel teacher code
```

因此模型可以联合决定“方向信息在当前任务里该用多少”。

这比强制把 representation 分成 even/odd 两个不可约子空间具有更弱但更任务导向的 inductive bias。

---

## 4.3 Run4 核心待验证假设

### H1：变化存在性可靠 ≠ 方向符号可靠

当前 teacher 的：

```text
trust_change
```

依赖：

- `max(q1,q2)`
- SAM coverage
- boundary/local/geometry 三个**无符号**变化量之间的不确定性

但没有显式回答：

```text
signed residual 的正负号是否可靠？
```

例如 local：

\[
m_l=0.55|\Delta a_1|+0.45|\Delta a_2|
\]

而 signed：

\[
d_l=0.55\Delta a_1+0.45\Delta a_2
\]

若两个 cue 一个正一个负：

```text
m_l 很大：确实有变化
|d_l| 很小：方向互相抵消
```

当前代码仍可以给这个像素较高 `trust_change`，然后让 odd/mixed branch 学一个模糊的方向。

这是 CR-SRD 的直接机制动机。

### H2：N0 中 even/odd 在 global cap 下可能竞争辅助预算

当前：

\[
L_{N0}=L_{even}+L_{odd}
\]

先相加，再进入：

```python
hsd_lambda * factor * aux_raw
→ capped_auxiliary(..., max_ratio=0.12)
```

如果 odd loss 较大且噪声较高，它可能让整体 auxiliary 更频繁撞到 cap，然后共同缩放 even + odd 梯度。

**当前无法证明这一点**，因为没有 Run3 原始 step-level `aux_raw / aux_ratio / z2_even / z2_odd` 完整日志供本次审计。

因此 Run4 应记录 cap hit rate 和梯度诊断，但**不能把改 `0.12` cap 当作方法创新**。

---

## 4.4 当前无法判断

当前材料不足以回答：

- Run3 各实验 best validation step；
- best 是否集中在末端；
- 12k 后 val F1 是否仍显著上升；
- N0 的 odd loss 是否长期大于 even；
- N0 是否高频触发 cap；
- main/odd/even 梯度是否真正冲突；
- N4 是否曾在服务器上运行但尚未回收；
- Run3 shell 是否具有安全的完成跳过/并发锁/精确 resume 逻辑。

---

# 5. 当前代码审查：P0 / P1 / P2

# P0：正确性与实验归因

## P0-1：`z2_srd` 同时改变主训练图——必须补 clean joint-BN control

当前 `a2net.py`：

```python
if self.training and self.auxiliary_mode == "z2_srd":
    merged = backbone(cat(x1, x2))
```

SWA 也一样合并 `2B` 前向。

公开 GitHub 当前版本也明确写明这是为了让 BN statistics independent of temporal ordering。

### 影响

Run3 与 clean anchor 的差值实际是：

\[
\Delta =
\text{Z2 auxiliary}
+
\text{joint temporal BN training}
+
\text{budget/version effects}
\]

不是单独的：

\[
\Delta = \text{Z2 auxiliary}
\]

### 最小验证

新增：

```text
J0_JointBN_Clean
auxiliary_mode = none
joint_temporal_bn = on
```

除此之外与 Run3 N2 完全相同。

这是 Run4 **必须有的 P0 对照**。

---

## P0-2：现有 Auxiliary Toggle 检查存在盲区

现有检查：

```text
same z2 model
compute_auxiliary=False
vs
compute_auxiliary=True
```

结果 0 很好，证明 forward 中真正计算 auxiliary 不回写 prediction。

但 root `model.training=True` 时仍满足：

```text
auxiliary_mode == z2_srd
```

所以两次都进入 joint temporal path。

### 最小改进

Run4 smoke 应同时检查：

```text
A. CR compute_aux False vs True       → exact 0
B. deploy before vs after switch      → <1e-6
C. temporal swap main prediction      → <5e-6
D. clean separate-BN vs clean joint-BN
   → 只记录差异，不要求相等
```

D 是实验因子，不是 correctness assertion。

---

## P0-3：Teacher Cache replay 数据流目前未发现明显错误

当前顺序：

```text
RGB A/B
  ↓
Scale
RandomCropResize
Flip
RandomExchange
  ↓
student

SAM2 cached t1/t2
  ↓ 同一 aug_state replay
Scale
Crop/resize
Flip
Exchange(t1↔t2)
  ↓
derive relational fields
```

`instance_id` 使用 nearest，其余连续结构图使用 bilinear；RandomExchange 后明确交换 `t1/t2`。

**结论：目前没有证据支持存在 A/B/label/cache 几何错位。**

---

## P0-4：部署删除路径正确

当前：

```python
switch_to_deploy():
    del training_auxiliary
    auxiliary_mode = "none"
```

随后：

- deploy consistency；
- inference parameter exact check；
- FLOPs tolerance check；
- deployed test；
- 正式 TEST RESULTS block。

这是当前代码中较完整的一部分。

---

# P1：方法瓶颈

## P1-1：`trust_change` 没有 directional sign reliability

这是最重要的 P1。

现有：

\[
w_{change}=
\max(q_1,q_2)^\gamma
\cdot coverage
\cdot(1-uncertainty)
\]

其中 uncertainty 来源于**无符号** boundary/local/geometry change channels 的不一致。

但：

```text
“有变化吗？”
和
“T2-T1 的符号是什么？”
```

不是同一个置信问题。

Run4 应对此做机制级处理，而不是调 loss weight。

---

## P1-2：even / odd 梯度竞争目前不可观测

代码只记录：

```text
z2_even
z2_odd
z2_even_feature
z2_odd_feature
z2_even_odd_cosine
```

这里的 `even_odd_cosine` 是 latent feature diagnostic，不是：

```text
∇L_even · ∇L_odd
```

因此无法判断优化冲突。

### Run4 低成本诊断

每隔例如 500 optimizer steps，只在一个固定 shared backbone tensor 上运行：

\[
\cos(g_m,g_a)=
\frac{g_m^\top g_a}
{\|g_m\|\|g_a\|+\epsilon}
\]

只记录，不改变 optimizer。

若 CR 方法成功，论文中可用它解释；若无差异，也不影响主方法。

---

## P1-3：N3 的命名容易被误解为“只删 group projection”

N2 vs N3 是比较 signed/mixed vs invariant control，比较干净。

但：

```text
N1 group-even
vs
N3 invariant control
```

同时改变了：

- pair representation；
- Z2 ordered encoder；
- head topology；
- 训练参数量。

因此不能把 N1−N3 单独解释成：

> “group projection 的贡献”。

论文表格应把 N3 描述为：

```text
Invariant abs/product control
```

而不是过度解释为纯 `No Group Projection` component ablation。

---

## P1-4：N4 是必要而非可选的机制闭环

没有 N4，无法区分：

```text
odd target 本身质量差
vs
even/odd 联合优化冲突
```

建议 Run4 Stage1 顺带补 N4，但把它定位成：

> **Run3 mechanism close-out diagnostic**

而不是 Run4 contribution。

---

# P2：实验与复现

## P2-1：checkpoint 只在 epoch boundary 保存

当前 `last_checkpoint.pth` 每 epoch 保存。

中途崩溃会丢失本 epoch 已完成的 optimizer steps。

这不是结果正确性错误，但在 WHU 每个 loader epoch 92 steps、SYSU 187 steps 的情况下，可以接受；若希望更强健，可增加低频 step checkpoint，但不应成为论文贡献。

## P2-2：12k checkpoint 不能直接 resume 成 20k/40k

`validate_resume()` 会核对：

```text
experiment
dataset_name
batch_size
max_steps
seed
```

同时 LR schedule 和 HSD multiplier 都依赖总 `max_steps`。

所以：

> **12k → 20k/40k 必须 fresh run。**

不能把 12k checkpoint 续训到 40k 然后称为“同协议 40k”。

## P2-3：best checkpoint 选择逻辑正确，但日志缺 best_step 汇总

当前按 validation F1 保存 best，再载入 best 做最终 test。

建议 Run4 增加显式：

```text
best_val_f1
best_global_step
best_effective_epoch
```

防止日后只能从大量 epoch log 反推。

## P2-4：Shell 复现行为未完成审计

由于缺 `Run3_all_shell_scripts.txt`，当前无法确认：

- shell 是否读取完整 TEST block 后 skip；
- 多次启动会不会覆盖；
- `--resume` 如何定位；
- physical GPU 1 是否统一映射；
- 是否存在并发碰撞。

这部分将在第 12 节给出 Run4 的安全设计，但属于**设计建议，不是对 Run3 shell 的事实描述**。

---

# 6. 2024–2026 文献与官方代码对照

本地 `参考文献_all.txt` 已先行检索，并额外联网核验论文主页/出版社/官方 GitHub。

|工作|状态 / venue|核心机制|对本项目真正有用的部分|不应直接搬运|来源|
|---|---|---|---|---|---|
|BFD, *Burden-Free Distillation From Foundation Model for Efficient Remote Sensing Change Detection*|IEEE TGRS 2025，已发表|DFM 多层空间关系匹配 + Patch Contrastive Distillation；foundation model 只在训练期使用|“大 teacher 只训练期存在、部署零负担”与本项目高度一致；多层结构关系也相关|在线抽取大模型 feature、直接复制 DFM/PCD 会改变现有离线 cache 叙事| [Official GitHub](https://github.com/Younger-hua/Burden-Free-Distillation)|
|Mask-CDKD|ISPRS JPRS 236, 2026，已发表|SAM ViT-L + MMoA，ViT-S，75% MIM/MAE，多层 KD，single-stage bidirectional adaptation，动态 loss schedule|不稳定 teacher supervision 应进行“适应/放松”，多层信息有互补性|任务是单时相 LULC segmentation；在线 SAM/MAE/MMoA 太重；不是二时相交换问题| [Official GitHub](https://github.com/whujader/mask_cdkd?utm_source=chatgpt.com)|
|MaskSub, *Masking meets Supervision*|CVPR 2025|main branch 正常监督，masked sub-branch 使用 relaxed self-distillation|“困难/受扰动分支不能机械使用同强度硬 target”这一训练原则|随机 RGB masking 本身没有 CD 特异性，直接移植创新性不足| [Official GitHub](https://github.com/naver-ai/augsub?utm_source=chatgpt.com)|
|CrossKD|CVPR 2024|避免 GT 与 teacher prediction 给同一 student head 产生矛盾优化目标|提醒我们区分“teacher supervision 是否与任务目标兼容”，可用于解释 gradient conflict|其 cross-head detector 机制与 CD 不匹配| [Official GitHub](https://github.com/jbwang1997/CrossKD?utm_source=chatgpt.com)|
|DistillSemiCD|IEEE TGRS 2026，已发表|DINOv2 teacher + 可靠区域 confidence refinement + semi-supervised CD|“foundation prior 应只作用于可靠区域”与 CR routing 思想相关|半监督/pseudo-label 流程与当前 fully-supervised + offline SAM cache 不同| [Official repository](https://github.com/Sean1005-x/DistillSemiCD)|
|SEED, *Exchange Is All You Need for Remote Sensing Change Detection*|arXiv:2601.07805，**预印本**|把 feature exchange 解释为参数无关 permutation，并研究时相信息保存|支持“exchange property 本身值得明确建模/验证”的研究方向|尚非同行评审；不能作为 Run4 核心新颖性背书||
|LWGANet|AAAI 2026|轻量 grouped attention，面向遥感空间/通道冗余|支撑当前学生 backbone 的轻量化定位|不解决 teacher supervision reliability||

两个代码开放性细节需要特别说明：

- BFD 的论文明确给出了官方 GitHub，但截至本次检查该仓库只有 README、1 commit，没有可审查的实现文件。因此不能根据仓库反推论文具体实现。
- DistillSemiCD 已有 TGRS 2026 论文，但当前官方仓库仍只有“code will be made publicly available”的 README。

---

# 7. Mask-CDKD 源码专项审查

上传的 `mask_cdkd_all_python_code.txt` 包含 16 个 Python 文件，已经逐类检查，包括 PyTorch 与 LuoJiaNET 两套实现。

## 7.1 实际训练图

```text
Target-domain unlabeled 1024×1024 image
              │
      75% patch masking
              │
      ┌───────┴─────────┐
      │                 │
SAM ViT-L teacher     ViT-S student
frozen backbone      12 transformer blocks
+ trainable MMoA     + 4-layer MAE decoder
+ teacher MAE
      │                 │
T layers 6/12/18      S layers 3/6/9
      └──── Feature KD ─┘
              │
 Teacher MAE reconstruction
 Student MAE reconstruction
              │
 adaptive state schedule
```

源码中的 `MaskCDKDDistiller` 明确：

```text
teacher_layers = (6,12,18)
student_layers = (3,6,9)
```

teacher backbone 冻结，但 MMoA 以及 teacher reconstruction branch 保留可训练参数；teacher 不被整体 `no_grad/stop_gradient` 包裹，因为存在 bidirectional adaptation。

动态 loss 状态大致是：

```text
early  : KD 0.20 / T-MAE 0.40 / S-MAE 0.40
middle : KD 0.60 / T-MAE 0.20 / S-MAE 0.20
late   : KD 0.70 / T-MAE 0.15 / S-MAE 0.15
```



## 7.2 为什么不能直接迁移

本项目与 Mask-CDKD 的本质差异：

|Mask-CDKD|当前 SAM-HSD|
|---|---|
|单时相语义分割 domain adaptation|双时相 binary CD|
|在线 SAM ViT-L teacher|SAM2 离线 cache|
|teacher adapter/MAE 可训练|teacher evidence 无梯度|
|RGB masked reconstruction|结构残差监督|
|ViT-S 约 29M|A2Net-LWGANet-L0 部署 2.913M|
|1024² 约 120G|256² 约 2.75–2.77G|
|源域偏差问题|时相方向/结构可靠性问题|

因此以下内容**否决直接移植**：

```text
MMoA
online SAM teacher
teacher MAE
student MAE decoder
75% RGB patch masking
Mask-CDKD 动态 λ schedule
```

可以迁移的只有抽象思想：

> **在 teacher information 不可靠的位置，不应该以同等方式强制 student 拟合。**

Run4 必须把这一思想重新定义为**二时相变化检测特有的 sign-coherence problem**，否则创新性不足。

---

# 8. Run4 候选方案比较与唯一选择

评分：5 = 最优。

|候选|新颖性|针对 Run3 失败|实现风险低|训练成本友好|零部署开销|可证伪性|论文叙事|
|---|---:|---:|---:|---:|---:|---:|---:|
|**A. CR-SRD：一致性路由 signed/magnitude supervision**|4.5|**5.0**|4.0|5.0|5.0|**5.0**|**5.0**|
|B. even/odd gradient surgery / PCGrad|3.0|5.0|3.0|4.0|5.0|4.0|3.0|
|C. MaskSub-style random structural masking|3.5|3.5|4.0|4.5|5.0|4.0|3.5|
|D. stage-specialized boundary/local/geometry routing|2.5|3.0|4.5|5.0|5.0|4.0|2.5|

## 唯一首选：CR-SRD

**全名：Coherence-Routed Structural Residual Distillation**  
**中文：一致性路由结构残差蒸馏**

一句话贡献：

> **CR-SRD 根据 SAM2 二时相结构残差内部的方向一致性，在像素/结构通道级动态选择 signed-direction supervision 或 sign-free magnitude supervision，使学生只在 teacher 方向可靠时学习时相方向，在方向模糊时仍保留变化强度知识，并在部署时完全删除该机制。**

为什么不是 PCGrad：

- PCGrad 只处理优化症状；
- 更像训练算法；
- 很难形成“为什么 teacher odd 语义失败”的结构故事；
- 即使有效，也不说明 teacher target 本身是否合理。

为什么不是随机 masking：

- 与 MaskSub 太接近；
- 对二时相 CD 特异性弱；
- 不直接解释 N2>N3 和 N0<N1。

为什么不是 stage routing：

- 当前 Run2/Run3 已存在 stage weights/channel routes；
- 容易变成结构组合和权重设计；
- 学术增量不足。

---

# 9. CR-SRD 数学定义、数据流与贡献边界

## 9.1 当前 teacher 结构量

令三个 change structural channels：

\[
k\in\{b,l,g\}
\]

分别为：

```text
b = boundary
l = local relation
g = geometry
```

当前已经有 unsigned magnitude：

\[
m_b \in [0,1]
\]

\[
m_l=
0.55|\Delta a_{r1}|+
0.45|\Delta a_{r2}|
\]

\[
m_g=
0.25|\Delta a_{r4}|
+0.25|\Delta depth|
+0.30|\Delta scale|
+0.20|\Delta compactness|
\]

以及 signed aggregate：

\[
d_b,d_l,d_g\in[-1,1]
\]

其中例如：

\[
d_l=
0.55\Delta a_{r1}
+0.45\Delta a_{r2}
\]

交换时相：

\[
d_k(T_2,T_1)=-d_k(T_1,T_2)
\]

而：

\[
m_k(T_2,T_1)=m_k(T_1,T_2)
\]

---

## 9.2 新增：Directional Coherence

定义：

\[
c_k=
\operatorname{clip}
\left(
\frac{|d_k|}
{m_k+\epsilon},
0,1
\right)
\]

解释：

### `c_k ≈ 1`

组成该 structural channel 的方向 cue 基本一致：

```text
有变化
并且
变化方向可信
```

### `c_k ≈ 0`

多个 cue 的绝对变化都可能很强，但正负相互抵消：

```text
有变化
但是
方向不可信
```

由于：

\[
| \sum_i w_i\Delta_i |
\le
\sum_i w_i|\Delta_i|
\]

所以 local/geometry 的该比例天然有合理范围。

更重要的是：

\[
c_k(T_2,T_1)=c_k(T_1,T_2)
\]

即 coherence 本身是交换不变量。

---

## 9.3 Signed directional target

仍使用当前 N2 定义：

\[
t^{dir}_k=
\frac{1+d_k}{2}
\in[0,1]
\]

交换时：

\[
t^{dir}_k(T_2,T_1)
=
1-t^{dir}_k(T_1,T_2)
\]

---

## 9.4 Sign-free magnitude target

N2 head 对前三个 channel 仍输出：

\[
q_k\in[0,1]
\]

从该 signed-coded prediction 提取：

\[
\hat m_k=
|2q_k-1|
\]

这表示：

> 只关心 student 预测离 0.5 有多远，而不关心它选择正还是负。

---

## 9.5 Coherence-Routed Loss

对每个 change channel：

\[
\ell_k=
c_k
\rho(q_k,t^{dir}_k)
+
(1-c_k)
\rho(|2q_k-1|,m_k)
\]

其中 \(\rho\) 为 Smooth-L1。

再使用当前：

\[
w=w_{trust\_change}
\]

做 weighted mean：

\[
L_k=
\frac{
\sum_x
w(x)\ell_k(x)
}{
\sum_xw(x)+\epsilon
}
\]

stable channel **完全不变**：

\[
L_s=
WM[
\rho(q_s,s),
w_{trust\_stable}
]
\]

stage weights 也冻结为现有：

```text
0.35 / 0.30 / 0.20 / 0.15
```

最终：

\[
L_{CR}
=
\sum_j \alpha_j
\left(
L_{j,b}+L_{j,l}+L_{j,g}+L_{j,s}
\right)
\]

继续使用现有：

```text
hsd_lambda = 0.06
max_ratio  = 0.12
```

Stage1 不对这些权重做网格搜索。

---

## 9.6 为什么这不是“简单 confidence mask”

普通 confidence masking：

```text
不可信 → 丢掉
```

CR-SRD：

```text
方向可信
→ 学方向 + 强度

方向不可信
→ 不学方向
→ 仍学 sign-free 变化强度
```

也就是说它不是 binary filtering，而是：

> **根据 teacher information 的可辨识性改变监督语义。**

这比简单 threshold/mask 更能形成方法贡献。

---

## 9.7 交换性质

|量|T1↔T2 后性质|
|---|---|
|magnitude \(m\)|invariant|
|coherence \(c\)|invariant|
|trust|invariant|
|signed residual \(d\)|anti-equivariant|
|directional target|complement|
|magnitude target|invariant|
|stable target|invariant|

因此 CR-SRD 没有破坏现有 Z2 逻辑。

---

## 9.8 训练图

```text
             ┌──────────────── 主学生路径 ────────────────┐
T1 ──┐       │                                            │
     ├─ joint 2B Backbone ─ joint 2B SWA ─ |TFM diff| ─ Decoder ─ Main Loss
T2 ──┘       │                                            │
             └────────────────────────────────────────────┘
                    ▲
                    │ gradients from auxiliary
                    │
SAM2 cached T1/T2
        │
geometry/exchange replay
        │
Relational Structure Builder
        │
┌────────────────────────────────────────────┐
│ magnitude:  m_b, m_l, m_g                 │
│ signed:     d_b, d_l, d_g                 │
│ trust:      w_change, w_stable             │
│ NEW:        c = |d| / (m + eps)           │
└────────────────────────────────────────────┘
        │
per-stage student residual probes
        │
N2-style shared mixed head
        │
┌──────────── coherence routing ─────────────┐
│ c high → signed directional target         │
│ c low  → sign-free magnitude target        │
└────────────────────────────────────────────┘
        │
CR auxiliary loss
```

Teacher Cache 全程没有梯度。

---

## 9.9 部署图

```text
T1 ─ Backbone ─ SWA ─┐
                     ├─ TFM ─ Decoder ─ Change Map
T2 ─ Backbone ─ SWA ─┘
```

不存在：

```text
SAM2
Teacher Cache
CR-SRD
probe
coherence
distillation head
```

### 预期复杂度

如果严格复用 N2 topology：

```text
训练参数目标：2,921,370
部署参数：    2,913,094
部署 FLOPs：  2.75–2.77G
```

前者是**实现验收目标，不是当前已测 Run4 事实**。

---

## 9.10 与历史方法的本质差异

### vs Run2

Run2 研究：

```text
exchange-invariant structural code
residual encoder
decoder correction
```

但没有研究：

> signed residual 本身何时方向可靠。

### vs Run3

Run3：

```text
N0 strict even + odd
N1 even
N2 all-pixel mixed signed
N3 invariant
```

CR-SRD：

```text
保留 N2 的统一表示
但不再假设所有可信 change pixel 都有可信方向
```

### vs Mask-CDKD

Mask-CDKD 是：

```text
image masking + MAE + cross-domain adaptation
```

CR-SRD 是：

```text
bi-temporal structural sign identifiability routing
```

两者的任务、teacher interaction、监督位置、训练成本均不同。

### vs DistillSemiCD

DistillSemiCD 的 confidence 是 pseudo-label / reliable-region 问题；CR 的 confidence 是：

> **同一 teacher structural residual 内部“变化存在性与时相方向可辨识性”的解耦。**

### vs BFD

BFD 蒸馏 foundation features/relations；CR 不需要在线 foundation feature，仅使用已经存在的 offline SAM2 structural cache。

---

## 9.11 最容易失败的三个条件

### Failure A：coherence 几乎全是 1

那么：

```text
CR ≈ N2
```

说明方向大多已经可靠，Run4 新机制没有作用空间。

**诊断：**

```text
mean(c)
p10/p50/p90
fraction(c<0.25)
fraction(c>0.75)
```

若低 coherence 区域 <5%，且 CR-N2 ≈0，则否证方法必要性。

### Failure B：coherence 几乎全是 0

那么：

```text
CR ≈ magnitude/invariant supervision
```

可能退化向 N3。

若 directional fraction <20%，同时 F1 靠近 N3，则说明 SAM2 structural signed representation 本身方向质量不足，应放弃 signed 主故事。

### Failure C：magnitude fallback 的 sign ambiguity 造成 q 不稳定

当 \(c\) 很低：

\[
|2q-1|\rightarrow m
\]

有两个对称解。

如果 direction signal 太弱，q 可能产生不稳定 sign。

**诊断：**

```text
q histogram
aux swap complement error
gradient cosine
Recall/Precision
```

如果出现明显训练振荡或某数据集 F1 下降超过预注册阈值，直接判定 CR 失败，而不是继续叠加新模块。

---

# 10. 可直接交给 Coding Agent 的 `models/` 修改规格

原则：

```text
不修改 Run1–Run3 现有语义
新模式单独命名
所有关键因素显式开关
默认不开启时旧模型行为完全不变
```

## 10.1 `models/a2net.py`

### 新接口

```python
A2Net_LWGANet_L0(
    ...,
    auxiliary_mode="none",
    cr_srd_cfg=None,
    joint_temporal_bn="auto",
)
```

允许：

```text
joint_temporal_bn:
    auto
    on
    off
```

规则：

```python
auto:
    z2_srd / cr_srd -> on
    others           -> off
```

这保证旧 Run3 行为不变。

新增：

```python
self.use_joint_temporal_bn
```

把目前：

```python
if self.training and self.auxiliary_mode == "z2_srd":
```

改成：

```python
if self.training and self.use_joint_temporal_bn:
```

在 backbone 与 SWA 两处统一使用。

### 新 auxiliary mode

```text
cr_srd
```

实例化：

```python
self.training_auxiliary = CRSRDAdapter(**cr_srd_cfg)
```

### deploy

```python
switch_to_deploy():
    del training_auxiliary
    auxiliary_mode = "none"
    use_joint_temporal_bn = False
```

部署输出必须与删除前 eval 输出 `<1e-6`。

---

## 10.2 `models/distill/sam_hsd/temporal_evidence.py`

在当前 `ExchangeInvariantStructuralEvidence` 结果上增加：

```python
direction_coherence: Tensor[B,3,H,W]
```

实现：

```python
c_boundary = abs(signed_boundary) / (boundary + eps)
c_local = abs(signed_local) / (local + eps)
c_geometry = abs(signed_geometry) / (geometry + eps)

coherence = torch.cat([...], dim=1).clamp(0, 1)
```

必须是参数无关计算。

新增 diagnostics：

```text
coherence_mean
coherence_p25
coherence_p50
coherence_p75
coherence_low_frac      # < 0.25
coherence_mid_frac
coherence_high_frac     # > 0.75
```

不要写入 Teacher Cache 文件；动态派生即可。

---

## 10.3 `models/distill/sam_hsd/losses.py`

增加：

```python
coherence_routed_structural_loss(
    prediction,
    evidence,
    channel_weights,
    use_coherence_gate=True,
    use_magnitude_fallback=True,
)
```

行为：

### Full

```python
dir_loss = smooth_l1(pred[:,:3], directional_target)
mag_loss = smooth_l1(abs(2*pred[:,:3]-1), magnitude_target)

change_loss = coherence * dir_loss \
            + (1-coherence) * mag_loss
```

### Mask-only ablation

```python
change_loss = coherence * dir_loss
```

### Gate off

必须恢复当前 N2 change loss：

```text
use_coherence_gate=False
→ original mixed signed structural_code_loss
```

这用于 numerical equivalence smoke。

stable loss 完全复用现有代码。

---

## 10.4 新增 `models/distill/sam_hsd/cr_srd.py`

建议类：

```python
class CRStructuralEncoderHSD(nn.Module)
class CRSRDAdapter(nn.Module)
```

### Student topology

必须与 N2 相同：

```text
4 × SpatialResidualProbe
hidden = 16
spatial_adapter = False

pair:
[right-left, left*right]

control head:
Conv(2h → h, 1)
GELU
Conv(h → 4, 1)
Sigmoid
```

不得增加 gating MLP、attention 或 learnable router。

### 输出 diagnostics

至少：

```text
total
direction
magnitude
stable

coherence_mean
coherence_low_frac
coherence_high_frac

stage1...stage4 routed losses
trust_change
trust_stable
```

---

## 10.5 `models/distill/sam_hsd/__init__.py`

导出：

```python
CRSRDAdapter
```

## 10.6 `models/distill/__init__.py`

继续向外导出：

```python
CRSRDAdapter
```

---

## 10.7 `models/scripts/train.py`

### 新参数

```text
--joint_temporal_bn {auto,on,off}

--cr_use_coherence_gate
--cr_use_magnitude_fallback
```

建议显式使用 `0/1` 或 `BooleanOptionalAction`，不要依赖模糊默认。

### 固定参数，不做 Stage1 搜索

```text
hidden = 16
stage_weights = 0.35,0.30,0.20,0.15
hsd_lambda = 0.06
hsd_max_ratio = 0.12
trust_gamma = 1.0
boundary_tolerance = 2
```

### 新增日志

```text
global_step
examples_seen
effective_epoch
lr

main_loss
aux_raw
aux_weighted
aux_factor
aux_ratio

aux_cap_scale
aux_cap_hit
rolling_cap_hit_rate

cr_direction_loss
cr_magnitude_loss
cr_stable_loss

coherence_mean
coherence_p25/p50/p75
coherence_low/mid/high_frac

best_val_f1
best_global_step
```

### 可选低频梯度诊断

```text
--diag_grad_interval 500
```

每 500 steps 在一个固定 backbone parameter 上计算：

```text
main_grad_norm
aux_grad_norm
main_aux_grad_cos
```

只记录，不做 PCGrad。

---

## 10.8 新增 `models/tools/smoke_cr_srd.py`

必须测试：

```text
1. coherence ∈ [0,1]
2. coherence swap invariant
3. signed target swap complement
4. magnitude swap invariant
5. stable swap invariant
6. CR loss finite
7. auxiliary → backbone has nonzero grad
8. auxiliary params have grad
9. teacher pack has no grad
10. main prediction exchange error <5e-6
11. auxiliary toggle error == 0
12. switch_to_deploy error <1e-6
13. deployed params == 2,913,094
14. CR training params == 2,921,370
15. deployed FLOPs ∈ 2.75–2.77G
```

同时做一个重要 numerical test：

```text
CR topology + gate off
vs
existing N2
```

复制相同 state dict 后，loss/output 应在浮点容差内一致。

---

## 10.9 新增 `models/tools/dry_run_cr_srd.py`

使用真实 SYSU/WHU cache，跑 5–20 optimizer steps：

验证：

```text
sample_id 完整
cache coverage 完整
replay shape 正确
T1/T2 exchange 正确
coherence 非 NaN
coherence distribution 非退化
aux grad 非零
cap diagnostics 有效
GPU memory 可接受
```

不保存正式 checkpoint，不覆盖任何现有实验。

---

## 10.10 不应修改

除 joint-BN 显式化外，Run4 不应改：

```text
models/backbone/lwganet.py
main Decoder
TFM
main BCE/Dice definition
SAM2 Cache schema
Run1/Run2/Run3 recipe semantics
```

---

# 11. Run4 紧凑实验矩阵

# Stage 0：正确性验收

|ID|任务|成功标准|
|---|---|---|
|S0-A|static / import / compile|无 import/syntax error|
|S0-B|random smoke|全部交换/梯度 assertion 通过|
|S0-C|real-cache SYSU dry run|cache/replay/coherence/grad 正常|
|S0-D|real-cache WHU dry run|同上|
|S0-E|deploy test|error `<1e-6`，params exact|
|S0-F|aux toggle|exact `0`|

任意一项失败：

```text
禁止启动 Stage1
```

---

# Stage 1：12k 机制筛选

固定：

```text
datasets = SYSU, WHU
seed = 2333
steps = 12000
batch = 64
```

|ID|名称|唯一研究问题|唯一变化|
|---|---|---|---|
|J0|JointBN Clean|Run3 joint-BN 本身带来多少变化？|clean + joint BN on|
|J1|N2 Reproduction|强 anchor 是否可复现？|现有 N2 原样|
|J2|CR MaskOnly|只抑制低 coherence signed supervision 是否有效？|CR gate on，fallback off|
|J3|**CR Full**|低 coherence 区使用 magnitude fallback 是否进一步有效？|fallback on|
|J4|N4 Odd Only Close-out|N0 失败来自 odd 本身还是分支冲突？|N4 existing recipe|

总计：

```text
5 configs × 2 datasets = 10 runs
```

---

## Stage1 预注册成功门槛

### CR Full 对 N2

进入下一阶段要求：

\[
\Delta F1 \ge +0.25
\]

在**两个数据集分别成立**；

或者：

```text
两数据集平均 ΔF1 ≥ +0.40 pp
且任一数据集 ΔF1 ≥ 0
```

同时：

```text
mean ΔIoU > 0
mean ΔKappa > 0
```

不接受只有 OA 微涨而 F1/IoU 无提升。

### magnitude fallback 的因子有效性

若要在论文中声称 fallback 有贡献：

```text
J3 − J2 平均 F1 ≥ +0.15 pp
```

若 J2≈J3：

> fallback 因子没有获得证据，应简化方法，不再通过调 fallback 权重救结果。

### Precision / Recall guard

若某数据集：

```text
Recall 或 Precision 下降 >1.5 pp
```

而 F1 增益又不足，则认为存在明显副作用。

---

## J4 的解释树

### 若 J4 本身也差

```text
N4 weak
N0 weak
```

支持：

> odd/sign teacher information 本身噪声较大。

### 若 J4 强而 N0 弱

```text
N4 strong
N0 weak
```

支持：

> even/odd 联合优化竞争或 cap 竞争。

### 若 J4 与 N0 都强于 N1，但现有结果不符

则需要检查 fresh reproduction/version effects。

---

# Stage 1B：低成本预算诊断

只有以下任一情况成立才启动：

```text
J3 已通过 Stage1 gate
或
12k 排名随末段 validation 明显变化
```

只跑：

```text
N2 20k
CR Full 20k
× SYSU/WHU
× seed2333
= 4 runs
```

**必须 fresh training。**

不允许：

```text
resume 12k checkpoint → 20k
```

---

# Stage 2：正式 3 seeds

若预算诊断支持 40k，则：

```text
methods  = N2, CR Full
datasets = SYSU, WHU
steps    = 40000
seeds    = {2333, 3407, 4519}
```

共：

```text
2 × 2 × 3 = 12 formal runs
```

报告：

```text
Recall mean±std
Precision mean±std
OA mean±std
F1 mean±std
IoU mean±std
Kappa mean±std

以及同 seed:
CR − N2 paired difference
```

### 正式成功门槛

每个数据集：

```text
paired mean ΔF1 ≥ +0.25 pp
至少 2/3 seeds ΔF1 > 0
任何 seed 不得 < -0.20 pp
paired mean ΔIoU > 0
paired mean ΔKappa > 0
```

n=3 不建议用夸张的显著性宣称；重点报告 effect consistency 和 paired deltas。

---

## Stage 2 之后才决定是否补正式 clean anchor

若 CR Full 通过：

```text
J0 JointBN Clean
× 3 seeds
× 2 datasets
```

再补 6 runs，用来建立最终论文归因：

```text
Clean JointBN
→ N2
→ CR-SRD
```

这比用旧 Run1/2 clean anchor 更严格。

---

# 12. 训练步数 / effective epoch / 收敛诊断

环境说明给出：

```text
SYSU train = 12,000
WHU  train = 5,947
```

Teacher Cache 也分别完整覆盖 12,000 / 5,947。

当前：

```text
batch_size = 64
gradient accumulation = 无
optimizer.step = 每个 batch 一次
drop_last = True
```

所以 effective batch size：

\[
B_{eff}=64
\]

用户指定公式：

\[
effective\ epochs
=
\frac{steps\times64}{N_{train}}
\]

得到：

|steps|SYSU effective epochs|WHU effective epochs|
|---:|---:|---:|
|12,000|**64.00**|**129.14**|
|20,000|106.67|215.23|
|40,000|213.33|430.47|

这已经非常重要：

> 同样 12k optimizer steps，WHU 的名义数据曝光约是 SYSU 的 2 倍。

---

## 12.1 `drop_last=True` 的真实 loader 行为

### SYSU

\[
\lfloor12000/64\rfloor=187
\]

每个 loader epoch 实际消费：

```text
187 × 64 = 11,968
```

随机丢 32 个样本。

因此：

```text
12k steps ≈ 64.17 loader passes
20k       ≈106.95
40k       ≈213.90
```

### WHU

\[
\lfloor5947/64\rfloor=92
\]

实际：

```text
92 × 64 = 5,888
```

每轮随机丢 59 个样本。

所以：

```text
12k ≈130.43 loader passes
20k ≈217.39
40k ≈434.78
```

因为每轮 shuffle，被 drop 的身份会变化，不等于永久缺失固定样本。

---

## 12.2 两时相 augmentation 不改变 effective batch

RandomExchange：

```text
50% 交换 T1/T2
```

是 augmentation，不是增加一个独立 training example。

因此不能把 batch 64 解释成 128。

joint BN 的 `2B` 是网络内部 BN 统计 batch：

```text
BN observes 128 temporal images
```

但 optimizer 仍只处理：

```text
64 image pairs / step
```

所以 effective batch 仍是 64 pair samples。

---

## 12.3 12k 到底够不够？

### 可以确认

12k 已经能够把 Run3 配方明显拉开：

```text
N2−N0:
+0.9782 / +0.5797 F1

N2−N3:
+0.6893 / +0.7031 F1
```

因此：

> **12k 作为机制筛选预算是有信息量的。**

### 不能确认

没有完整 curve 时不能说：

```text
12k 已充分收敛
```

也不能说：

```text
12k 明显欠拟合
```

尤其 WHU 12k 已是约 129 nominal effective epochs。

---

## 12.4 Run4 预算决策规则

### 12k 可视为足够筛选，如果 20k 诊断显示

- best val step 在总预算前 70% 出现；
- 最后 20% 的 best-val 改进 `<0.10 pp`；
- 20k 相对 12k 的同协议增益 `<0.20 pp`；
- N2/CR 排名不变。

### 应提升至 40k，如果

任一条件成立：

```text
best validation step 位于最后 15%
```

或：

```text
最后 20% smoothed val F1
仍以 >0.05 pp / 1k steps 的趋势上升
```

或：

```text
20k 比 12k 提升 >0.20 pp
```

### 过拟合诊断

若：

```text
validation F1 从 peak 下跌 ≥0.30 pp
并持续 ≥10% budget
同时 train main loss 继续下降
```

则更像过拟合，不应该机械延长训练。

---

## 12.5 Run4 必须新增的收敛日志

至少：

```text
global_step
examples_seen
effective_epoch
lr

train_main
train_aux_raw
train_aux_weighted
cap_scale
cap_hit

val_R / P / OA / F1 / IoU / Kappa
best_F1
best_step

coherence distribution
CR dir / mag loss

gradient norm/cosine（低频）
```

这能使“12k/20k/40k”从拍脑袋预算变成可证据化决策。

---

# 13. Checkpoint / Log 目录是否应该扁平化

当前 Run3 示例：

```text
Run3/
└── N0_Z2_SRD_Full/
    └── SYSU-CD-256/
        └── steps_12000/
            └── seed_2333/
                └── train_log.txt
```

用户希望：

```text
Run4/<experiment>/<dataset>/train_log.txt
```

---

## 13.1 如果真的永远只有一个预算 + 一个 seed

**技术上可以完全扁平化。**

但必须至少具备：

```text
manifest/config 完全匹配
存在结果则拒绝覆盖
运行中的 lock
完整 TEST RESULTS 才算 complete
incomplete 才允许 exact resume
```

否则第二次启动很容易覆盖已有实验。

---

## 13.2 正式多 seed 后完全扁平化一定会碰撞

例如：

```text
CR_Full/SYSU/train_log.txt
```

无法同时表示：

```text
seed2333
seed3407
seed4519
12k
20k
40k
```

把 seed/budget 只写进日志正文也不够，因为 checkpoint 文件名仍冲突。

---

## 13.3 推荐折中：单层叶目录

最终建议：

```text
outputs/
└── LS-Rep_BCD_RSML_3/
    └── saved_models/
        └── SAM-HSD/
            └── Run4/
                └── J3_CR_SRD_Full/
                    ├── SYSU-CD-256/
                    │   ├── s12000_seed2333/
                    │   ├── s40000_seed2333/
                    │   ├── s40000_seed3407/
                    │   └── s40000_seed4519/
                    └── WHU-CD-256/
                        └── ...

checkpoints/
└── LS-Rep_BCD_RSML_3/
    └── saved_models/
        └── SAM-HSD/
            └── Run4/
                └── J3_CR_SRD_Full/
                    └── SYSU-CD-256/
                        └── s40000_seed2333/
                            ├── last_checkpoint.pth
                            └── best_model_F1=....pth
```

日志与 checkpoint 使用**完全相同的 relative leaf**。

---

## 13.4 命名规范

```text
<experiment>/<dataset>/s<max_steps>_seed<seed>/
```

例：

```text
J3_CR_SRD_Full/SYSU-CD-256/s40000_seed2333/
```

不要写：

```text
final
new
test2
best
try_again
```

实验协议发生实质变化就创建新 experiment ID。

---

## 13.5 建议 leaf 内文件

```text
train_log.txt
run_manifest.json
.complete
```

checkpoint root：

```text
last_checkpoint.pth
best_model_F1=xxxxxx.pth
run_manifest.json
```

`.complete` 只有在：

```text
=== END TEST RESULTS ===
```

正常完成后才原子创建。

---

## 13.6 Launcher 幂等逻辑伪代码

```bash
leaf="${EXP}/${DATASET}/s${STEPS}_seed${SEED}"

log_dir="${OUTPUT_ROOT}/Run4/${leaf}"
ckpt_dir="${CKPT_ROOT}/Run4/${leaf}"

acquire_lock "${log_dir}.lock" || exit 1

if manifest_mismatch; then
    exit 2                    # 永不覆盖
fi

if complete_test_block_exists; then
    echo "completed; skip"
    exit 0
fi

if valid_last_checkpoint_exists; then
    RESUME="--resume ${ckpt_dir}/last_checkpoint.pth"
elif existing_nonempty_leaf; then
    echo "incomplete leaf without valid resume; refuse"
    exit 3
else
    RESUME=""
fi

exec env CUDA_VISIBLE_DEVICES=1 \
  /home/yqwang/miniforge3/envs/lsrep/bin/python \
  -m models.scripts.train \
  --gpu_id 0 \
  ...
```

---

## 13.7 关键原则

**不要：**

```text
rm -rf existing leaf
覆盖 train_log
覆盖 checkpoint
自动把 12k resume 成 40k
移动 Run3 结果
```

**要：**

```text
同参数 complete → skip
同参数 incomplete + valid checkpoint → resume
任何关键参数不一致 → hard fail
```

由于 Run3 shell 未上传，以上是 **Run4 推荐规范，而非对当前 shell 已有行为的确认**。

---

# 14. 验证清单、GPU 1 启动顺序与风险

## 14.1 静态检查

修改 Python 后：

```bash
python -m compileall models
```

修改任何 shell 后：

```bash
bash -n <script>
```

并重新生成相应 shell 汇总文件。

---

## 14.2 Random-input smoke

全部通过后才能读真实 Cache。

验收：

```text
shape
NaN/Inf
swap
grad
toggle
deploy
params
FLOPs
```

---

## 14.3 Real-cache dry run

分别：

```text
SYSU
WHU
```

验证不同 change ratio 下 coherence distribution 是否合理。

WHU change pixels 明显更稀疏，因此不能只看 SYSU。

---

## 14.4 GPU 1 正确映射

环境文档明确指出：

```bash
CUDA_VISIBLE_DEVICES=1
```

后物理 GPU 1 会在程序内部成为逻辑：

```text
cuda:0
```

因此启动应该是：

```bash
CUDA_VISIBLE_DEVICES=1 \
/home/yqwang/miniforge3/envs/lsrep/bin/python \
-m models.scripts.train \
--gpu_id 0 ...
```

不是：

```text
CUDA_VISIBLE_DEVICES=1 --gpu_id 1
```



---

## 14.5 单卡执行顺序

物理 GPU 1 串行：

```text
Stage0 smoke
    ↓
SYSU real-cache dry
    ↓
WHU real-cache dry
    ↓
Stage1 J0–J4
    ↓
分析预注册 gate
    ↓
必要时 20k N2/CR budget diagnostic
    ↓
决定正式预算
    ↓
Stage2:
seed2333 SYSU: N2 → CR
seed2333 WHU : N2 → CR

seed3407 SYSU: N2 → CR
seed3407 WHU : N2 → CR

seed4519 SYSU: N2 → CR
seed4519 WHU : N2 → CR
```

把 paired methods 相邻执行，可以减少机器状态/代码版本漂移。

---

## 14.6 主要风险表

|风险|等级|监控|决策|
|---|---|---|---|
|joint-BN 混杂未解决|P0|J0|不补 J0 不做 clean 因果宣称|
|coherence 退化到全 0/1|高|distribution|否决 CR 必要性|
|magnitude sign ambiguity|中高|q histogram / swap|失败则停止，不叠模块|
|aux cap 高频饱和|中|cap_hit_rate|先解释，不把调 cap 当创新|
|12k 排名受预算影响|中|20k diagnostic|fresh budget comparison|
|单 seed 偶然性|高|3 seeds paired|Stage2 才做正式结论|
|目录覆盖|高|manifest/lock|hard fail|
|旧 Run2 版本污染解释|中|来源标签|只作 hypothesis|
|部署结构意外改变|P0|exact params/deploy error|直接拒绝提交|

---

# 15. 可直接交给 Coding Agent 的实施任务书

## 目标

在**不改变 A2Net-LWGANet-L0 部署图、部署参数 2,913,094、部署 FLOPs 约 2.75–2.77G**的前提下，实现 Run4 的 `CR-SRD`，并同时修复 Run3 暴露出的 joint-BN 实验归因问题。

## 必须完成

### Task 1：显式化 temporal BN policy

修改：

```text
models/a2net.py
models/scripts/train.py
```

新增：

```text
joint_temporal_bn = auto/on/off
```

要求：

```text
旧 z2_srd + auto 的数值行为不变
旧 none + auto 行为不变
```

### Task 2：添加 coherence evidence

修改 teacher evidence builder：

```text
direction_coherence[B,3,H,W]
```

必须 parameter-free，交换 invariant。

### Task 3：实现 CR loss

新增：

```text
direction loss
magnitude fallback
stable loss
```

显式开关：

```text
cr_use_coherence_gate
cr_use_magnitude_fallback
```

### Task 4：实现 `CRSRDAdapter`

Student train topology 与 N2 完全相同。

不得新增 learnable router。

### Task 5：日志

新增：

```text
coherence
dir/mag loss
cap hit/scale
best step
effective epoch
```

低频 gradient cosine 作为可选 diagnostic。

### Task 6：smoke

写：

```text
models/tools/smoke_cr_srd.py
```

全部验收条件必须 assertion 化。

### Task 7：real cache dry run

写：

```text
models/tools/dry_run_cr_srd.py
```

只读 cache，不改 cache。

### Task 8：Run4 recipes

在 `train_scripts/SAM-HSD/Run4/` 实现：

```text
J0 JointBN Clean
J1 N2 Repro
J2 CR MaskOnly
J3 CR Full
J4 N4 OddOnly
```

每个配置只有声明的唯一变量不同。

### Task 9：安全 launcher

使用：

```text
<experiment>/<dataset>/s<steps>_seed<seed>
```

实现：

```text
manifest
complete skip
exact resume
lock
refuse overwrite
```

不得删除 Run3。

### Task 10：提交前

执行：

```text
python compile/smoke
real-cache dry run
deploy test
bash -n
regenerate shell summary
git diff
git status
git diff --cached
```

确保不暂存：

```text
datasets
teacher cache
checkpoint
weights
logs
prediction outputs
```

---

# 16. 立即执行顺序

1. **先补齐实验归因基础设施，而不是马上写 CR loss**：把 `joint_temporal_bn` 从 `auxiliary_mode=="z2_srd"` 中解耦为显式 policy，并用 smoke 确认 `auto` 不改变旧 Run3 行为。
2. 实现 `direction_coherence = |signed|/(magnitude+eps)`，只增加 teacher evidence 和 diagnostics，不先增加任何 learnable module。
3. 在 N2 原有 topology 上实现 CR routed loss；`gate off` 必须能数值回退到当前 N2。
4. 完成 random smoke：交换性质、梯度、toggle、deploy、params、FLOPs。
5. 在物理 GPU 1 上做 SYSU/WHU real-cache dry run，先看 coherence 分布。如果它几乎全 0 或全 1，**在正式训练前就重新评估 hypothesis**。
6. 建好 Run4 单层 leaf 目录和 no-overwrite launcher；shell 完成后执行 `bash -n` 并重生成 shell 汇总。
7. 跑 Stage1：J0、J1、J2、J3、J4 × SYSU/WHU × 12k/seed2333。
8. 严格按预注册阈值决定是否继续；若 CR 没达到门槛，不调一轮 loss weight 网格“救方法”。
9. 若通过，fresh 跑 N2/CR 的 20k budget diagnostic；根据 best-step/末段趋势决定正式 20k 还是 40k。
10. Stage2 用固定 3 seeds 做 N2 vs CR paired formal runs，输出 mean±std 和 seed-wise paired difference。
11. 主方法通过后，再补 3-seed JointBN Clean，为论文建立：
    ```text
    clean → mixed signed → coherence-routed
    ```
    的完整因果链。
12. 最后统一从每个 `train_log.txt` 的最后完整 TEST RESULTS block 抽取正式表格，再形成论文结果，不使用 validation best 行替代 test。

---

# 17. 仍需补充的证据

当前并非“无”。

## 最高优先级：`Run3_all_shell_scripts.txt`

本次附件中不存在。补充后可以进一步精确审查：

```text
路径生成
complete skip
best/last checkpoint 定位
resume 参数
并发碰撞
GPU 映射
重复启动
shell 幂等性
```

## 第二优先级：Run3 N0–N3 的 8 个完整 `train_log.txt`

目前 Excel 只够确认最终 test。

若要正式回答：

```text
12k 是否不足
最佳 validation step 在哪里
N0 odd 是否长期过大
cap 是否饱和
末段是否过拟合
```

需要完整训练日志。

尤其应抽取：

```text
main
aux_raw
aux_weighted
aux_ratio
aux_factor

z2_even
z2_odd
z2_even_feature
z2_odd_feature
z2_even_odd_cosine

val F1/IoU
global_step
lr
```

## 第三优先级：`lsrep` 环境快照

环境说明详尽记录了 `cd_base`，但最新项目使用：

```text
/home/yqwang/miniforge3/envs/lsrep
```

如果最终论文需要严格环境复现，建议补：

```bash
python -V
python -c "import torch; print(torch.__version__, torch.version.cuda)"
pip freeze
```

但不要修改 `cd_base`。

## 可选：N4 日志

如果 N4 实际已经在服务器运行、只是尚未进入 Excel，应直接补其最后完整正式日志；若没有，则按 Run4 J4 正式补跑。

---

# 最终研究判断

当前最重要的论文故事不应继续写成：

> “Z2 群分解能够提升变化检测。”

因为 Run3 数值暂不支持这个主张。

更符合现有证据的故事是：

> **二时相 SAM2 结构中，“变化强度”与“变化方向”的可靠性并不等价。Run3 证明保留 signed information 有价值，但把 direction 作为独立 odd branch 全局强监督反而降低性能；共享 mixed representation 更有效。因此 Run4 应研究如何在方向可靠时使用 signed teacher knowledge、在方向不可靠时退化为 sign-free structural supervision。**

这条路线同时解释了：

```text
N2 > N3
N2 > N1
N1 > N0
```

并且可以通过：

```text
N4
CR MaskOnly
CR Full
JointBN Clean
3-seed paired formal matrix
```

用很少的实验被证伪或支持。

如果 CR-SRD 成立，最终贡献可以保持为一个清楚的、二时相变化检测特有的训练期机制，而不是 module stacking；如果它不成立，也能在 Stage1/Stage1B 以有限 GPU 成本尽早停止，而不会进入无边界调参。

**部署约束保持不变：SAM2 / Teacher Cache / CR-SRD 全部训练期删除，目标部署参数仍为 `2,913,094`，256×256 FLOPs 仍控制在约 `2.75–2.77G`。**