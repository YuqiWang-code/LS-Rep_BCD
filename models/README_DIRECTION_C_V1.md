# 方向 C 代码交付说明

本包基于执行时 GitHub 提交 [`c1d43561a35efedeefce6142d514f139d2238863`](https://github.com/YuqiWang-code/LS-Rep_BCD/tree/c1d43561a35efedeefce6142d514f139d2238863)，复用已有材料，没有重新开展文献调研。`implementation_version=direction_c_v1`。这是方向 C 的首个可运行实验实现；通过程序检查不代表已经证明精度收益。

## 1. 替换范围

`models.zip` 内是完整 `models/`。先将服务器原 `models/` 移到项目外备份，再解压放入项目根目录；不要直接覆盖混合，否则旧文件仍会残留。其余数据、Cache、权重及历史结果目录不需要移动。包内不含数据、预训练权重或测试环境依赖。

保留 B0（clean baseline）与 C0（方向 C）两个入口。原 backbone、SWA/TFM/decoder 主计算路径、四尺度 GT loss 保留；清理旧蒸馏实现、旧 H/R/N 实验分支及旧专用工具。项目根目录旧脚本如仍传入旧实验名或 HSD 参数，需要改用本说明命令。

| 文件 | 实现职责 |
|---|---|
| `a2net.py` | 原学生路径、训练辅助接口、`switch_to_deploy()` |
| `distill/diagnostics.py` | `build_cd_difficulty()`：六维困难画像及空间困难权重 |
| `distill/routing.py` | `estimate_teacher_fit()`、`DirectionC`、带硬 Reject 的损失聚合 |
| `distill/losses.py` | 两位教师各自的辅助头、response/boundary/relation loss |
| `distill/teacher_cache.py` | 两套离线 Cache 的配对、覆盖与格式校验 |
| `datasets/cache_transforms.py` | Cache 几何增强重放 |
| `scripts/train.py` | B0/C0 训练、原日志/保存协议及新增诊断字段 |
| `utils/checkpoint.py` | 学生/辅助头/Router、两个优化器及 RNG 状态 |
| `tools/validate_teacher_cache.py` | 只读检查全部训练 Cache |
| `tools/dry_run_direction_c.py` | 真实 Cache 单批前后向与部署检查 |
| `tools/smoke_direction_c.py` | 无数据的必要功能检查 |
| `tools/export_deploy.py` | 导出只含学生的权重 |

## 2. 已落地的设计选择

原材料未确定的适配度、Router 学习信号和 relation 语义，按以下明确约定实现，不能写成某篇论文已经验证的结论。

**诊断。** `h` 依次为边界漏检率、内部碎裂度、小变化漏检率、背景误报率、预测二元熵、GT 变化面积比。全部从 `sg(P)` 和训练 GT 计算。边界半径默认 2，小目标阈值默认 64 像素；碎裂根据 GT 连通区域内预测的连通片数量计算。无对应 GT 支持时指标为 0，并在返回值中提供有效标志。连通域用 SciPy 在 CPU 上计算。

**可靠性。** 用困难空间权重加权教师 quality/confidence，形成 `q_k`。`d_k` 是该教师损失与 GT 代理损失在最终 decoder 特征处的梯度加权余弦相似度；空间权重关注边界漏检、碎裂、小变化漏检、误报和不确定位置。`d` 的范围是 [-1,1]，`r_k=q_k*max(d_k,0)`。GT 代理使用最终输出的逐样本 BCE+Dice；学生主损失仍是原四尺度 BCE+Dice。`d` 是局部一阶适配度代理，并非真实参数更新后的收益，也未证明能预测跨数据集收益。

**Router 与 stop-gradient。** MLP 为 `8 -> 16 -> 3`，输出 SAM/OV/Reject 的 softmax 权重。h、q、d、r 均不向学生反传，Router 的输入显式 detach；用于学生蒸馏损失的权重也 detach。Router 独立用交叉熵学习标签：若最大 `q*d > utility_margin`（默认 0.02），标签是该教师，否则是 Reject。Router 有独立 Adam；不通过最小化自己加权后的教师损失学习，以免直接通过减小权重降低目标。

**Reject。** 每个样本取三个输出中的 argmax；若为 Reject，有效教师权重全部置零，否则保留两个原 softmax 教师权重，不重新归一化。学生更新目标为 `L_GT + kd_lambda * mean_i(sum_k(w_eff[i,k]*L[i,k]))`，默认 `kd_lambda=0.06`。Reject 样本对学生只提供 GT 监督；Router 仍接受自身交叉熵监督。辅助损失先逐样本计算再加权，避免某个样本 Reject 后仍泄漏教师梯度。原始教师损失仍会被计算和记录，以构造适配度；归零的是教师对学生总损失的贡献。

**两个独立教师。** SAM 为边界 BCE + 0.1 × 同实例邻接关系损失，t1/t2 各自处理，绝不直接比较两个时相的 instance ID 数值。OV 为 soft_change BCE + 0.1 × 空间余弦 Gram 损失，l1/l2 平均。辅助头只读取现有 decoder 特征，训练参数共增加 1,365，其中 Router 195。

**relation 的边界。** 文档只给出了 OV relation 的 8 通道形状，没有完整生成器语义。因此使用池化到 8×8 空间后的两两余弦关系，比较空间 Gram，未将通道臆定为八邻域方向或 DINO 原始特征。该比较对统一通道置换不变，但这不能证明任意时相变换下缓存都正确；真实 Cache 的生成语义仍须结合生成器确认。当前将 OV 缓存视为无序时相对的变化描述。

**冷启动限制。** 辅助头随机初始化，梯度代理可能不稳定，初期甚至长期全 Reject 都可能发生。实现不强迫教师参与；请同时检查 `target_reject_ratio`、`reject_ratio`、`d_*` 与 `router_accuracy`，区分教师代理不适配与 Router 未学会标签。缓存 q 也不自动等同于已校准的变化置信度。这些属于需要真实实验评估的方法问题。

## 3. 数据、Cache 与权重

沿用数据目录中的 `A/`、`B/`、`label/`、`list/train.txt`、`list/val.txt`、`list/test.txt`。保留原图像读取/归一化及标签阈值协议；输入为已对齐的 256×256 图像，GT 继续使用原 0/255 标签。不重划分数据，不生成伪 GT，不读取验证/测试 GT 来训练 Router。

C0 只读取训练 split 的 SAMStruct 与 OVCDistill。两个 manifest 的 sample ID 集合必须与训练列表一致；不接受缺失后静默降级。GT、A/B、SAM、OV 同步重放裁剪和翻转，instance ID 使用最近邻；OV 保留 l1/l2 原生空间分辨率。时相交换同步交换 SAM 的 t1/t2，OV 按上述无序时相对约定处理。不得将旧缓存再套一套独立随机增强。

读取约定：`manifest.json` 含 `cache_version=1`、`source_dataset`、`config_hash`、`entries`（ID -> 相对 `train/` 的文件路径）。SAM 的 `teacher_type=sam2_struct_v2`。每个 pt 含 `sample_id` 和 `meta.config_hash`。SAM 的 `t1/t2` 各含 `[1,H,W]` 的 `instance_id/boundary/quality`；OV 的 `soft_change/confidence/relation` 各含 `l1/l2`，通道数分别 1/1/8。概率/quality 在 [0,1]；ID 为非负 int32/int64。加载器校验格式和范围，不自动裁掉异常值或改写原 Cache。

继续使用已有 `lwganet_l0_e299.pth` 初始化学生，不需要新的 SAM/DINO 预训练权重；教师已由离线 Cache 提供。新辅助头和 Router 随机初始化。依赖沿用 PyTorch、NumPy、OpenCV、SciPy，THOP 用于 FLOPs 检查；不在线导入 UNIC、DiffPAN 或 SAM 推理工程。

增强改成按 seed/epoch/sample ID 对应索引确定随机状态，B0/C0 在本包内使用相同数据顺序和增强；这和历史版本随机数消费顺序并非逐步相同。正式对照应重跑本包的 B0，不能将旧 B0 当成逐随机状态一致的对照。

## 4. 服务器命令

在项目根目录及原训练环境执行。以下使用 SYSU，WHU 可修改 `cd_dataset`；文档未确认 CDD/LEVIR 的两套缓存齐备，不默认它们可以直接运行 C0。

```bash
cd /home/yqwang/projects/LS-Rep_BCD_RSML_3
cd_dataset=SYSU
cd_data_root="/home/yqwang/datasets/CD/${cd_dataset}-CD-256"
cd_sam_root="/home/yqwang/datasets/CD_teacher_cache/SAMStruct/sam2.1_hiera_large/${cd_dataset}-CD-256"
cd_ov_root="/home/yqwang/datasets/CD_teacher_cache/OVCDistill/dinov2_vitb14/${cd_dataset}-CD-256"
cd_pretrained="/home/yqwang/projects/LS-Rep_BCD_RSML_3/pre-trained_weights/lwganet_l0_e299.pth"
cd_ckpt_root="/home/yqwang/checkpoints/LS-Rep_BCD_RSML_3/DirectionC"
cd_log_root="/home/yqwang/outputs/LS-Rep_BCD_RSML_3/DirectionC"

# 首次运行前：全训练集 Cache 只读校验，然后真实单批检查。
python -m models.tools.validate_teacher_cache \
  --data_root "$cd_data_root" --dataset_name "$cd_dataset" \
  --sam_cache_root "$cd_sam_root" --ov_cache_root "$cd_ov_root"
python -m models.tools.dry_run_direction_c \
  --data_root "$cd_data_root" --dataset_name "$cd_dataset" \
  --sam_cache_root "$cd_sam_root" --ov_cache_root "$cd_ov_root" \
  --pretrained_path "$cd_pretrained" --device cuda:0 --batch_size 2

# B0 与 C0 使用同一协议，结果分别保存在新目录。
python -m models.scripts.train --experiment B0 \
  --dataset_name "$cd_dataset" --data_root "$cd_data_root" \
  --pretrained_path "$cd_pretrained" --gpu_id 0 \
  --batch_size 64 --max_steps 40000 --seed 2333 \
  --save_dir "$cd_ckpt_root/B0/$cd_dataset/steps_40000/seed_2333" \
  --log_file "$cd_log_root/B0/$cd_dataset/steps_40000/seed_2333/train_log.txt"

python -m models.scripts.train --experiment C0 \
  --dataset_name "$cd_dataset" --data_root "$cd_data_root" \
  --pretrained_path "$cd_pretrained" --gpu_id 0 \
  --sam_cache_root "$cd_sam_root" --ov_cache_root "$cd_ov_root" \
  --batch_size 64 --max_steps 40000 --seed 2333 \
  --kd_lambda 0.06 --router_lr 0.001 --utility_margin 0.02 \
  --save_dir "$cd_ckpt_root/C0/$cd_dataset/steps_40000/seed_2333" \
  --log_file "$cd_log_root/C0/$cd_dataset/steps_40000/seed_2333/train_log.txt"
```

若 batch 64 在真实 GPU 上不足，需同时调整 B0/C0 并使用新的对照目录；本地未测显存。本实现每批为适配度额外计算一次 GT 代理梯度和两个教师梯度，并进行 CPU 连通域诊断，训练速度不能由部署 FLOPs 推断。

恢复训练：使用原启动命令加 `--resume <该次运行目录>/last_checkpoint.pth`，保留同样实验、数据、Cache、超参、save_dir 和 log_file。恢复从保存的 epoch 边界开始；不提供任意 batch 中断位置恢复。新版本严格检查配置与列表/manifest 指纹，旧蒸馏 checkpoint 不能恢复为 C0，本包也不把旧格式 B0 当作新格式恢复点。

导出已由验证集 F1 选择的最佳 checkpoint：

```bash
python -m models.tools.export_deploy \
  --checkpoint /实际运行目录/best_model_F1=实际数值.pth \
  --output /实际运行目录/student_deploy.pth
```

导出文件保留 `model` 权重字典，仅含学生；推理模型用 `A2Net_LWGANet_L0(pretrained=False)` 严格加载该字典。它不再含训练状态，不能作为恢复训练 checkpoint。

## 5. 日志与保存

保留 `train_log.txt` 默认文件名、原 epoch/指标记录结构、`last_checkpoint.pth`、`best_model_F1={F1:.6f}.pth`、验证 F1 严格变好时更新最佳模型，以及 `=== TEST RESULTS ===` / `=== END TEST RESULTS ===` 块和原指标名称。日志首行方法标题更新为通用 A2Net 标题。路径仍由 `save_dir` 和 `log_file` 控制，上面的新子目录用于隔离旧实验。

新增 h 六项、q/d/r、教师 softmax/有效权重、Reject 比例、目标 Reject 比例、Router 准确率及 CE、独立教师损失等字段。checkpoint 的 `model` 同时包含辅助头及 Router；主 `optimizer` 包含学生/辅助头，新增 `router_optimizer`，保留 epoch/global_step/best_val_f1/args/RNG，并增加格式版本字段。保存格式兼容约定不等于旧方法参数可直接恢复。

## 6. 已完成验证与未验证范围

本地 CPU / PyTorch 2.14.0+cpu，真实 A2Net 模型，合成输入及按文档构造的磁盘 Cache：

- 导入、前向/反向通过；B0/C0 在相同初始化下训练预测和主路径 BN 状态一致。
- 强制 Reject 后所有教师贡献为 0，学生梯度与纯 GT baseline 一致；两个教师分别启用时都能向学生及辅助头反传。
- Router 独立收到非零梯度，其更新不向学生/辅助头反传；h/q/d/r/有效权重切断学生梯度。
- 空 GT、碎裂、零 quality、适配度正负号、relation 通道置换及几何重放检查通过。
- 学生/辅助头/Router/两个优化器及 RNG 保存恢复通过；恢复后下一优化步与不中断路径一致。
- B0/C0 各完成两步完整训练、验证最佳模型保存、test 结果块、已完成 checkpoint 重载与部署导出。
- 辅助分支删除前后预测最大误差 **0**；部署参数 **2,913,094**，本环境 THOP **2.767634G**（256×256）。保留原图及既有约 2.75G 的计数口径差异。

上述结果证明实现路径可运行，不是遥感数据集指标。未验证 RSML-3 实际数据/Cache 内容、已有预训练文件加载、CUDA 数值和显存、长程恢复、多进程真实 I/O 吞吐、teacher reliability 的实际有效性及 C0 相对 B0 的收益。部署图不含任何 Cache、诊断器、Router 或辅助头。
