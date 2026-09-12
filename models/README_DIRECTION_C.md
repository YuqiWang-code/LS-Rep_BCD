# DART-R-TS 代码包（C1 主方案）

基于 GitHub `98c58266a99e73e340ae844a81aca366613fda4b`。附件与该提交全部 65 个 Python 文件的 AST 一致。完整诊断、数学定义、实验表与限制见压缩包根目录 `DELIVERY_REPORT.md`。

- **运行改进版用 C1**；B0 是 clean baseline，C0 保留旧梯度路由，C0F 只修复旧 C0 的增强，C2–C5 是机制消融。
- **C1 不再训练 MLP/辅助头**：使用 SAM 结构引导的变化概率传输、OV 软变化提案、GT 检查的逐像素收益路由；部署完全删除辅助对象。
- C1 训练/部署均为 **2,913,094 参数**。CPU 实测 256×256 THOP **2.7676336G**；不能据此宣称真实数据精度提升。
- `routing_cfg={'mechanism':'task_space'}` 才实例化新机制；省略时仍为 legacy，避免旧工具静默改变语义。
- 新版 `implementation_version=dart_r_ts_v2`。旧 v1 checkpoint 可导出学生，不能当作新机制的精确恢复点。不要从 C0 best 接着训练当作 C1 的 40k 公平结果。
- Cache 只读；不含、不调用 Cache 重建流程。教师 relation 字段仍被校验，但 C1 不使用未确认语义的 OV Gram 监督。
- `README_DIRECTION_C_V1.md` 仅用于旧版本归档，其实验状态和日期不适用于 C1。

在服务器原 `lsrep` 环境和项目根目录运行：

```bash
python -m models.tools.smoke_task_space --device cuda:0
python -m models.tools.dry_run_task_space \
  --dataset_name SYSU --data_root /home/yqwang/datasets/CD/SYSU-CD-256 \
  --sam_cache_root /home/yqwang/datasets/CD_teacher_cache/SAMStruct/sam2.1_hiera_large/SYSU-CD-256 \
  --ov_cache_root /home/yqwang/datasets/CD_teacher_cache/OVCDistill/dinov2_vitb14/SYSU-CD-256 \
  --pretrained_path /home/yqwang/projects/LS-Rep_BCD_RSML_3/pre-trained_weights/lwganet_l0_e299.pth \
  --device cuda:0 --batch_size 2
bash models/scripts/run_task_space.sh B0 SYSU 0 2333
bash models/scripts/run_task_space.sh C1 SYSU 0 2333
```

恢复 C1：原相同命令最后加 `resume`：

```bash
bash models/scripts/run_task_space.sh C1 SYSU 0 2333 resume
```

默认 batch64 / 40k / seed2333；`DART_STEPS`、`DART_BATCH` 仅供明确改变实验预算时使用。checkpoint 与日志分开存入 `DART-R-TS/<ID>/<dataset>/steps_<steps>/seed_<seed>`。四数据集必须各做真实 Cache dry run；本交付仅验证了合成磁盘 Cache，没有访问 RSML-3。
