# AGENTS.md — LS-Rep_BCD_RSML_3 / DART-R-TS

> Last updated: 2026-09-13
> 本文件是当前项目中 AI/Coding Agent 的工作约束。当前唯一活动实验是 `train_scripts/DART-R/`（方向 C），主方法为 **C1 DART-R-TS（Task-Space DART-R，任务空间困难感知教师路由与拒绝）**；学生模型 A2Net-LWGANet-L0，部署参数 2,913,094 不变。项目长期处于「方法有效性探寻阶段」，主线结论随时可能被新实验推翻；不得把本文档中的任何结果当作已定稿的论文结论。不记录或继承旧实验（SAM-HSD、旧方向 C 的 C0 梯度路由等已归档）、旧服务器结果表或历史完成状态。

## GitHub 提交流程

完成 `models/` 或 `AGENTS.md` 的修改并通过必要检查后，按以下顺序提交到 GitHub：

```bash
git add models AGENTS.md docs/experiment_metrics.xlsx "docs/RSML-3_服务器环境与变化检测数据统一说明.md"
git status
git commit -m "Update code"
git push
```

执行 `git commit` 前必须先检查 `git status`，确认暂存区只包含本次准备提交的文件；不要把权重、缓存、数据集、日志或无关改动加入提交。

## 研究定位与优先级

本项目服务于 211 高校硕士研究生的学术论文研究与实验验证，不以通用软件工程建设为主要目标。Agent 应将具有论文价值的**模型创新和方法性改进**置于最高优先级，重点关注网络结构、特征表征与交互、训练期辅助机制及其可解释性。

1. 优先提出机制明确、区别于现有模块且能形成论文贡献点的方法创新；每项方案都要说明创新动机、作用机制、与已有方法的差异、可证伪假设及最小消融实验。
2. 损失函数和超参数仅作为公平训练、稳定收敛与验证机制有效性的配套手段；除非有明确的新理论或新机制，否则不得把调参结果包装成核心创新。
3. 工程修改只需服务于实验正确性、可复现性和必要的运行效率。
4. 评价方案优先考虑学术新颖性、机制合理性、实验可验证性和论文叙事完整性，警惕无机理的模块堆叠与只追求单次指标的改动。

## 1. 作用域与权威来源

当前唯一活动实验族：

```text
train_scripts/DART-R/           # 方向 C：DART-R-TS（B0 / C1 / C2）
```

（`train_scripts/SAM-HSD/` 及旧方向 C 的 C0 梯度路由均为历史归档，不再活动。）

按以下优先级判断事实：

1. 当前代码与 shell 参数；
2. `train_scripts/DART-R/Run2/README.md`（当前实验协议）；
3. `docs/RSML-3_服务器环境与变化检测数据统一说明.md`（环境/数据/路径）；
4. 本文件中的摘要。

不得从旧项目复制实验结果、GPU/存储配置、完成状态或结论。需要报告指标时，必须重新读取当前服务器上对应运行的正式 test block。

## 2. RSML-3 项目环境

```yaml
project:
  name: LS-Rep_BCD_RSML_3
  server: RSML-3
  user: yqwang
  path: /home/yqwang/projects/LS-Rep_BCD_RSML_3

environment:
  name: lsrep          # clone 自 cd_base
  path: /home/yqwang/miniforge3/envs/lsrep
  python: 3.10.21
  pytorch: 2.14.0+cu132
  cuda_runtime: "13.2"

gpu:
  count: 2
  model: NVIDIA GeForce RTX 5090   # Blackwell, sm_120, 每卡 32GB
```

环境使用：

```bash
source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate lsrep
cd /home/yqwang/projects/LS-Rep_BCD_RSML_3
```

环境规则：

- 训练、验证、推理和复杂度统计固定使用 `lsrep`，不要直接使用或污染 `cd_base`。
- 不要自行降级、替换或无约束升级 PyTorch/CUDA。
- 缺失的普通 Python 依赖只安装到 `lsrep`；CUDA Extension 必须先确认兼容 PyTorch 2.14.0、CUDA 13.2 与 `sm_120`。

## 3. 服务器目录约定

| 用途 | 路径 |
|---|---|
| 项目代码 | `/home/yqwang/projects/LS-Rep_BCD_RSML_3` |
| 数据集 | `/share_datasets/CD` |
| Teacher Cache | `/share_datasets/CD_teacher_cache` |
| DART-R-TS checkpoint | `/share_datasets/yqwang/checkpoints/LS-Rep_BCD_RSML_3/DART-R-TS` |
| DART-R-TS 训练/测试日志 | `/home/yqwang/outputs/LS-Rep_BCD_RSML_3/DART-R-TS` |
| 预训练权重 | `/home/yqwang/projects/LS-Rep_BCD_RSML_3/pre-trained_weights` |

（旧 SAM-HSD / DirectionC 的 checkpoint 也在 `/share_datasets/yqwang/checkpoints/` 下，仅作归档，不再活动。）

`.vscode/sftp.json` 用于手动上传代码，自动上传保持关闭；该配置忽略 `pre-trained_weights/` 和常见权重文件。

## 4. DART-R-TS（方向 C）

方向 C 主方法经历一次迭代：Run1 的 C0（梯度路由 DART-R）已判无效，现迭代为 **C1 DART-R-TS（Task-Space DART-R）**。学生模型仍叫 A2Net-LWGANet-L0，DART-R-TS 只是训练期机制。`models/scripts/train.py` 注册以下入口：

| ID | 机制 | 说明 |
|---|---|---|
| B0 | none | clean baseline，无教师 |
| C1 | task_space | **当前主方法**（DART-R-TS） |
| C2 | task_space / quality 路由 | 关掉 Brier 收益准入的负对照 |
| C0 / C0F | legacy 梯度路由 | 历史（已判无效）/ 旧重放修复对照 |
| C3 / C4 / C5 | task_space | 单教师 / 无困难加权消融（暂不跑） |

**C1 DART-R-TS 机制**（`models/distill/task_space.py`，辅助参数 **0**，训练=部署=2,913,094）：不训练 MLP/辅助头；把 SAM 结构经 leave-one-out 实例传输转译为变化概率提案，OV 提供 soft_change 提案；用精确 Brier 相对收益 `u=(p−y)²−(s−y)²` 做**逐像素**准入（只接受 `u>0` 的教师），`L=L_GT+0.06·KL(soft-target‖p)`，困难诊断仅作空间加权。同时修复 Cache 重放的 P0 问题（`aligned` 重放）。

**Run1（旧 C0）判定**：C0 相对 B0 平均 ΔF1 ≈ −0.16（4 数据集 3 降 1 平），失败诊断是梯度余弦代理 d≈0 → `r=q·max(d,0)`≈0 → Router 学会「几乎全 Reject」（reject_ratio 0.84–0.90，router_accuracy≈0.995）。是教师代理不适配，不是 Router 没学会。

**Run2 消融计划**（`train_scripts/DART-R/Run2/README.md`）：验证 C1 相对 B0 的有效性。3 组（B0 / C1 / C2）× 4 数据集 = 12 run，batch **128**、40k、seed 2333；双卡并行——GPU 0 跑 SYSU+WHU、GPU 1 跑 CDD+LEVIR，各卡内 B0→C1→C2 串行（脚本自带「跳过已完成 + 自动 `--resume`」）。判定：C1 vs B0 为整体有效性；C1 vs C2 为收益准入是否有用。单 seed 仅作筛选，论文级主比较 seed `2333/3407/5871`。

## 5. 模型与部署约束

- 主学生模型是 A2Net-LWGANet-L0；SAM2 Teacher Cache 及所有训练辅助只允许训练期使用。
- 推理参数量保持 `2,913,094`；256×256 FLOPs 实测约 `2.7676G`（`±0.03G` 容差，不能仅因这一级统计差异阻断最终测试）。
- 打开或关闭训练辅助不得改变主预测；`switch_to_deploy()` 必须移除训练辅助参数，部署前后最大误差 `< 1e-6`。
- 不得把 Teacher map 注入部署期主特征路径。
- A/B/label 与 Teacher Cache 必须重放完全一致的 crop、resize、flip 和时相交换。
- 实验语义集中在 `models/scripts/train.py`；wrapper 只负责选择实验、数据集、GPU 和运行目录。

核心代码：

```text
models/a2net.py
models/backbone/lwganet.py
models/decoder/a2net_decoder.py
models/datasets/cd_dataset.py
models/datasets/cache_transforms.py
models/distill/task_space.py
models/distill/routing.py
models/distill/losses.py
models/distill/diagnostics.py
models/distill/teacher_cache.py
models/losses/combined_loss.py
models/scripts/train.py
models/tools/validate_teacher_cache.py
models/tools/smoke_task_space.py
models/tools/dry_run_task_space.py
models/tools/integration_task_space.py
models/tools/export_deploy.py
```

## 6. Teacher Cache 规则

- 根 `/share_datasets/CD_teacher_cache/`，两套 teacher：SAMStruct（`sam2_struct_v2`）与 OVCDistill（DINOv2 ViT-B/14），四数据集 train 全覆盖。
- 先跑 `validate_teacher_cache` + `dry_run_task_space`；验证通过不重建缓存。
- 不得删除或覆盖数据集、Teacher Cache、checkpoint，除非用户明确授权并已核对绝对路径。
- **有效性状态**：缓存文件本身可用，但「teacher 能否提升主模型」仍需 C1 vs B0 同协议对照验证；在得到证据前，不得把 teacher 描述为有效组件。

## 7. 启动、恢复与结果纪律

推荐顺序：

1. 激活 `lsrep` 并进入项目目录；
2. 跑 `smoke_task_space`（CUDA 契约测试）；
3. 四数据集各做 `validate_teacher_cache` + `dry_run_task_space`（真实 Cache 单批前后向 + 梯度诊断）；
4. 启动 Run2 队列（双卡 tmux）；
5. 从 `train_log.txt` 收集正式 test 指标。

每个实验目录使用：

- `last_checkpoint.pth`：精确恢复（脚本检测到则自动加 `--resume`）；
- `best_model_F1=*.pth`：验证集选择出的最佳模型；
- `train_log.txt`：训练、恢复和最终测试记录。

只有 `train_log.txt` 同时出现 `=== TEST RESULTS ===` 与 `=== END TEST RESULTS ===` 才算完成。

结果规则：

- 正式指标只能来自最终 test block，不能使用验证集最佳行或 checkpoint 文件名替代。
- 报告 Recall、Precision、OA、F1、IoU、Kappa，以及训练/部署参数量和 FLOPs。
- 单数据集、单种子或阈值以下差异不能被表述为普适提升。

## 8. Agent 工作规则

1. 修改前读取当前代码、Run2 README 与实际日志，不依赖旧项目记忆。
2. 方法创新优先，但每项主张必须对应明确机制、可证伪假设和最小消融。
3. 保持推理图、Teacher Cache replay 顺序和部署约束不变。
4. 修改 shell 后执行 `bash -n` 语法检查。
5. 修改模型后至少跑 smoke test；涉及数据或缓存时再跑 real-cache dry run。
6. 不恢复已归档的旧实验模块、路径、指标或环境配置。
7. 不在 `cd_base` 中安装项目专属依赖，不擅自更换 PyTorch/CUDA 栈。
8. 任何删除数据、缓存或 checkpoint 的操作都需要用户明确授权和精确路径校验。
