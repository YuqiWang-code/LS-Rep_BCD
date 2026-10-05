# CATA-CD v2

**Capability-Validated Adaptive Teacher Agent for Lightweight Change Detection**（新主线）。

完整方案见 `docs/temporary/CATA-CD_v2_扩展教师池_验证驱动Agent_完整方案_20261005.md`。

## 目录结构

```text
teacher_adaptation/   数据集适配性验证（Stage 2：C0/C1 anchor + Wave-A 教师能力矩阵）
Run1/                 第一次迭代（Agent 选教师后从头重跑 M1）
Run2/                 （后续迭代）
```

## 主线流程（5 阶段）

1. **Stage 0**：干净基线 C0（A2Net 2.913M）与 C1（C0+DCA 3.280M）。
2. **Stage 1**：Teacher Cache 兼容性 audit（`models/tools/audit_teacher_cache.py`）。
3. **Stage 2**：`teacher_adaptation/` 逐个教师包 × 四数据集 40K，得到能力矩阵 `U[D,T]=TV*−C1`。
4. **Stage 3**：冻结 Teacher Capability Registry（research vs agent-train 分离，防 test leakage）。
5. **Stage 4/5**：LODO 离线 bandit 选教师 → `Run1/` 从头重跑 M1。

## 硬约束

- 学生 A2Net-LWGANet-L0；DCA 首版 width=128（约 0.37M）。
- 部署参数：C0=2,913,094；C1=3,280,274；**有效推理参数 < 5M**。
- 教师/Cache/Translator/Agent 全部训练期使用，不进部署图。
- 从头训练：同一 ImageNet 预训练 + seed 2333，完整 40K，禁止 checkpoint 微调/续训作为实验组。
