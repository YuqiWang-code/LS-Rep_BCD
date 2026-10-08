# FA-SCRD Implementation Specification

## Failure-Aware Symmetric Change-Relation Distillation for A2Net-LWGANet-L0

## 1. Core Idea

FA-SCRD uses a fixed, independent, CD-adapted DINOv3 ViT-B/16 teacher to generate offline bi-temporal feature cache. During A2Net training, only symmetric change-relation knowledge is distilled, with failure-aware soft weighting. Teacher, cache loader, router and KD modules are removed during deployment.

Training graph:

```
T1,T2
 |
 +----------------+
 |                |
DINOv3 Teacher    A2Net Student
+LoRA Adapter     LWGANet-L0
 |                |
FT1,FT2           SWA->TFM
 |                |
CT=|Norm(FT1)-Norm(FT2)|    CS
 |                |
 +------Relation KD---------+
              |
        update student only
```

Deployment:

```
T1,T2
 |
A2Net-LWGANet-L0
 |
SWA
 |
TFM
 |
Decoder
 |
Change map
```

Removed:
- Teacher
- Cache
- Adapter
- Router
- KD loss

---

# 2. Teacher Definition

## Backbone

DINOv3 ViT-B/16:

```
/home/yqwang/projects/LS-Rep_BCD_RSML_3/pre-trained_weights/
dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

Parameter:
86M

Student:
2.9M


## Adapter

Use LoRA.

Insert:

```
Transformer Attention:
Q projection
V projection
```

Freeze:
- patch embedding
- transformer blocks
- MLP
- LayerNorm

LoRA rank:

```
r=8
```

Expected trainable parameters:

<0.5M


## Feature extraction

Input:

```
[B,3,256,256]
```

Mid feature:

```
Block 6 output

[B,768,16,16]
```

Deep feature:

```
Block 11 output

[B,768,16,16]
```

---

# 3. Cache Schema

Directory:

```
datasets/CD_teacher_cache/

FA_SCRD_DINOv3_CD/
    CDD/
    LEVIR/
    SYSU/
    WHU/
```

Each sample:

```python
{
 "meta":{
   "dataset": "...",
   "checkpoint_hash": "...",
   "resolution":256,
   "normalization":"imagenet"
 },

 "t1_mid",
 "t2_mid",

 "t1_deep",
 "t2_deep",

 "teacher_change_logit"
}
```

All features stored as FP16.

teacher_change_logit is only used for reliability estimation.

---

# 4. Synchronized Replay

All transformations must jointly apply to:

```
Image
Label
Teacher Cache
```

Crop:

```
A/B/Y/F_T1/F_T2
```

must share coordinates.

Temporal exchange:

Before:

```
(T1,T2)
```

After:

```
T1 <- T2
T2 <- T1

F_T1 <- F_T2
F_T2 <- F_T1
```

---

# 5. KD Objective

Teacher change relation:

\[
C_T^l=|Norm(F_{T1}^l)-Norm(F_{T2}^l)|
\]

Student:

\[
C_S^l
\]

Affinity:

\[
A=Softmax(XX^T/	au)
\]

Relation loss:

\[
L_{rel}=KL(A_T||A_S)
\]

Default:

```
lambda_mid=1.0
lambda_deep=1.0
tau=0.07
```

---

# 6. Failure-aware Weight

Student difficulty:

\[
d_r=mean(|p_s-y|)
\]

detach:

```
d_r = stopgrad(d_r)
```

Errors:

\[
e_S^r=BCE(p_S,y)
\]

\[
e_T^r=BCE(p_T,y)
\]

Advantage:

\[
a_r=\sigma((e_S^r-e_T^r)/	au_a)
\]

Default:

```
tau_a=0.5
```

Weight:

\[
w_r=stopgrad(d_r a_r)
\]

---

# 7. Final Loss

\[
L=L_{seg}+\lambda_{KD}\sum_r w_rL_{rel}^r
\]

Default:

```
lambda_KD=0.5
```

---

# 8. Code Modification Plan

New:

```
models/distill/

fa_scrd_teacher.py
relation_kd.py
failure_router.py
cache_dataset.py
```

Interfaces:

```python
symmetric_relation_kd(
    student_feature,
    teacher_feature,
    weight
)
```

```python
compute_failure_weight(
    pred_student,
    pred_teacher,
    label
)
```

train.py:

```
total_loss =
seg_loss +
lambda_kd * kd_loss
```

Do not modify deployment forward.

---

# 9. Ablation

Fixed:

```
seed=2333
batch=64
40000 steps
```

First run:

```
SYSU
WHU
```

## C0

Clean A2Net

## C1

C0 + joint temporal BN

## A1

C1 + fixed teacher relation KD

No failure weighting.

## M1

A1 + failure-aware weighting

Full FA-SCRD.

Success:

M1 > C1 on SYSU and WHU.

Failure:

- A1 fails: teacher transfer invalid.
- A1 succeeds but M1 fails: remove router.
- Only one dataset improves: treat as dataset-specific trick.

---

# 10. Deployment Verification

Required:

```
Params = 2,913,094
FLOPs = 2.75-2.77G
```

Check:

- no teacher in deploy checkpoint
- no cache dependency
- auxiliary switch does not change prediction

Requirement:

```
max(abs(pred_before - pred_after)) < 1e-6
```

Temporal swap:

```
prediction(T1,T2)
≈
prediction(T2,T1)
```

---

# Final Principle

Keep only:

```
training-time teacher cache
+
symmetric relation KD
+
failure-aware weighting
```

Reject:

```
online teacher
EMA teacher
teacher residual
pixel hard gate
teacher prediction injection
multi-teacher routing
```
