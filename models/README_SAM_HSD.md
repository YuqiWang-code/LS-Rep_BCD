# SAM-HSD / EIR-HSD / Z2-SRD model package

The inference graph is always the unchanged A2Net-LWGANet-L0 student:

```text
T1/T2 -> shared LWGANet-L0 -> SWA -> absolute-difference TFM -> A2Net decoder
```

Deployment is fixed at **2,913,094 parameters / approximately 2.75--2.77G THOP FLOPs**
(2.767634G in the RSML-3 `lsrep` environment). Every teacher builder,
probe, side head and auxiliary loss lives below `training_auxiliary` and is deleted by
`switch_to_deploy()` without changing the main output.

## Run2 EIR-HSD

`ExchangeInvariantStructuralEvidence` converts the replayed T1/T2 SAM partitions into one
four-channel code:

```text
[boundary residual, local relation residual, object geometry residual, stable consensus]
```

The default code is permutation-invariant and temporal-exchange invariant. Quality,
coverage and semantic disagreement produce continuous change/stable trust; missing coverage
abstains. `ExchangeInvariantEncoderHSD` predicts the same code with either simple 1×1 probes
(R2) or detachable spatial residual probes (R3/R5–R8). `ResidualDecoderHSD` uses one shared
64→16→4 structure side head, GT-conditioned feature relations and an optional student-error
FN/FP correction. Structural regression does not directly constrain the main change mask.

The unified registry in `models/scripts/train.py` supports both historical H0–H8 and Run2:

```text
R0 clean anchor
R1 exact Run1 unsigned reference
R2 exchange-invariant code with simple encoder probes
R3 spatial residual encoder
R4 residual decoder structure head
R5 full EIR-HSD
R6 R5 with fixed 0.5/0.5 fusion
R7 R5 without residual correction
R8 R5 with signed directional structural code
```

Measured train/deploy parameter counts are checked by `models/tools/smoke_eir_hsd.py`:

```text
R0  2,913,094 -> 2,913,094
R1  2,920,791 -> 2,913,094
R2  2,921,370 -> 2,913,094
R3  2,922,970 -> 2,913,094
R4  2,914,330 -> 2,913,094
R5–R8 2,924,206 -> 2,913,094
```

Dataset geometry and temporal exchange are replayed on cached maps before relational fields
are derived. This order is mandatory. Formal test metrics are appended to `train_log.txt`;
no inference image directory is generated.

## Run3 Z2-SRD

`Z2StructuralEncoderHSD` treats temporal exchange as the two-element group Z2. A shared
ordered-pair encoder is evaluated as `(T1,T2)` and `(T2,T1)`, then projected into an even
latent `(u12+u21)/2` and an odd latent `(u12-u21)/2`. The even head regresses the existing
four-channel R2 code; the bias-free odd head regresses signed boundary/local/geometry
residuals. Both are encoder-only training auxiliaries and are removed for deployment.

The Run3 registry keeps each ablation explicit:

```text
N0 group projection with separated even + odd heads
N1 group projection with even head only
N2 signed residual mixed into one four-channel head (negative control)
N3 abs-residual/product head without group projection (R2-style control)
N4 group projection with odd head only
```

`models/tools/smoke_z2_srd.py` checks the exact even/odd group laws, teacher exchange
behavior, N0--N4 routing and gradients, paired training-state invariance, auxiliary removal,
and the fixed 2,913,094-parameter deployment graph. The real-cache CUDA gate is
`models/tools/dry_run_z2_srd.py`.
