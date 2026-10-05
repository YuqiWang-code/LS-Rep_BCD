# UniverSat Base

**Model:** UniverSat Base (embed dim 768, ~201 M parameters).

**Original repository:** https://github.com/gastruc/UniverSat

**Paper:** *UniverSat: Resolution- and Modality-Agnostic Transformers for Earth Observation* — accepted at **NeurIPS 2026**.

**Weights:** `pre-trained_weights/universat_base.safetensors` (433 tensors, all float32, keys prefixed with `model.`).

---

## Minimal importable encoder

The full research codebase has been removed from this directory. A minimal,
self-contained PyTorch encoder that loads these weights and runs as a frozen
teacher lives in:

```
models/thirdparty/universat/
```

```python
from models.thirdparty.universat import build_encoder, META

model = build_encoder("pre-trained_weights/universat_base.safetensors").eval()
features = model.forward_features(x)          # x: [B, 3, H, W], already normalized
dense = features["dense"]                     # [B, 768, H/4, W/4]
```

It re-implements the UniverSat Base encoder (Universal Patch Encoder + trunk:
Bi_ACA_in, 12 gated self-attention blocks with qk-norm + 2D RoPE + LayerScale,
Bilinear_out, and the sub-patch CA_Sub cross-attention) using only `torch`, and
loads the released checkpoint via `safetensors` (stripping the `model.` prefix,
strict=False).

## META facts (hard-coded single-modality spot-RGB)

| Field            | Value                                                              |
| ---------------- | ------------------------------------------------------------------ |
| `name`           | UniverSat Base (spot-RGB)                                          |
| `mean`           | `(0.485, 0.456, 0.406)` (ImageNet-style [0,1] defaults — see note) |
| `std`            | `(0.229, 0.224, 0.225)`                                            |
| `input_size`     | `(256, 256)`                                                       |
| `color_order`    | `RGB`                                                              |
| `feature_stride` | `4`                                                                |

Spot configuration (SPOT-6/7, 1 m/pixel, 3 RGB bands — wavelengths 0.665 / 0.56 /
0.49 µm): patch size 40 m (`scale = 4.0`, `patch_px = 40`), `subpatch = 10`.
Dense output is produced at `feature_stride = 4` (`= patch_px / subpatch`), i.e.
`forward_features` returns `[B, 768, H/4, W/4]`.

**Normalization note:** upstream UniverSat normalizes each dataset with
per-channel z-scores computed on raw 0–255 values (`NORM_spot_patch.json`, which
is *not* shipped in the repo). The `META.mean/std` above are ImageNet-style
`[0,1]`-scale defaults matching this repository's CDD pipeline; swap in
PASTIS-HD SPOT6 statistics for exact parity. `forward_features` applies no
normalization — the caller must normalize the input beforehand.

**Input caveat:** the 40 px patch unfold crops the trailing `H % 40` px, so
square inputs that are multiples of 40 (e.g. 240) avoid cropping; 256 uses
240 px and drops 16 px on the right/bottom edge.
