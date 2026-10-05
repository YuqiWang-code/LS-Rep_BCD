# MaRS (simplified)

- **Model**: MaRS-Base RGB SwinV2 encoder (`swinv2_base_window8_256`)
- **Original GitHub**: https://github.com/WanderRainy/MaRS
- **Paper**: "MaRS: A Multi-Modality Very-High-Resolution Remote Sensing Foundation Model with Cross-Granularity Meta-Modality Learning" — **AAAI 2026** (CCF-A)
- **Weight**: `pre-trained_weights/mars_base_rgb_encoder_only.pth` (plain state_dict, encoder-only, no head)

## META facts

| Field | Value |
| --- | --- |
| name | MaRS-Base RGB SwinV2 encoder (swinv2_base_window8_256) |
| mean | `(87.01, 91.52, 83.51)` (found in `mars_dataset.py` `RGB_MEAN`, **0-255 scale**) |
| std | `(62.66, 54.58, 53.11)` (found in `mars_dataset.py` `RGB_STD`, **0-255 scale**) |
| input_size | `(256, 256)` |
| color_order | `RGB` |
| feature_stride | `16` ("dense" = 1/16 stage, 512 ch) |
| architecture | embed_dim=128, depths=(2,2,18,2), num_heads=(4,8,16,32), window_size=8, patch_size=4 |

Multi-scale features returned by the encoder:

- `dense` — stride 16, 512 ch (1/16 stage)
- `s8` — stride 8, 256 ch
- `s32` — stride 32, 1024 ch (optional; final LayerNorm applied)

Normalization note: the repo applies `(x - mean) / std` to float images in **`[0, 255]`**
(`torchvision.ToTensor()` does **not** divide a float32 ndarray by 255). In `[0, 1]` scale the
equivalents are `mean=(0.34122, 0.35890, 0.32749)`, `std=(0.24573, 0.21404, 0.20827)`.

## Minimal importable encoder

The self-contained, `timm`-free PyTorch encoder lives at:

```
models/thirdparty/mars/
├── __init__.py   # re-exports build_encoder and META
└── model.py      # vendored SwinV2 (SwinTransformerV2) implementation
```

Usage:

```python
import torch
from models.thirdparty.mars import build_encoder, META

m = build_encoder('pre-trained_weights/mars_base_rgb_encoder_only.pth')
m.eval()
x = torch.randn(1, 3, *META.input_size)  # already normalized per META
with torch.no_grad():
    feats = m.forward_features(x)  # {"dense", "s8", "s32"}
```

Caveat: the released encoder-only checkpoint strips the final LayerNorm (`norm.weight` /
`norm.bias`), so that module is freshly initialized and only affects the optional `s32` output.