# AnySat

**Model name:** AnySat — "AnySat: An Earth Observation Model for Any Resolutions, Scales, and Modalities"
**Original GitHub:** https://github.com/gastruc/AnySat
**Paper:** CVPR 2025 (Highlight)

**Weight file:** `pre-trained_weights/AnySat.pth` (a dict with a single `'state_dict'` key, 418 tensors).
This is the AnySat **base** model (`embed_dim=768, depth=6, num_heads=12`).

## META facts

| field          | value                                          |
|----------------|------------------------------------------------|
| modality       | `spot` (single-date 3-channel RGB, 1 m GSD)    |
| mean           | `(0.485, 0.456, 0.406)` (ImageNet RGB — assumed, see note) |
| std            | `(0.229, 0.224, 0.225)` (ImageNet RGB — assumed) |
| input_size     | `(256, 256)`                                    |
| color_order    | `RGB`                                           |
| feature_stride | `10` (patch size)                               |
| dense output   | `[B, 1536, 25, 25]` fp32 for 256×256 input     |

*Note:* the AnySat repo does **not** define a fixed spot-RGB mean/std (the demo only
normalizes aerial/S1/S2 with dataset statistics, and Pastis spot tiles are read raw
0–255 with optional per-fold statistics). ImageNet RGB is therefore used as the
documented fallback. The dense channel dim is `1536 = 768 (cross-attention context
token) + 768 (per-patch sub-patch detail token)`; the token grid is
`floor((H-10)/10)+1`, so a 256 input yields a 25×25 grid (the last ~6 px are ignored).

## Minimal importable encoder

The full repo was simplified to just this README and the LICENSE file. The minimal,
self-contained importable encoder (spot RGB path only) lives at:

```
models/thirdparty/anysat/
```

Usage:

```python
import torch
from models.thirdparty.anysat import build_encoder, META

m = build_encoder('pre-trained_weights/AnySat.pth')
m.eval()
x = torch.randn(1, 3, META.input_size[0], META.input_size[1])  # already normalized per META
with torch.no_grad():
    out = m.forward_features(x)   # {"dense": [1, 1536, 25, 25] fp32}
```

`forward_features` hard-codes the `spot` modality at a fixed GSD of 1.0 m with
`patch_size=10` and `scale=1`; the returned `dense` map is produced exactly like the
official `forward_release(..., output='dense', output_modality='spot')` path.
