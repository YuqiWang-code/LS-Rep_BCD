# RADIO (C-RADIOv3-B) — trimmed reference copy

This directory is a trimmed copy of the NVIDIA RADIO repository. The full
source has been removed; only the LICENSE is retained. A minimal, importable,
self-contained encoder lives in `models/thirdparty/radio/`.

## Model

- **Model name:** C-RADIOv3-B (checkpoint `arch`: `vit_base_patch16_v2_224`)
- **Original GitHub:** https://github.com/NVlabs/RADIO
- **Weight filename:** `pre-trained_weights/c-radio_v3-b_half.pth` (fp16 checkpoint)

## META facts

- **name:** `RADIO-C-RADIOv3-B (vit_base_patch16_v2_224)`
- **mean / std:** `(0, 0, 0)` / `(1, 1, 1)` — normalization is *internal*
  (the encoder's `InputConditioner` subtracts CLIP mean `[0.4815, 0.4578, 0.4082]`
  and divides by CLIP std `[0.2686, 0.2613, 0.2758]`, restored from the checkpoint).
  Feed raw `[0, 1]` images.
- **input_size:** `(224, 224)` (any resolution that is a multiple of 16 is accepted)
- **color_order:** `RGB`
- **feature_stride:** `16`
- **dense output:** `{"dense": [B, 768, H/16, W/16]}` in float32

## Note

The minimal importable encoder lives in `models/thirdparty/radio/`. It is
self-contained (only `torch` is required — no `timm`/`transformers`/
`huggingface`), rebuilds the C-RADIOv3-B ViT-Base + Conditional Position
Embedding (CPE) architecture, and loads the `base_model.*` + `input_conditioner.*`
state dict (skipping the training-only `_loss_states.*`, `_heads.*`,
`_feature_projections.*`, `adaloss`, `lean_in_expected_losses`,
`weight_balance_mask` keys and casting fp16 → fp32).

```python
from models.thirdparty.radio import build_encoder, META

m = build_encoder('pre-trained_weights/c-radio_v3-b_half.pth').eval()
x = ...  # [B, 3, H, W] in [0, 1]
out = m.forward_features(x)   # out['dense']: [B, 768, H/16, W/16] fp32
```
