# RemoteCLIP — ViT-L/14 Visual Encoder (minimal)

This folder retains only the upstream license. The full upstream repository has
been slimmed down because this project needs only the **visual transformer**
(ViT-L/14) as a frozen teacher backbone for binary change detection — not the
text tower.

- **Model**: RemoteCLIP ViT-L/14 visual encoder (OpenCLIP / OpenAI-CLIP state-dict layout).
- **Original GitHub**: https://github.com/ChenDelong1999/RemoteCLIP
- **Paper**: *RemoteCLIP: A Vision Language Foundation Model for Remote Sensing*,
  IEEE Transactions on Geoscience and Remote Sensing (TGRS), 2024.
  DOI: [10.1109/TGRS.2024.3390838](https://doi.org/10.1109/TGRS.2024.3390838)
- **Weight file**: `pre-trained_weights/RemoteCLIP-ViT-L-14.pt`
- **License**: Apache License 2.0 (see `LICENSE`).

## Architecture / preprocessing facts

| Fact | Value |
| --- | --- |
| Backbone | CLIP ViT-L/14 (pre-norm) |
| Embed / width | 1024 |
| Depth | 24 |
| Heads | 16 |
| Patch size / feature stride | 14 |
| Input size | 224 × 224 |
| Patch grid | 16 × 16 |
| Color order | RGB |
| Normalization mean | `(0.48145466, 0.4578275, 0.40821073)` |
| Normalization std | `(0.26862954, 0.26130258, 0.27577711)` |

The checkpoint stores the visual tower under `visual.*` keys
(`visual.conv1.weight`, `visual.class_embedding`, `visual.positional_embedding`,
`visual.ln_pre.*`, `visual.transformer.resblocks.*`, `visual.ln_post.*`,
`visual.proj`). The text tower (`transformer.*`, `token_embedding`,
`ln_final`, `text_projection`, `positional_embedding`, `logit_scale`) is not
needed and is dropped when loading.

## Import

The minimal, self-contained encoder (only `torch` required; no `timm`,
`transformers`, or repo imports) lives at:

```
models/thirdparty/remoteclip/
```

```python
from models.thirdparty.remoteclip import build_encoder, META

model = build_encoder("pre-trained_weights/RemoteCLIP-ViT-L-14.pt").eval()
# model.forward_features(img) -> {"dense": [B, 1024, 16, 16], "global": [B, 1024]}
```
