# SAM 2.1 Hiera Large — image encoder only

This directory used to contain a full checkout of the SAM 2 repository. It has
been reduced to the license files plus this README; the minimal, importable
encoder lives in `models/thirdparty/sam2/`.

- **Model:** SAM 2.1 Hiera Large (image encoder only — no mask decoder, memory,
  or prompt machinery).
- **Original GitHub:** https://github.com/facebookresearch/sam2
- **Weight file:** `pre-trained_weights/sam2.1_hiera_large.pt`
  (`torch.load(..., weights_only=False)` returns a dict with a single `"model"`
  key; the image encoder weights are the `image_encoder.*` keys, 594 in total).

## Encoder facts (META)

| Field | Value |
| --- | --- |
| `name` | `SAM 2.1 Hiera Large (image encoder only)` |
| `mean` | `(0.485, 0.456, 0.406)` |
| `std` | `(0.229, 0.224, 0.225)` |
| `input_size` | `(256, 256)` |
| `color_order` | `RGB` |
| `feature_stride` | `16` |

The model does **not** normalize internally. Feed it RGB float32 images in
`[0, 1]` that have already been normalized with the ImageNet `mean`/`std` above
(this matches SAM 2's `SAM2Transforms`, which applies
`torchvision Normalize(mean, std)` after `ToTensor`).

`forward_features(x)` returns a dict of fp32 feature maps:

- `"dense"` — stride-16 feature (`[B, 256, H/16, W/16]`, the SAM 2
  `vision_features`), 256 channels.
- `"s8"` — stride-8 feature (`[B, 256, H/8, W/8]`).
- `"s4"` — stride-4 feature (`[B, 256, H/4, W/4]`).

For the verified `input_size=(256, 256)` these are
`(1, 256, 16, 16)`, `(1, 256, 32, 32)`, and `(1, 256, 64, 64)`.

## Usage

```python
import torch
from models.thirdparty.sam2 import build_encoder, META

m = build_encoder("pre-trained_weights/sam2.1_hiera_large.pt").eval()
x = torch.randn(1, 3, *META.input_size)  # already normalized per META
with torch.no_grad():
    feats = m.forward_features(x)
dense = feats["dense"]  # [1, 256, 16, 16] fp32
```

## License

See `LICENSE` (and `LICENSE_cctorch` for the `cctorch` component that was part
of the original repository).
