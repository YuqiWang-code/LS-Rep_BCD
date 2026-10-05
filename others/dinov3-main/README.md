# DINOv3（精简参考）

- 模型：DINOv3 ViT-B/16 LVD-1689M 与 ViT-L/16 SAT-493M（Meta 官方技术发布/preprint，非 CCF-A 论文）
- 官方 GitHub：https://github.com/facebookresearch/dinov3
- 权重：
  - `pre-trained_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`（neutral LVD）
  - `pre-trained_weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`（卫星预训练 SAT-493M）
- 用途：CATA-CD 教师池 T3（neutral）与 T4（satellite）。

## 本项目使用方式

最小可加载编码器已抽取到 `models/thirdparty/dinov3/`（自包含，仅 torch）：

```python
from models.thirdparty.dinov3 import build_encoder_vitb, build_encoder_vitl
m = build_encoder_vitb("pre-trained_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")  # B/16
l = build_encoder_vitl("pre-trained_weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")  # L/16
d = m.forward_features(x)  # {"dense": [B,768,16,16]} (L: [B,1024,16,16])
```

- 归一化：ImageNet RGB mean=(0.485,0.456,0.406) / std=(0.229,0.224,0.225)；输入 256×256
- 构建参数：`n_storage_tokens=4, mask_k_bias=True, layerscale_init=1e-5`（L/16 另开 `untie_global_and_local_cls_norm`）
- 本目录仅保留 LICENSE 与本说明，原完整仓库见官方 GitHub。
