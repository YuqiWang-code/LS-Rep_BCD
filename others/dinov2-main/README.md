# DINOv2（精简参考）

- 模型：DINOv2 ViT-B/14（通用 self-supervised dense representation）
- 官方 GitHub：https://github.com/facebookresearch/dinov2
- 权重：`pre-trained_weights/dinov2_vitb14_pretrain.pth`（官方 `dinov2_vitb14_pretrain.pth`）
- 用途：CATA-CD 教师池 T2（relation package），离线生成 compact cache。

## 本项目使用方式

最小可加载编码器已抽取到 `models/thirdparty/dinov2/`（自包含，仅 torch）：

```python
from models.thirdparty.dinov2 import build_encoder, META
m = build_encoder("pre-trained_weights/dinov2_vitb14_pretrain.pth")
d = m.forward_features(x)  # {"dense": [B,768,16,16], "global": [B,768]}
```

- 归一化：ImageNet RGB mean=(0.485,0.456,0.406) / std=(0.229,0.224,0.225)（[0,1] 尺度）
- 输入 224×224（patch 14 → 16×16 grid）；pos_embed 从 37×37 双三次插值到 16×16
- 本目录仅保留 LICENSE 与本说明，原完整仓库见官方 GitHub。
