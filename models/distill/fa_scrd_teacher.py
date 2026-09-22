"""FA-SCRD fixed teacher: DINOv3 ViT-B/16 + LoRA (Q/V) + lightweight change head.

The teacher is a training-time-only, fully-offline knowledge source. It is
never part of the deployed student graph and is never updated by the student.

    xA, xB -> DINOv3 ViT-B/16 (frozen backbone + LoRA Q/V)
              -> mid/deep bi-temporal features
              -> |Norm(F_T1) - Norm(F_T2)| -> lightweight change head
              -> teacher_change_logit

Feature levels (both at 1/16 resolution for 256x256 input):
    mid  = block index 5  ("Block 6")  -> [B, 768, 16, 16]
    deep = block index 11 ("Block 11") -> [B, 768, 16, 16]
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_THIRDPARTY_DIR = _PROJECT_ROOT / "models" / "thirdparty"
if str(_THIRDPARTY_DIR) not in sys.path:
    sys.path.insert(0, str(_THIRDPARTY_DIR))

from dinov3.models.vision_transformer import vit_base  # noqa: E402


class LoRAQKV(nn.Module):
    """Low-rank adapter applied to the Q and V slices of a fused qkv Linear.

    The K slice receives no adaptation, matching the FA-SCRD spec. The B
    matrices are zero-initialized so the adapter is the identity at init.
    """

    def __init__(self, base_qkv: nn.Module, dim: int, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base_qkv = base_qkv
        self.dim = dim
        self.in_features = dim  # SelfAttention.compute_attention reads self.qkv.in_features
        self.scaling = alpha / r

        self.lora_A = nn.Linear(dim, r, bias=False)
        self.lora_B_q = nn.Linear(r, dim, bias=False)
        self.lora_B_v = nn.Linear(r, dim, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B_q.weight)
        nn.init.zeros_(self.lora_B_v.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.base_qkv(x)
        h = self.lora_A(x)
        dq = self.lora_B_q(h) * self.scaling
        dv = self.lora_B_v(h) * self.scaling
        delta = torch.cat([dq, torch.zeros_like(dq), dv], dim=-1)
        return qkv + delta


class ChangeHead(nn.Module):
    """Fuse mid/deep bi-temporal diffs into a single change logit at full res."""

    def __init__(self, dim: int = 768, out_size: int = 256):
        super().__init__()
        self.out_size = out_size
        self.proj_mid = nn.Conv2d(dim, 64, 1)
        self.proj_deep = nn.Conv2d(dim, 64, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, diff_mid: torch.Tensor, diff_deep: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.proj_mid(diff_mid), self.proj_deep(diff_deep)], dim=1)
        x = self.fuse(x)
        return F.interpolate(x, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)


class FASCRDTeacher(nn.Module):
    """Fixed DINOv3 ViT-B/16 teacher with LoRA + change head."""

    def __init__(
        self,
        weight_path: str,
        lora_r: int = 8,
        img_size: int = 256,
        mid_block: int = 5,
        deep_block: int = 11,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.mid_block = mid_block
        self.deep_block = deep_block

        self.backbone = vit_base(
            patch_size=16,
            img_size=img_size,
            pos_embed_rope_base=100,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
            qkv_bias=True,
            layerscale_init=1e-5,
            norm_layer="layernormbf16",
            ffn_layer="mlp",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=4,
            mask_k_bias=True,
        )

        self._load_weights(weight_path)
        self._apply_lora(lora_r)
        self.head = ChangeHead(dim=self.backbone.embed_dim, out_size=img_size)

    def _load_weights(self, weight_path: str) -> None:
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[teacher] missing_keys ({len(missing)}): {missing[:8]}")
        if unexpected:
            print(f"[teacher] unexpected_keys ({len(unexpected)}): {unexpected[:8]}")

    def _apply_lora(self, r: int) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        for block in self.backbone.blocks:
            attn = block.attn
            dim = attn.qkv.in_features
            attn.qkv = LoRAQKV(attn.qkv, dim, r=r)

    def extract_features(self, x: torch.Tensor):
        """Return (mid, deep) reshaped patch features for one image.

        Args:
            x: [B, 3, H, W]

        Returns:
            (mid, deep): each [B, embed_dim, H/16, W/16]
        """
        outputs = self.backbone.get_intermediate_layers(
            x,
            n=[self.mid_block, self.deep_block],
            reshape=True,
            norm=True,
        )
        return outputs[0], outputs[1]

    def forward_change_logit(self, xA: torch.Tensor, xB: torch.Tensor) -> torch.Tensor:
        """Bi-temporal change logit for a T1/T2 pair."""
        mid_a, deep_a = self.extract_features(xA)
        mid_b, deep_b = self.extract_features(xB)
        diff_mid = (mid_a - mid_b).abs()
        diff_deep = (deep_a - deep_b).abs()
        return self.head(diff_mid, diff_deep)

    @property
    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
