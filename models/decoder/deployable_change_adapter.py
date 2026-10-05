"""Deployable Change Adapter (DCA).

A small, temporal-symmetric, deployable module inserted between the
TemporalFusionModule change features ``(c2, c3, c4, c5)`` (all 64 channels) and
the Decoder. Each scale mixes three experts (detail / context / noise) with a
global gate computed from symmetric pooled statistics of the change features,
so the whole module is invariant to T1/T2 exchange.

The DCA reads ONLY student change features — never teacher features, teacher
maps, cache, or labels — so it stays in the deploy graph. At ``width=128`` it
adds ~0.37M params, keeping total deploy params well under the 5M budget.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _depthwise(in_ch: int, out_ch: int, k: int, padding: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, k, stride=1, padding=padding, groups=in_ch, bias=False)


class _Expert(nn.Module):
    """One expert branch: ``in_d -> width -> width`` with a residual option."""

    def __init__(self, in_d: int, width: int, kind: str):
        super().__init__()
        self.kind = kind
        self.residual = kind == "noise"
        self.proj_in = nn.Conv2d(in_d, width, 1, bias=False)
        self.act = nn.GELU()
        if kind == "context":
            self.dw = _depthwise(width, width, 5, padding=2)
        else:  # detail / noise
            self.dw = _depthwise(width, width, 3, padding=1)
        self.pw = nn.Conv2d(width, width, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.proj_in(x)
        y = self.act(shortcut)
        y = self.dw(y)
        y = self.pw(y)
        if self.residual:
            y = y + shortcut
        return y


class DeployableChangeAdapter(nn.Module):
    """4-scale, 3-expert gated mixture adapter over TFM change features."""

    def __init__(self, in_d: int = 64, width: int = 128):
        super().__init__()
        self.in_d = in_d
        self.width = width
        self.num_scales = 4
        self.num_experts = 3

        self.experts = nn.ModuleList([
            nn.ModuleList([_Expert(in_d, width, kind) for kind in ("detail", "context", "noise")])
            for _ in range(self.num_scales)
        ])
        self.out_proj = nn.ModuleList([
            nn.Conv2d(width, in_d, 1) for _ in range(self.num_scales)
        ])
        # Gate inputs are GAP statistics of symmetric change features only.
        self.gate = nn.Sequential(
            nn.Linear(self.num_scales * in_d, 64),
            nn.GELU(),
            nn.Linear(64, self.num_scales * self.num_experts),
        )

    def forward(self, c2, c3, c4, c5):
        feats = (c2, c3, c4, c5)
        gap = torch.cat([f.mean(dim=(2, 3)) for f in feats], dim=1)  # [B, 4*in_d]
        weights = self.gate(gap).view(-1, self.num_scales, self.num_experts).softmax(dim=2)

        outs = []
        for s in range(self.num_scales):
            x = feats[s]
            y = sum(
                weights[:, s, k].view(-1, 1, 1, 1) * self.experts[s][k](x)
                for k in range(self.num_experts)
            )
            outs.append(self.out_proj[s](y))
        return outs[0], outs[1], outs[2], outs[3]


def dca_param_count(module: nn.Module) -> int:
    """Number of learnable parameters of a DCA instance."""
    return sum(p.numel() for p in module.parameters())


__all__ = ["DeployableChangeAdapter", "dca_param_count"]
