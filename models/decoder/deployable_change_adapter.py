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
    """4-scale, 3-expert gated mixture adapter over TFM change features.

    ``gate_mode`` (S-PCG ladder, review §5.1):

    ``legacy``   S0 -- the original single global gate over ``GAP(c_s)`` for all
                 four scales. This path is kept bit-for-bit identical so that the
                 existing C1 anchors stay valid.
    ``stats``    S1 -- per-scale gate that sees only the symmetric difference
                 statistic ``GAP(d_s)`` with ``d_s = |a_s - b_s|``.
    ``sympcg``   S2 -- per-scale gate that additionally sees the *symmetric common
                 context* ``GAP(m_s)`` with ``m_s = (a_s + b_s) / 2`` and the
                 pooled phase-difference ``v_s = |GAP(a_s) - GAP(b_s)|``.

    ``S1 -> S2`` is the clean single-variable contrast: identical architecture,
    identical parameter flow, only the gate's *inputs* change. All gate inputs are
    exchange-symmetric by construction, so T1/T2 swap invariance is preserved.
    """

    def __init__(self, in_d: int = 64, width: int = 128, gate_mode: str = "legacy",
                 gate_hidden: int = 64):
        super().__init__()
        if gate_mode not in ("legacy", "stats", "sympcg"):
            raise ValueError(f"unknown gate_mode {gate_mode!r}")
        self.in_d = in_d
        self.width = width
        self.num_scales = 4
        self.num_experts = 3
        self.gate_mode = gate_mode

        self.experts = nn.ModuleList([
            nn.ModuleList([_Expert(in_d, width, kind) for kind in ("detail", "context", "noise")])
            for _ in range(self.num_scales)
        ])
        self.out_proj = nn.ModuleList([
            nn.Conv2d(width, in_d, 1) for _ in range(self.num_scales)
        ])
        if gate_mode == "legacy":
            # Gate inputs are GAP statistics of symmetric change features only.
            self.gate = nn.Sequential(
                nn.Linear(self.num_scales * in_d, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, self.num_scales * self.num_experts),
            )
        else:
            in_gate = in_d if gate_mode == "stats" else 3 * in_d
            self.gate_s = nn.ModuleList([
                nn.Sequential(nn.Linear(in_gate, gate_hidden), nn.GELU(),
                              nn.Linear(gate_hidden, self.num_experts))
                for _ in range(self.num_scales)
            ])

    def _gate_weights(self, feats, pair):
        if self.gate_mode == "legacy":
            gap = torch.cat([f.mean(dim=(2, 3)) for f in feats], dim=1)  # [B, 4*in_d]
            return self.gate(gap).view(-1, self.num_scales, self.num_experts).softmax(dim=2)
        if pair is None:
            raise ValueError(f"gate_mode={self.gate_mode!r} needs the pre-fusion pair (a_s, b_s)")
        a_feats, b_feats = pair
        outs = []
        for s in range(self.num_scales):
            a, b = a_feats[s], b_feats[s]
            d = (a - b).abs()
            gap_d = d.mean(dim=(2, 3))
            if self.gate_mode == "stats":
                inp = gap_d
            else:
                m = 0.5 * (a + b)
                v = (a.mean(dim=(2, 3)) - b.mean(dim=(2, 3))).abs()
                inp = torch.cat([gap_d, m.mean(dim=(2, 3)), v], dim=1)
            outs.append(self.gate_s[s](inp))
        return torch.stack(outs, dim=1).softmax(dim=2)   # [B, 4, 3]

    def forward(self, c2, c3, c4, c5, pair=None):
        feats = (c2, c3, c4, c5)
        weights = self._gate_weights(feats, pair)

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
