#!/usr/bin/env python3
"""PSFB: Privileged Spatial Feature Bridge (fourth-round main mechanism).

Report: ``docs/temporary/CATA-CD v2 第四轮完整中文研究报告.md`` §5.

What it does
------------
Run3's CTP-Critic let the teacher influence the student only through an auxiliary
1-channel critic's KL/BCE. PSFB instead builds a **teacher-conditioned shadow view
of the real decoder's input** and supervises it with the real task loss, plus a
64-channel feature-transfer term that pushes the teacher-free view toward the
teacher-conditioned one:

    Z        = GN(C3)                       student DCA output at stride 8, 64ch
    U_T      = Resize32(P([E, M]))          teacher adapter, train-only, from v3 cache
    Delta_T  = R([Z, U~])   U~ in {U_T, 0, RollBatch(U_T,1)}
    H_T      = C3 + alpha * Delta_T
    H_0      = C3 + alpha * Delta_0         (same R, U~ = 0)
    L        = L_main + lambda_s * L_shadow(H_T) + lambda_f * L_feat(H_0, sg(H_T))

The main prediction path ``Decoder(C2, C3, C4, C5)`` is never given the teacher;
the shadow forward is a second, train-only pass through the *shared* decoder with
**temporary BatchNorm buffers**, so the deployable student is untouched in shape,
parameters and FLOPs.

Two deviations from the report, both deliberate and documented
-------------------------------------------------------------
1. **Teacher-independent initialisation.** The report says a teacher's ZERO arm
   cannot serve as another teacher's CTRL because "P's parameter count and RNG
   stream differ". That is right, but it is fixable rather than something to pay
   for with extra runs: ``P``'s first conv shape depends on the teacher channel
   count, so constructing it consumes a different number of RNG draws and every
   *later* module (notably the shared ``R``) then initialises differently across
   teachers. Here ``P`` and ``R`` are drawn from **separate, teacher-independent
   generators**, so ``R`` (and anything built after the bridge) is bit-identical
   across teachers. Per-teacher ZERO arms are still recommended as insurance, but
   cross-teacher comparisons are no longer confounded by initialisation.

2. **Diagnostics for the self-defeating path.** ``L_feat``'s gradient reaches ``R``
   only through ``H_0 = C3 + R([Z, 0])``, so the ``U`` slice of ``R`` receives no
   gradient from it: the term trains R's zero-input branch toward the current
   ``H_T``. If ``||H_T - H_0||`` collapses early then ``R`` has learned to ignore
   ``U`` and ``L_feat`` has nothing left to transfer. The bridge therefore exposes
   ``||H_T - H_0||``, ``||Delta_T||`` and ``||H_T - C3||`` so the trainer can log
   them every epoch and treat a collapse as a failure, not a success.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ("real", "zero", "shuffle")
#: Fixed, teacher-independent seeds. Kept as module constants so every run in a
#: comparison re-derives the same R regardless of which teacher it reads.
P_SEED = 2333
R_SEED = 2334
DEFAULT_D = 32
DEFAULT_GN_GROUPS = 1


@dataclass
class PSFBOutput:
    h_t: torch.Tensor          # teacher-conditioned shadow feature  [B,64,32,32]
    h_0: torch.Tensor          # teacher-free shadow feature        [B,64,32,32]
    u: torch.Tensor            # teacher adapter output after mode  [B,d,32,32]
    delta_t: torch.Tensor
    delta_0: torch.Tensor
    stats: dict


def _seeded_weight(shape, fan_in: int, seed: int) -> torch.Tensor:
    """Draw a weight tensor from a fixed, teacher-independent generator.

    Construction happens on CPU (the bridge is moved to the device afterwards), so
    a CPU generator is always the right one here.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    w = torch.empty(*shape)
    w.normal_(0.0, float(fan_in) ** -0.5, generator=g)
    return w


class PrivilegedSpatialBridge(nn.Module):
    """Train-only teacher bridge at c3. Never registered inside the student."""

    def __init__(self, stu_ch: int = 64, teach_ch: int = 768, d: int = DEFAULT_D,
                 alpha: float = 1.0, mode: str = "real", r_use_gelu: bool = False):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        self.mode = mode
        self.d = int(d)
        self.alpha = float(alpha)
        self.teach_ch = int(teach_ch)

        # nn.Conv2d's own reset_parameters() draws from the GLOBAL RNG, and P's
        # first conv has a teacher-dependent shape. Without restoring the state,
        # a teacher with more channels would shift every subsequent draw, making
        # the bridge's mere presence teacher-dependent. Save/restore around the
        # whole construction so it is RNG-neutral (review finding #1).
        _rng_state = torch.get_rng_state()
        try:
            self.gn = nn.GroupNorm(DEFAULT_GN_GROUPS, stu_ch)

            # --- P: teacher adapter, drawn from its OWN generator ---
            p = nn.Conv2d(2 * teach_ch, d, 1, bias=False)
            with torch.no_grad():
                p.weight.copy_(_seeded_weight(p.weight.shape, 2 * teach_ch, P_SEED))
            self.P = nn.Sequential(p, nn.GroupNorm(DEFAULT_GN_GROUPS, d), nn.GELU())

            # --- R: shared residual, drawn from a TEACHER-INDEPENDENT generator ---
            layers: list[nn.Module] = [nn.Conv2d(stu_ch + d, stu_ch, 1)]
            if r_use_gelu:
                layers.append(nn.GELU())
            r = nn.Sequential(*layers)
            last = [m for m in r if isinstance(m, nn.Conv2d)][-1]
            with torch.no_grad():
                last.weight.copy_(_seeded_weight(last.weight.shape, stu_ch + d, R_SEED))
                if last.bias is not None:
                    last.bias.zero_()
            self.R = r

            # GroupNorm/P's norm default init is shape-dependent; pin it to fixed
            # constants so nothing about the bridge varies with the teacher either.
            with torch.no_grad():
                for m in self.modules():
                    if isinstance(m, nn.GroupNorm):
                        m.weight.fill_(1.0)
                        m.bias.zero_()
        finally:
            torch.set_rng_state(_rng_state)

    # ------------------------------------------------------------------ helpers
    def encode_teacher(self, evidence: torch.Tensor, context: torch.Tensor,
                       size: tuple[int, int]) -> torch.Tensor:
        """Channel-mix at the teacher's NATIVE grid, then resample the narrow U.

        Same memory-motivated ordering as Run3's critic: the wide tensors stay at
        native resolution and only the d-channel summary is upsampled.
        """
        x = torch.cat([evidence.float(), context.float()], dim=1)
        u = self.P(x)
        if u.shape[-2:] != tuple(size):
            u = F.interpolate(u, size=size, mode="bilinear", align_corners=False)
        return u

    def _apply_mode(self, u: torch.Tensor) -> torch.Tensor:
        if self.mode == "zero":
            return torch.zeros_like(u)
        if self.mode == "shuffle":
            if u.shape[0] < 2:
                return u
            return torch.roll(u, shifts=1, dims=0)
        return u

    # ----------------------------------------------------------------- forward
    def forward(self, c3: torch.Tensor, evidence: torch.Tensor,
                context: torch.Tensor) -> PSFBOutput:
        z = self.gn(c3)
        u_raw = self.encode_teacher(evidence, context, c3.shape[-2:])
        u = self._apply_mode(u_raw)

        delta_t = self.R(torch.cat([z, u], dim=1))
        delta_0 = self.R(torch.cat([z, torch.zeros_like(u)], dim=1))
        h_t = c3 + self.alpha * delta_t
        h_0 = c3 + self.alpha * delta_0

        with torch.no_grad():
            stats = {
                "u_norm": float(u.detach().abs().mean()),
                "delta_t_norm": float(delta_t.detach().abs().mean()),
                "delta_0_norm": float(delta_0.detach().abs().mean()),
                # collapse indicator: if this hits ~0 the U slice of R is ignored
                "ht_h0_gap": float((h_t - h_0).detach().abs().mean()),
                "ht_c3_gap": float((h_t - c3).detach().abs().mean()),
                "u_vs_zero_gap": float((u - 0).detach().abs().mean()) if self.mode == "real" else 0.0,
            }
        return PSFBOutput(h_t=h_t, h_0=h_0, u=u, delta_t=delta_t, delta_0=delta_0,
                          stats=stats)


def feature_transfer_loss(h_0: torch.Tensor, h_t: torch.Tensor,
                          kind: str = "cosine") -> torch.Tensor:
    """Push the teacher-free view toward the stop-gradient teacher-conditioned view.

    ``kind`` is frozen per experiment ("cosine" or "smoothl1"); the report forbids
    trying both and reporting the better one as an independent contribution.
    """
    tgt = F.normalize(h_t.detach(), dim=1)
    src = F.normalize(h_0, dim=1)
    if kind == "cosine":
        return (1.0 - (src * tgt).sum(dim=1)).mean()
    if kind == "smoothl1":
        return F.smooth_l1_loss(src, tgt)
    raise ValueError(f"unknown feature-transfer kind {kind!r}")


def shadow_decoder_forward(decoder: nn.Module, c2, h, c4, c5):
    """Run the SHARED decoder on a substituted c3 with temporary BN buffers.

    The decoder contains BatchNorm (``a2net_decoder.py``), so a plain second call
    would update the main path's ``running_mean``/``running_var``/
    ``num_batches_tracked`` and thereby change the deployed model. ``functional_call``
    with cloned buffers keeps the shadow pass's statistics local: gradients still
    reach the shared decoder weights (that is the intended transfer), but the
    buffers the main path will ship are never written.
    """
    from torch.func import functional_call

    params = dict(decoder.named_parameters())
    buffers = {k: v.detach().clone() for k, v in decoder.named_buffers()}
    return functional_call(decoder, (params, buffers), (c2, h, c4, c5))


def bridge_param_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def bridge_init_fingerprint(module: nn.Module) -> str:
    """SHA of R's weights only: must be identical across teachers (see docstring #1)."""
    h = hashlib.sha256()
    for k, v in sorted(module.R.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


__all__ = [
    "MODES",
    "P_SEED",
    "R_SEED",
    "PSFBOutput",
    "PrivilegedSpatialBridge",
    "bridge_init_fingerprint",
    "bridge_param_count",
    "feature_transfer_loss",
    "shadow_decoder_forward",
]
