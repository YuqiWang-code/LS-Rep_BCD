#!/usr/bin/env python3
"""Synthetic smoke test for CATA-CD v2.

Verifies the deployable student (C0 / C1), temporal swap symmetry, deploy
no-op, DCA parameter budget, and (optionally) a teacher package build + encode.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models import A2Net_LWGANet_L0  # noqa: E402
from models.decoder.deployable_change_adapter import dca_param_count  # noqa: E402

EXPECTED_C0_PARAMS = 2_913_094
PARAM_CAP = 5_000_000
SWAP_ATOL = 1e-6


def check_student(dca_mode: str) -> None:
    model = A2Net_LWGANet_L0(pretrained=False, dca_mode=dca_mode).eval()
    n = sum(p.numel() for p in model.parameters())
    assert n < PARAM_CAP, f"params {n:,} exceed {PARAM_CAP:,}"
    if dca_mode == "none":
        assert n == EXPECTED_C0_PARAMS, f"C0 params {n:,} != {EXPECTED_C0_PARAMS:,}"
    else:
        dca = sum(p.numel() for p in model.dca.parameters())
        assert 200_000 < dca < 600_000, f"unexpected DCA params {dca:,}"

    torch.manual_seed(0)
    a = torch.randn(2, 3, 256, 256)
    b = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        ab = model(a, b)
        ba = model(b, a)
        swap = max((x - y).abs().max().item() for x, y in zip(ab, ba))
        assert swap < SWAP_ATOL, f"swap error {swap:.2e}"

        before = model(a, b)
        model.switch_to_deploy()
        after = model(a, b)
        deploy = max((x - y).abs().max().item() for x, y in zip(before, after))
        assert deploy < SWAP_ATOL, f"deploy error {deploy:.2e}"

        _, change = model(a, b, return_change_features=True)
        shapes = [tuple(c.shape) for c in change]
        assert all(s[1] == 64 for s in shapes), shapes

    print(f"[smoke] dca_mode={dca_mode}: params={n:,} (DCA={dca_param_count(model.dca) if model.dca is not None else 0:,}) "
          f"swap={swap:.2e} deploy={deploy:.2e} change_shapes={shapes}")


def check_teacher(teacher_id: str, weight_dir: str) -> None:
    from models.distill.teacher_package import build_teacher_package
    package, meta, cin = build_teacher_package(teacher_id, weight_dir, device="cpu")
    xa = torch.rand(1, 3, meta.input_size[0], meta.input_size[1])
    xb = torch.rand(1, 3, meta.input_size[0], meta.input_size[1])
    out = package.encode_pair(xa, xb)
    print(f"[smoke] teacher={teacher_id}: cin={cin} local_change={tuple(out['local_change'].shape)} "
          f"confidence={tuple(out['confidence'].shape)} global_desc={tuple(out['global_desc'].shape)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dca", default="none", choices=("none", "moe128"))
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--weight_dir", default="pre-trained_weights")
    args = ap.parse_args()

    check_student(args.dca)
    if args.teacher:
        check_teacher(args.teacher, args.weight_dir)
    print("[smoke] ALL PASSED")


if __name__ == "__main__":
    main()
