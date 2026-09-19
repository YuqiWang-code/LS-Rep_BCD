"""Synthetic smoke for UASAM (Uncertainty-Aware BT-SAM) geometry hypothesis.

Diagnosis H3 claims the current BT-SAM structural prior treats SAM mask
boundary jitter / small registration error as real structural change, which
hurts small buildings (WHU / LEVIR) disproportionately.

This is a pure-geometry concept test (no model, no cache). It compares:

    current : symmetric difference of instance occupancy   (BT-SAM today)
    uasam   : symmetric difference minus a tolerance band
              D1 = T1 \\ Dilate(T2, tau)
              D2 = T2 \\ Dilate(T1, tau)
              tau = boundary_radius (default 2 px)

Spec (diagnosis doc §5.4 step 4):
    1. A small building shifted 1-2 px must NOT be mass-classified as change.
    2. Real appearance / disappearance must still be change.

Run:  python models/tools/smoke_uasam.py
"""

from __future__ import annotations

import numpy as np


def binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation with a square (2r+1) structuring element."""
    H, W = mask.shape
    out = np.zeros_like(mask)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ty0, ty1 = max(0, dy), min(H, H + dy)
            tx0, tx1 = max(0, dx), min(W, W + dx)
            sy0, sy1 = max(0, -dy), min(H, H - dy)
            sx0, sx1 = max(0, -dx), min(W, W - dx)
            out[ty0:ty1, tx0:tx1] |= mask[sy0:sy1, sx0:sx1]
    return out


def current_change(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """Current BT-SAM: one-sided occupancy = symmetric difference."""
    return t1 != t2


def uasam_change(t1: np.ndarray, t2: np.ndarray, radius: int) -> np.ndarray:
    """UASAM: residual beyond a geometric tolerance band."""
    t1_only = t1 & ~binary_dilate(t2, radius)
    t2_only = t2 & ~binary_dilate(t1, radius)
    return t1_only | t2_only


def building(h: int, w: int, y0: int, x0: int) -> np.ndarray:
    """A solid rectangular building on a 64x64 canvas."""
    m = np.zeros((64, 64), dtype=bool)
    m[y0:y0 + h, x0:x0 + w] = True
    return m


def report(name, t1, t2, radius):
    cur = current_change(t1, t2)
    uas = uasam_change(t1, t2, radius)
    obj = int((t1 | t2).sum())
    print(
        f"{name:<38} "
        f"obj={obj:5d}px  "
        f"current_change={int(cur.sum()):5d}px  "
        f"uasam_change={int(uas.sum()):5d}px"
    )
    return cur, uas, obj


def main():
    radius = 2
    print("=" * 90)
    print("UASAM synthetic smoke (tau = boundary_radius = 2 px)")
    print("=" * 90)

    # 1. Small building jitter: 6x6 building shifted by (0, 2)
    t1 = building(6, 6, 20, 20)
    t2 = building(6, 6, 20, 22)
    _, uas, obj = report("small 6x6 shift(0,2)", t1, t2, radius)
    assert int(uas.sum()) == 0, "small jitter must vanish under tolerance band"

    # 2. Small building jitter: shift (1, 1) diagonal
    t1 = building(6, 6, 20, 20)
    t2 = building(6, 6, 21, 21)
    _, uas, obj = report("small 6x6 shift(1,1)", t1, t2, radius)
    assert int(uas.sum()) == 0, "diagonal jitter must vanish under tolerance band"

    # 3. Large building jitter (contrast): same 2px shift, much smaller fraction
    t1 = building(30, 30, 17, 17)
    t2 = building(30, 30, 17, 19)
    _, uas, obj = report("large 30x30 shift(0,2)", t1, t2, radius)
    assert int(uas.sum()) == 0, "large jitter must vanish under tolerance band"

    # 4. Appearance: building appears only in T2
    t1 = np.zeros((64, 64), dtype=bool)
    t2 = building(10, 10, 27, 27)
    _, uas, obj = report("appearance 10x10 (T1 empty)", t1, t2, radius)
    assert int(uas.sum()) == int(t2.sum()), "appearance must be fully preserved"

    # 5. Disappearance: building present only in T1
    t1 = building(10, 10, 27, 27)
    t2 = np.zeros((64, 64), dtype=bool)
    _, uas, obj = report("disappearance 10x10 (T2 empty)", t1, t2, radius)
    assert int(uas.sum()) == int(t1.sum()), "disappearance must be fully preserved"

    print("=" * 90)
    print("PASS: tolerance band removes boundary jitter, preserves true change")
    print("=" * 90)


if __name__ == "__main__":
    main()
