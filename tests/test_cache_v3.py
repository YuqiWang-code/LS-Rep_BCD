#!/usr/bin/env python3
"""Unit tests for Cache v3 (E/M symmetric teacher evidence).

    python tests/test_cache_v3.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.distill.cache_v3 import (  # noqa: E402
    CACHE_SCHEMA_VERSION_V3,
    CacheV3Meta,
    TeacherCacheV3Reader,
    apply_cache_state_v3,
    save_sample_v3,
    symmetric_evidence,
)


def _feats(seed=0, c=8, h=4, w=5, b=2):
    torch.manual_seed(seed)
    return torch.randn(b, c, h, w), torch.randn(b, c, h, w)


# ------------------------------------------------------------------- evidence
def test_evidence_and_context_are_exchange_symmetric():
    fa, fb = _feats()
    e1, m1 = symmetric_evidence(fa, fb)
    e2, m2 = symmetric_evidence(fb, fa)
    assert torch.equal(e1, e2), "E must be invariant under A<->B"
    assert torch.equal(m1, m2), "M must be invariant under A<->B"


def test_evidence_is_abs_diff_of_channel_l2_normalised_features():
    fa, fb = _feats(seed=3)
    e, _ = symmetric_evidence(fa, fb)
    na, nb = torch.nn.functional.normalize(fa, dim=1), torch.nn.functional.normalize(fb, dim=1)
    assert torch.allclose(e, (na - nb).abs(), atol=1e-5)


def test_identical_pairs_give_zero_evidence_and_self_context():
    fa, _ = _feats(seed=5)
    e, m = symmetric_evidence(fa, fa)
    assert float(e.abs().max()) < 1e-6
    assert torch.allclose(m, torch.nn.functional.normalize(fa, dim=1), atol=1e-6)


def test_accepts_3d_and_4d_inputs():
    fa, fb = _feats(b=1)
    e4, m4 = symmetric_evidence(fa, fb)
    e3, m3 = symmetric_evidence(fa[0], fb[0])
    assert e4.shape == (1, 8, 4, 5) and e3.shape == (8, 4, 5)
    assert torch.allclose(e4[0], e3, atol=1e-6)


# ---------------------------------------------------------------------- replay
def test_replay_applies_crop_and_flip_to_spatial_tensors():
    cache = {"evidence": torch.arange(2 * 3 * 4).float().reshape(2, 3, 4),
             "context": torch.ones(2, 3, 4)}
    st = {"flip_h": True}
    out = apply_cache_state_v3(cache, st, src_h=4, src_w=4)
    assert torch.equal(out["evidence"], torch.flip(cache["evidence"], dims=[-1]))
    assert torch.equal(out["context"], torch.flip(cache["context"], dims=[-1]))


def test_replay_refuses_raw_temporal_taps():
    """Storing raw F_A/F_B would require an explicit A<->B exchange; refuse silently."""
    cache = {"feature_a": torch.zeros(2, 3, 4), "feature_b": torch.zeros(2, 3, 4)}
    try:
        apply_cache_state_v3(cache, {"flip_h": True})
    except ValueError as e:
        assert "raw temporal taps" in str(e)
        return
    raise AssertionError("raw temporal taps must be rejected, not silently replayed")


# ---------------------------------------------------------------------- meta
def test_meta_round_trip(tmp_path=None):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "manifest.json"
        m = CacheV3Meta(teacher_id="dinov3_sat", dataset="LEVIR", split="train",
                        weight_file="w.pth", weight_hash="abc", color_order="RGB",
                        source_size=256, native_h=16, native_w=16, channels=1024,
                        n_samples=7120, split_fingerprint="fp")
        p.write_text(m.to_json(), encoding="utf-8")
        back = CacheV3Meta.from_json(p)
        assert back.channels == 1024 and back.native_h == 16
        assert tuple(back.taps) == ("evidence", "context")
        assert back.schema_version == CACHE_SCHEMA_VERSION_V3


def test_meta_rejects_wrong_schema_version():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "manifest.json"
        p.write_text(json.dumps({"schema_version": 2, "teacher_id": "x"}), encoding="utf-8")
        try:
            CacheV3Meta.from_json(p)
        except ValueError as e:
            assert "schema_version" in str(e)
            return
        raise AssertionError("v2 manifest must not be accepted as v3")


def test_reader_refuses_non_train_split(tmp_path=None):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        r = TeacherCacheV3Reader(d, "SYSU", "dinov3_sat")
        for split in ("val", "test"):
            try:
                r.load(split, "00000.png")
            except PermissionError as e:
                assert "refusing to read" in str(e)
                continue
            raise AssertionError(f"split={split} must be refused")


# ------------------------------------------------------------------ round trip
def test_save_load_round_trip_is_fp16():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "dinov3_sat" / "SYSU" / "train"
        ev, ctx = symmetric_evidence(*_feats(seed=9, c=8))
        save_sample_v3(out, "00000.png", ev[0], ctx[0])
        obj = torch.load(out / "00000.png.pt", map_location="cpu", weights_only=False)
        assert obj["evidence"].dtype == torch.float16
        assert obj["context"].dtype == torch.float16
        assert obj["evidence"].shape == (8, 4, 5)
        assert torch.allclose(obj["evidence"].float(), ev[0], atol=1e-3)


def test_atomic_write_leaves_no_tmp_file():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        ev, ctx = symmetric_evidence(*_feats(seed=11, c=4))
        save_sample_v3(out, "x.png", ev[0], ctx[0])
        leftovers = list(out.glob("*.tmp"))
        assert not leftovers, leftovers


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed.append(t.__name__)
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
