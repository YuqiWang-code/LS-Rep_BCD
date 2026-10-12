#!/usr/bin/env python3
"""P0 audit for the third round: what is actually inside the v2 teacher caches?

Report step: "先做 P0 读写审计，不做冗长'证明一定有效'诊断：检查 teacher v2/v3
Cache 实际目录、样本名、native taps 与空间对齐".

This tool reads ONLY the existing v2 cache tree (never writes to it) and reports,
per teacher and dataset:
  * whether the directory/manifest exists and how many .pt files there are;
  * the exact tensor keys, shapes and dtypes of one sample;
  * the manifest's meta (weight hash, normalization, native h/w, channels);
  * whether the cache sample_ids match list/train.txt exactly;
  * the effective native grid, so we can group teachers by real spatial resolution.

Usage:
    python others/_cache_audit.py --cache_root /share_datasets/CD_teacher_cache/CATA_CD_v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

TEACHERS = ["sam2", "dinov2", "dinov3_lvd", "dinov3_sat", "remoteclip", "mars",
            "anysat", "universat", "radio"]
DATASETS = ["SYSU", "WHU", "CDD", "LEVIR"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", default="/share_datasets/CD_teacher_cache/CATA_CD_v2")
    ap.add_argument("--data_root", default="/share_datasets/CD")
    ap.add_argument("--split", default="train")
    ap.add_argument("--teachers", nargs="+", default=TEACHERS)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--max_report", type=int, default=6)
    args = ap.parse_args()

    root = Path(args.cache_root)
    rows = []
    for t in args.teachers:
        for ds in args.datasets:
            d = root / t / ds / args.split
            man = d / "manifest.json"
            pts = sorted(d.glob("*.pt")) if d.is_dir() else []
            rec = {"teacher": t, "dataset": ds, "dir_exists": d.is_dir(),
                   "n_pt": len(pts), "manifest_exists": man.is_file()}
            if man.is_file():
                m = json.loads(man.read_text(encoding="utf-8"))
                rec["manifest_keys"] = sorted(m.keys())
                rec["manifest"] = {k: v for k, v in m.items()
                                   if not isinstance(v, (list, dict)) or k in ("meta", "normalize")}
            if pts:
                try:
                    obj = torch.load(pts[0], map_location="cpu", weights_only=False)
                    rec["sample_file"] = pts[0].name
                    rec["tensors"] = {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                                      for k, v in obj.items() if isinstance(v, torch.Tensor)}
                    rec["non_tensor_keys"] = sorted(k for k in obj if not isinstance(k, torch.Tensor))
                    # sample-id universe vs the dataset list
                    ids = {p.stem for p in pts}
                    lst = Path(args.data_root) / f"{ds}-CD-256" / "list" / f"{args.split}.txt"
                    if lst.is_file():
                        want = {ln.strip().split("/")[-1] for ln in
                                lst.read_text(encoding="utf-8").splitlines() if ln.strip()}
                        rec["list_n"] = len(want)
                        rec["cache_ids_missing_from_list"] = len(ids - want)
                        rec["list_missing_from_cache"] = len(want - ids)
                        rec["example_cache_id"] = sorted(ids)[0] if ids else None
                        rec["example_list_id"] = sorted(want)[0] if want else None
                except Exception as e:  # noqa: BLE001
                    rec["load_error"] = f"{type(e).__name__}: {e}"
            rows.append(rec)

    print("=" * 100)
    print("V2 CACHE AUDIT")
    print("=" * 100)
    print(f"{'teacher':12s} {'ds':6s} {'n_pt':>6s} {'manifest':>8s} {'tensors (shape @ dtype)':s}")
    for r in rows:
        if not r["dir_exists"]:
            print(f"{r['teacher']:12s} {r['dataset']:6s} {'-':>6s} {'-':>8s} MISSING DIR")
            continue
        ts = r.get("tensors", {})
        desc = "; ".join(f"{k}{v['shape']}@{v['dtype']}" for k, v in ts.items())
        print(f"{r['teacher']:12s} {r['dataset']:6s} {r['n_pt']:6d} "
              f"{str(r['manifest_exists']):>8s} {desc}")
        if r.get("load_error"):
            print(f"{'':12s} {'':6s} LOAD ERROR: {r['load_error']}")
        if "cache_ids_missing_from_list" in r:
            flag = "" if (r["cache_ids_missing_from_list"] == 0 and
                          r["list_missing_from_cache"] == 0) else "   <-- ID MISMATCH"
            print(f"{'':12s} {'':6s} ids: list={r['list_n']} "
                  f"cache_not_in_list={r['cache_ids_missing_from_list']} "
                  f"list_not_in_cache={r['list_missing_from_cache']} "
                  f"ex_cache={r['example_cache_id']} ex_list={r['example_list_id']}{flag}")

    # native grid summary (from one dataset per teacher)
    print()
    print("=" * 100)
    print("NATIVE GRID SUMMARY (SYSU row; confidence is [1,h,w] at the teacher's own resolution)")
    print("=" * 100)
    seen = {}
    for r in rows:
        if r["dataset"] != "SYSU":
            continue
        ts = r.get("tensors", {})
        c = ts.get("confidence", {}).get("shape")
        lc = ts.get("local_change", {}).get("shape")
        seen[r["teacher"]] = {"confidence": c, "local_change": lc,
                              "manifest_meta": r.get("manifest", {}).get("meta")}
        print(f"  {r['teacher']:12s} confidence={c}  local_change={lc}")
        mm = r.get("manifest", {})
        if mm:
            print(f"  {'':12s} manifest: {json.dumps(mm, ensure_ascii=False)[:300]}")

    out = PROJECT_ROOT / "outputs" / "CATA-CD" / "cache_audit_v2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[audit] wrote {out}")


if __name__ == "__main__":
    main()
