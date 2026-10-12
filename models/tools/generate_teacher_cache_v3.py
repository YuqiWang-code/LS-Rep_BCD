#!/usr/bin/env python3
"""Generate the incremental Teacher Cache v3 (E/M pair evidence) for CTP-Critic.

Report §7.2. Writes ONLY to a fresh root (``CATA_CD_v3`` by default); it never
touches, and refuses to write into, the v2 tree.

**Data path is deliberately identical to the v2 generator** (``generate_teacher_cache_v2``):
images are read straight from disk with OpenCV, converted BGR->RGB, resized to 256,
scaled to [0,1], resized to the teacher's input size, and then normalised with the
*teacher's own* mean/std. It must NOT go through ``get_loader``: that pipeline
already applies the dataset's 6-channel normalisation (``Normalize`` + ``ToTensor``),
so feeding it to the teacher would normalise twice and produce garbage features.

Design notes that follow the report's P0 list:

* **incremental** -- one ``(teacher, dataset)`` pair per invocation; no need to
  materialise all nine teachers before training starts;
* **atomic** -- each sample is written to ``.tmp`` then renamed, and the manifest
  is written last, so a partial run is detectable and resumable;
* **traceable** -- the manifest carries the teacher weight SHA256, the dataset
  split fingerprint, native h/w, channel count and the replay contract;
* **train only** -- only ``split=train`` is generated.

Cost warning (measured from the v2 manifests): full native grid E/M is cheap for
RADIO/DINOv2 (14x14x768 ~ 20 GiB across four datasets) but UniverSat at 64x64x768
would be ~411 GiB. Generate per experiment, not for the whole pool.

Usage:
    python -m models.tools.generate_teacher_cache_v3 \
        --teacher_package dinov3_sat --dataset_name LEVIR \
        --data_root /share_datasets/CD/LEVIR-CD-256 \
        --weight_dir pre-trained_weights --out_root /share_datasets/CD_teacher_cache/CATA_CD_v3 \
        --batch_size 4 --gpu_id 1
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.distill.cache_v3 import (  # noqa: E402
    CACHE_SCHEMA_VERSION_V3,
    CacheV3Meta,
    file_sha256,
    save_sample_v3,
    symmetric_evidence,
)
from models.distill.teacher_package import TEACHER_REGISTRY, build_teacher_package  # noqa: E402

SPLIT = "train"


def read_rgb_01(path: Path, size: int) -> np.ndarray:
    """Read an image as float32 RGB in [0,1] — identical to the v2 generator."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"Failed to read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return img.astype(np.float32) / 255.0


def split_fingerprint(data_root: Path, split: str) -> str:
    return hashlib.sha256((data_root / "list" / f"{split}.txt").read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate incremental Teacher Cache v3")
    p.add_argument("--teacher_package", required=True, choices=sorted(TEACHER_REGISTRY))
    p.add_argument("--dataset_name", required=True, choices=("SYSU", "WHU", "CDD", "LEVIR"))
    p.add_argument("--data_root", required=True)
    p.add_argument("--weight_dir", default=str(PROJECT_ROOT / "pre-trained_weights"))
    p.add_argument("--out_root", default="/share_datasets/CD_teacher_cache/CATA_CD_v3")
    p.add_argument("--forbid_root", default="/share_datasets/CD_teacher_cache/CATA_CD_v2",
                   help="refuse to write anywhere inside this tree (v2 protection)")
    p.add_argument("--split", default=SPLIT, choices=("train",))
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--max_samples", type=int, default=0, help="0 = all")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verify_against", default=None,
                   help="optional v2 cache dir; recompute E from raw files and compare "
                        "the max-channel response with the stored v2 confidence")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_root = Path(args.out_root).resolve()
    forbid = Path(args.forbid_root).resolve()
    if forbid in out_root.parents or out_root == forbid:
        raise SystemExit(f"[v3] refusing to write inside the v2 protection tree {forbid}")

    out_dir = out_root / args.teacher_package / args.dataset_name / args.split
    man_path = out_dir / "manifest.json"
    if man_path.is_file() and not args.overwrite:
        print(f"[v3] manifest already exists, nothing to do: {man_path}")
        print("     (pass --overwrite to regenerate)")
        return

    device = torch.device(f"cuda:{args.gpu_id}" if args.device == "cuda" else "cpu")
    package, meta, cin = build_teacher_package(args.teacher_package, args.weight_dir, device="cpu")
    package = package.to(device).eval()
    tin = tuple(meta.input_size)
    weight_path = Path(args.weight_dir) / TEACHER_REGISTRY[args.teacher_package].weight
    weight_hash = file_sha256(weight_path) if weight_path.is_file() else "MISSING"
    fp = split_fingerprint(data_root, args.split)

    sample_ids = [ln.strip() for ln in
                  (data_root / "list" / f"{args.split}.txt").read_text(encoding="utf-8").splitlines()
                  if ln.strip()]
    if args.max_samples > 0:
        sample_ids = sample_ids[: args.max_samples]
    print(f"[v3] {args.teacher_package}/{args.dataset_name}: {len(sample_ids)} samples, "
          f"teacher input {tin}, v2 projection_in={cin}")

    def load_pair(sid: str):
        a = read_rgb_01(data_root / "A" / sid, 256)
        b = read_rgb_01(data_root / "B" / sid, 256)
        if tin != (256, 256):
            a = cv2.resize(a, (tin[1], tin[0]), interpolation=cv2.INTER_LINEAR)
            b = cv2.resize(b, (tin[1], tin[0]), interpolation=cv2.INTER_LINEAR)
        xa = torch.from_numpy(a.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        xb = torch.from_numpy(b.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        xa = (xa - torch.tensor(meta.mean, device=device).view(1, 3, 1, 1)) / \
             torch.tensor(meta.std, device=device).view(1, 3, 1, 1)
        xb = (xb - torch.tensor(meta.mean, device=device).view(1, 3, 1, 1)) / \
             torch.tensor(meta.std, device=device).view(1, 3, 1, 1)
        return xa, xb

    with torch.no_grad():
        xa0, xb0 = load_pair(sample_ids[0])
        fa = package.encoder.forward_features(xa0)[package.feature_key]
        fb = package.encoder.forward_features(xb0)[package.feature_key]
        e0, _ = symmetric_evidence(fa[0], fb[0])
        native_h, native_w = int(e0.shape[-2]), int(e0.shape[-1])
        channels = int(e0.shape[0])
        confidence_v2_check = float(e0.max())
    print(f"[v3] native C={channels} grid={native_h}x{native_w}")

    # Optional equivalence check against the v2 cache for the same sample, so a
    # silent data-path regression cannot slip through.
    if args.verify_against:
        v2p = Path(args.verify_against) / args.teacher_package / args.dataset_name / args.split \
            / f"{sample_ids[0]}.pt"
        if v2p.is_file():
            ref = torch.load(v2p, map_location="cpu", weights_only=False)["confidence"].float()
            # e0 is [C,h,w]; v2's confidence is max over channels -> [1,h,w]
            got = e0.max(dim=0, keepdim=True)[0].detach().cpu()
            if ref.shape == got.shape:
                mad = float((ref - got).abs().max())
                rel = mad / max(float(ref.abs().max()), 1e-9)
                print(f"[v3] max-channel response vs v2 confidence: max|diff|={mad:.6f} "
                      f"(rel {rel:.2e}) -> {'OK: v3 E reproduces v2 confidence' if mad < 5e-3 else 'MISMATCH -- check the data path'}")
            else:
                print(f"[v3] v2 confidence shape {tuple(ref.shape)} vs recomputed {tuple(got.shape)}; "
                      f"skipping numeric check")
        else:
            print(f"[v3] verify_against: no v2 file at {v2p}")

    out_dir.mkdir(parents=True, exist_ok=True)
    bs, n_done, t0 = args.batch_size, 0, time.time()
    with torch.no_grad():
        for start in range(0, len(sample_ids), bs):
            chunk = sample_ids[start:start + bs]
            for sid in chunk:
                xa, xb = load_pair(sid)
                fa = package.encoder.forward_features(xa)[package.feature_key]
                fb = package.encoder.forward_features(xb)[package.feature_key]
                ev, ctx = symmetric_evidence(fa[0], fb[0])
                save_sample_v3(out_dir, sid, ev, ctx)
                n_done += 1
                if n_done % 200 == 0:
                    el = time.time() - t0
                    print(f"  [v3] {n_done}/{len(sample_ids)}  {n_done / max(el, 1e-6):.2f} samples/s",
                          flush=True)

    man = CacheV3Meta(
        teacher_id=args.teacher_package, dataset=args.dataset_name, split=args.split,
        weight_file=TEACHER_REGISTRY[args.teacher_package].weight, weight_hash=weight_hash,
        color_order="RGB", source_size=256, native_h=native_h, native_w=native_w,
        channels=channels, n_samples=n_done, split_fingerprint=fp,
        extra={"v2_projection_in": int(cin),
               "data_path": "direct file read, BGR->RGB, /255, teacher mean/std (mirrors v2)",
               "symmetry_check_sample0": confidence_v2_check},
    )
    man_path.write_text(man.to_json(), encoding="utf-8")
    print(f"[v3] wrote {man_path} ({n_done} samples, {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
