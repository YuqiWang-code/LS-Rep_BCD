#!/usr/bin/env python3
"""J0 determinism forensics: why do two same-config, same-seed runs diverge?

Step 3 of the 2026-10-08 review plan (section 4.2 / 8.1 P0).

The plan's J0 stage says: before spending any more 40K runs, re-run the SAME
configuration twice for 100-1000 steps and record

    1. initial weight hashes        -> are the two runs initialised identically?
    2. data-order / augmentation hashes -> is the input stream identical?
    3. loss / gradient traces       -> at which step does the first divergence occur?
    4. numeric backend flags        -> what could even make it nondeterministic?

This tool produces one JSON per run; ``--compare a.json b.json`` then reports the
FIRST step at which the two runs disagree, which is the actionable output.

It deliberately reuses ``models.scripts.train``'s own factories (``set_seed``,
``build_model``, ``build_optimizer``, the loader construction, the loss) so the
audited procedure is byte-for-byte the procedure that produced the 44 existing
runs. It does NOT write checkpoints and does NOT touch the production output tree.

Usage:
    python -m models.tools.audit_reproducibility \
        --experiment_id J0-REP-a --dataset_name SYSU \
        --data_root /share_datasets/CD/SYSU-CD-256 \
        --dca_mode moe128 --teacher_package none \
        --pretrained --pretrained_path pre-trained_weights/lwganet_l0_e299.pth \
        --gpu_id 0 --batch_size 64 --steps 200 --seed 2333 \
        --out outputs/CATA-CD/Run2/j0/SYSU_a.json

    python -m models.tools.audit_reproducibility --compare a.json b.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models import build_loss  # noqa: E402
from models.datasets.cd_dataset import get_loader  # noqa: E402
from models.scripts.train import (  # noqa: E402
    CataCacheDataset,
    build_model,
    build_optimizer,
    multiscale_loss,
    set_seed,
    unpack,
)
from models.distill.cache_v2 import TeacherCacheReaderV2  # noqa: E402

# A tiny fixed slice of the parameter vector: cheap to hash every step, and any
# real numerical divergence shows up in it almost immediately.
SLICE = 4096


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tensor_sha(t: torch.Tensor) -> str:
    return sha_bytes(t.detach().to("cpu").contiguous().numpy().tobytes())


def state_dict_hashes(model: torch.nn.Module) -> dict:
    """Whole-model hash plus one hash per top-level module group."""
    sd = model.state_dict()
    full = hashlib.sha256()
    groups: dict[str, list[str]] = {}
    for k in sorted(sd):
        full.update(k.encode())
        full.update(sd[k].detach().to("cpu").contiguous().numpy().tobytes())
        groups.setdefault(k.split(".")[0], []).append(k)
    out = {"state_dict_full": full.hexdigest()}
    for g, keys in groups.items():
        h = hashlib.sha256()
        for k in keys:
            h.update(k.encode())
            h.update(sd[k].detach().to("cpu").contiguous().numpy().tobytes())
        out[f"group_{g}"] = {"sha256": h.hexdigest(),
                             "n_tensors": len(keys),
                             "n_params": int(sum(sd[k].numel() for k in keys))}
    return out


def numeric_env() -> dict:
    """Everything that can make identical code produce different numbers."""
    import torch.backends.cudnn as cudnn

    env = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cudnn_deterministic": bool(cudnn.deterministic),
        "cudnn_benchmark": bool(cudnn.benchmark),
        "cudnn_allow_tf32": bool(cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    }
    # Which other processes share this GPU right now? Concurrency changes cuBLAS
    # workspace availability and therefore kernel selection.
    env["gpu_total_mib"] = None
    env["gpu_used_mib"] = None
    try:
        import subprocess
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        env["nvidia_smi"] = q.stdout.strip().splitlines()
    except Exception as e:  # noqa: BLE001
        env["nvidia_smi"] = f"unavailable: {e}"
    return env


def batch_fingerprint(image: torch.Tensor, label: torch.Tensor) -> dict:
    """Cheap but sensitive fingerprint of one augmented batch."""
    im0 = image[:1].detach().to("cpu").contiguous()
    lb0 = label[:1].detach().to("cpu").contiguous()
    return {
        "image0_sha256": tensor_sha(im0),
        "label0_sha256": tensor_sha(lb0),
        "image_sum": float(image.detach().float().sum()),
        "image_absmax": float(image.detach().float().abs().max()),
        "label_sum": float(label.detach().float().sum()),
    }


def param_trace(model: torch.nn.Module, aux) -> dict:
    """Cheap global scalars + one slice hash, checked every step."""
    ps = [p for p in model.parameters() if p.requires_grad]
    if aux is not None:
        ps += [p for p in aux.parameters() if p.requires_grad]
    flat = torch.cat([p.detach().reshape(-1)[:SLICE] for p in ps[:4]])
    total = float(sum(float(p.detach().float().sum()) for p in ps))
    return {"slice_sha256": tensor_sha(flat), "param_sum": total}


def grad_norm(model, aux, device) -> float:
    sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq += float(p.grad.detach().float().pow(2).sum())
    if aux is not None:
        for p in aux.parameters():
            if p.grad is not None:
                sq += float(p.grad.detach().float().pow(2).sum())
    return float(np.sqrt(sq))


def build_args(ns: argparse.Namespace) -> argparse.Namespace:
    """Reproduce exactly the attribute set models.scripts.train expects."""
    ns.main_loss_weights = (1.0, 1.0, 1.0, 1.0)
    ns.dice_reduction = "batch"
    ns.use_teacher = ns.teacher_package != "none"
    ns.inWidth = 256
    ns.inHeight = 256
    ns.lr = 5e-4
    ns.lr_mode = "poly"
    ns.step_loss = 30
    ns.weight_decay = 1e-4
    ns.kd_cap = 2.0
    ns.rho = 0.25
    ns.lambda_max = 1.0
    ns.implementation_version = "cata_cd_v2"
    return ns


def run(ns: argparse.Namespace) -> dict:
    args = build_args(ns)
    if args.device == "cuda":
        torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}" if args.device == "cuda" else "cpu")

    env = numeric_env()
    set_seed(args.seed)
    # Apply the diagnostic switches AFTER the project's own set_seed() so that the
    # captured snapshot describes what actually ran.
    import torch.backends.cudnn as cudnn
    if getattr(args, "no_tf32", False):
        cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    if getattr(args, "det_algos", False):
        torch.use_deterministic_algorithms(True)
    # set_seed() flips cudnn.deterministic/benchmark, so the effective numeric
    # configuration can only be captured AFTER it runs. Re-snapshot those flags.
    env.update(numeric_env())

    model = build_model(args).to(device)
    student_params = sum(p.numel() for p in model.parameters())

    aux = None
    if args.use_teacher:
        from models.distill.kd import ChangeEvidenceHead
        aux = ChangeEvidenceHead(64, 128, scale_index=2).to(device)

    init = state_dict_hashes(model)
    if aux is not None:
        for k, v in state_dict_hashes(aux).items():
            init[f"aux_{k}"] = v

    optimizer = build_optimizer(args, list(model.parameters()) +
                                (list(aux.parameters()) if aux is not None else []))
    criterion = build_loss(args.dice_reduction)

    if args.use_teacher:
        reader = TeacherCacheReaderV2(args.teacher_cache_root, args.dataset_name)
        base = get_loader(args.data_root, os.path.join(args.data_root, "list", "train.txt"),
                          batchsize=1, trainsize=args.inWidth, num_workers=args.num_workers,
                          return_state=True, seed=args.seed).dataset
        train_loader = torch.utils.data.DataLoader(
            CataCacheDataset(base, reader, "train"), batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
            generator=torch.Generator().manual_seed(args.seed), persistent_workers=False)
    else:
        train_loader = get_loader(args.data_root, os.path.join(args.data_root, "list", "train.txt"),
                                  batchsize=args.batch_size, trainsize=args.inWidth,
                                  num_workers=args.num_workers, seed=args.seed)

    model.train()
    if aux is not None:
        aux.train()
    if hasattr(train_loader.dataset, "set_epoch"):
        train_loader.dataset.set_epoch(0)
    if getattr(train_loader, "generator", None) is not None:
        train_loader.generator.manual_seed(args.seed + 0)

    trace, gs = [], 0
    for batch in train_loader:
        if gs >= args.steps:
            break
        if args.use_teacher:
            image, target = batch[0], batch[1]
            local_change = batch[2].to(device, non_blocking=True)
            confidence = batch[3].to(device, non_blocking=True)
        else:
            image, target = unpack(batch)

        fp = batch_fingerprint(image, target)
        rng_sha = sha_bytes(torch.get_rng_state().numpy().tobytes())

        pre = image[:, :3].to(device, non_blocking=True)
        post = image[:, 3:6].to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        predictions, change = model(pre, post, return_change_features=True)
        main_loss = multiscale_loss(predictions, target, criterion, args.main_loss_weights)

        kd_loss, lam = torch.zeros((), device=device), torch.zeros((), device=device)
        if args.use_teacher:
            from models.distill.kd import dense_change_kd, gradient_budget_lambda
            kd_loss = dense_change_kd(aux(change), local_change, confidence, cap=args.kd_cap)
            lam = gradient_budget_lambda(main_loss, kd_loss, change[2], rho=args.rho,
                                         lam_max=args.lambda_max)

        total = main_loss + lam * kd_loss if args.use_teacher else main_loss
        total.backward()
        gnorm = grad_norm(model, aux, device)
        optimizer.step()

        entry = {
            "step": gs,
            "loss_total": float(total.detach()),
            "loss_main": float(main_loss.detach()),
            "loss_kd": float(kd_loss.detach()),
            "lam": float(lam.detach()),
            "grad_norm": gnorm,
            "rng_sha256": rng_sha,
            **{f"batch_{k}": v for k, v in fp.items()},
            **{f"param_{k}": v for k, v in param_trace(model, aux).items()},
        }
        trace.append(entry)
        gs += 1
        if gs % 25 == 0:
            print(f"  [j0] step {gs}/{args.steps} loss={entry['loss_total']:.6f}", flush=True)

    return {
        "_meta": {
            "purpose": "J0 determinism forensics (review section 4.2)",
            "experiment_id": args.experiment_id,
            "dataset": args.dataset_name,
            "dca_mode": args.dca_mode,
            "teacher_package": args.teacher_package,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "steps": gs,
            "student_params": student_params,
            "num_workers": args.num_workers,
            "switches": {"det_algos": bool(getattr(args, "det_algos", False)),
                         "no_tf32": bool(getattr(args, "no_tf32", False)),
                         "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")},
            "data_fingerprint": {s: sha_bytes((Path(args.data_root) / "list" / f"{s}.txt").read_bytes())
                                 for s in ("train", "val", "test")},
        },
        "numeric_env": env,
        "init_hashes": init,
        "trace": trace,
    }


def compare(paths: list[str]) -> int:
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    print(f"[j0-compare] comparing {len(reports)} runs")
    for r in reports:
        m = r["_meta"]
        e = r["numeric_env"]
        sw = m.get("switches", {})
        print(f"  - {m['experiment_id']}/{m['dataset']} steps={m['steps']} "
              f"torch={e['torch']} cudnn_det={e['cudnn_deterministic']} "
              f"cudnn_tf32={e['cudnn_allow_tf32']} det_algos={e['deterministic_algorithms_enabled']} "
              f"CUBLAS_WS={e['CUBLAS_WORKSPACE_CONFIG']} "
              f"switches={sw}")

    # 1) initial weights
    base = reports[0]["init_hashes"]
    init_ok = True
    for r in reports[1:]:
        diff = [k for k in base if k.endswith("sha256") or isinstance(base[k], str)]
        bad = [k for k in diff if base.get(k) != r["init_hashes"].get(k)]
        if bad:
            init_ok = False
            print(f"  INIT MISMATCH vs {r['_meta']['experiment_id']}: {bad}")
    print(f"\n[1] initial weights identical across runs: {init_ok}")

    # 2/3) per-step divergence
    n = min(len(r["trace"]) for r in reports)
    first_div = None
    for i in range(n):
        rows = [r["trace"][i] for r in reports]
        for key in ("batch_image0_sha256", "batch_label0_sha256", "rng_sha256", "loss_total",
                    "grad_norm", "param_slice_sha256"):
            vals = [row.get(key) for row in rows]
            same = all(v == vals[0] for v in vals) if isinstance(vals[0], (str, type(None))) \
                else (max(vals) - min(vals) <= 0.0)
            if not same:
                first_div = {"step": i, "field": key, "values": vals}
                break
        if first_div:
            break
    if first_div is None:
        print(f"[2] first divergence: NONE within {n} steps -> the two runs are "
              f"step-for-step reproducible on this configuration")
    else:
        print(f"[2] FIRST DIVERGENCE at step {first_div['step']} in '{first_div['field']}': "
              f"{first_div['values']}")
    if reports[0]["trace"]:
        t0, t1 = reports[0]["trace"][-1], reports[-1]["trace"][-1]
        print(f"[3] final loss after {n} steps: run0={t0['loss_total']:.8f} "
              f"run1={t1['loss_total']:.8f} (delta={t1['loss_total']-t0['loss_total']:+.3e})")
    return 0 if init_ok else 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs="+", default=None,
                    help="compare two or more audit JSONs instead of running")
    ap.add_argument("--experiment_id", default="J0-REP")
    ap.add_argument("--dataset_name", default="SYSU", choices=("CDD", "LEVIR", "SYSU", "WHU"))
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--dca_mode", default="moe128", choices=("none", "moe128"))
    ap.add_argument("--teacher_package", default="none")
    ap.add_argument("--teacher_cache_root", default=None)
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2333)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--pretrained_path", default=None)
    ap.add_argument("--out", default=None)
    # --- determinism switches, used ONLY to isolate the nondeterministic kernel.
    # The review (§4.2) forbids silently changing the production protocol: these
    # flags exist to find WHICH switch reproduces the baseline, not to redefine it.
    ap.add_argument("--det_algos", action="store_true",
                    help="torch.use_deterministic_algorithms(True)")
    ap.add_argument("--no_tf32", action="store_true",
                    help="disable cudnn + matmul TF32 (TF32 backward kernels are a "
                         "prime suspect for forward-identical/gradient-different runs)")
    ns = ap.parse_args()

    if ns.compare:
        raise SystemExit(compare(ns.compare))
    if not ns.data_root:
        raise SystemExit("--data_root required")
    if not ns.out:
        raise SystemExit("--out required")

    rep = run(ns)
    out = Path(ns.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"[j0] wrote {out} ({rep['_meta']['steps']} steps)")
    print("[j0] numeric-env flags to note:")
    for k in ("cudnn_deterministic", "cudnn_benchmark", "cudnn_allow_tf32", "matmul_allow_tf32",
              "float32_matmul_precision", "deterministic_algorithms_enabled",
              "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED"):
        print(f"     {k} = {rep['numeric_env'][k]}")


if __name__ == "__main__":
    main()
