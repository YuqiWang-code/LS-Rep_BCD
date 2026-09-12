"""Clean baseline / direction-C trainer; legacy logging and checkpoint conventions."""

from __future__ import annotations

import argparse
import datetime
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

try:
    from thop import profile
except ImportError:
    profile = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models import A2Net_LWGANet_L0, build_loss
from models.datasets.cd_dataset import get_loader, get_test_loader
from models.distill import PairedTeacherCache
from models.distill.diagnostics import H_NAMES
from models.utils.checkpoint import build_checkpoint, restore_rng_state, save_checkpoint_atomic
from models.utils.logger import TrainingLogger
from models.utils.metrics import ConfuseMatrixMeter
from models.utils.scheduler import adjust_learning_rate


EXPECTED_DEPLOY_PARAMS = 2_913_094
EXPECTED_DEPLOY_FLOPS = 2.7475e9
# THOP's operator accounting varies slightly across the supported PyTorch
# environment. RSML-3 consistently reports 2.767634G for this unchanged graph.
DEPLOY_FLOPS_ATOL = 0.03e9


EXPERIMENTS = {
    "B0": {"name": "Baseline_A2Net_LWGANet_L0", "auxiliary_mode": "none"},
    "C0": {"name": "C0_Difficulty_Reliable_Routing_Reject", "auxiliary_mode": "direction_c"},
    "C0F": {"name": "C0F_Legacy_AlignedReplay", "auxiliary_mode": "direction_c"},
    "C1": {"name": "C1_DART_R_TaskSpace", "auxiliary_mode": "direction_c"},
    "C2": {"name": "C2_TaskSpace_QualityOnly", "auxiliary_mode": "direction_c"},
    "C3": {"name": "C3_TaskSpace_OVOnly", "auxiliary_mode": "direction_c"},
    "C4": {"name": "C4_TaskSpace_SAMOnly", "auxiliary_mode": "direction_c"},
    "C5": {"name": "C5_TaskSpace_NoDifficulty", "auxiliary_mode": "direction_c"},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="A2Net clean baseline / direction C"
    )
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument(
        "--dataset_name",
        required=True,
        choices=["CDD", "LEVIR", "SYSU", "WHU"],
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--sam_cache_root", default=None)
    parser.add_argument("--ov_cache_root", default=None)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--router_lr", type=float, default=1e-3)
    parser.add_argument("--router_hidden", type=int, default=16)
    parser.add_argument("--utility_margin", type=float, default=.02)
    parser.add_argument("--boundary_radius", type=int, default=2)
    parser.add_argument("--small_area", type=int, default=64)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pretrained_path", default=None)
    parser.add_argument("--inWidth", type=int, default=256)
    parser.add_argument("--inHeight", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=40000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr_mode", default="poly", choices=["poly", "step"])
    parser.add_argument("--step_loss", type=int, default=30)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--backbone_lr_mult", type=float, default=1.0)
    parser.add_argument("--dice_reduction", default="batch", choices=["sample", "batch"])
    parser.add_argument("--main_loss_weights", default="1,1,1,1")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--kd_lambda", type=float, default=0.06)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--log_file", default="train_log.txt")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2333)
    args = parser.parse_args()
    args.main_loss_weights = tuple(float(value) for value in args.main_loss_weights.split(","))
    if len(args.main_loss_weights) != 4:
        raise ValueError("main_loss_weights must contain four comma-separated values")
    if args.batch_size <= 0 or args.max_steps <= 0 or args.num_workers < 0:
        raise ValueError("batch_size/max_steps must be positive and num_workers non-negative")
    if args.kd_lambda < 0 or args.router_lr <= 0 or args.lr <= 0:
        raise ValueError("Loss weight must be nonnegative and learning rates positive")
    if args.inWidth != args.inHeight or args.inWidth != 256:
        raise ValueError("Formal trainer preserves the existing 256x256 protocol")
    if args.pretrained and not args.pretrained_path:
        raise ValueError("--pretrained_path is required unless --no-pretrained is given")
    if args.router_hidden < 1 or args.boundary_radius < 1 or args.small_area < 1:
        raise ValueError("Invalid direction-C dimensions/thresholds")
    if not 0 <= args.utility_margin < 1:
        raise ValueError("utility_margin must be in [0,1)")
    if any(not math.isfinite(v) or v < 0 for v in args.main_loss_weights):
        raise ValueError("Invalid main loss weights")
    recipe = EXPERIMENTS[args.experiment]
    args.experiment_name = recipe["name"]
    args.auxiliary_mode = recipe["auxiliary_mode"]
    args.mechanism = "task_space" if args.experiment in {"C1", "C2", "C3", "C4", "C5"} else "legacy"
    args.cache_replay = "legacy" if args.experiment in {"B0", "C0"} else "aligned"
    args.implementation_version = "dart_r_ts_v2"
    args.routing_signal = "relative_brier_gain" if args.mechanism=="task_space" else "feature_gradient_cosine"
    args.reject_unit = "pixel" if args.mechanism=="task_space" else "image"
    args.legacy_d_r_router_fields_active = args.mechanism=="legacy"
    if args.auxiliary_mode == "direction_c" and not (args.sam_cache_root and args.ov_cache_root):
        raise ValueError("C0 requires --sam_cache_root AND --ov_cache_root")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def nested_to(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: nested_to(item, device) for key, item in value.items()}
    return value


def nested_add_batch_to(value, device):
    """Batch one Dataset sample without materializing a full training batch."""
    if torch.is_tensor(value):
        return value.unsqueeze(0).to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: nested_add_batch_to(item, device) for key, item in value.items()}
    return value


def unpack_batch(batch):
    if len(batch) == 2:
        return batch[0], batch[1], None, None
    if len(batch) == 3:
        return batch[0], batch[1], batch[2], None
    if len(batch) == 4:
        return batch[0], batch[1], batch[2], batch[3]
    raise ValueError(f"Unexpected batch structure with {len(batch)} fields")


def multiscale_loss(predictions, target, criterion, weights):
    return sum(weight * criterion(prediction, target)
               for weight, prediction in zip(weights, predictions))


def build_optimizer(args, model):
    grouped = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or name.startswith("training_auxiliary.router."):
            continue
        scale = args.backbone_lr_mult if name.startswith("backbone.") else 1.0
        grouped.setdefault(scale, []).append(parameter)
    groups = [
        {"params": parameters, "lr": args.lr * scale, "lr_scale": scale,
         "weight_decay": args.weight_decay}
        for scale, parameters in grouped.items()
    ]
    return torch.optim.Adam(groups, lr=args.lr, betas=(0.9, 0.99), eps=1e-8)


def build_router_optimizer(args, model):
    if not model.use_training_auxiliary or not hasattr(model.training_auxiliary, "router"):
        return None
    return torch.optim.Adam(model.training_auxiliary.router.parameters(), lr=args.router_lr)


def build_model(args):
    if getattr(args, 'mechanism', 'legacy') == 'task_space':
        cfg = dict(mechanism='task_space', boundary_radius=args.boundary_radius,
                   small_area=args.small_area, policy='quality' if args.experiment=='C2' else 'advantage',
                   teacher={'C3':'ov', 'C4':'sam'}.get(args.experiment, 'both'),
                   difficulty=args.experiment!='C5')
        return A2Net_LWGANet_L0(pretrained=args.pretrained, pretrained_path=args.pretrained_path,
                                auxiliary_mode=args.auxiliary_mode, routing_cfg=cfg)
    cfg = dict(hidden=args.router_hidden, utility_margin=args.utility_margin,
               boundary_radius=args.boundary_radius, small_area=args.small_area)
    return A2Net_LWGANet_L0(pretrained=args.pretrained, pretrained_path=args.pretrained_path,
                            auxiliary_mode=args.auxiliary_mode, routing_cfg=cfg)


def train_epoch(args, loader, model, criterion, optimizer, epoch, global_step, device,
                router_optimizer=None):
    model.train()
    loader.dataset.set_epoch(epoch)
    loader.generator.manual_seed(args.seed + epoch)
    meter = ConfuseMatrixMeter(n_class=2)
    keys = ("total", "main", "aux_raw", "aux_weighted", "aux_ratio", "router_loss",
            "reject_ratio", "target_reject_ratio", "router_accuracy", "sam_boundary", "sam_relation",
            "ov_response", "ov_relation", "data_time", "step_time")
    keys += tuple("h_"+name for name in H_NAMES)
    keys += tuple(prefix+name for prefix in ("q_", "d_", "r_", "w_", "effective_") for name in ("sam", "ov"))
    keys += ("w_reject", "pixel_reject_ratio", "image_reject_ratio", "accepted_change_ratio",
             "accepted_bg_ratio", "effective_mass", "student_brier", "sam_transport", "ov_task")
    keys += tuple(prefix+name for prefix in ("proposal_gain_", "relative_gain_", "eligible_ratio_",
                  "available_ratio_", "proposal_brier_") for name in ("sam", "ov"))
    keys += ("probe_cls_kd_gt_ratio", "probe_cls_kd_gt_cosine")
    totals = dict.fromkeys(keys, 0.)
    batches, last_lr = 0, args.lr
    data_started = time.perf_counter()
    probe_ratio, probe_cosine = 0., 0.
    for batch in loader:
        data_elapsed = time.perf_counter()-data_started
        step_started = time.perf_counter()
        if global_step >= args.max_steps:
            break
        image,target,_,pack = unpack_batch(batch)
        pre,post=image[:,:3].to(device),image[:,3:6].to(device)
        target=target.to(device).float()
        pack=nested_to(pack,device) if pack is not None else None
        last_lr=adjust_learning_rate(args,optimizer,epoch,global_step,len(loader))
        optimizer.zero_grad(set_to_none=True)
        if router_optimizer is not None:router_optimizer.zero_grad(set_to_none=True)
        predictions,auxiliary=model(pre,post,target=target,teacher_pack=pack,
                                    compute_auxiliary=args.auxiliary_mode!="none")
        main_loss=multiscale_loss(predictions,target,criterion,args.main_loss_weights)
        zero=main_loss.new_zeros(())
        detail=auxiliary.get("direction_c",{})
        aux_raw=detail.get("total",zero)
        aux_weighted=args.kd_lambda*aux_raw
        router_loss=detail.get("router_loss",zero)
        loss=main_loss+aux_weighted
        if batches == 0 and detail:
            # Diagnostic ONLY: exact weighted KD and actual four-scale GT loss
            # at final classifier weights. Never used to gate or train the router.
            cls = model.decoder.cls.weight
            gm = torch.autograd.grad(main_loss, cls, retain_graph=True)[0].detach().float()
            gk = torch.autograd.grad(aux_weighted, cls, retain_graph=True, allow_unused=True)[0]
            if gk is not None:
                gk = gk.detach().float()
                probe_ratio = float(gk.norm()/gm.norm().clamp_min(1e-12))
                probe_cosine = float((gm*gk).sum()/(gm.norm()*gk.norm()).clamp_min(1e-12))
        if not torch.isfinite(loss) or not torch.isfinite(router_loss):
            raise FloatingPointError("Non-finite training loss; checkpoint not overwritten")
        loss.backward()
        if router_optimizer is not None:
            # Router sees only detached features/statistics; this has no student gradient.
            router_loss.backward()
        optimizer.step()
        if router_optimizer is not None:router_optimizer.step()
        prediction=(predictions[0].detach()>.5).long()
        current_f1=meter.update_cm(prediction.cpu().numpy(),target.cpu().numpy())
        aux_ratio=aux_weighted.detach()/main_loss.detach().clamp_min(1e-8)
        values={key:zero for key in keys}
        values.update(total=loss,main=main_loss,aux_raw=aux_raw,aux_weighted=aux_weighted,
                      aux_ratio=aux_ratio,router_loss=router_loss)
        if detail and "target_action" in detail:
            action=detail["action"];target_action=detail["target_action"]
            values.update(reject_ratio=(action==2).float().mean(),
                          target_reject_ratio=(target_action==2).float().mean(),
                          router_accuracy=(action==target_action).float().mean())
            for j,name in enumerate(H_NAMES):values["h_"+name]=detail["h"][:,j].mean()
            for key,prefix in (("q","q_"),("d","d_"),("r","r_"),
                               ("weights","w_"),("effective_weights","effective_")):
                for j,name in enumerate(("sam","ov")):values[prefix+name]=detail[key][:,j].mean()
            values["w_reject"]=detail["weights"][:,2].mean()
            for name in ("sam_boundary","sam_relation","ov_response","ov_relation"):
                values[name]=detail[name]
        elif detail:
            for j,name in enumerate(H_NAMES): values["h_"+name]=detail["h"][:,j].mean()
            for key,prefix in (("q","q_"),("weights","w_"),("effective_weights","effective_"),
                               ("proposal_gain","proposal_gain_"),("relative_gain","relative_gain_"),
                               ("eligible_ratio","eligible_ratio_"),("available_ratio","available_ratio_"),
                               ("proposal_brier","proposal_brier_")):
                for j,name in enumerate(("sam","ov")): values[prefix+name]=detail[key][:,j].mean()
            values["w_reject"]=detail["weights"][:,2].mean()
            values["reject_ratio"]=detail["pixel_reject_ratio"]
            for name in ("pixel_reject_ratio", "image_reject_ratio", "accepted_change_ratio",
                         "accepted_bg_ratio", "effective_mass", "student_brier", "sam_transport", "ov_task"):
                values[name]=detail[name]
        values["data_time"]=zero.new_tensor(data_elapsed)
        values["probe_cls_kd_gt_ratio"]=zero.new_tensor(probe_ratio)
        values["probe_cls_kd_gt_cosine"]=zero.new_tensor(probe_cosine)
        values["step_time"]=zero.new_tensor(time.perf_counter()-step_started)
        for key,value in values.items():totals[key]+=float(value.detach())
        batches+=1;global_step+=1
        if global_step%5==0:
            print(f"\rstep [{global_step}/{args.max_steps}] F1={current_f1:.3f} "
                  f"lr={last_lr:.7f} loss={float(loss.detach()):.3f} "
                  f"aux_ratio={float(aux_ratio):.3f} reject={float(values['reject_ratio']):.3f} "
                  f"data={data_elapsed:.3f}s",end="")
        data_started=time.perf_counter()
    if not batches:raise RuntimeError("No training batches; check batch size and max_steps")
    averages = {key:value/batches for key,value in totals.items()}
    if getattr(args, 'mechanism', 'legacy') == 'task_space':
        # Never fabricate v1 cosine/reliability/router accuracy values for v2.
        inactive = {'router_loss','router_accuracy','target_reject_ratio',
                    'sam_boundary','sam_relation','ov_response','ov_relation',
                    'd_sam','d_ov','r_sam','r_ov'}
        averages = {key:value for key,value in averages.items() if key not in inactive}
    return (averages,meter.get_scores(),last_lr,global_step)


@torch.no_grad()
def evaluate(loader, model, criterion, weights, device):
    model.eval()
    meter = ConfuseMatrixMeter(n_class=2)
    losses = []
    for batch in loader:
        image, target, _, _ = unpack_batch(batch)
        pre = image[:, :3].to(device, non_blocking=True)
        post = image[:, 3:6].to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).float()
        predictions = model(pre, post)
        loss = multiscale_loss(predictions, target, criterion, weights)
        meter.update_cm((predictions[0] > 0.5).long().cpu().numpy(), target.cpu().numpy())
        losses.append(float(loss))
    return sum(losses) / max(len(losses), 1), meter.get_scores()


@torch.no_grad()
def deploy_consistency(model, loader, device):
    model.eval()
    image, _, _, _ = unpack_batch(next(iter(loader)))
    pre, post = image[:1, :3].to(device), image[:1, 3:6].to(device)
    before = model(pre, post)
    model.switch_to_deploy().eval()
    after = model(pre, post)
    max_error = max((left - right).abs().max().item() for left, right in zip(before, after))
    if max_error >= 1e-6:
        raise RuntimeError(f"Deploy consistency failed: max_error={max_error:.8e}")
    return max_error


@torch.no_grad()
def auxiliary_toggle_consistency(model, dataset, device):
    """Prove the loss-only branch cannot alter the student's main output."""
    sample = dataset[0]
    image, target, _, teacher_pack = unpack_batch(sample)
    image = image.unsqueeze(0).to(device, non_blocking=True)
    target = target.unsqueeze(0).to(device, non_blocking=True).float()
    teacher_pack = nested_add_batch_to(teacher_pack, device) \
        if teacher_pack is not None else None

    # Keep every child module in deterministic eval mode while making only the
    # root execute its training return/auxiliary branch.
    model.eval()
    model.training = True
    without_auxiliary, _ = model(
        image[:, :3], image[:, 3:6], target=target,
        teacher_pack=teacher_pack, compute_auxiliary=False,
    )
    with_auxiliary, _ = model(
        image[:, :3], image[:, 3:6], target=target,
        teacher_pack=teacher_pack, compute_auxiliary=True,
    )
    model.training = False
    max_error = max(
        (left - right).abs().max().item()
        for left, right in zip(without_auxiliary, with_auxiliary)
    )
    if max_error != 0.0:
        raise RuntimeError(f"Auxiliary changed the main output: max_error={max_error:.8e}")
    return max_error


@torch.no_grad()
def test_deployed(args, loader, model, device):
    model.eval()
    meter = ConfuseMatrixMeter(n_class=2)
    for batch in loader:
        image, target, _, _ = unpack_batch(batch)
        output = model(
            image[:, :3].to(device, non_blocking=True),
            image[:, 3:6].to(device, non_blocking=True),
        )[0]
        meter.update_cm(
            (output > 0.5).long().cpu().numpy(), target.cpu().numpy(),
        )
    infer_params = sum(parameter.numel() for parameter in model.parameters())
    flops = None
    if profile is not None:
        dummy = torch.randn(1, 3, args.inHeight, args.inWidth, device=device)
        flops, _ = profile(model, inputs=(dummy, dummy), verbose=False)
    return meter.get_scores(), infer_params, flops


def append_test_results(logger, args, scores, train_params, infer_params, flops,
                        auxiliary_error, deploy_error, elapsed):
    """The test record lives at the end of train_log.txt; no test_infer directory."""
    logger.log_message("=" * 100)
    logger.log_message("=== TEST RESULTS ===")
    logger.log_message("Metric Split: test")
    logger.log_message(f"Dataset: {args.dataset_name}")
    logger.log_message(f"Experiment: {args.experiment}/{args.experiment_name}")
    logger.log_message(f"Train Params: {train_params / 1e6:.4f}M")
    logger.log_message(f"Infer Params: {infer_params / 1e6:.4f}M")
    logger.log_message(f"FLOPs: {flops / 1e9:.4f}G" if flops is not None else "FLOPs: unavailable")
    logger.log_message(f"Auxiliary toggle max error: {auxiliary_error:.8e}")
    logger.log_message(f"Deploy max error: {deploy_error:.8e}")
    labels = {
        "recall": "Recall", "precision": "Precision", "F1": "F1",
        "IoU": "IoU", "OA": "OA", "Kappa": "Kappa",
    }
    for key, label in labels.items():
        logger.log_message(f"{label}: {scores[key]:.6f}")
    logger.log_message(f"Total time: {elapsed}")
    logger.log_message("=== END TEST RESULTS ===")


def validate_resume(args, checkpoint):
    if checkpoint.get("format_version") != 2:
        raise ValueError("Legacy checkpoint cannot resume direction-C package; start a fresh run")
    saved = checkpoint.get("args", {})
    keys=("implementation_version", "experiment", "dataset_name", "batch_size", "max_steps", "seed",
          "lr", "lr_mode", "step_loss", "weight_decay", "backbone_lr_mult", "dice_reduction",
          "main_loss_weights", "router_lr", "router_hidden", "utility_margin", "boundary_radius",
          "small_area", "kd_lambda", "inWidth", "inHeight", "data_fingerprint", "cache_fingerprint",
          "mechanism", "cache_replay")
    for key in keys:
        if saved.get(key) != getattr(args,key,None):
            raise ValueError(f"Resume configuration mismatch for {key}: {saved.get(key)!r} vs {getattr(args,key,None)!r}")


def main():
    args = parse_args()
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use --device cpu only for local verification")
        torch.cuda.set_device(args.gpu_id)
    device = torch.device(f"cuda:{args.gpu_id}" if args.device=="cuda" else "cpu")
    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args).to(device)
    train_params = sum(parameter.numel() for parameter in model.parameters())
    auxiliary_params = sum(
        parameter.numel() for name, parameter in model.named_parameters()
        if name.startswith("training_auxiliary.")
    )
    optimizer = build_optimizer(args, model)
    router_optimizer = build_router_optimizer(args, model)
    criterion = build_loss(args.dice_reduction)

    import hashlib
    args.data_fingerprint={split:hashlib.sha256((Path(args.data_root)/"list"/(split+".txt")).read_bytes()).hexdigest()
                           for split in ("train","val","test")}
    split_ids={split:[v.strip() for v in (Path(args.data_root)/"list"/(split+".txt")).read_text().splitlines() if v.strip()]
               for split in ("train","val","test")}
    for split,ids in split_ids.items():
        if not ids or len(ids)!=len(set(ids)):
            raise ValueError(f"Empty split or duplicate IDs in {split}")
    if any(set(split_ids[a]) & set(split_ids[b]) for a,b in
           (("train","val"),("train","test"),("val","test"))):
        raise ValueError("Sample IDs overlap between train/val/test")
    train_list=[v.strip() for v in (Path(args.data_root)/"list/train.txt").read_text().splitlines() if v.strip()]
    teacher_cache = PairedTeacherCache(args.sam_cache_root,args.ov_cache_root,args.dataset_name,train_list) \
        if args.auxiliary_mode == "direction_c" else None
    args.cache_fingerprint=teacher_cache.fingerprint if teacher_cache else None
    train_loader = get_loader(
        args.data_root, os.path.join(args.data_root, "list", "train.txt"),
        batchsize=args.batch_size, trainsize=args.inWidth,
        num_workers=args.num_workers, teacher_cache=teacher_cache, seed=args.seed,
        cache_replay=args.cache_replay,
    )
    if len(train_loader)==0:
        raise ValueError("Training dataset smaller than batch_size with drop_last=True")
    val_loader = get_test_loader(
        args.data_root, os.path.join(args.data_root, "list", "val.txt"),
        batchsize=args.batch_size, testsize=args.inWidth,
        num_workers=args.num_workers, return_meta=True,
    )
    test_loader = get_test_loader(
        args.data_root, os.path.join(args.data_root, "list", "test.txt"),
        batchsize=args.batch_size, testsize=args.inWidth,
        num_workers=args.num_workers, return_meta=True,
    )
    args.max_epochs = int(np.ceil(args.max_steps / len(train_loader)))

    start_epoch, global_step, best_val_f1 = 0, 0, -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_resume(args, checkpoint)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if router_optimizer is not None:
            if checkpoint.get("router_optimizer") is None:raise ValueError("Missing router optimizer state")
            router_optimizer.load_state_dict(checkpoint["router_optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        best_val_f1 = checkpoint["best_val_f1"]
        restore_rng_state(checkpoint.get("rng"))

    log_path = Path(args.log_file)
    if not log_path.is_absolute():
        log_path = save_dir / log_path
    if not args.resume and (log_path.exists() or (save_dir/"last_checkpoint.pth").exists()):
        raise FileExistsError("Existing run found: use --resume or a fresh save/log directory")
    logger = TrainingLogger(
        str(log_path),
        {
            **vars(args),
            "train_params": f"{train_params / 1e6:.4f}M",
            "auxiliary_params": f"{auxiliary_params / 1e6:.6f}M",
            "n_train": len(train_loader.dataset), "n_val": len(val_loader.dataset),
            "n_test": len(test_loader.dataset),
        },
        append=bool(args.resume),
    )
    best_path = None
    if args.resume:
        candidate = save_dir / f"best_model_F1={best_val_f1:.6f}.pth"
        if candidate.is_file():
            best_path = candidate
    last_path = save_dir / "last_checkpoint.pth"
    started = datetime.datetime.now()

    for epoch in range(start_epoch, args.max_epochs):
        if global_step >= args.max_steps:break
        losses, _, lr, global_step = train_epoch(
            args, train_loader, model, criterion, optimizer,
            epoch, global_step, device, router_optimizer,
        )
        val_loss, val_scores = evaluate(
            val_loader, model, criterion, args.main_loss_weights, device,
        )
        is_best = val_scores["F1"] > best_val_f1
        if is_best:
            best_val_f1 = val_scores["F1"]
        checkpoint = build_checkpoint(
            model, optimizer, epoch, global_step, best_val_f1, args, router_optimizer,
        )
        save_checkpoint_atomic(checkpoint, last_path)
        if is_best:
            new_best = save_dir / f"best_model_F1={best_val_f1:.6f}.pth"
            save_checkpoint_atomic(checkpoint, new_best)
            if best_path is not None and best_path != new_best and best_path.name.startswith("best_model_F1="):
                best_path.unlink(missing_ok=True)
            best_path = new_best
        logger.log_epoch(
            epoch, args.max_epochs, losses,
            {"f1": val_scores["F1"], "iou": val_scores["IoU"],
             "kappa": val_scores["Kappa"], "recall": val_scores["recall"],
             "precision": val_scores["precision"], "oa": val_scores["OA"]},
            lr, torch.cuda.max_memory_allocated(device) / 1e9 if device.type=="cuda" else 0., is_best,
        )
        logger.log_message(f"Val loss: {val_loss:.6f}; global_step: {global_step}")
        if global_step >= args.max_steps:
            break

    if best_path is None or not best_path.is_file():
        raise RuntimeError("Training finished without a validation-selected best checkpoint")
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    auxiliary_error = auxiliary_toggle_consistency(model, train_loader.dataset, device)
    deploy_error = deploy_consistency(model, test_loader, device)
    scores, infer_params, flops = test_deployed(args, test_loader, model, device)
    if infer_params != EXPECTED_DEPLOY_PARAMS:
        raise RuntimeError(f"Unexpected deploy parameter count: {infer_params:,}")
    if flops is not None and abs(flops - EXPECTED_DEPLOY_FLOPS) > DEPLOY_FLOPS_ATOL:
        raise RuntimeError(f"Unexpected deploy FLOPs: {flops / 1e9:.6f}G")
    append_test_results(
        logger, args, scores, train_params, infer_params, flops,
        auxiliary_error, deploy_error, datetime.datetime.now() - started,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
