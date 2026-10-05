#!/usr/bin/env python3
"""Deploy RDT-CD Run2 to RSML-3: sync code/scripts, smoke, dry run, launch training.

Usage:
    python others/_deploy_run2.py sync     # SFTP upload models/*.py + Run2 scripts
    python others/_deploy_run2.py smoke    # run smoke (D1 / D1NG / D2 variants)
    python others/_deploy_run2.py dry      # short real-data dry run (D1/SYSU)
    python others/_deploy_run2.py train    # launch two tmux training queues
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import paramiko

HOST = "172.18.232.202"
PORT = 22
USER = "yqwang"
PW = "yq009996"
REMOTE_ROOT = "/home/yqwang/projects/LS-Rep_BCD_RSML_3"
LOCAL_ROOT = Path(__file__).resolve().parent.parent.parent
CONDA = "source /home/yqwang/miniforge3/etc/profile.d/conda.sh && conda activate lsrep"


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, USER, PW, timeout=60)
    return c


def run(c, cmd, stream=True, timeout=None):
    _in, out, err = c.exec_command(cmd, timeout=timeout)
    o = out.read().decode(errors="replace")
    e = err.read().decode(errors="replace")
    if stream:
        if o:
            sys.stdout.write(o)
        if e:
            sys.stdout.write(e)
    return o, e, out.channel.recv_exit_status()


def sync(c):
    sftp = c.open_sftp()
    uploads = []
    for p in sorted((LOCAL_ROOT / "models").rglob("*.py")):
        rel = p.relative_to(LOCAL_ROOT)
        if "__pycache__" in rel.parts:
            continue
        uploads.append(rel.as_posix())
    for p in sorted((LOCAL_ROOT / "train_scripts" / "RDT-CD" / "Run2").rglob("*")):
        if p.is_file():
            uploads.append(p.relative_to(LOCAL_ROOT).as_posix())

    dirs = sorted({os.path.dirname(u) for u in uploads})
    for d in dirs:
        run(c, f"mkdir -p {REMOTE_ROOT}/{d}", stream=False)
    n = 0
    for u in uploads:
        sftp.put(str(LOCAL_ROOT / u), f"{REMOTE_ROOT}/{u}")
        n += 1
    sftp.close()
    print(f"Uploaded {n} files ({len(dirs)} dirs).")


def smoke(c):
    cmds = [
        "D1 (default, cache + full SCGR)",
        "python -m models.tools.smoke_dynamic_teacher --device cuda --gpu_id 0",
        "D1NG (--no-gradient-gate)",
        "python -m models.tools.smoke_dynamic_teacher --device cuda --gpu_id 0 --no-gradient-gate",
        "D2 (--no-cache-conditioning)",
        "python -m models.tools.smoke_dynamic_teacher --device cuda --gpu_id 0 --no-cache-conditioning",
    ]
    for label, cmd in zip(cmds[0::2], cmds[1::2]):
        print(f"\n{'=' * 70}\nSMOKE: {label}\n{'=' * 70}")
        full = f"{CONDA} && cd {REMOTE_ROOT} && {cmd}"
        _, _, code = run(c, full, stream=True)
        print(f"\n[smoke exit={code}] {label}")
        if code != 0:
            sys.exit(f"smoke failed ({label})")


def dry(c):
    exp, ds = "D1", "SYSU"
    ds_folder = f"{ds}-CD-256"
    save_dir = f"/home/yqwang/outputs/LS-Rep_BCD_RSML_3/RDT-CD/_drytest/{exp}/{ds}"
    cmd = (
        f"python -m models.scripts.train --experiment {exp} "
        f"--dataset_name {ds} --data_root /share_datasets/CD/{ds_folder} "
        f"--pretrained_path {REMOTE_ROOT}/pre-trained_weights/lwganet_l0_e299.pth "
        f"--sam_cache_root /share_datasets/CD_teacher_cache/SAMStruct/sam2.1_hiera_large/{ds_folder} "
        f"--ov_cache_root /share_datasets/CD_teacher_cache/OVCDistill/dinov2_vitb14/{ds_folder} "
        f"--gpu_id 0 --batch_size 64 --max_steps 5 --seed 2333 "
        f"--save_dir {save_dir} --log_file train_log.txt"
    )
    print(f"\n{'=' * 70}\nDRY RUN: {exp}/{ds} (max_steps=5, real data + cache)\n{'=' * 70}")
    run(c, f"rm -rf {save_dir}", stream=False)
    full = f"{CONDA} && cd {REMOTE_ROOT} && {cmd}"
    _, _, code = run(c, full, stream=True, timeout=1800)
    print(f"\n[dry exit={code}]")
    if code != 0:
        sys.exit("dry run failed")


def train(c):
    for gpu, name, script in (
        (0, "rdtcd2_gpu0", "train_scripts/RDT-CD/Run2/run_gpu0_sysu_whu.sh"),
        (1, "rdtcd2_gpu1", "train_scripts/RDT-CD/Run2/run_gpu1_cdd_levir.sh"),
    ):
        # kill stale session with same name if any, then launch fresh
        run(c, f"tmux kill-session -t {name} 2>/dev/null; true", stream=False)
        cmd = (
            f"tmux new -s {name} -d "
            f"'cd {REMOTE_ROOT} && {CONDA} && bash {script}'"
        )
        run(c, cmd, stream=False)
        print(f"Launched tmux session: {name} (gpu {gpu})")
    print("\nTraining queues launched. Verify with: tmux ls")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    stage = sys.argv[1]
    c = connect()
    try:
        if stage == "sync":
            sync(c)
        elif stage == "smoke":
            smoke(c)
        elif stage == "dry":
            dry(c)
        elif stage == "train":
            train(c)
        else:
            sys.exit(__doc__)
    finally:
        c.close()


if __name__ == "__main__":
    main()
