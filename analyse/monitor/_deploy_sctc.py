#!/usr/bin/env python3
"""Deploy SCTC Run1 to RSML-3: sync code/scripts, smoke, dry run, launch training.

Usage:
    python -B analyse/monitor/_deploy_sctc.py sync     # SFTP upload models/*.py + SCTC/Run1 scripts
    python -B analyse/monitor/_deploy_sctc.py smoke    # run smoke_temporal_calibration on GPU 0
    python -B analyse/monitor/_deploy_sctc.py dry      # short real-data dry run (S2/SYSU, max_steps 5)
    python -B analyse/monitor/_deploy_sctc.py train    # launch Phase-1 tmux queue on GPU 0
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
PRETRAINED = f"{REMOTE_ROOT}/pre-trained_weights/lwganet_l0_e299.pth"


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
    for p in sorted((LOCAL_ROOT / "train_scripts" / "SCTC" / "Run1").rglob("*")):
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
    cmd = "python -m models.tools.smoke_temporal_calibration --device cuda --gpu_id 0"
    print(f"\n{'=' * 70}\nSMOKE: S0/S1/S2 SCTC synthetic contract\n{'=' * 70}")
    full = f"{CONDA} && cd {REMOTE_ROOT} && {cmd}"
    _, _, code = run(c, full, stream=True, timeout=1800)
    print(f"\n[smoke exit={code}]")
    if code != 0:
        sys.exit("smoke failed")


def dry(c):
    exp, ds = "S2", "SYSU"
    ds_folder = f"{ds}-CD-256"
    save_dir = f"/home/yqwang/outputs/LS-Rep_BCD_RSML_3/SCTC/_drytest/{exp}/{ds}"
    cmd = (
        f"python -m models.scripts.train --experiment {exp} "
        f"--dataset_name {ds} --data_root /share_datasets/CD/{ds_folder} "
        f"--pretrained --pretrained_path {PRETRAINED} "
        f"--gpu_id 0 --batch_size 64 --max_steps 5 --seed 2333 "
        f"--save_dir {save_dir} --log_file train_log.txt"
    )
    print(f"\n{'=' * 70}\nDRY RUN: {exp}/{ds} (max_steps=5, real data, verifies FLOPs + pipeline)\n{'=' * 70}")
    run(c, f"rm -rf {save_dir}", stream=False)
    full = f"{CONDA} && cd {REMOTE_ROOT} && {cmd}"
    _, _, code = run(c, full, stream=True, timeout=1800)
    print(f"\n[dry exit={code}]")
    if code != 0:
        sys.exit("dry run failed")
    # Clean up the throwaway dry-test directory.
    run(c, f"rm -rf {save_dir}", stream=False)
    print("dry-test directory removed.")


def train(c):
    name = "sctc1_gpu0"
    script = "train_scripts/SCTC/Run1/run_gpu0_whu_sysu.sh"
    run(c, f"tmux kill-session -t {name} 2>/dev/null; true", stream=False)
    cmd = f"tmux new -s {name} -d 'cd {REMOTE_ROOT} && {CONDA} && bash {script}'"
    run(c, cmd, stream=False)
    print(f"Launched tmux session: {name} (gpu 0)")
    print("\nVerify with: tmux ls  (expect sctc1_gpu0) and pgrep -af models.scripts.train")


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
