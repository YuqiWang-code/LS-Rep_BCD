#!/usr/bin/env python3
"""Deploy SAGE-CD Run1 to RSML-3: sync, smoke, prepare cache, train.

Usage:
    python -B analyse/monitor/_deploy_sage.py sync
    python -B analyse/monitor/_deploy_sage.py smoke
    python -B analyse/monitor/_deploy_sage.py cache   # generate SYSU/WHU DINO semantic cache
    python -B analyse/monitor/_deploy_sage.py train
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
    for p in sorted((LOCAL_ROOT / "train_scripts" / "SAGE-CD").rglob("*")):
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
    cmd = "python -m models.tools.smoke_sage --device cuda --gpu_id 0"
    print(f"\n{'=' * 70}\nSMOKE: SAGE-CD\n{'=' * 70}")
    _, _, code = run(c, f"{CONDA} && cd {REMOTE_ROOT} && {cmd}", stream=True, timeout=1200)
    print(f"\n[smoke exit={code}]")
    if code != 0:
        sys.exit("smoke failed")


def cache(c):
    script = "train_scripts/SAGE-CD/Run2/prepare_sam_cache.sh"
    print(f"\n{'=' * 70}\nSAM CACHE: generate SYSU/WHU structural cache\n{'=' * 70}")
    _, _, code = run(c, f"{CONDA} && cd {REMOTE_ROOT} && bash {script} 0", stream=True, timeout=7200)
    print(f"\n[cache exit={code}]")
    if code != 0:
        sys.exit("cache prep failed")


def train(c):
    for name, script in [("sage2_gpu0_sysu", "train_scripts/SAGE-CD/Run2/run_gpu0_sysu.sh"),
                         ("sage2_gpu0_whu", "train_scripts/SAGE-CD/Run2/run_gpu0_whu.sh")]:
        run(c, f"tmux kill-session -t {name} 2>/dev/null; true", stream=False)
        run(c, f"tmux new -s {name} -d 'cd {REMOTE_ROOT} && {CONDA} && bash {script} 0'", stream=False)
        print(f"Launched tmux session: {name} (gpu 0)")


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
        elif stage == "cache":
            cache(c)
        elif stage == "train":
            train(c)
        else:
            sys.exit(__doc__)
    finally:
        c.close()


if __name__ == "__main__":
    main()
