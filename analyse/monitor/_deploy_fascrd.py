#!/usr/bin/env python3
"""Deploy FA-SCRD Run1 to RSML-3: sync, smoke, prepare teacher cache, train.

Usage:
    python -B analyse/monitor/_deploy_fascrd.py sync      # upload models/*.py (incl thirdparty) + FA-SCRD/Run1 scripts
    python -B analyse/monitor/_deploy_fascrd.py smoke     # run smoke_fa_scrd on GPU 0
    python -B analyse/monitor/_deploy_fascrd.py teacher   # fine-tune teacher + generate SYSU/WHU cache
    python -B analyse/monitor/_deploy_fascrd.py train     # launch Phase-1 training queue on GPU 0
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
DINO3_WEIGHT = f"{REMOTE_ROOT}/pre-trained_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"


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
    for p in sorted((LOCAL_ROOT / "train_scripts" / "FA-SCRD" / "Run1").rglob("*")):
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
    cmd = f"python -m models.tools.smoke_fa_scrd --device cuda --gpu_id 0 --weight_path {DINO3_WEIGHT}"
    print(f"\n{'=' * 70}\nSMOKE: FA-SCRD\n{'=' * 70}")
    full = f"{CONDA} && cd {REMOTE_ROOT} && {cmd}"
    _, _, code = run(c, full, stream=True, timeout=1200)
    print(f"\n[smoke exit={code}]")
    if code != 0:
        sys.exit("smoke failed")


def teacher(c):
    script = "train_scripts/FA-SCRD/Run1/prepare_teacher_cache.sh"
    print(f"\n{'=' * 70}\nTEACHER PREP: fine-tune + generate SYSU/WHU cache\n{'=' * 70}")
    full = f"{CONDA} && cd {REMOTE_ROOT} && bash {script} 0"
    _, _, code = run(c, full, stream=True, timeout=7200)
    print(f"\n[teacher exit={code}]")
    if code != 0:
        sys.exit("teacher prep failed")


def train(c):
    name = "fascrd1_gpu0"
    script = "train_scripts/FA-SCRD/Run1/run_gpu0_sysu_whu.sh"
    run(c, f"tmux kill-session -t {name} 2>/dev/null; true", stream=False)
    cmd = f"tmux new -s {name} -d 'cd {REMOTE_ROOT} && {CONDA} && bash {script} 0'"
    run(c, cmd, stream=False)
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
        elif stage == "teacher":
            teacher(c)
        elif stage == "train":
            train(c)
        else:
            sys.exit(__doc__)
    finally:
        c.close()


if __name__ == "__main__":
    main()
