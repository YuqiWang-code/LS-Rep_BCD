#!/usr/bin/env python3
"""CATA-CD v2 deploy helper: sync code, bash-n, launch/monitor training on RSML-3.

Subcommands:
    status    GPU / weights / datasets / cache / tmux
    sync      SFTP-upload models/, train_scripts/CATA-CD/, analyse/, tests/, README.md
    bashn     bash -n every .sh under train_scripts/CATA-CD
    launch    (re)start cache-gen + training tmux queues
    monitor   tail each run's train_log + nvidia-smi + tmux ls
    sh CMD    run an arbitrary shell command on the server (conda env + project cwd)
    py CMD    run an arbitrary python command in the project conda env
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parent.parent
REMOTE = "/home/yqwang/projects/LS-Rep_BCD_RSML_3"
CONDA = "source /home/yqwang/miniforge3/etc/profile.d/conda.sh && conda activate lsrep"

_UPLOAD_DIRS = [
    ("models", "models"),
    ("train_scripts/CATA-CD", "train_scripts/CATA-CD"),
    ("analyse", "analyse"),
    ("tests", "tests"),
    ("others", "others"),
]
_UPLOAD_FILES = ["README.md"]


def conn():
    cfg = json.loads((ROOT / ".vscode" / "sftp.json").read_text())
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(cfg["host"], cfg["port"], cfg["username"], cfg["password"], timeout=30)
    return c, cfg


def run(c, cmd, timeout=1800):
    _i, o, e = c.exec_command(cmd, timeout=timeout)
    out = o.read().decode(errors="replace")
    err = e.read().decode(errors="replace")
    return out, err


def sync(c):
    sftp = c.open_sftp()

    def mkdirs(remote_dir):
        parts = remote_dir.strip("/").split("/")
        cur = ""
        for p in parts:
            cur += "/" + p
            try:
                sftp.stat(cur)
            except IOError:
                sftp.mkdir(cur)

    def put(local: Path, remote: str):
        mkdirs(remote.rsplit("/", 1)[0])
        data = local.read_bytes()
        if local.suffix == ".sh":
            data = data.replace(b"\r\n", b"\n")  # force LF
        with sftp.open(remote, "wb") as fh:
            fh.write(data)
        print(f"  up {local} -> {remote}")

    for local_rel, remote_rel in _UPLOAD_DIRS:
        local_base = ROOT / local_rel
        for f in sorted(local_base.rglob("*")):
            if not f.is_file():
                continue
            if "__pycache__" in f.parts or f.suffix in {".pyc", ".pth", ".pt", ".safetensors"}:
                continue
            rel = f.relative_to(ROOT).as_posix()
            put(f, f"{REMOTE}/{rel}")
    for f in _UPLOAD_FILES:
        put(ROOT / f, f"{REMOTE}/{f}")
    sftp.close()
    print("[sync] done")


def bashn(c):
    out, err = run(c, f"cd {REMOTE} && for s in $(find train_scripts/CATA-CD -name '*.sh'); do bash -n $s && echo OK $s || echo FAIL $s; done")
    print(out)
    if err.strip():
        print("STDERR:", err[-2000:])


def status(c):
    cmds = (
        "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader",
        "ls /share_datasets/CD_teacher_cache/",
        "tmux ls 2>&1",
    )
    for cmd in cmds:
        out, err = run(c, cmd, timeout=60)
        print(f"$ {cmd}\n{out}")
        if err.strip():
            print("STDERR:", err[:500])


def launch(c):
    run(c, "tmux kill-session -t cata_gpu0 2>/dev/null; tmux kill-session -t cata_gpu1 2>/dev/null; tmux kill-session -t cata_cache 2>/dev/null; true")
    gpu0 = (f"cd {REMOTE} && {CONDA} && bash train_scripts/CATA-CD/teacher_adaptation/run_gpu0.sh 0 "
            f"> /home/yqwang/outputs/LS-Rep_BCD_RSML_3/CATA-CD/launcher_gpu0.log 2>&1")
    gpu1 = (f"cd {REMOTE} && {CONDA} && bash train_scripts/CATA-CD/teacher_adaptation/run_gpu1.sh 1 "
            f"> /home/yqwang/outputs/LS-Rep_BCD_RSML_3/CATA-CD/launcher_gpu1.log 2>&1")
    cache = (f"cd {REMOTE} && {CONDA} && bash train_scripts/CATA-CD/teacher_adaptation/prepare_all_caches.sh 1 "
             f"> /home/yqwang/outputs/LS-Rep_BCD_RSML_3/CATA-CD/launcher_cache.log 2>&1")
    run(c, f"mkdir -p /home/yqwang/outputs/LS-Rep_BCD_RSML_3/CATA-CD")
    for name, cmd in [("cata_gpu0", gpu0), ("cata_gpu1", gpu1), ("cata_cache", cache)]:
        out, err = run(c, f"tmux new-session -d -s {name} \"{cmd}\"")
        print(f"[launch] {name}: {out.strip() or 'started'} {err.strip()[:200]}")
    import time
    time.sleep(12)
    print("---- tmux after 12s ----")
    out, _ = run(c, "tmux ls 2>&1; echo ===; ps aux | grep -E 'models.scripts.train|generate_teacher_cache_v2' | grep -v grep | awk '{print $2, $12, $13, $14}'")
    print(out)


def monitor(c):
    out, _ = run(c, "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; echo ===TMUX===; tmux ls 2>&1")
    print(out)
    out, _ = run(c, f"for f in $(find /home/yqwang/outputs/LS-Rep_BCD_RSML_3/CATA-CD -name train_log.txt 2>/dev/null); do echo \"== $f ==\"; tail -n 3 $f; done")
    print(out)
    out, _ = run(c, "for d in /share_datasets/CD_teacher_cache/CATA_CD_v2/*/; do t=$(basename $d); n=$(find $d -name '*.pt' 2>/dev/null | wc -l); echo \"$t: $n cache files\"; done")
    print(out)


def sh(c, cmd: str):
    """Run an arbitrary shell command on the server, inside the project conda env."""
    out, err = run(c, f"cd {REMOTE} && {CONDA} && {cmd}")
    print(out, end="")
    if err.strip():
        print("--- STDERR ---")
        print(err, end="")
    return out, err


def sh_main(c):
    sh(c, " ".join(sys.argv[2:]))


def py_main(c):
    sh(c, "python " + " ".join(sys.argv[2:]))


def pull_main(c):
    """Download remote files: pull <remote_path> <local_path> ... (pairs)."""
    sftp = c.open_sftp()
    pairs = sys.argv[2:]
    for i in range(0, len(pairs), 2):
        remote, local = pairs[i], Path(pairs[i + 1])
        local.parent.mkdir(parents=True, exist_ok=True)
        sftp.get(remote, str(local))
        print(f"  down {remote} -> {local} ({local.stat().st_size} bytes)")
    sftp.close()


if __name__ == "__main__":
    sub = sys.argv[1] if len(sys.argv) > 1 else "status"
    c, cfg = conn()
    try:
        {"status": status, "sync": sync, "bashn": bashn, "launch": launch,
         "monitor": monitor, "sh": sh_main, "py": py_main, "pull": pull_main}[sub](c)
    finally:
        c.close()
