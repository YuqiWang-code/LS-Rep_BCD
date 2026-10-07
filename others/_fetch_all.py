#!/usr/bin/env python3
"""Download every CATA-CD train_log.txt (teacher_adaptation + Run1 + noise) to local outputs/CATA-CD/."""
import json
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parent.parent
cfg = json.loads((ROOT / ".vscode" / "sftp.json").read_text())
REMOTE_BASE = "/home/yqwang/outputs/LS-Rep_BCD_RSML_3/CATA-CD"
LOCAL_BASE = ROOT / "outputs" / "CATA-CD"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(cfg["host"], cfg["port"], cfg["username"], cfg["password"], timeout=30)
sftp = c.open_sftp()

_i, o, e = c.exec_command(f"find {REMOTE_BASE} -name train_log.txt", timeout=120)
remotes = [p for p in o.read().decode().splitlines() if p.strip()]
n = 0
for r in remotes:
    rel = Path(r).relative_to(REMOTE_BASE)
    local = LOCAL_BASE / rel
    local.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(r, str(local))
    n += 1
print(f"downloaded {n} logs -> {LOCAL_BASE}")
sftp.close()
c.close()
