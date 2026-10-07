#!/usr/bin/env python3
"""Download all CATA-CD train_log.txt from the server into local outputs/CATA-CD/."""
import json
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parent.parent
cfg = json.loads((ROOT / ".vscode" / "sftp.json").read_text())
REMOTE_BASE = "/home/yqwang/outputs/LS-Rep_BCD_RSML_3/CATA-CD/teacher_adaptation"
LOCAL_BASE = ROOT / "outputs" / "CATA-CD" / "teacher_adaptation"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(cfg["host"], cfg["port"], cfg["username"], cfg["password"], timeout=30)
sftp = c.open_sftp()

EXPS = ["C0", "C1", "TV-SAM", "TV-D2", "TV-D3N", "TV-D3S", "TV-RCLIP", "TV-MARS", "TV-ANYSAT", "TV-UNISAT", "TV-RADIO"]
DSS = ["SYSU", "WHU", "CDD", "LEVIR"]

n = 0
for exp in EXPS:
    for ds in DSS:
        remote = f"{REMOTE_BASE}/{exp}/{ds}/train_log.txt"
        local = LOCAL_BASE / exp / ds / "train_log.txt"
        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            sftp.get(remote, str(local))
            n += 1
        except IOError as e:
            print(f"MISS {exp}/{ds}: {e}")
print(f"downloaded {n} logs -> {LOCAL_BASE}")
sftp.close()
c.close()
