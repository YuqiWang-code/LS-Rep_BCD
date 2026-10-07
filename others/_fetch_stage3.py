#!/usr/bin/env python3
"""Download Stage-3 registry inputs (signatures + probes) from the server."""
import json
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parent.parent
cfg = json.loads((ROOT / ".vscode" / "sftp.json").read_text())
REMOTE = "/share_datasets/CD_teacher_cache/CATA_CD_v2"
LOCAL = ROOT / "outputs" / "CATA-CD"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(cfg["host"], cfg["port"], cfg["username"], cfg["password"], timeout=30)
sftp = c.open_sftp()

for sub in ("signatures", "probes"):
    dst = LOCAL / sub
    dst.mkdir(parents=True, exist_ok=True)
    names = sftp.listdir(f"{REMOTE}/{sub}")
    for n in names:
        if n.endswith(".json"):
            sftp.get(f"{REMOTE}/{sub}/{n}", str(dst / n))
    print(f"{sub}: {len([n for n in names if n.endswith('.json')])} files -> {dst}")

sftp.close()
c.close()
