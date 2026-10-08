#!/usr/bin/env python3
"""Download Stage-3 inputs (v1/v2 signatures + probes) from the server."""
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

for sub in ("signatures", "probes", "signatures_v2", "probes_v2"):
    try:
        names = [n for n in sftp.listdir(f"{REMOTE}/{sub}") if n.endswith(".json")]
    except IOError:
        print(f"{sub}: not found")
        continue
    dst = LOCAL / sub
    dst.mkdir(parents=True, exist_ok=True)
    for n in names:
        sftp.get(f"{REMOTE}/{sub}/{n}", str(dst / n))
    print(f"{sub}: {len(names)} files -> {dst}")

# teacher metadata (written by the module; keep a JSON copy for traceability)
meta = json.loads((ROOT / "models" / "agent" / "teacher_metadata.py").read_text().split("TEACHER_METADATA")[1]
                  .split("}")[0].split("=", 1)[1].replace("'", '"') + "}") if False else None
try:
    import sys
    sys.path.insert(0, str(ROOT))
    from models.agent.teacher_metadata import as_json
    (LOCAL / "teacher_metadata.json").write_text(json.dumps(as_json(), indent=2), encoding="utf-8")
    print("teacher_metadata.json written")
except Exception as exc:  # noqa: BLE001
    print("teacher_metadata skipped:", exc)

sftp.close()
c.close()
