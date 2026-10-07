#!/usr/bin/env python3
"""Check whether identical-config runs diverge (non-determinism) from the first epochs."""
import json
from pathlib import Path

import paramiko

cfg = json.loads((Path(__file__).resolve().parent.parent / ".vscode" / "sftp.json").read_text())
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(cfg["host"], cfg["port"], cfg["username"], cfg["password"], timeout=30)


def run(cmd, t=120):
    _i, o, e = c.exec_command(cmd, timeout=t)
    return o.read().decode(errors="replace")


M1 = "/home/yqwang/outputs/LS-Rep_BCD_RSML_3/CATA-CD/Run1"
TA = "/home/yqwang/outputs/LS-Rep_BCD_RSML_3/CATA-CD/teacher_adaptation"

for ds in ["SYSU", "WHU"]:
    print(f"===== {ds}: first 3 epoch lines =====")
    print(" M1 :", run(f"grep 'Epoch \\[' {M1}/{ds}/train_log.txt | head -n 3").replace("\n", "\n      "))
    print(" C1 :", run(f"grep 'Epoch \\[' {TA}/C1/{ds}/train_log.txt | head -n 3").replace("\n", "\n      "))

print("===== 同配置 run 在 WHU 上的 test F1 分布 =====")
for exp in ["C0", "C1", "TV-SAM", "TV-D2", "TV-D3N", "TV-D3S", "TV-RCLIP", "TV-MARS", "TV-ANYSAT", "TV-UNISAT", "TV-RADIO"]:
    txt = run(f"cat {TA}/{exp}/WHU/train_log.txt 2>/dev/null")
    import re
    m = re.search(r"^F1: ([\d.]+)", txt, re.M)
    print(f"  {exp:10s} WHU F1 = {float(m.group(1))*100:.2f}" if m else f"  {exp:10s} --")
c.close()
