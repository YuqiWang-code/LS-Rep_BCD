#!/usr/bin/env python3
"""Extract the Teacher Capability Matrix (TV-* vs C1) from completed runs."""
import json
import re
from pathlib import Path

import paramiko

cfg = json.loads((Path(__file__).resolve().parent.parent / ".vscode" / "sftp.json").read_text())
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(cfg["host"], cfg["port"], cfg["username"], cfg["password"], timeout=30)


def run(cmd, t=180):
    _i, o, e = c.exec_command(cmd, timeout=t)
    return o.read().decode(errors="replace")


base = "/home/yqwang/outputs/LS-Rep_BCD_RSML_3/CATA-CD/teacher_adaptation"
EXPS = ["C0", "C1", "TV-SAM", "TV-D2", "TV-D3N", "TV-D3S", "TV-RCLIP", "TV-MARS", "TV-ANYSAT", "TV-UNISAT", "TV-RADIO"]
DSS = ["SYSU", "WHU", "CDD", "LEVIR"]


def parse(txt):
    m = re.search(r"=== TEST RESULTS ===\n(.*?)=== END TEST RESULTS ===", txt, re.S)
    if not m:
        return None
    vals = {}
    for line in m.group(1).splitlines():
        line = line.strip()
        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip()
            if k in ("F1", "IoU", "Recall", "Precision", "OA", "Kappa"):
                vals[k] = float(v.strip())
    return vals


results = {}
for exp in EXPS:
    results[exp] = {}
    for ds in DSS:
        txt = run(f"cat {base}/{exp}/{ds}/train_log.txt 2>/dev/null")
        results[exp][ds] = parse(txt)

print("数据集 | C0 F1 | C1 F1 | 各教师 F1（相对 C1 的 Δ）")
print("-" * 110)
for ds in DSS:
    c0 = results["C0"][ds]
    c1 = results["C1"][ds]
    if not c0 or not c1:
        print(f"{ds}: C0/C1 missing")
        continue
    parts = [f"{ds}: C0={c0['F1']*100:.2f} C1={c1['F1']*100:.2f}"]
    for exp in EXPS[2:]:
        r = results[exp][ds]
        if r:
            d = r["F1"] - c1["F1"]
            parts.append(f"{exp}={r['F1']*100:.2f}({d*100:+.2f})")
        else:
            parts.append(f"{exp}=--")
    print(" ".join(parts))

print()
print("IoU 版本（相对 C1）")
print("-" * 110)
for ds in DSS:
    c1 = results["C1"][ds]
    if not c1:
        continue
    parts = [f"{ds}: C1={c1['IoU']*100:.2f}"]
    for exp in EXPS[2:]:
        r = results[exp][ds]
        if r:
            d = r["IoU"] - c1["IoU"]
            parts.append(f"{exp}={r['IoU']*100:.2f}({d*100:+.2f})")
        else:
            parts.append(f"{exp}=--")
    print(" ".join(parts))

c.close()
