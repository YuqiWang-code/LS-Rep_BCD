#!/usr/bin/env python3
"""Final state probe: server GPUs/jobs + local CATA-CD artifacts."""
import json
import re
import statistics as st
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parent.parent
cfg = json.loads((ROOT / ".vscode" / "sftp.json").read_text())
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(cfg["host"], cfg["port"], cfg["username"], cfg["password"], timeout=30)


def run(cmd, t=120):
    _i, o, e = c.exec_command(cmd, timeout=t)
    return o.read().decode(errors="replace")


CATA = "/home/yqwang/outputs/LS-Rep_BCD_RSML_3/CATA-CD"
print("===== GPU =====")
print(run("nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader"))
print("===== tmux =====")
print(run("tmux ls 2>&1"))
print("===== running train/cache procs =====")
print(run("ps aux | grep -E 'models.scripts.train|generate_teacher_cache' | grep -v grep | wc -l").strip())

print("\n===== CATA-CD 各组完成情况 =====")
for grp in ["teacher_adaptation", "Run1", "noise"]:
    n = run(f"grep -rl 'END TEST RESULTS' {CATA}/{grp} 2>/dev/null | wc -l").strip()
    tot = run(f"find {CATA}/{grp} -name train_log.txt 2>/dev/null | wc -l").strip()
    print(f"  {grp}: {n}/{tot} 完成")

print("\n===== 噪声底汇总（复算） =====")
NOISE = f"{CATA}/noise"
TA = f"{CATA}/teacher_adaptation"
for ds in ["SYSU", "WHU", "CDD", "LEVIR"]:
    def f1(p):
        t = run(f"cat {p} 2>/dev/null")
        m = re.search(r"^F1: ([\d.]+)", t, re.M)
        return float(m.group(1)) * 100 if (m and "END TEST RESULTS" in t) else None
    vals = {k: f1(f"{TA}/C1/{ds}/train_log.txt") if k == "C1" else f1(f"{NOISE}/{k}/{ds}/train_log.txt")
            for k in ["C1", "R1", "R2", "R3"]}
    present = [v for v in vals.values() if v is not None]
    sigma = st.stdev(present) if len(present) > 1 else float("nan")
    print(f"  {ds}: " + "  ".join(f"{k}={v:.2f}" for k, v in vals.items() if v is not None) +
          f"   sigma={sigma:.3f}pp")
c.close()
