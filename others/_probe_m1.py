#!/usr/bin/env python3
"""Probe M1 (Run1) progress + GPU status + ETA."""
import json
import re
from pathlib import Path

import paramiko

cfg = json.loads((Path(__file__).resolve().parent.parent / ".vscode" / "sftp.json").read_text())
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(cfg["host"], cfg["port"], cfg["username"], cfg["password"], timeout=30)


def run(cmd, t=120):
    _i, o, e = c.exec_command(cmd, timeout=t)
    return o.read().decode(errors="replace")


def fmt(sec):
    sec = int(sec)
    return f"{sec // 3600}h{(sec % 3600) // 60}m"


base = "/home/yqwang/outputs/LS-Rep_BCD_RSML_3/CATA-CD/Run1"
print("===== GPU =====")
print(run("nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader"))

print("===== tmux =====")
print(run("tmux ls 2>&1"))

print("===== M1 进度 / ETA =====")
out = run("ps -o pid,etimes,cmd -C python 2>/dev/null | grep 'experiment_id M1'")
seen = {}
for line in out.splitlines():
    m = re.search(r"--dataset_name (\w+)", line)
    if not m:
        continue
    cols = line.split()
    try:
        el = int(cols[1])
    except (IndexError, ValueError):
        continue
    ds = m.group(1)
    if ds not in seen or el > seen[ds]:
        seen[ds] = el

for ds in ["SYSU", "WHU", "CDD", "LEVIR"]:
    txt = run(f"cat {base}/{ds}/train_log.txt 2>/dev/null")
    if "=== END TEST RESULTS ===" in txt:
        m = re.search(r"=== TEST RESULTS ===\n(.*?)=== END", txt, re.S)
        f1 = re.search(r"^F1: ([\d.]+)", m.group(1), re.M) if m else None
        print(f"M1/{ds}: DONE  F1={float(f1.group(1))*100:.2f}" if f1 else f"M1/{ds}: DONE")
        continue
    steps = re.findall(r"global_step: (\d+)", txt)
    step = int(steps[-1]) if steps else 0
    el = seen.get(ds)
    if step > 0 and el:
        remain = (40000 - step) / (step / el)
        print(f"M1/{ds}: step {step}/40000 ({step/400:.1f}%), 已跑 {fmt(el)}, 预计还需 {fmt(remain)}")
    else:
        print(f"M1/{ds}: 启动中/尚未到第一次 val")
c.close()
