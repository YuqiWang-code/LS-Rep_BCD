#!/usr/bin/env python3
"""Probe noise-floor run progress + ETA + partial sigma."""
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
NOISE = f"{CATA}/noise"
TA = f"{CATA}/teacher_adaptation"
DATASETS = ["SYSU", "WHU", "CDD", "LEVIR"]
REPS = ["R1", "R2", "R3"]

print("===== GPU =====")
print(run("nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader"))

print("===== 进度 / ETA =====")
out = run("ps -o pid,etimes,cmd -C python 2>/dev/null | grep 'experiment_id R'")
seen = {}
for line in out.splitlines():
    m = re.search(r"--dataset_name (\w+)", line)
    e = re.search(r"--experiment_id (\w+)", line)
    if not (m and e):
        continue
    cols = line.split()
    try:
        el = int(cols[1])
    except (IndexError, ValueError):
        continue
    key = f"{e.group(1)}/{m.group(1)}"
    seen[key] = max(seen.get(key, 0), el)

print("  仍在跑:", ", ".join(sorted(seen.keys())) or "(none)")

print("\n===== test F1 (已完成) =====")
vals = {}
for ds in DATASETS:
    row = {}
    t = run(f"cat {TA}/C1/{ds}/train_log.txt 2>/dev/null")
    m = re.search(r"^F1: ([\d.]+)", t, re.M)
    if m:
        row["C1"] = float(m.group(1)) * 100
    for r in REPS:
        t = run(f"cat {NOISE}/{r}/{ds}/train_log.txt 2>/dev/null")
        if "=== END TEST RESULTS ===" in t:
            m = re.search(r"^F1: ([\d.]+)", t, re.M)
            if m:
                row[r] = float(m.group(1)) * 100
    vals[ds] = row
    s = "  ".join(f"{k}={v:.2f}" for k, v in row.items())
    extra = ""
    if len(row) > 1:
        vs = list(row.values())
        extra = f"   [{len(vs)} samples] sigma={st.stdev(vs):.3f}pp range={max(vs)-min(vs):.3f}pp"
    print(f"  {ds}: {s}{extra}")

print("\n===== 未完成的进度 =====")
for key, el in sorted(seen.items()):
    exp, ds = key.split("/")
    t = run(f"cat {NOISE}/{exp}/{ds}/train_log.txt 2>/dev/null")
    steps = re.findall(r"global_step: (\d+)", t)
    step = int(steps[-1]) if steps else 0
    if step:
        print(f"  {key}: step {step}/40000 ({step/400:.1f}%), 已跑 {el//3600}h{(el%3600)//60}m, "
              f"还需 ~{(40000-step)/(step/el)/3600:.1f}h")
    else:
        print(f"  {key}: 启动中")
c.close()
