#!/usr/bin/env python3
"""Probe remaining runs and estimate ETA from elapsed time + progress."""
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


def parse_etime(s):
    """[[DD-]hh:]mm:ss -> seconds"""
    s = s.strip()
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = [int(x) for x in s.split(":")]
    while len(parts) < 3:
        parts = [0] + parts
    return days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def fmt(sec):
    sec = int(sec)
    return f"{sec // 3600}h{(sec % 3600) // 60}m"


print("===== GPU =====")
print(run("nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader")[0])

print("===== 剩余 run 的进度与 ETA =====")
out = run("ps -o pid,etimes,cmd -C python 2>/dev/null | grep 'experiment_id TV-UNISAT' | grep -v grep")
base = "/home/yqwang/outputs/LS-Rep_BCD_RSML_3/CATA-CD/teacher_adaptation"
seen = {}
for line in out.splitlines():
    if "--dataset_name" not in line:
        continue
    m = re.search(r"--dataset_name (\w+)", line)
    if not m:
        continue
    ds = m.group(1)
    # elapsed seconds column
    cols = line.split()
    try:
        elapsed = int(cols[1])
    except (IndexError, ValueError):
        continue
    if ds in seen:  # keep the largest elapsed (the main process)
        continue
    seen[ds] = elapsed

for ds, elapsed in sorted(seen.items()):
    log = run(f"cat {base}/TV-UNISAT/{ds}/train_log.txt 2>/dev/null")
    steps = re.findall(r"global_step: (\d+)", log)
    step = int(steps[-1]) if steps else 0
    if step <= 0:
        print(f"TV-UNISAT/{ds}: step=? (还没到第一次 val)")
        continue
    rate = step / elapsed  # steps per second
    remain = (40000 - step) / rate
    print(f"TV-UNISAT/{ds}: step {step}/40000 ({step/400:.1f}%), 已跑 {fmt(elapsed)}, "
          f"速率 {rate:.2f} step/s, 预计还需 {fmt(remain)}")

print("===== 全部完成统计 =====")
done = 0
for exp in ["C0", "C1", "TV-SAM", "TV-D2", "TV-D3N", "TV-D3S", "TV-RCLIP", "TV-MARS", "TV-ANYSAT", "TV-UNISAT", "TV-RADIO"]:
    for ds in ["SYSU", "WHU", "CDD", "LEVIR"]:
        t = run(f"grep -c 'END TEST RESULTS' {base}/{exp}/{ds}/train_log.txt 2>/dev/null")
        if t.strip() == "1":
            done += 1
print(f"完成 {done}/44")
c.close()
