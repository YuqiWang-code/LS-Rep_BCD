#!/usr/bin/env python3
"""Poll the server until all 12 noise-floor runs finish, then report sigma.

Prints a final summary and exits. Runs as a long-lived background job.
"""
import json
import re
import statistics as st
import time
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parent.parent
cfg = json.loads((ROOT / ".vscode" / "sftp.json").read_text())

CATA = "/home/yqwang/outputs/LS-Rep_BCD_RSML_3/CATA-CD"
NOISE = f"{CATA}/noise"
TA = f"{CATA}/teacher_adaptation"
RUN1 = f"{CATA}/Run1"

DATASETS = ["SYSU", "WHU", "CDD", "LEVIR"]
REPS = ["R1", "R2", "R3"]
# M1 used teacher=none only for SYSU/WHU (CDD->dinov3_lvd, LEVIR->anysat)
M1_IS_C1CONFIG = {"SYSU", "WHU"}


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(cfg["host"], cfg["port"], cfg["username"], cfg["password"], timeout=30)
    return c


def f1_of(c, path):
    try:
        _i, o, _e = c.exec_command(f"cat {path} 2>/dev/null", timeout=60)
        txt = o.read().decode(errors="replace")
    except Exception:
        return None
    if "=== END TEST RESULTS ===" not in txt:
        return None
    m = re.search(r"^F1: ([\d.]+)", txt, re.M)
    return float(m.group(1)) * 100 if m else None


def main():
    c = connect()
    n_done = 0
    for attempt in range(90):  # ~13.5h
        c2 = None
        try:
            done = 0
            for ds in DATASETS:
                for r in REPS:
                    if f1_of(c, f"{NOISE}/{r}/{ds}/train_log.txt") is not None:
                        done += 1
            ts = time.strftime("%H:%M:%S")
            print(f"[watch {attempt}] {ts} noise runs complete: {done}/12", flush=True)
            if done == 12:
                n_done = done
                break
        except Exception as exc:  # noqa: BLE001
            print(f"[watch {attempt}] connection issue: {exc}", flush=True)
            try:
                c.close()
            except Exception:
                pass
            time.sleep(30)
            c = connect()
            continue
        time.sleep(540)

    print("\n===== NOISE FLOOR (C1 config, test F1 pp) =====", flush=True)
    rows = {}
    for ds in DATASETS:
        vals_3way = {"C1": f1_of(c, f"{TA}/C1/{ds}/train_log.txt")}
        for r in REPS:
            vals_3way[r] = f1_of(c, f"{NOISE}/{r}/{ds}/train_log.txt")
        m1 = f1_of(c, f"{RUN1}/{ds}/train_log.txt")
        present = {k: v for k, v in vals_3way.items() if v is not None}
        vals = list(present.values())
        sigma_3way = st.stdev(vals) if len(vals) > 1 else float("nan")
        # mixed: add the M1 run (same config but 2-way concurrency) for SYSU/WHU
        mixed = list(vals)
        if ds in M1_IS_C1CONFIG and m1 is not None:
            mixed.append(m1)
        sigma_mixed = st.stdev(mixed) if len(mixed) > 1 else float("nan")
        rows[ds] = (present, m1, sigma_3way, sigma_mixed)
        print(f"\n{ds}:")
        print("  " + "  ".join(f"{k}={v:.2f}" for k, v in present.items()) +
              (f"   M1(2-way,同配置)={m1:.2f}" if m1 is not None else ""))
        print(f"  n(3-way)={len(vals)}  sigma_3way={sigma_3way:.3f}pp   "
              f"range={max(vals)-min(vals):.3f}pp")
        print(f"  n(mixed)={len(mixed)}  sigma_mixed={sigma_mixed:.3f}pp")

    print("\n===== 对照：教师效应量级 =====", flush=True)
    print("  max |teacher test delta| = 0.68pp (RADIO->SYSU); DCA WHU = 1.38pp")
    all3 = [r[2] for r in rows.values() if r[2] == r[2]]
    if all3:
        print(f"  平均 sigma_3way = {st.mean(all3):.3f}pp")
    print("\n[watch] DONE", flush=True)
    c.close()


if __name__ == "__main__":
    main()
