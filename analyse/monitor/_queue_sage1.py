"""Smart monitor for SAGE-CD Run1 (G0/G1/G2 x SYSU/WHU) on RSML-3.

Run:  python -B analyse/monitor/_queue_sage1.py
"""

import json
import os
import re
import time

import paramiko

HOST = "172.18.232.202"
USER = "yqwang"
PW = "yq009996"
PORT = 22

BASE = "/home/yqwang/outputs/LS-Rep_BCD_RSML_3/SAGE-CD/Run1"
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".sage1_state.json")
MAX_STEPS = 40000
ORDER = [(exp, ds) for exp in ("G0", "G1", "G2") for ds in ("SYSU", "WHU")]

RUN_LOOP = r'''
BASE="/home/yqwang/outputs/LS-Rep_BCD_RSML_3/SAGE-CD/Run1"
for PAIR in __PAIRS__; do
  EXP=${PAIR%%:*}; DS=${PAIR#*:}
    P="$BASE/$EXP/$DS/train_log.txt"
    if [ ! -f "$P" ]; then printf 'R|%s|%s|none|0|0||\n' "$EXP" "$DS"; continue; fi
    DONE=$(grep -c '^=== END TEST RESULTS ===' "$P" 2>/dev/null || true)
    STEP=$(grep 'global_step:' "$P" 2>/dev/null | tail -1 | sed -n 's/.*global_step: *\([0-9][0-9]*\).*/\1/p')
    ERR=$(grep -cE 'Traceback|RuntimeError|ValueError|FileNotFoundError|CUDA out of memory|Non-finite|FloatingPointError' "$P" 2>/dev/null || true)
    if [ "$DONE" = "1" ]; then
      F1=$(awk '/^=== TEST RESULTS ===/{f=1} f && /^F1:/{v=$2} /^=== END TEST RESULTS ===/{f=0} END{print v}' "$P")
      IO=$(awk '/^=== TEST RESULTS ===/{f=1} f && /^IoU:/{v=$2} /^=== END TEST RESULTS ===/{f=0} END{print v}' "$P")
      printf 'R|%s|%s|done|%s|%s|%s|%s\n' "$EXP" "$DS" "$STEP" "$ERR" "${F1:-}" "${IO:-}"
    elif [ -n "$STEP" ]; then
      printf 'R|%s|%s|running|%s|%s||\n' "$EXP" "$DS" "$STEP" "$ERR"
    else
      printf 'R|%s|%s|wait|0|%s||\n' "$EXP" "$DS" "$ERR"
    fi
  done
done
'''.replace("__PAIRS__", " ".join(f"{e}:{d}" for e, d in ORDER))


def _run(client, cmd, timeout=45):
    _, out, err = client.exec_command(cmd, timeout=timeout)
    t = out.read().decode("utf-8", "replace").strip()
    err.read()
    return t


def _load_state():
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (ValueError, OSError):
        return {}


def _save_state(state):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
    os.replace(tmp, STATE_FILE)


def _parse_procs(raw):
    seen = set()
    procs = []
    for line in raw.splitlines():
        if "models.scripts.train_sage" not in line:
            continue
        m_e = re.search(r"--experiment\s+(\S+)", line)
        m_d = re.search(r"--dataset_name\s+(\S+)", line)
        m_g = re.search(r"--gpu_id\s+(\S+)", line)
        if not (m_e and m_d):
            continue
        key = (m_e.group(1), m_d.group(1), m_g.group(1) if m_g else "?")
        if key not in seen:
            seen.add(key)
            procs.append(key)
    return procs


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PW, timeout=20)
    try:
        server_time = _run(c, 'date "+%Y-%m-%d %H:%M:%S %z"')
        tmux = _run(c, "tmux ls 2>&1")
        gpu = _run(c, "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader")
        procs_raw = _run(c, "pgrep -af 'models.scripts.train_sage' 2>/dev/null")
        run_raw = _run(c, RUN_LOOP)
    finally:
        c.close()

    procs = _parse_procs(procs_raw)
    live = {f"{e}/{d}" for e, d, _ in procs}
    prev = _load_state()
    now = time.time()
    new_state = {}
    rows = []
    counts = {"done": 0, "running": 0, "wait": 0, "error": 0, "none": 0}
    for exp, ds in ORDER:
        rec = None
        for line in run_raw.splitlines():
            p = line.split("|")
            if len(p) >= 7 and p[0] == "R" and p[1] == exp and p[2] == ds:
                rec = p
                break
        if rec is None:
            rec = ["R", exp, ds, "none", "0", "0", "", ""]
        status, step, err = rec[3], int(rec[4] or 0), int(rec[5] or 0)
        f1, iou = rec[6] or "", rec[7] or ""
        if status != "done" and err > 0:
            status = "error"
        key = f"{exp}/{ds}"
        note = ""
        pct = ""
        if status == "running":
            pct = f"{100.0 * step / MAX_STEPS:5.1f}%"
            counts["running"] += 1
            if key not in live:
                note = "DEAD(no process)"
            else:
                p = prev.get(key)
                if p and p.get("step") == step and (now - p.get("ts", 0)) > 5400:
                    note = "STALLED"
            new_state[key] = {"step": step, "ts": now}
        elif status == "done":
            counts["done"] += 1
            note = f"F1={f1} IoU={iou}" if f1 else ""
        elif status == "error":
            counts["error"] += 1
            note = "ERROR(see log)"
        elif status == "none":
            counts["none"] += 1
        else:
            counts["wait"] += 1
        rows.append((exp, ds, status, step, pct, note))
    _save_state(new_state)

    print("SERVER TIME:  " + server_time)
    print()
    print("== tmux ==")
    print(tmux if tmux else "(none)")
    print()
    print("== live train_sage processes ==")
    if procs:
        for e, d, g in sorted(procs):
            print(f"  {e}/{d}  (gpu {g})")
    else:
        print("  (none)")
    print()
    print(f"== {len(ORDER)} runs ==")
    for exp, ds, status, step, pct, note in rows:
        detail = f"step={step} ({pct})" if pct else step if status == "done" else "-"
        print(f"  [{exp}/{ds}] {status:<7} {detail:<20} {note}")
    print()
    print(f"  => {counts['done']}/{len(ORDER)} done, {counts['running']} running, {counts['wait']} wait, {counts['error']} error, {counts['none']} none")
    print()
    print("== GPU ==")
    for g in gpu.splitlines():
        print("  " + g)
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
