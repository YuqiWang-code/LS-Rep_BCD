"""Smart monitor for SAGE-CD Run2 (R0-R4 x SYSU/WHU) on RSML-3.

Run:  python -B analyse/monitor/_queue_sage2.py
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

BASE = "/home/yqwang/outputs/LS-Rep_BCD_RSML_3/SAGE-CD/Run2"
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".sage2_state.json")
MAX_STEPS = 40000
ORDER = [(e, d) for e in ("R0", "R1", "R2", "R3", "R4") for d in ("SYSU", "WHU")]

RUN_LOOP = r'''
BASE="/home/yqwang/outputs/LS-Rep_BCD_RSML_3/SAGE-CD/Run2"
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


def _run(c, cmd, t=45):
    _, o, e = c.exec_command(cmd, timeout=t)
    x = o.read().decode("utf-8", "replace").strip(); e.read(); return x


def _load():
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        return json.load(open(STATE_FILE, "r", encoding="utf-8"))
    except Exception:
        return {}


def _save(s):
    tmp = STATE_FILE + ".tmp"
    json.dump(s, open(tmp, "w", encoding="utf-8"), indent=2); os.replace(tmp, STATE_FILE)


def _procs(raw):
    seen, out = set(), []
    for line in raw.splitlines():
        if "train_sage" not in line:
            continue
        m1 = re.search(r"--experiment\s+(\S+)", line); m2 = re.search(r"--dataset_name\s+(\S+)", line)
        if not (m1 and m2):
            continue
        k = (m1.group(1), m2.group(1))
        if k not in seen:
            seen.add(k); out.append(k)
    return out


def main():
    c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PW, timeout=20)
    try:
        st = _run(c, 'date "+%Y-%m-%d %H:%M:%S %z"')
        tmux = _run(c, "tmux ls 2>&1")
        gpu = _run(c, "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader")
        procs_raw = _run(c, "pgrep -af 'train_sage' 2>/dev/null")
        rr = _run(c, RUN_LOOP)
    finally:
        c.close()

    procs = _procs(procs_raw); live = {f"{e}/{d}" for e, d in procs}
    prev = _load(); now = time.time(); ns = {}
    rows = []; cnt = {"done": 0, "running": 0, "wait": 0, "error": 0, "none": 0}
    for e, d in ORDER:
        rec = None
        for ln in rr.splitlines():
            p = ln.split("|")
            if len(p) >= 7 and p[0] == "R" and p[1] == e and p[2] == d:
                rec = p; break
        if rec is None:
            rec = ["R", e, d, "none", "0", "0", "", ""]
        stt, step, err = rec[3], int(rec[4] or 0), int(rec[5] or 0)
        f1, io = rec[6] or "", rec[7] or ""
        if stt != "done" and err > 0:
            stt = "error"
        k = f"{e}/{d}"; note = ""; pct = ""
        if stt == "running":
            pct = f"{100.0 * step / MAX_STEPS:5.1f}%"; cnt["running"] += 1
            if k not in live:
                note = "DEAD(no process)"
            else:
                p = prev.get(k)
                if p and p.get("step") == step and (now - p.get("ts", 0)) > 5400:
                    note = "STALLED"
            ns[k] = {"step": step, "ts": now}
        elif stt == "done":
            cnt["done"] += 1; note = f"F1={f1} IoU={io}" if f1 else ""
        elif stt == "error":
            cnt["error"] += 1; note = "ERROR"
        elif stt == "none":
            cnt["none"] += 1
        else:
            cnt["wait"] += 1
        rows.append((e, d, stt, step, pct, note))
    _save(ns)

    print("SERVER TIME:  " + st)
    print("\n== tmux =="); print(tmux if tmux else "(none)")
    print("\n== live train_sage ==")
    if procs:
        for e, d in sorted(procs):
            print(f"  {e}/{d}")
    else:
        print("  (none)")
    print(f"\n== {len(ORDER)} runs ==")
    for e, d, stt, step, pct, note in rows:
        det = f"step={step} ({pct})" if pct else step if stt == "done" else "-"
        print(f"  [{e}/{d}] {stt:<7} {det:<20} {note}")
    print(f"\n  => {cnt['done']}/{len(ORDER)} done, {cnt['running']} running, {cnt['wait']} wait, {cnt['error']} error, {cnt['none']} none")
    print("\n== GPU ==")
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
