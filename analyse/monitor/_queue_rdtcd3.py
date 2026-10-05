"""Smart monitor for RDT-CD Run3 training queues on RSML-3.

Connects once over SSH, snapshots server time / tmux / GPU / live python
processes, then classifies each of the 12 runs (B0|R3A|R3 x SYSU|WHU|CDD|LEVIR)
into:

    done     -- final `=== END TEST RESULTS ===` block present (F1/IoU shown)
    running  -- has global_step progress (shows step + % of 40000)
    wait     -- log exists but no progress yet, or not reached by the queue
    error    -- log contains a traceback/exception and is NOT done
    none     -- no log file yet

Step snapshots are persisted to `.rdtcd3_state.json` so a run that stops
advancing across checks is flagged as STALLED (vs. a run that is simply slow).

Run:  python -B analyse/monitor/_queue_rdtcd3.py
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

BASE = "/home/yqwang/outputs/LS-Rep_BCD_RSML_3/RDT-CD/Run3"
STATE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".rdtcd3_state.json"
)

MAX_STEPS = 40000
ORDER = (
    [(exp, ds) for exp in ("B0", "R3A", "R3") for ds in ("SYSU", "WHU", "CDD", "LEVIR")]
    + [("R3O", "SYSU"), ("R3O", "WHU")]
)

ERROR_RE = re.compile(
    r"Traceback|RuntimeError|ValueError|FileNotFoundError|CUDA out of memory|"
    r"Non-finite|FloatingPointError"
)

# Single remote loop that dumps one `R|...` record per run. Far fewer
# round-trips than the old per-run greps, and it also extracts final metrics.
RUN_LOOP = r'''
BASE="/home/yqwang/outputs/LS-Rep_BCD_RSML_3/RDT-CD/Run3"
for PAIR in B0:SYSU B0:WHU B0:CDD B0:LEVIR R3A:SYSU R3A:WHU R3A:CDD R3A:LEVIR R3:SYSU R3:WHU R3:CDD R3:LEVIR R3O:SYSU R3O:WHU; do
  EXP=${PAIR%%:*}; DS=${PAIR#*:}
    P="$BASE/$EXP/$DS/train_log.txt"
    if [ ! -f "$P" ]; then
      printf 'R|%s|%s|none|0|0||\n' "$EXP" "$DS"
      continue
    fi
    DONE=$(grep -c '^=== END TEST RESULTS ===' "$P" 2>/dev/null || true)
    STEP=$(grep 'global_step:' "$P" 2>/dev/null | tail -1 \
           | sed -n 's/.*global_step: *\([0-9][0-9]*\).*/\1/p')
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
'''


def _run(client, cmd, timeout=45):
    _, out, err = client.exec_command(cmd, timeout=timeout)
    text = out.read().decode("utf-8", "replace").strip()
    err.read()  # drain so the channel doesn't block
    return text


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
    """Map live `models.scripts.train` processes to deduped (exp, ds, gpu).

    DataLoader workers are forked from the main process and share its argv, so
    pgrep returns several identical lines per run; dedupe by (exp, ds, gpu).
    The `bash -c "pgrep ..."` wrapper itself also matches the pattern but has
    no --experiment/--dataset_name, so it is skipped.
    """
    seen = set()
    procs = []
    for line in raw.splitlines():
        if "models.scripts.train" not in line:
            continue
        m_exp = re.search(r"--experiment\s+(\S+)", line)
        m_ds = re.search(r"--dataset_name\s+(\S+)", line)
        m_gpu = re.search(r"--gpu_id\s+(\S+)", line)
        if not (m_exp and m_ds):
            continue
        key = (m_exp.group(1), m_ds.group(1), m_gpu.group(1) if m_gpu else "?")
        if key in seen:
            continue
        seen.add(key)
        procs.append(key)
    return procs


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PW, timeout=20)

    try:
        server_time = _run(client, 'date "+%Y-%m-%d %H:%M:%S %z"')
        tmux = _run(client, "tmux ls 2>&1")
        gpu = _run(
            client,
            "nvidia-smi --query-gpu=index,memory.used,memory.total,"
            "utilization.gpu --format=csv,noheader",
        )
        procs_raw = _run(client, "pgrep -af 'models.scripts.train' 2>/dev/null")
        run_raw = _run(client, RUN_LOOP)
    finally:
        client.close()

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
            parts = line.split("|")
            if len(parts) >= 7 and parts[0] == "R" and parts[1] == exp \
                    and parts[2] == ds:
                rec = parts
                break
        if rec is None:
            rec = ["R", exp, ds, "none", "0", "0", "", ""]

        status = rec[3]
        step = int(rec[4] or 0)
        err = int(rec[5] or 0)
        f1 = rec[6] or ""
        iou = rec[7] or ""

        # A run with an exception in the log that never produced the final
        # block is a crash, not "wait" and not "running".
        if status != "done" and err > 0:
            status = "error"

        key = f"{exp}/{ds}"
        note = ""
        pct = ""
        if status == "running":
            pct = f"{100.0 * step / MAX_STEPS:5.1f}%"
            counts["running"] += 1
            if key not in live:
                # Log still advances but no python process is alive: the
                # process was killed (OOM / SIGKILL) without a traceback.
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

    # ---- report ---------------------------------------------------------
    print("SERVER TIME:  " + server_time)
    print()
    print("== tmux ==")
    print(tmux if tmux else "(no tmux sessions)")
    print()
    print("== live train processes ==")
    if procs:
        for e, d, g in sorted(procs):
            print(f"  {e}/{d}  (gpu {g})")
    else:
        print("  (none)")
    print()
    print(f"== {len(ORDER)} runs ==")
    for exp, ds, status, step, pct, note in rows:
        detail = f"step={step} ({pct})" if pct else step if status == "done" else "-"
        line = f"  [{exp}/{ds}] {status:<7} {detail:<20} {note}"
        print(line)
    print()
    print(f"  => {counts['done']}/{len(ORDER)} done, {counts['running']} running, "
          f"{counts['wait']} wait, {counts['error']} error, {counts['none']} none")
    print()
    print("== GPU ==")
    for gline in gpu.splitlines():
        print("  " + gline)
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
