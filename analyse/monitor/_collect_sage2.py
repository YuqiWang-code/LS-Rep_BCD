"""SFTP-download completed SAGE-CD Run2 logs to local outputs/.

Run:  python -B analyse/monitor/_collect_sage2.py
"""

import paramiko
from pathlib import Path

HOST = "172.18.232.202"
USER = "yqwang"
PW = "yq009996"
PORT = 22

REMOTE_BASE = "/home/yqwang/outputs/LS-Rep_BCD_RSML_3/SAGE-CD/Run2"
LOCAL_ROOT = Path(__file__).resolve().parent.parent.parent
LOCAL_BASE = LOCAL_ROOT / "outputs" / "SAGE-CD" / "Run2"
ORDER = [(e, d) for e in ("R0", "R1", "R2", "R3", "R4") for d in ("SYSU", "WHU")]


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PW, timeout=20)
    sftp = c.open_sftp()
    done = 0
    for e, d in ORDER:
        remote = f"{REMOTE_BASE}/{e}/{d}/train_log.txt"
        local = LOCAL_BASE / e / d / "train_log.txt"
        _, out, _ = c.exec_command(f"grep -c '^=== END TEST RESULTS ===' {remote} 2>/dev/null || true")
        if out.read().decode("utf-8", "replace").strip() != "1":
            print(f"[skip] {e}/{d}")
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        sftp.get(remote, str(local))
        done += 1
        print(f"[ok] {e}/{d} -> {local.relative_to(LOCAL_ROOT).as_posix()}")
    sftp.close(); c.close()
    print(f"Downloaded {done}/{len(ORDER)} logs.")


if __name__ == "__main__":
    main()
