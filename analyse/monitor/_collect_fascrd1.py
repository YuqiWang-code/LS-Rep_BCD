"""SFTP-download completed FA-SCRD Run1 train_log.txt files to local outputs/.

Run:  python -B analyse/monitor/_collect_fascrd1.py
"""

import paramiko
from pathlib import Path

HOST = "172.18.232.202"
USER = "yqwang"
PW = "yq009996"
PORT = 22

REMOTE_BASE = "/home/yqwang/outputs/LS-Rep_BCD_RSML_3/FA-SCRD/Run1"
LOCAL_ROOT = Path(__file__).resolve().parent.parent.parent
LOCAL_BASE = LOCAL_ROOT / "outputs" / "FA-SCRD" / "Run1"

ORDER = [(exp, ds) for exp in ("C0", "C1", "A1", "M1") for ds in ("SYSU", "WHU", "CDD", "LEVIR")]


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PW, timeout=20)
    sftp = client.open_sftp()

    done = 0
    for exp, ds in ORDER:
        remote = f"{REMOTE_BASE}/{exp}/{ds}/train_log.txt"
        local = LOCAL_BASE / exp / ds / "train_log.txt"
        _, out, _ = client.exec_command(f"grep -c '^=== END TEST RESULTS ===' {remote} 2>/dev/null || true")
        if out.read().decode("utf-8", "replace").strip() != "1":
            print(f"[skip] {exp}/{ds} not complete")
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        sftp.get(remote, str(local))
        done += 1
        print(f"[ok] {exp}/{ds} -> {local.relative_to(LOCAL_ROOT).as_posix()}")

    sftp.close()
    client.close()
    print(f"Downloaded {done}/{len(ORDER)} completed logs.")


if __name__ == "__main__":
    main()
