#!/usr/bin/env python3
"""One-shot status probe for CATA-CD runs on RSML-3."""
import json
from pathlib import Path

import paramiko

cfg = json.loads((Path(__file__).resolve().parent.parent / ".vscode" / "sftp.json").read_text())
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(cfg["host"], cfg["port"], cfg["username"], cfg["password"], timeout=30)


def run(cmd, t=120):
    _i, o, e = c.exec_command(cmd, timeout=t)
    return o.read().decode(errors="replace"), e.read().decode(errors="replace")


print("===== GPU =====")
print(run("nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader")[0])

print("===== tmux =====")
print(run("tmux ls 2>&1")[0])

print("===== 训练/缓存进程 =====")
print(run("ps aux | grep -E 'models.scripts.train|generate_teacher_cache_v2' | grep -v grep | awk '{print $2, $13, $14, $15, $16}'")[0])

print("===== cache 进度 =====")
out, _ = run(
    "for d in /share_datasets/CD_teacher_cache/CATA_CD_v2/*/; do "
    "t=$(basename $d); "
    "for ds in SYSU WHU CDD LEVIR; do "
    "n=$(find $d$ds/train -name '*.pt' 2>/dev/null | wc -l); "
    "if [ -f $d$ds/train/manifest.json ]; then st=DONE; else st=running; fi; "
    "echo \"$t/$ds: $n $st\"; "
    "done; done"
)
print(out)

print("===== 训练日志 =====")
out, _ = run(
    "for f in $(find /home/yqwang/outputs/LS-Rep_BCD_RSML_3/CATA-CD/teacher_adaptation -name train_log.txt 2>/dev/null); "
    "do echo \"== $f ==\"; tail -n 2 $f; done"
)
print(out)

c.close()
