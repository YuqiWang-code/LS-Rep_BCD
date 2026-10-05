import paramiko
host='172.18.232.202'; user='yqwang'; pw='yq009996'
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(host, port=22, username=user, password=pw, timeout=20)
def run(cmd):
    i,o,e=c.exec_command(cmd, timeout=30); return o.read().decode('utf-8','replace').strip()

BASE='/home/yqwang/projects/LS-Rep_BCD_RSML_3'
print('SERVER:', run('date "+%Y-%m-%d %H:%M:%S %z"'))
print('=== tmux ==='); print(run('tmux ls 2>&1'))

print('=== SAM cache ===')
for d in ['SAMStruct/sam2.1_hiera_large/CDD-CD-256','SAMStruct/sam2.1_hiera_large/LEVIR-CD-256']:
    n = run('ls /home/yqwang/datasets/CD_teacher_cache/%s/train/ 2>/dev/null | wc -l' % d)
    m = run('ls /home/yqwang/datasets/CD_teacher_cache/%s/manifest.json 2>/dev/null | wc -l' % d)
    print('  %s pt=%s manifest=%s' % (d.split('/')[-1], n, m))

print('=== 8 runs 状态 ===')
order = [('B0','SYSU'),('C0','SYSU'),('B0','WHU'),('C0','WHU'),('B0','CDD'),('C0','CDD'),('B0','LEVIR'),('C0','LEVIR')]
for exp, ds in order:
    lg = run('cat /home/yqwang/outputs/LS-Rep_BCD_RSML_3/DirectionC/%s/%s/steps_40000/seed_2333/train_log.txt 2>/dev/null | grep -c "^=== END TEST RESULTS ==="' % (exp, ds))
    g = run('grep global_step /home/yqwang/outputs/LS-Rep_BCD_RSML_3/DirectionC/%s/%s/steps_40000/seed_2333/train_log.txt 2>/dev/null | tail -1' % (exp, ds))
    sess = run('tmux ls 2>/dev/null | grep -c "^%s_%s:"' % (exp, ds))
    status = 'DONE' if lg=='1' else ('RUN step='+g.split('global_step: ')[-1] if g else ('IN-TMUX' if sess=='1' else 'WAIT'))
    print('  %s_%s: %s' % (exp, ds, status[:60]))

print('=== GPU ==='); print(run('nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader'))
c.close()
