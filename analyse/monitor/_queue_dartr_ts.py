import paramiko
host='172.18.232.202'; user='yqwang'; pw='yq009996'
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(host, port=22, username=user, password=pw, timeout=20)
def run(cmd):
    i,o,e=c.exec_command(cmd, timeout=30); return o.read().decode('utf-8','replace').strip()

BASE='/home/yqwang/outputs/LS-Rep_BCD_RSML_3/DART-R-TS'
print('SERVER:', run('date "+%Y-%m-%d %H:%M:%S %z"'))
print('=== tmux ==='); print(run('tmux ls 2>&1'))
print('=== 12 runs 状态 ===')
order = [(exp, ds) for exp in ['B0','C1','C2'] for ds in ['SYSU','WHU','CDD','LEVIR']]
done = 0
for exp, ds in order:
    p = f'{BASE}/{exp}/{ds}/steps_40000/seed_2333/train_log.txt'
    lg = run(f'grep -c "^=== END TEST RESULTS ===" {p} 2>/dev/null')
    g = run(f'grep "global_step" {p} 2>/dev/null | tail -1')
    step = g.split('global_step: ')[-1] if 'global_step' in g else ''
    if lg == '1':
        status = 'DONE'; done += 1
    elif step:
        status = f'RUN step={step}'
    else:
        status = 'WAIT'
    print(f'  {exp}/{ds}: {status}')
print(f'  => {done}/12 done')
print('=== GPU ==='); print(run('nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader'))
c.close()
