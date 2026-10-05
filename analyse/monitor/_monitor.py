import paramiko
host='172.18.232.202'; user='yqwang'; pw='yq009996'
c=paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(host, port=22, username=user, password=pw, timeout=20)
def run(cmd):
    i,o,e=c.exec_command(cmd, timeout=25); return o.read().decode('utf-8','replace').strip()
print('SERVER:', run('date "+%Y-%m-%d %H:%M:%S %z"'))
print('=== tmux ==='); print(run('tmux ls 2>&1'))
print('=== cache .pt ===')
for d in ['OVCDistill/dinov2_vitb14/CDD-CD-256','OVCDistill/dinov2_vitb14/LEVIR-CD-256','SAMStruct/sam2.1_hiera_large/CDD-CD-256','SAMStruct/sam2.1_hiera_large/LEVIR-CD-256']:
    n = run('ls /home/yqwang/datasets/CD_teacher_cache/%s/train/ 2>/dev/null | wc -l' % d)
    m = run('ls /home/yqwang/datasets/CD_teacher_cache/%s/manifest.json 2>/dev/null | wc -l' % d)
    print('  %s pt=%s manifest=%s' % (d.split('/')[-1], n, m))
print('=== ablation ===')
for e in ['B0','C0']:
    g = run('grep global_step /home/yqwang/outputs/LS-Rep_BCD_RSML_3/DirectionC/%s/SYSU/steps_40000/seed_2333/train_log.txt 2>/dev/null | tail -1' % e)
    print('  %s SYSU: %s' % (e, g[:110]))
print('=== GPU ==='); print(run('nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader'))
c.close()
