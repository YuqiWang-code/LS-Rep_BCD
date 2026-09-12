"""Temporary synthetic on-disk dataset/cache -> trainer -> test block -> export.
Never points at the user's dataset/cache/checkpoint directories.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import cv2
import numpy as np
import torch
from models.tools.smoke_direction_c import synthetic_pack


def make_fixture(root):
    data=root/'SYSU-CD-256';sam=root/'sam';ov=root/'ov'
    for folder in ('A','B','label','list'):(data/folder).mkdir(parents=True,exist_ok=True)
    for folder in (sam,ov):(folder/'train').mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(2333)
    entries={};torch.manual_seed(2333)
    for split in ('train','val','test'):
        names=[f'{split}_{i}.png' for i in range(2)]
        (data/'list'/f'{split}.txt').write_text('\n'.join(names)+'\n')
        for name in names:
            y,pack=synthetic_pack(batch=1,size=256)
            a=rng.integers(0,256,(256,256,3),dtype=np.uint8)
            b=rng.integers(0,256,(256,256,3),dtype=np.uint8)
            cv2.imwrite(str(data/'A'/name),a);cv2.imwrite(str(data/'B'/name),b)
            cv2.imwrite(str(data/'label'/name),(y[0,0].numpy()*255).astype(np.uint8))
            if split=='train':
                filename=hashlib.sha1(name.encode()).hexdigest()+'.pt';entries[name]=filename
                for kind,path in (('sam',sam),('ov',ov)):
                    content={k:{kk:vv[0].cpu() for kk,vv in v.items()} for k,v in pack[kind].items()}
                    content.update(sample_id=name,meta={'config_hash':'synthetic-test-only'})
                    torch.save(content,path/'train'/filename)
    for kind,path in (('sam',sam),('ov',ov)):
        manifest={'cache_version':1,'source_dataset':'SYSU','config_hash':'synthetic-test-only',
                  'teacher_type':'sam2_struct_v2' if kind=='sam' else 'synthetic_ov', 'entries':entries}
        (path/'manifest.json').write_text(json.dumps(manifest))
    return data,sam,ov


def run():
    project=Path(__file__).resolve().parents[2]
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    results={}
    with tempfile.TemporaryDirectory(prefix='dartr_ts_fixture_') as temporary:
        root=Path(temporary);data,sam,ov=make_fixture(root)
        before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for base in (data,sam,ov) for p in base.rglob('*') if p.is_file()}
        def execute(argv):
            proc=subprocess.run([sys.executable,*argv],cwd=project,env=env,text=True,capture_output=True)
            if proc.returncode:raise RuntimeError(proc.stdout+'\n'+proc.stderr)
            return proc.stdout
        for experiment in ('B0','C1'):
            save=root/'checkpoints'/experiment;log=root/'logs'/experiment/'train_log.txt'
            command=['-m','models.scripts.train','--experiment',experiment,'--dataset_name','SYSU',
                     '--data_root',str(data),'--sam_cache_root',str(sam),'--ov_cache_root',str(ov),
                     '--no-pretrained','--device','cpu','--batch_size','2','--max_steps','2',
                     '--num_workers','0','--save_dir',str(save),'--log_file',str(log)]
            execute(command)
            text=log.read_text();assert '=== END TEST RESULTS ===' in text
            block=text.rsplit('=== TEST RESULTS ===',1)[1].split('=== END TEST RESULTS ===')[0]
            assert all(label+':' in block for label in ('Recall','Precision','OA','F1','IoU','Kappa'))
            ckpt=torch.load(save/'last_checkpoint.pth',map_location='cpu',weights_only=False)
            assert ckpt['global_step']==2 and ckpt['router_optimizer'] is None
            # Completion resume must reload the validation-selected best and retain log blocks.
            execute(command+['--resume',str(save/'last_checkpoint.pth')])
            assert log.read_text().count('=== END TEST RESULTS ===')==2
            export=root/f'{experiment}_deploy.pth'
            execute(['-m','models.tools.export_deploy','--checkpoint',str(next(save.glob('best_model_F1=*.pth'))),
                     '--output',str(export)])
            deployed=torch.load(export,map_location='cpu',weights_only=False)
            assert deployed['deploy_parameters']==2_913_094
            assert not any(k.startswith('training_auxiliary.') for k in deployed['model'])
            results[experiment]={'train_steps':2,'formal_block_format':True,'completed_resume':True,'export':True}
        output=execute(['-m','models.tools.dry_run_task_space','--dataset_name','SYSU','--data_root',str(data),
                        '--sam_cache_root',str(sam),'--ov_cache_root',str(ov),'--device','cpu'])
        results['disk_fixture_dry_run']=json.loads(output)
        after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for base in (data,sam,ov) for p in base.rglob('*') if p.is_file()}
        assert before==after
    report={'status':'passed','scope':'synthetic disk fixture, NOT RSML-3 real cache',
            'dataset_and_cache_unchanged':True,'results':results}
    print(json.dumps(report,indent=2));return report


if __name__=='__main__':run()
