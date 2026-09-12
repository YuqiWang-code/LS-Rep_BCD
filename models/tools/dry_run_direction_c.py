"""One real-cache training batch and deployment check; writes no dataset/cache/checkpoint."""
import argparse
import copy
import json
from pathlib import Path
import torch
from models import A2Net_LWGANet_L0,build_loss
from models.distill.teacher_cache import PairedTeacherCache
from models.datasets.cd_dataset import get_loader
from models.scripts.train import nested_to,auxiliary_toggle_consistency


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data_root',required=True);p.add_argument('--dataset_name',required=True)
    p.add_argument('--sam_cache_root',required=True);p.add_argument('--ov_cache_root',required=True)
    p.add_argument('--pretrained_path');p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch_size',type=int,default=2);p.add_argument('--num_workers',type=int,default=0)
    args=p.parse_args();device=torch.device(args.device);torch.set_num_threads(2)
    entries=[v.strip() for v in (Path(args.data_root)/'list/train.txt').read_text().splitlines() if v.strip()]
    cache=PairedTeacherCache(args.sam_cache_root,args.ov_cache_root,args.dataset_name,entries)
    loader=get_loader(args.data_root,'train.txt',batchsize=args.batch_size,
                      num_workers=args.num_workers,teacher_cache=cache)
    if not len(loader):raise ValueError('Dataset smaller than batch_size')
    image,gt,ids,pack=next(iter(loader));image,gt=image.to(device),gt.to(device)
    pack=nested_to(pack,device)
    model=A2Net_LWGANet_L0(pretrained=bool(args.pretrained_path),pretrained_path=args.pretrained_path,
                           auxiliary_mode='direction_c').to(device).train()
    criterion=build_loss('batch');pred,aux=model(image[:,:3],image[:,3:],target=gt,teacher_pack=pack)
    detail=aux['direction_c'];loss=sum(criterion(x,gt) for x in pred)+.06*detail['total']
    loss.backward();detail['router_loss'].backward()
    grad=sum(float(p.grad.abs().sum()) for p in model.training_auxiliary.router.parameters() if p.grad is not None)
    assert grad>0 and torch.isfinite(loss)
    observed={key:detail[key].detach().cpu().tolist() for key in ('h','q','d','r','weights','target_action','action')}
    model.zero_grad(set_to_none=True)
    _,aux=model(image[:,:3],image[:,3:],target=gt,teacher_pack=pack,force_action=2)
    assert float(aux['direction_c']['total'].detach())==0
    toggle=auxiliary_toggle_consistency(model,loader.dataset,device)
    model.eval()
    with torch.no_grad():before=model(image[:1,:3],image[:1,3:])
    model.switch_to_deploy()
    with torch.no_grad():after=model(image[:1,:3],image[:1,3:])
    error=max(float((x-y).abs().max()) for x,y in zip(before,after))
    assert error<1e-6
    print(json.dumps({'status':'passed','sample_ids':list(ids),'router_grad_l1':grad,
                      'reject_teacher_loss':0.,'auxiliary_toggle_error':toggle,
                      'deploy_max_error':error,'diagnostics':observed},indent=2))


if __name__=='__main__':main()
