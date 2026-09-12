"""Read-only real-cache dry run for C1. Does not save/update datasets or caches.
The report is an implementation diagnostic, never a formal accuracy result.
"""
import argparse
import json
from pathlib import Path
import torch
from models import A2Net_LWGANet_L0, build_loss
from models.distill import PairedTeacherCache
from models.datasets.cd_dataset import get_loader
from models.scripts.train import nested_to, auxiliary_toggle_consistency


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data_root',required=True)
    p.add_argument('--dataset_name',required=True,choices=['CDD','LEVIR','SYSU','WHU'])
    p.add_argument('--sam_cache_root',required=True);p.add_argument('--ov_cache_root',required=True)
    p.add_argument('--pretrained_path');p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch_size',type=int,default=2);p.add_argument('--num_workers',type=int,default=0)
    args=p.parse_args();device=torch.device(args.device);torch.set_num_threads(2);torch.manual_seed(2333)
    entries=[v.strip() for v in (Path(args.data_root)/'list/train.txt').read_text().splitlines() if v.strip()]
    cache=PairedTeacherCache(args.sam_cache_root,args.ov_cache_root,args.dataset_name,entries)
    loader=get_loader(args.data_root,'train.txt',batchsize=args.batch_size,num_workers=args.num_workers,
                      teacher_cache=cache,cache_replay='aligned')
    if not len(loader):raise ValueError('Dataset smaller than batch_size')
    image,gt,ids,pack=next(iter(loader));image,gt=image.to(device),gt.to(device)
    pack=nested_to(pack,device)
    model=A2Net_LWGANet_L0(pretrained=bool(args.pretrained_path),pretrained_path=args.pretrained_path,
                           auxiliary_mode='direction_c',routing_cfg={'mechanism':'task_space'}).to(device).train()
    criterion=build_loss('batch')
    pred,aux=model(image[:,:3],image[:,3:],target=gt,teacher_pack=pack)
    detail=aux['direction_c'];main_loss=sum(criterion(x,gt) for x in pred)
    # Measure actual student gradient contribution, with the same coefficient as C1.
    cls=model.decoder.cls.weight
    gm=torch.autograd.grad(main_loss,cls,retain_graph=True)[0]
    gk=torch.autograd.grad(.06*detail['total'],cls,retain_graph=True)[0]
    ratio=float(gk.norm()/gm.norm().clamp_min(1e-12))
    cosine=float((gm*gk).sum()/(gm.norm()*gk.norm()).clamp_min(1e-12))
    loss=main_loss+.06*detail['total'];loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None)
    observed={key:detail[key].detach().cpu().tolist() for key in (
        'q','effective_weights','pixel_reject_ratio','image_reject_ratio',
        'accepted_change_ratio','accepted_bg_ratio','proposal_gain','relative_gain',
        'eligible_ratio','available_ratio','student_brier','proposal_brier','effective_mass')}
    model.zero_grad(set_to_none=True)
    _,aux=model(image[:,:3],image[:,3:],target=gt,teacher_pack=pack,force_action=2)
    assert aux['direction_c']['total'].item()==0
    toggle=auxiliary_toggle_consistency(model,loader.dataset,device)
    model.eval()
    with torch.no_grad():before=model(image[:1,:3],image[:1,3:])
    model.switch_to_deploy()
    with torch.no_grad():after=model(image[:1,:3],image[:1,3:])
    error=max(float((x-y).abs().max()) for x,y in zip(before,after))
    assert error<1e-6 and sum(x.numel() for x in model.parameters())==2_913_094
    print(json.dumps({'status':'passed','sample_ids':list(ids),'cache_fingerprint':cache.fingerprint,
                      'scope':'one batch implementation check; not accuracy validation',
                      'cls_kd_gt_grad_ratio':ratio,'cls_kd_gt_grad_cosine':cosine,
                      'no_kd_on_this_batch':ratio==0.,'auxiliary_toggle_error':toggle,
                      'deploy_max_error':error,'diagnostics':observed},indent=2))


if __name__=='__main__':main()
