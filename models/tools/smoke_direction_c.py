"""CPU/CUDA checks of the real network and the direction-C contracts.
Run: python -m models.tools.smoke_direction_c [--device cuda]
"""
from __future__ import annotations
import argparse
import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from models import A2Net_LWGANet_L0, build_loss
from models.distill.diagnostics import build_cd_difficulty
from models.distill.losses import spatial_gram
from models.distill.routing import estimate_teacher_fit
from models.datasets.cache_transforms import replay_teacher_pack
from models.scripts.train import build_optimizer, build_router_optimizer, validate_resume
from models.utils.checkpoint import build_checkpoint, save_checkpoint_atomic, restore_rng_state


def synthetic_pack(batch=2,size=64,device='cpu'):
    y=torch.zeros(batch,1,size,size,device=device)
    y[:,:,size//4:3*size//4,size//4:3*size//4]=1
    yy,xx=torch.meshgrid(torch.arange(size,device=device),torch.arange(size,device=device),indexing='ij')
    ids=(1+(yy//max(size//4,1))*4+xx//max(size//4,1)).int()[None,None].repeat(batch,1,1,1)
    edge=((yy%max(size//4,1)==0)|(xx%max(size//4,1)==0)).float()[None,None].repeat(batch,1,1,1)
    sam={t:{'instance_id':ids.clone(),'boundary':edge.clone(),'quality':torch.ones_like(y)*.8} for t in ('t1','t2')}
    ov={k:{} for k in ('soft_change','confidence','relation')}
    for level,n in (('l1',32),('l2',16)):
        ov['soft_change'][level]=torch.nn.functional.interpolate(y,(n,n),mode='nearest')*.8+.1
        ov['confidence'][level]=torch.ones(batch,1,n,n,device=device)*.85
        ov['relation'][level]=torch.randn(batch,8,n,n,device=device)
    return y,{'sam':sam,'ov':ov}


def norm(parameters):
    return sum(float(p.grad.abs().sum()) for p in parameters if p.grad is not None)


def run(device='cpu'):
    torch.set_num_threads(2)
    torch.manual_seed(2333)
    baseline=A2Net_LWGANet_L0(pretrained=False).to(device)
    torch.manual_seed(2333)
    model=A2Net_LWGANet_L0(pretrained=False,auxiliary_mode='direction_c').to(device)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in baseline.state_dict().items())
    a,b=torch.randn(2,3,64,64,device=device),torch.randn(2,3,64,64,device=device)
    gt,pack=synthetic_pack(device=device)
    # Actual student graph, including training-state BN, must match baseline.
    baseline.train();model.train()
    bp,_=baseline(a,b)
    pred,aux=model(a,b,target=gt,teacher_pack=pack,force_action=2)
    detail=aux['direction_c']
    assert all(torch.equal(x,y) for x,y in zip(bp,pred))
    assert all(torch.equal(v,dict(model.named_buffers())[k]) for k,v in baseline.named_buffers())
    criterion=build_loss('batch')
    gt_loss=sum(criterion(x,gt) for x in pred)
    (gt_loss+detail['total']).backward()
    sum(criterion(x,gt) for x in bp).backward()
    assert float(detail['total'].detach())==0 and torch.count_nonzero(detail['effective_weights'])==0
    assert norm(model.training_auxiliary.heads.parameters())==0
    for (n,p) in baseline.named_parameters():
        q=dict(model.named_parameters())[n]
        if p.grad is not None:assert torch.equal(p.grad,q.grad),n
    # Decision statistics have no autograd path to P/F; router learns separately.
    model.zero_grad(set_to_none=True)
    pred,aux=model(a,b,target=gt,teacher_pack=pack)
    detail=aux['direction_c']
    for key in ('h','q','d','r','effective_weights'):
        assert not detail[key].requires_grad,key
    detail['router_loss'].backward()
    assert norm(model.training_auxiliary.router.parameters())>0
    assert norm(model.training_auxiliary.heads.parameters())==0
    assert norm(p for n,p in model.named_parameters() if not n.startswith('training_auxiliary.'))==0
    # Both teachers individually produce real student/head gradients.
    for teacher in (0,1):
        model.zero_grad(set_to_none=True)
        _,aux=model(a,b,target=gt,teacher_pack=pack,force_action=teacher)
        loss=aux['direction_c']['total'];loss.backward()
        assert torch.isfinite(loss) and norm(model.training_auxiliary.heads.parameters())>0
        assert norm(model.backbone.parameters())>0
        assert norm(model.training_auxiliary.router.parameters())==0
    # Unit-level suitability sign: GT-consistent vs GT-opposed teacher.
    f=torch.zeros(2,1,8,8,device=device,requires_grad=True);p=f.sigmoid();y=torch.zeros_like(p)
    good=torch.nn.functional.binary_cross_entropy(p,y,reduction='none').mean((1,2,3))
    bad=torch.nn.functional.binary_cross_entropy(p,1-y,reduction='none').mean((1,2,3))
    fit=estimate_teacher_fit(torch.stack((good,bad),1),f,p,y,torch.ones_like(y))
    assert torch.all(fit[:,0]>.99) and torch.all(fit[:,1]<-.99)
    # Empty GT, fragmentation and empty coverage are finite and explicit.
    empty=build_cd_difficulty(torch.full_like(gt,.1),torch.zeros_like(gt))
    assert torch.isfinite(empty['h']).all() and not empty['valid'][:,:3].any()
    split=gt.clone();split[:,:,:,31:33]=0
    diag=build_cd_difficulty(split,gt)
    assert (diag['h'][:,1]>0).all()
    zero_pack=copy.deepcopy(pack)
    for t in ('t1','t2'):zero_pack['sam'][t]['quality'].zero_()
    for level in ('l1','l2'):zero_pack['ov']['confidence'][level].zero_()
    _,aux=model(a,b,target=gt,teacher_pack=zero_pack)
    assert torch.isfinite(aux['direction_c']['loss_per_teacher']).all()
    assert torch.all(aux['direction_c']['target_action']==2)
    # Opaque relation channels: Gram is invariant to a common channel permutation.
    rel=pack['ov']['relation']['l1']
    assert torch.allclose(spatial_gram(rel),spatial_gram(rel[:,torch.randperm(8)]),atol=1e-6)
    single={kind:{k:({kk:vv[0].cpu() for kk,vv in v.items()}) for k,v in data.items()} for kind,data in pack.items()}
    replay=replay_teacher_pack(single,{'flip_h':True,'exchange':True})
    assert torch.equal(replay['sam']['t1']['instance_id'],single['sam']['t2']['instance_id'].flip(-1))
    assert replay['ov']['soft_change']['l1'].shape[-2:]==(32,32)
    # Independent optimizer state and checkpoint round trip.
    args=SimpleNamespace(lr=5e-4,backbone_lr_mult=1.,weight_decay=1e-4,router_lr=1e-3,
                         implementation_version='direction_c_v1')
    optimizer=build_optimizer(args,model);ropt=build_router_optimizer(args,model)
    model.zero_grad(set_to_none=True)
    pred,aux=model(a,b,target=gt,teacher_pack=pack,force_action=0)
    (sum(criterion(x,gt) for x in pred)+.06*aux['direction_c']['total']).backward()
    aux['direction_c']['router_loss'].backward();optimizer.step();ropt.step()
    checkpoint=build_checkpoint(model,optimizer,0,1,.5,args,ropt)
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'last_checkpoint.pth';save_checkpoint_atomic(checkpoint,path)
        loaded=torch.load(path,map_location=device,weights_only=False)
        restored=copy.deepcopy(model);restored.load_state_dict(loaded['model'])
        ro=build_router_optimizer(args,restored);ro.load_state_dict(loaded['router_optimizer'])
        so=build_optimizer(args,restored);so.load_state_dict(loaded['optimizer'])
        assert len(ro.state)>0 and len(so.state)>0
        restore_rng_state(loaded['rng']);r1=torch.rand(8)
        restore_rng_state(loaded['rng']);r2=torch.rand(8);assert torch.equal(r1,r2)
        # The next identical optimization step must match after state restore.
        def next_step(m,opt,router_opt):
            opt.zero_grad(set_to_none=True);router_opt.zero_grad(set_to_none=True)
            ps,ad=m(a,b,target=gt,teacher_pack=pack,force_action=0)
            (sum(criterion(x,gt) for x in ps)+.06*ad['direction_c']['total']).backward()
            ad['direction_c']['router_loss'].backward();opt.step();router_opt.step()
        next_step(model,optimizer,ropt)
        next_step(restored,so,ro)
        assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
        validate_resume(args,loaded)
        try:validate_resume(args,{'model':{}})
        except ValueError:pass
        else:raise AssertionError('legacy resume was not rejected')
    # Deployment: exact parameter count, prediction identity, no auxiliary state.
    model.eval()
    with torch.no_grad():before=model(a,b)
    train_params=sum(p.numel() for p in model.parameters())
    model.switch_to_deploy().eval()
    with torch.no_grad():after=model(a,b)
    error=max(float((x-y).abs().max()) for x,y in zip(before,after))
    assert error==0 and sum(p.numel() for p in model.parameters())==2_913_094
    assert not any(k.startswith('training_auxiliary') for k in model.state_dict())
    from thop import profile
    dummy=torch.zeros(1,3,256,256,device=device)
    with torch.no_grad():flops,_=profile(model,inputs=(dummy,dummy),verbose=False)
    assert abs(flops-2.7475e9)<.03e9
    report={'status':'passed','device':device,'torch':torch.__version__,
            'train_parameters':train_params,'deploy_parameters':2_913_094,'flops_256':flops,
            'deploy_max_error':error,'checks':['train BN/main path equality','Reject GT-only gradients',
             'router-only gradients','both teacher gradients','fit sign','empty GT/coverage',
             'fragmentation','relation Gram invariance','geometry replay',
             'model and two optimizer checkpoint','identical next step after restore','RNG restore','legacy resume rejection','deployment']}
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu')
    run(parser.parse_args().device)
