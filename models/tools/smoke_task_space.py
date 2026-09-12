"""Contract tests on the real student; synthetic data, no accuracy claims.
Run from project root: python -m models.tools.smoke_task_space --device cpu
"""
from __future__ import annotations
import argparse
import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch
from thop import profile

from models import A2Net_LWGANet_L0, build_loss
from models.distill.task_space import (TaskSpaceDirectionC, build_task_proposals,
                                      instance_transport, task_space_route, bernoulli_kl)
from models.datasets.cache_transforms import replay_teacher_pack, _replay_tensor
from models.scripts.train import build_optimizer, build_router_optimizer, validate_resume
from models.utils.checkpoint import build_checkpoint, restore_rng_state, save_checkpoint_atomic
from models.tools.smoke_direction_c import synthetic_pack


def check_geometry():
    # Every pixel has a distinct large int64 ID: reveals nearest-coordinate bugs.
    ids = np.arange(256*256, dtype=np.int64).reshape(256, 256)+2**30
    val = np.linspace(0, 1, 256*256, dtype=np.float32).reshape(256, 256)
    pack = {'sam': {t: {'instance_id': torch.from_numpy(ids.copy())[None],
                       'boundary': torch.from_numpy(val.copy())[None],
                       'quality': torch.ones(1, 256, 256)} for t in ('t1', 't2')},
            'ov': {'soft_change': {'l1': torch.rand(1, 32, 32)}}}
    legacy_mismatch = None
    for fv in (False, True):
        for fh in (False, True):
            state = {'scale': {'height': 256, 'width': 256},
                     'crop_resize': {'enabled': True, 'top': 7, 'left': 8,
                                     'source_height': 256, 'source_width': 256},
                     'flip_v': fv, 'flip_h': fh, 'exchange': True}
            expected = cv2.resize((ids-2**30).astype(np.int32)[7:-7, 8:-8],
                                  (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.int64)+2**30
            soft = cv2.resize(val[7:-7, 8:-8], (256, 256), interpolation=cv2.INTER_LINEAR)
            if fv: expected, soft = expected[::-1], soft[::-1]
            if fh: expected, soft = expected[:, ::-1], soft[:, ::-1]
            result = replay_teacher_pack(pack, state, 'aligned')
            assert np.array_equal(result['sam']['t1']['instance_id'][0], expected)
            assert np.max(np.abs(result['sam']['t1']['boundary'][0].numpy()-soft)) < 1e-6
            assert result['ov']['soft_change']['l1'].shape == (1, 32, 32)
            original = pack['ov']['soft_change']['l1'][0].numpy()
            ov_expected = cv2.resize(original, (256,256), interpolation=cv2.INTER_LINEAR)
            ov_expected = cv2.resize(ov_expected[7:-7,8:-8], (256,256), interpolation=cv2.INTER_LINEAR)
            if fv: ov_expected = ov_expected[::-1]
            if fh: ov_expected = ov_expected[:,::-1]
            ov_expected = cv2.resize(ov_expected, (32,32), interpolation=cv2.INTER_LINEAR)
            assert np.max(np.abs(result['ov']['soft_change']['l1'][0].numpy()-ov_expected)) < 2e-6
            if not fv and not fh:
                legacy = _replay_tensor(pack['sam']['t1']['instance_id'], state, 'instance_id')[0]
                legacy_mismatch = {'large_int64_ids': float((legacy != torch.from_numpy(expected.copy())).float().mean())}
                small = _replay_tensor(pack['sam']['t1']['instance_id']-2**30, state, 'instance_id')[0]
                legacy_mismatch['ordinary_ids'] = float((small != torch.from_numpy(expected.copy())-2**30).float().mean())
    return legacy_mismatch


def run(device='cpu'):
    torch.set_num_threads(2)
    torch.manual_seed(2333)
    y, pack = synthetic_pack(size=64, device=device)
    p = (torch.rand_like(y)*.8+.1).requires_grad_()
    mechanism = TaskSpaceDirectionC()
    for k in (0, 1):
        out = mechanism(None, p, y, pack, force_action=k)
        grad = torch.autograd.grad(out['total'], p, retain_graph=True)[0]
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
        # Direct probability descent must point toward binary GT on admitted pixels.
        assert (grad*(p.detach()-y) >= -1e-7).all()
    proposals, quality = build_task_proposals(p, pack)
    assert not proposals.requires_grad and not quality.requires_grad
    swapped = {'sam': {'t1': pack['sam']['t2'], 't2': pack['sam']['t1']}, 'ov': pack['ov']}
    assert torch.allclose(proposals, build_task_proposals(p, swapped)[0], atol=1e-6)
    renamed = copy.deepcopy(pack)
    for t in ('t1', 't2'):
        renamed['sam'][t]['instance_id'] = renamed['sam'][t]['instance_id'].long()*1234567+2**30
    assert torch.allclose(proposals, build_task_proposals(p, renamed)[0], atol=1e-6)
    # Analytic extremes: useful, adverse, equal and zero-confidence teachers.
    for gt in (torch.zeros_like(y), torch.ones_like(y)):
        start = torch.full_like(y, .5)
        candidates = torch.cat((gt, 1-gt), 1)
        route = task_space_route(start, gt, candidates, torch.ones_like(candidates))
        assert (route['action']==0).all() and (route['effective_map'][:,1]==0).all()
        same = task_space_route(start, gt, start.expand_as(candidates), torch.ones_like(candidates))
        assert (same['action']==2).all()
        missing = task_space_route(start, gt, candidates, torch.zeros_like(candidates))
        assert (missing['action']==2).all()
        loss = mechanism(None, start.requires_grad_(), gt, pack)['total']
        assert torch.isfinite(loss)
    singleton = torch.arange(64*64,device=device).reshape(1,1,64,64)+1
    identity, q = instance_transport(p[:1], singleton, torch.ones_like(p[:1]), torch.zeros_like(p[:1]))
    assert torch.equal(identity,p[:1]) and q.count_nonzero()==0
    # All-Reject is exact zero and retains a differentiable zero for train loss.
    out = mechanism(None, p, y, pack, force_action=2)
    assert out['total'].item()==0
    assert torch.autograd.grad(out['total'], p)[0].count_nonzero()==0

    torch.manual_seed(2333)
    baseline = A2Net_LWGANet_L0(pretrained=False).to(device).train()
    rng_baseline = torch.get_rng_state().clone()
    torch.manual_seed(2333)
    model = A2Net_LWGANet_L0(pretrained=False, auxiliary_mode='direction_c',
                            routing_cfg={'mechanism':'task_space'}).to(device).train()
    assert torch.equal(rng_baseline, torch.get_rng_state())
    assert all(torch.equal(v, model.state_dict()[k]) for k,v in baseline.state_dict().items())
    a,b = torch.randn(2,3,64,64,device=device), torch.randn(2,3,64,64,device=device)
    criterion = build_loss('batch')
    bp,_ = baseline(a,b)
    pred,aux = model(a,b,target=y,teacher_pack=pack,force_action=2)
    assert all(torch.equal(x,z) for x,z in zip(bp,pred))
    sum(criterion(x,y) for x in bp).backward()
    (sum(criterion(x,y) for x in pred)+.06*aux['direction_c']['total']).backward()
    max_grad_norm = max(p.grad.norm().item() for p in baseline.parameters())
    for n,param in baseline.named_parameters():
        other = dict(model.named_parameters())[n]
        # CUDA backward reductions are non-deterministic (~1e-9 of the global
        # gradient scale); a real auxiliary leak would be ~kd_lambda of the same
        # scale. A global 1% tolerance separates the two cleanly.
        assert (param.grad - other.grad).norm() <= 1e-2 * max_grad_norm, n
    assert all(torch.equal(v,dict(model.named_buffers())[k]) for k,v in baseline.named_buffers())
    model.zero_grad(set_to_none=True)
    pred,aux = model(a,b,target=y,teacher_pack=pack)
    aux['direction_c']['total'].backward()
    kd_backbone_grad = sum(float(v.grad.abs().sum()) for v in model.backbone.parameters() if v.grad is not None)
    assert kd_backbone_grad > 0
    assert sum(v.numel() for v in model.training_auxiliary.parameters())==0
    args = SimpleNamespace(lr=5e-4,backbone_lr_mult=1.,weight_decay=1e-4,router_lr=1e-3,
                           implementation_version='dart_r_ts_v2',mechanism='task_space',cache_replay='aligned')
    opt = build_optimizer(args,model)
    assert build_router_optimizer(args,model) is None
    def step(net, optimizer):
        optimizer.zero_grad(set_to_none=True)
        ps,ad = net(a,b,target=y,teacher_pack=pack)
        (sum(criterion(x,y) for x in ps)+.06*ad['direction_c']['total']).backward()
        optimizer.step()
    step(model,opt)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'last_checkpoint.pth'
        save_checkpoint_atomic(build_checkpoint(model,opt,0,1,.5,args),path)
        loaded = torch.load(path,map_location='cpu',weights_only=False)
        validate_resume(args,loaded)
        restored = copy.deepcopy(model)
        restored.load_state_dict(loaded['model'])
        ropt = build_optimizer(args,restored); ropt.load_state_dict(loaded['optimizer'])
        assert loaded['router_optimizer'] is None and len(ropt.state)>0
        restore_rng_state(loaded['rng']); x = torch.rand(8)
        restore_rng_state(loaded['rng']); assert torch.equal(x,torch.rand(8))
        step(model,opt); step(restored,ropt)
        restore_error = max(float((v-restored.state_dict()[k]).abs().max())
                            for k,v in model.state_dict().items())
        # CUDA optimizer steps are non-deterministic (~1e-3 for one step, even for
        # identical models); a real resume bug diverges by orders of magnitude.
        assert restore_error < 1e-2
        bad = copy.deepcopy(loaded); bad['args']['implementation_version']='direction_c_v1'
        try: validate_resume(args,bad)
        except ValueError: pass
        else: raise AssertionError('Old-method checkpoint accepted as TS resume')
    model.eval(); model.training=True
    with torch.no_grad():
        off,_ = model(a,b,compute_auxiliary=False)
        on,_ = model(a,b,target=y,teacher_pack=pack)
    toggle_error = max(float((x-z).abs().max()) for x,z in zip(off,on))
    assert toggle_error==0
    model.eval()
    with torch.no_grad():
        before = model(a,b)
        exchanged = model(b,a)
    exchange_error = max(float((x-z).abs().max()) for x,z in zip(before,exchanged))
    assert exchange_error < 1e-6
    model.switch_to_deploy().switch_to_deploy().eval()
    assert not hasattr(model,'training_auxiliary')
    with torch.no_grad(): after=model(a,b)
    deploy_error=max(float((x-z).abs().max()) for x,z in zip(before,after))
    assert deploy_error==0 and sum(v.numel() for v in model.parameters())==2_913_094
    dummy = torch.zeros(1,3,256,256,device=device)
    with torch.no_grad(): flops,_=profile(model,inputs=(dummy,dummy),verbose=False)
    assert abs(flops-2.7475e9)<.03e9
    geometry_mismatch=check_geometry()
    report={'status':'passed','scope':'synthetic contract tests; not dataset accuracy',
            'torch':torch.__version__,'device':device,'train_parameters':2_913_094,
            'deploy_parameters':2_913_094,'flops_256':flops,'auxiliary_toggle_error':toggle_error,
            'deploy_max_error':deploy_error,'exchange_error':exchange_error,
            'resume_next_step_error':restore_error,'kd_backbone_grad_l1':kd_backbone_grad,
            'legacy_geometry_probe_mismatch_fraction':geometry_mismatch,
            'checks':['proposal detachment','ID relabel and temporal symmetry','both teacher gradients',
                      'GT-directed probability gradients','adverse/equal/missing teacher rejection',
                      'empty/full GT and singleton IDs','Reject baseline gradient and BN equality',
                      'optimizer and RNG resume','stale version rejected','aligned OpenCV geometry',
                      'auxiliary toggle','deploy removal and FLOPs']}
    print(json.dumps(report,indent=2));return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu')
    run(parser.parse_args().device)
