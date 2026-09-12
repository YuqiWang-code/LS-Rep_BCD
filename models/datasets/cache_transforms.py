"""Replay the existing symmetric crop/resize/flip/exchange on cached maps.

SAM uses the student raster resolution; OV retains its native multiscale grids.
OV relations are opaque features for spatial Gram loss (permutation invariant).
No guessed directional-channel permutation is applied.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def _replay_tensor(value,state,key=None,native_resolution=False):
    if value.ndim!=3:raise ValueError('Cache tensors must be [C,H,W]')
    dtype=value.dtype
    out_h,out_w=value.shape[-2:]
    if not native_resolution and state.get('scale'):
        out_h=int(state['scale']['height']);out_w=int(state['scale']['width'])
    # Grid in normalized source coordinates: exact symmetric crop fractions,
    # followed by flips. align_corners=False matches image pixel centres.
    yy=2*(torch.arange(out_h,dtype=torch.float32)+.5)/out_h-1
    xx=2*(torch.arange(out_w,dtype=torch.float32)+.5)/out_w-1
    crop=state.get('crop_resize',{})
    if crop.get('enabled',False):
        yy=yy*(1-2*crop['top']/crop['source_height'])
        xx=xx*(1-2*crop['left']/crop['source_width'])
    if state.get('flip_v',False):yy=-yy
    if state.get('flip_h',False):xx=-xx
    gy,gx=torch.meshgrid(yy,xx,indexing='ij')
    grid=torch.stack((gx,gy),dim=-1).unsqueeze(0)
    x=value.float().unsqueeze(0)
    if key=='instance_id' and value.max()>2**24:
        x=x.double();grid=grid.double()
    out=F.grid_sample(x,grid,mode='nearest' if key=='instance_id' else 'bilinear',
                      padding_mode='border',align_corners=False).squeeze(0)
    return out.to(dtype)


def replay_teacher_pack(pack,state):
    def visit(data,is_ov=False,parent=None):
        result={}
        for key,value in data.items():
            if isinstance(value,dict):result[key]=visit(value,is_ov or key=='ov',key)
            elif torch.is_tensor(value):result[key]=_replay_tensor(value,state,key,is_ov)
        if state.get('exchange',False) and 't1' in result and 't2' in result:
            result['t1'],result['t2']=result['t2'],result['t1']
        return result
    return visit(pack)
