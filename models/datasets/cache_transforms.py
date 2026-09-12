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


def _resize_aligned(value, size, integer=False):
    if tuple(value.shape[-2:]) == tuple(size):
        return value
    if integer:
        # OpenCV INTER_NEAREST uses floor(dst * src / dst_size). Preserve
        # arbitrary int64 IDs without a float conversion or round-to-nearest.
        h, w = value.shape[-2:]
        yi = torch.arange(size[0], device=value.device)*h//size[0]
        xi = torch.arange(size[1], device=value.device)*w//size[1]
        return value.index_select(-2, yi).index_select(-1, xi)
    return F.interpolate(value.float().unsqueeze(0), size=size, mode='bilinear',
                         align_corners=False).squeeze(0)


def _replay_aligned(value, state, key, native_resolution):
    """Replay Scale -> crop/resize -> flips; OV returns to its native grid.

    OV is first reconstructed on the student raster so fractional native-grid
    margins do not create different crop geometry. No cache is modified on disk.
    """
    if value.ndim != 3:
        raise ValueError('Cache tensors must be [C,H,W]')
    dtype, original = value.dtype, value.shape[-2:]
    integer = key == 'instance_id'
    scale = state.get('scale')
    crop = state.get('crop_resize', {})
    canvas = (int(scale['height']), int(scale['width'])) if scale else original
    x = _resize_aligned(value if integer else value.float(), canvas, integer)
    if crop.get('enabled', False):
        if tuple(canvas) != (crop['source_height'], crop['source_width']):
            raise ValueError('Cache crop source must match the scaled image raster')
        top, left = int(crop['top']), int(crop['left'])
        h, w = canvas
        if top < 0 or left < 0 or 2*top >= h or 2*left >= w:
            raise ValueError('Invalid cache crop')
        x = _resize_aligned(x[:, top:h-top, left:w-left], canvas, integer)
    if state.get('flip_v', False):
        x = x.flip(-2)
    if state.get('flip_h', False):
        x = x.flip(-1)
    if native_resolution:
        x = _resize_aligned(x, original, integer)
    return x.to(dtype)


def replay_teacher_pack(pack,state,version='aligned'):
    if version not in ('legacy', 'aligned'):
        raise ValueError('Unknown cache replay version')
    def visit(data,is_ov=False,parent=None):
        result={}
        for key,value in data.items():
            if isinstance(value,dict):result[key]=visit(value,is_ov or key=='ov',key)
            elif torch.is_tensor(value):
                result[key]=(_replay_tensor(value,state,key,is_ov) if version=='legacy'
                             else _replay_aligned(value,state,key,is_ov))
        if state.get('exchange',False) and 't1' in result and 't2' in result:
            result['t1'],result['t2']=result['t2'],result['t1']
        return result
    return visit(pack)
