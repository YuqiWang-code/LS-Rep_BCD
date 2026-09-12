"""Detached CD error statistics. All connected components use 4-connectivity."""
from __future__ import annotations
import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F

H_NAMES = ('boundary_miss', 'fragmentation', 'small_change_miss',
           'bg_false_alarm', 'uncertainty', 'change_ratio')


def masked_mean(x, mask):
    """Per-image mean; empty support is exactly zero (not a fabricated observation)."""
    dims = tuple(range(1, x.ndim))
    return (x * mask).sum(dims) / mask.expand_as(x).sum(dims).clamp_min(1.)


@torch.no_grad()
def build_cd_difficulty(prediction, target, boundary_radius=2, small_area=64):
    p = prediction.detach().float().clamp(1e-6, 1-1e-6)
    y = target.detach().float()
    if p.shape != y.shape or p.ndim != 4 or p.shape[1] != 1:
        raise ValueError('P and GT must be matching [B,1,H,W] tensors')
    if not torch.all((y == 0) | (y == 1)):
        raise ValueError('GT must be binary 0/1')
    k = 2 * boundary_radius + 1
    interior = 1 - F.max_pool2d(1-y, k, 1, boundary_radius)
    inner_boundary = (y-interior).clamp(0, 1)
    hard = (p > .5).float()
    fn, fp = y*(1-hard), (1-y)*hard
    small_np = np.zeros(tuple(y.shape), dtype=np.float32)
    fragmented_np = np.zeros_like(small_np)
    frag = []
    gt_np = y[:, 0].cpu().numpy().astype(bool)
    hard_np = hard[:, 0].cpu().numpy().astype(bool)
    for i, (gt, pr) in enumerate(zip(gt_np, hard_np)):
        labels, n = ndimage.label(gt)
        area = np.bincount(labels.ravel())
        is_small = (area <= small_area) & (area > 0); is_small[0] = False
        small_np[i, 0] = is_small[labels]
        pieces, npieces = ndimage.label(pr & gt)
        if npieces:
            parent = ndimage.maximum(labels, pieces, np.arange(1, npieces+1)).astype(int)
            count = np.bincount(parent, minlength=n+1)
        else:
            count = np.zeros(n+1, dtype=int)
        fragmented_np[i, 0] = ((count > 1)[labels]) & gt
        # Sum of extra predicted components divided by total possible pieces.
        frag.append(float(np.maximum(count[1:]-1, 0).sum() / np.maximum(count[1:], 1).sum()) if n else 0.)
    small = torch.from_numpy(small_np).to(p.device)
    fragmented = torch.from_numpy(fragmented_np).to(p.device)
    entropy = -(p*p.log()+(1-p)*(1-p).log()) / np.log(2.)
    h = torch.stack((masked_mean(fn, inner_boundary), p.new_tensor(frag),
                     masked_mean(fn, small), masked_mean(fp, 1-y),
                     entropy.mean((1,2,3)), y.mean((1,2,3))), dim=1)
    masks = {'boundary': inner_boundary, 'interior': interior, 'small': small,
             'background': 1-y, 'fragmented': fragmented}
    # Error locations plus uncertainty; no student gradient enters this weighting.
    importance = 1 + fn*inner_boundary + fragmented*interior + fn*small + fp + entropy
    valid = torch.stack([masks[n].sum((1,2,3)) > 0 for n in
                         ('boundary','interior','small','background')], dim=1)
    return {'h': h, 'masks': masks, 'importance': importance, 'valid': valid}
