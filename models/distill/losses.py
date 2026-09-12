"""Teacher-specific losses. Never regress numerical instance IDs.

OV relation is distilled through spatial cosine Gram matrices. Its eight channels
are treated as opaque features, NOT guessed to mean eight directional neighbours.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .diagnostics import masked_mean


def resize(x, size, mode='bilinear'):
    return F.interpolate(x.float(), size=size, mode=mode,
                         **({'align_corners': False} if mode == 'bilinear' else {}))


def spatial_gram(x, size=8):
    z = F.adaptive_avg_pool2d(x.float(), (size, size)).flatten(2)
    z = F.normalize(z, dim=1, eps=1e-6)
    return z.transpose(1,2) @ z


class TeacherHeads(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.sam_boundary = nn.Conv2d(channels, 1, 1)
        self.sam_relation = nn.Conv2d(channels, 8, 1)
        self.ov_change = nn.Conv2d(channels, 1, 1)
        self.ov_relation = nn.Conv2d(channels, 8, 1)

    def forward(self, feature, pack, importance):
        sam, ov = pack['sam'], pack['ov']
        h, w = sam['t1']['boundary'].shape[-2:]
        boundary = torch.maximum(sam['t1']['boundary'], sam['t2']['boundary']).detach().float()
        qualities = [sam[t]['quality'].detach().float() * (sam[t]['instance_id'] > 0)
                     for t in ('t1','t2')]
        q_sam = (qualities[0] + qualities[1]) * .5
        boundary_logits = resize(self.sam_boundary(feature), (h,w))
        boundary_loss = masked_mean(F.binary_cross_entropy_with_logits(
            boundary_logits, boundary, reduction='none'), q_sam)
        # Derive 4-connected right/down same-instance relations independently in T1/T2.
        embed = F.normalize(resize(self.sam_relation(feature), (h,w)), dim=1, eps=1e-6)
        relation_losses = []
        for t, quality in zip(('t1','t2'), qualities):
            ids = sam[t]['instance_id'].detach()
            for dim in (-1, -2):
                left = [slice(None)]*4; right = [slice(None)]*4
                left[dim] = slice(None,-1); right[dim] = slice(1,None)
                left, right = tuple(left), tuple(right)
                a, b = ids[left], ids[right]
                weight = torch.minimum(quality[left], quality[right]) * ((a > 0) & (b > 0))
                desired = (a == b).float()
                similarity = (1+(embed[left]*embed[right]).sum(1,keepdim=True))*.5
                relation_losses.append(masked_mean((similarity-desired).square(), weight))
        sam_relation = torch.stack(relation_losses).mean(0)
        sam_loss = boundary_loss + .1*sam_relation
        ov_losses, response_terms, relation_terms, q_ov_terms = [], [], [], []
        for level in ('l1','l2'):
            soft = ov['soft_change'][level].detach().float()
            conf = ov['confidence'][level].detach().float()
            rel = ov['relation'][level].detach().float()
            change_logits = resize(self.ov_change(feature), soft.shape[-2:])
            response = masked_mean(F.binary_cross_entropy_with_logits(change_logits,soft,reduction='none'),conf)
            student_gram = spatial_gram(self.ov_relation(feature))
            teacher_gram = spatial_gram(rel)
            q = F.adaptive_avg_pool2d(conf, (8,8)).flatten(2).squeeze(1)
            pair_weight = q.unsqueeze(2)*q.unsqueeze(1)
            # Ignore the diagonal: normalized self-similarity carries no supervision.
            pair_weight = pair_weight * (1-torch.eye(64,device=q.device).unsqueeze(0))
            relation = masked_mean((student_gram-teacher_gram).square(), pair_weight)
            ov_losses.append(response+.1*relation)
            response_terms.append(response); relation_terms.append(relation)
            q_ov_terms.append(masked_mean(resize(conf,importance.shape[-2:]),importance))
        q_sam_score = masked_mean(resize(q_sam,importance.shape[-2:]),importance)
        quality = torch.stack((q_sam_score,torch.stack(q_ov_terms).mean(0)),dim=1).detach()
        return torch.stack((sam_loss,torch.stack(ov_losses).mean(0)),dim=1), quality, {
            'sam_boundary':boundary_loss.mean(), 'sam_relation':sam_relation.mean(),
            'ov_response':torch.stack(response_terms).mean(),
            'ov_relation':torch.stack(relation_terms).mean(),
        }
