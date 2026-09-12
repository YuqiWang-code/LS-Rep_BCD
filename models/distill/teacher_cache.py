"""Validated read-only SAMStruct + OVCDistill caches, paired by sample ID."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch


def canonical_dataset(name):
    return str(name).upper().replace('-CD-256','').replace('_CD_256','')


class TeacherCache:
    def __init__(self, root, dataset_name, kind):
        self.root = Path(root)
        self.kind = kind
        self.manifest = json.loads((self.root/'manifest.json').read_text(encoding='utf-8'))
        m = self.manifest
        if m.get('cache_version') != 1 or not isinstance(m.get('entries'),dict) or not m['entries']:
            raise ValueError(f'{root}: expected cache_version=1 and nonempty entries mapping')
        if canonical_dataset(m.get('source_dataset')) != canonical_dataset(dataset_name):
            raise ValueError(f'{root}: source_dataset mismatch')
        if not m.get('config_hash'):
            raise ValueError(f'{root}: missing config_hash')
        if kind == 'sam' and m.get('teacher_type') != 'sam2_struct_v2':
            raise ValueError(f'{root}: expected SAM teacher_type=sam2_struct_v2')
        self.entries = m['entries']
        summary_path = self.root/'validation_summary.json'
        # Prior summary is useful evidence, but our own complete schema check is
        # available even when an OV generator did not produce that summary.
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            if summary.get('config_hash') != m['config_hash']:
                raise ValueError(f'{root}: validation config_hash mismatch')
        self.fingerprint = hashlib.sha256((self.root/'manifest.json').read_bytes()).hexdigest()

    def path_for(self, sample_id):
        if sample_id not in self.entries:
            raise KeyError(f'{self.kind}: cache missing sample {sample_id}')
        root = (self.root/'train').resolve()
        path = (root/self.entries[sample_id]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f'{sample_id}: unsafe cache relative path')
        return path

    def load(self, sample_id):
        path = self.path_for(sample_id)
        raw = torch.load(path,map_location='cpu',weights_only=False)
        if raw.get('sample_id') != sample_id:
            raise ValueError(f'{path}: sample_id mismatch')
        if raw.get('meta',{}).get('config_hash') != self.manifest['config_hash']:
            raise ValueError(f'{path}: config_hash mismatch')
        if self.kind == 'sam':
            pack = {t:{k:raw[t][k] for k in ('instance_id','boundary','quality')} for t in ('t1','t2')}
        else:
            pack = {k:{level:raw[k][level] for level in ('l1','l2')}
                    for k in ('soft_change','confidence','relation')}
        validate_pack(pack,self.kind,str(path))
        return pack


def validate_pack(pack,kind,source='cache'):
    def check(x,channels,bounded=False,integer=False):
        if not torch.is_tensor(x) or x.ndim != 3 or x.shape[0] != channels:
            raise ValueError(f'{source}: expected [{channels},H,W] cache tensor')
        if min(x.shape[-2:]) < 1 or not torch.isfinite(x).all():
            raise ValueError(f'{source}: empty/nonfinite cache')
        if bounded and (x.min()<0 or x.max()>1):
            raise ValueError(f'{source}: probability/quality field must be in [0,1], no silent clipping')
        if integer and (x.dtype not in (torch.int32,torch.int64) or x.min()<0):
            raise ValueError(f'{source}: instance IDs must be nonnegative int32/int64')
    if kind == 'sam':
        shapes=[]
        for t in ('t1','t2'):
            for k in ('instance_id','boundary','quality'):
                check(pack[t][k],1,k!='instance_id',k=='instance_id')
                shapes.append(pack[t][k].shape)
        if len(set(shapes))!=1:raise ValueError(f'{source}: SAM spatial shape mismatch')
    else:
        for level in ('l1','l2'):
            for k in ('soft_change','confidence','relation'):
                check(pack[k][level],8 if k=='relation' else 1,k!='relation')
            if len({pack[k][level].shape[-2:] for k in pack})!=1:
                raise ValueError(f'{source}: OV spatial shape mismatch')


class PairedTeacherCache:
    def __init__(self,sam_root,ov_root,dataset_name,train_list):
        self.sam=TeacherCache(sam_root,dataset_name,'sam')
        self.ov=TeacherCache(ov_root,dataset_name,'ov')
        expected=list(train_list)
        if len(expected)!=len(set(expected)):raise ValueError('Duplicate sample IDs in train list')
        for cache in (self.sam,self.ov):
            if set(cache.entries)!=set(expected):
                missing=set(expected)-set(cache.entries);extra=set(cache.entries)-set(expected)
                raise ValueError(f'{cache.kind}: train coverage mismatch: missing={len(missing)}, extra={len(extra)}')
            for sample_id in expected:
                if not cache.path_for(sample_id).is_file():
                    raise FileNotFoundError(cache.path_for(sample_id))
        self.entries=self.sam.entries
        self.fingerprint={'sam':self.sam.fingerprint,'ov':self.ov.fingerprint}

    def load(self,sample_id):
        return {'sam':self.sam.load(sample_id),'ov':self.ov.load(sample_id)}
