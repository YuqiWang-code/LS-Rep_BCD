"""Read-only, full two-teacher coverage/schema validation. Never rebuilds caches."""
import argparse
import json
from pathlib import Path
from models.distill.teacher_cache import PairedTeacherCache


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data_root',required=True);p.add_argument('--dataset_name',required=True)
    p.add_argument('--sam_cache_root',required=True);p.add_argument('--ov_cache_root',required=True)
    args=p.parse_args()
    entries=[v.strip() for v in (Path(args.data_root)/'list/train.txt').read_text().splitlines() if v.strip()]
    cache=PairedTeacherCache(args.sam_cache_root,args.ov_cache_root,args.dataset_name,entries)
    for i,sample_id in enumerate(entries):
        cache.load(sample_id)
        if (i+1)%1000==0:print(f'Validated {i+1}/{len(entries)} samples',flush=True)
    print(json.dumps({'status':'passed','n_samples':len(entries),'fingerprint':cache.fingerprint,
                      'scope':'coverage, IDs, hashes, tensor shapes/ranges; not teacher accuracy'},indent=2))


if __name__=='__main__':main()
