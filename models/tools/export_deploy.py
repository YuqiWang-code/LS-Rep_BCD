"""Export only the unchanged student from a format-v2 training checkpoint."""
import argparse
from pathlib import Path
import torch
from models import A2Net_LWGANet_L0
from models.utils.checkpoint import save_checkpoint_atomic


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    if Path(args.output).exists():raise FileExistsError('Choose a new output path')
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if checkpoint.get('format_version')!=2:raise ValueError('Expected a direction-C package checkpoint')
    state={k:v for k,v in checkpoint['model'].items() if not k.startswith('training_auxiliary.')}
    model=A2Net_LWGANet_L0(pretrained=False)
    model.load_state_dict(state,strict=True);model.switch_to_deploy().eval()
    assert sum(p.numel() for p in model.parameters())==2_913_094
    save_checkpoint_atomic({'model':model.state_dict(),'deploy_parameters':2_913_094,
                            'source_global_step':checkpoint['global_step']},args.output)
    print(args.output)


if __name__=='__main__':main()
