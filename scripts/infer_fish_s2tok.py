"""Generate summed S from semantic codes + reference codes, optionally decode audio."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from omegaconf import OmegaConf
from semantic2any.models.fish_s2tok import FishS2Tok
from semantic2any.data.fish_codes import read_codes
from semantic2any.utils.fish_tokenizer import load_fish_codec,FishCodeFeatures,decode_summed_features

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True,help='checkpoint-N/inference.pt');p.add_argument('--semantic-codes',required=True,help='integer .npy [T], codebook 0 only');p.add_argument('--reference-record',required=True,help='JSON index record with absolute/GCS codes path');p.add_argument('--fish-repo',required=True);p.add_argument('--fish-checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--wav');p.add_argument('--device',default='cuda');p.add_argument('--steps',type=int,default=25);p.add_argument('--cfg',type=float,default=0.7);p.add_argument('--seed',type=int,default=1234);args=p.parse_args()
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True);cfg=OmegaConf.create(state['config']);device=torch.device(args.device);torch.manual_seed(args.seed)
    codec=load_fish_codec(args.fish_repo,args.fish_checkpoint,cfg.codec.sha256).to(device)
    lookup=FishCodeFeatures.from_codec(codec).to(device)
    codes=np.load(args.semantic_codes,allow_pickle=False)
    if codes.ndim!=1 or not np.issubdtype(codes.dtype,np.integer) or not len(codes):raise ValueError('Semantic .npy must be nonempty integer [T]')
    refpath=Path(args.reference_record);rec=json.loads(refpath.read_text());ref=read_codes(rec,refpath.parent,int(cfg.data.max_reference_frames)).to(device)[None]
    model=FishS2Tok(cfg.model).to(device).eval();model.load_state_dict(state['model'],strict=True)
    model.cfm.setup_estimator_caches(2,len(codes)+ref.shape[1])
    with torch.inference_mode():
        semantic=lookup.semantic(torch.as_tensor(codes.astype(np.int64),device=device)[None]);_,reference=lookup(ref)
        summed=model.generate(semantic,reference,args.steps,args.cfg)
        if not torch.isfinite(summed).all():raise FloatingPointError('Nonfinite generated S')
        Path(args.output).parent.mkdir(parents=True,exist_ok=True);np.save(args.output,summed[0].cpu().numpy())
        if args.wav:
            import soundfile as sf
            # Prepend reference S to preserve causal codec context, then remove its audio.
            audio=decode_summed_features(codec,torch.cat([reference,summed],dim=1))
            audio=audio[...,reference.shape[1]*2048:]
            if not torch.isfinite(audio).all():raise FloatingPointError('Nonfinite decoded audio')
            Path(args.wav).parent.mkdir(parents=True,exist_ok=True);sf.write(args.wav,audio[0,0].cpu().numpy(),44100,subtype='FLOAT')

if __name__=='__main__':main()
