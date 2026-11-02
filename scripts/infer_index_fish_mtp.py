"""Autoregressive discrete Fish codes; optional frozen-codec waveform decode."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from semantic2any.models.index_fish_mtp import build_index_fish_model
from semantic2any.data.index_fish_codes import prepare_index_batch, fish_frames_for_duration
from semantic2any.data.fish_codes import read_codes
from semantic2any.utils.semantic_codecs import IndexTTS25CodeDecoder

def main():
    p=argparse.ArgumentParser()
    for name in ('checkpoint','index-codes','reference-record','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--duration-seconds',type=float,help='Original audio duration, if known; otherwise N/25')
    p.add_argument('--device',default='cuda')
    p.add_argument('--temperature',type=float,default=0)
    p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--wav')
    p.add_argument('--fish-repo')
    p.add_argument('--fish-checkpoint')
    args=p.parse_args()
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    cfg=state['config']; device=torch.device(args.device); torch.manual_seed(args.seed)
    raw=np.load(args.index_codes,allow_pickle=False)
    if raw.ndim!=1 or not len(raw) or not np.issubdtype(raw.dtype,np.integer):
        raise ValueError('Index codes must be nonempty integer [N]')
    duration=args.duration_seconds if args.duration_seconds is not None else len(raw)/25
    if abs(duration-len(raw)/25)>.081 or duration>cfg['data']['max_seconds']:
        raise ValueError('Duration inconsistent with Index codes or exceeds configured limit')
    frames=fish_frames_for_duration(duration)
    codes=torch.as_tensor(raw.astype(np.int64),device=device)[None]
    lengths=torch.tensor([len(raw)],device=device)
    decoder=IndexTTS25CodeDecoder(cfg['index']['bundle'],expected_sha256=cfg['index']['sha256']).to(device).eval() if cfg['model']['source_mode']=='decoded' else None
    refpath=Path(args.reference_record)
    ref=read_codes(json.loads(refpath.read_text()),refpath.parent,cfg['data']['max_reference_frames']).to(device)[None]
    model=build_index_fish_model(cfg['model'],cfg.get('architecture','cross_attention')).to(device).eval()
    model.load_state_dict(state['model'],strict=True)
    with torch.inference_mode():
        source,ids,sl=prepare_index_batch(cfg['model']['source_mode'],decoder,codes,lengths)
        result=model.generate(source,ids,sl,ref,torch.tensor([ref.shape[1]],device=device),frames,args.temperature)
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        np.save(args.output,result[0].cpu().numpy().astype(np.uint16))
        if args.wav:
            if not args.fish_repo or not args.fish_checkpoint:
                raise ValueError('--wav requires --fish-repo and --fish-checkpoint')
            from semantic2any.utils.fish_tokenizer import load_fish_codec,FishCodeFeatures,decode_summed_features
            import soundfile as sf
            codec=load_fish_codec(args.fish_repo,args.fish_checkpoint,cfg['codec']['sha256']).to(device)
            lookup=FishCodeFeatures.from_codec(codec).to(device)
            _,summed=lookup(torch.cat((ref,result),1))
            audio=decode_summed_features(codec,summed)[0,0,ref.shape[1]*2048:]
            audio=audio[:round(duration*44100)]
            if not torch.isfinite(audio).all():raise FloatingPointError('Nonfinite waveform')
            Path(args.wav).parent.mkdir(parents=True,exist_ok=True)
            sf.write(args.wav,audio.cpu().numpy(),44100,subtype='FLOAT')
if __name__=='__main__':main()
