"""CPU-only integration canary. Synthetic codes; not a quality benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from omegaconf import OmegaConf
from semantic2any.third_party.indextts.maskgct import RepCodec
from semantic2any.utils.semantic_codecs import IndexTTS25SemanticCodec

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--ddp',action='store_true');args=p.parse_args()
    root=Path(args.output_dir).resolve();root.mkdir(exist_ok=False,parents=True)
    repo=Path(__file__).resolve().parents[1]
    torch.set_num_threads(1);torch.manual_seed(42)
    arch=dict(codebook_size=32,hidden_size=8,codebook_dim=4,vocos_dim=8,vocos_intermediate_dim=16,vocos_num_layers=2,num_quantizers=1,downsample_scale=2)
    codec=RepCodec(**arch);bundle=root/'index.pt'
    torch.save({'arch':arch,'frames_per_code':2,'state_dict':{k:v for k,v in codec.state_dict().items() if k.startswith(IndexTTS25SemanticCodec.DECODE_PREFIXES)}},bundle)
    rng=np.random.default_rng(42)
    rng.integers(0,8,(8,10),dtype=np.uint16).tofile(root/'fish.bin')
    rng.integers(0,32,5,dtype=np.uint16).tofile(root/'index.bin')
    row={'duration_seconds':.2,'target':{'id':'a','codes':str(root/'fish.bin'),'num_frames':5},'reference':{'id':'b','codes':str(root/'fish.bin'),'num_frames':3,'offset_bytes':100},'index':{'id':'a','codes':str(root/'index.bin'),'num_frames':5}}
    (root/'train.jsonl').write_text((json.dumps(row)+'\n')*4);(root/'valid.jsonl').write_text(json.dumps(row)+'\n')
    (root/'ref.json').write_text(json.dumps(row['reference']));np.save(root/'index.npy',np.fromfile(root/'index.bin',dtype='<u2'))
    cfg=OmegaConf.load(repo/'configs/index_fish_mtp.yaml');cfg.index.bundle=str(bundle);cfg.index.sha256=hashlib.sha256(bundle.read_bytes()).hexdigest()
    cfg.model.update(dict(dim=16,heads=2,condition_layers=1,shared_layers=1,index_dim=8,index_vocab=32,codebook_sizes=[8]*10))
    cfg.data.train_manifest=str(root/'train.jsonl');cfg.data.valid_manifest=str(root/'valid.jsonl');cfg.data.num_workers=0
    cfg.train.batch_size=1;cfg.train.mixed_precision='no';cfg.train.save_every=2;cfg.train.validate_every=2;cfg.train.gradient_accumulation_steps=2
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    def run(argv,ddp=False):
        subprocess.run([sys.executable,*argv],cwd=repo,env={**env,**({'ACCELERATE_USE_CPU':'true'} if ddp else {})},check=True)
    def config(name,mode='raw'):
        cfg.train.output_dir=str(root/name);cfg.model.source_mode=mode;cfg.model.lookahead_seconds=.08 if mode=='raw' else None
        path=root/(name+'.yaml');OmegaConf.save(cfg,path);return str(path)
    train='trainers/train_index_fish_mtp.py'
    c=config('resume');run([train,'--config',c,'--max-steps','2'])
    run([train,'--config',c,'--resume',str(root/'resume/checkpoint-2'),'--max-steps','3'])
    run([train,'--config',config('continuous'),'--max-steps','3'])
    a=torch.load(root/'resume/checkpoint-3/inference.pt',weights_only=True)['model'];b=torch.load(root/'continuous/checkpoint-3/inference.pt',weights_only=True)['model']
    assert all(torch.equal(a[k],b[k]) for k in a),'Resume differs from uninterrupted training'
    run([train,'--config',config('decoded','decoded'),'--max-steps','2'])
    if args.ddp:
        run(['-m','torch.distributed.run','--standalone','--nproc_per_node=2',train,'--config',config('ddp'),'--max-steps','2'],ddp=True)
        info=json.loads((root/'ddp/checkpoint-2/trainer.json').read_text());assert info['world_size']==2
    run(['scripts/infer_index_fish_mtp.py','--checkpoint',str(root/'resume/checkpoint-3/inference.pt'),'--index-codes',str(root/'index.npy'),'--reference-record',str(root/'ref.json'),'--device','cpu','--output',str(root/'generated.npy')])
    generated=np.load(root/'generated.npy');assert generated.shape==(5,10) and generated.max()<8
    (root/'result.json').write_text(json.dumps({'passed':True,'resume_exact':True,'raw_and_decoded':True,'ddp':args.ddp,'generated_shape':list(generated.shape)},indent=2))
    print('INDEX_FISH_MTP_SMOKE_PASS',root)
if __name__=='__main__':main()
