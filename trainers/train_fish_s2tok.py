"""Train Fish S2Tok with frozen codebooks and the existing DiT/CFM backbone."""
import argparse,json,sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from semantic2any.data.fish_codes import FishPairedDataset,collate_fish_pairs
from semantic2any.models.fish_s2tok import FishS2Tok
from semantic2any.utils.fish_tokenizer import load_fish_codec,FishCodeFeatures


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--resume');p.add_argument('--max-steps',type=int);args=p.parse_args()
    cfg=OmegaConf.load(args.config);tr=cfg.train
    if int(tr.save_every)<=0 or int(tr.validate_every)<=0:raise ValueError('Checkpoint/validation intervals must be positive')
    acc=Accelerator(gradient_accumulation_steps=int(tr.gradient_accumulation_steps),mixed_precision=tr.mixed_precision,kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    if int(os.environ.get('WORLD_SIZE','1'))>1 and acc.num_processes==1:
        raise RuntimeError('Distributed launcher was not detected; use accelerate launch (CPU tests need ACCELERATE_USE_CPU=true)')
    set_seed(int(tr.seed),device_specific=True)
    out=Path(tr.output_dir)
    if not args.resume and out.exists() and any(out.glob('checkpoint-*')):
        raise ValueError('Output contains checkpoints; use --resume or a new output directory')
    if acc.is_main_process:
        out.mkdir(parents=True,exist_ok=True)
        OmegaConf.save(cfg,out/'config.yaml')
    acc.wait_for_everyone()
    codec=load_fish_codec(cfg.codec.repo,cfg.codec.checkpoint,cfg.codec.sha256)
    features=FishCodeFeatures.from_codec(codec).to(acc.device);del codec
    model=FishS2Tok(cfg.model)
    model.cfm.setup_estimator_caches(int(tr.batch_size)*2,int(cfg.data.max_target_frames)+int(cfg.data.max_reference_frames))
    opt=torch.optim.AdamW(model.parameters(),lr=float(tr.learning_rate),weight_decay=float(tr.weight_decay))
    def loader(path,shuffle):
        ds=FishPairedDataset(path,int(cfg.data.max_target_frames),int(cfg.data.max_reference_frames))
        # Fixed order permits exact epoch/batch skipping on resume; prepare manifests with a fixed shuffle upstream.
        return DataLoader(ds,batch_size=int(tr.batch_size),shuffle=False,num_workers=int(cfg.data.num_workers),collate_fn=collate_fish_pairs,pin_memory=True)
    train=loader(cfg.data.train_manifest,False);valid=loader(cfg.data.valid_manifest,False)
    model,opt,train,valid=acc.prepare(model,opt,train,valid)
    step=start_epoch=start_batch=0
    if args.resume:
        info=json.loads((Path(args.resume)/'trainer.json').read_text())
        if info.get('world_size',1)!=acc.num_processes:raise ValueError('Resume requires unchanged distributed world size')
        if info['config']!=OmegaConf.to_container(cfg,resolve=True):raise ValueError('Resume config differs from checkpoint')
        acc.load_state(args.resume);step=info['step'];start_epoch=info['epoch'];start_batch=info['next_batch']
    def loss_for(batch):
        # FP32 frozen lookups; only the trainable network participates in autocast.
        semantic,target=features(batch['target_codes']);_,reference=features(batch['reference_codes'])
        with acc.autocast():return model(semantic,target,reference,batch['target_lengths'],batch['reference_lengths'])
    def save(epoch,next_batch):
        dest=out/f'checkpoint-{step}'
        acc.save_state(str(dest))
        if acc.is_main_process:
            info={'world_size':acc.num_processes,'step':step,'epoch':epoch,'next_batch':next_batch,'config':OmegaConf.to_container(cfg,resolve=True)}
            (dest/'trainer.json').write_text(json.dumps(info,indent=2))
            torch.save({'model':acc.unwrap_model(model).state_dict(),'config':info['config'],'step':step},dest/'inference.pt')
        acc.wait_for_everyone()
    def validate():
        model.eval();values=[]
        # Keep validation noise deterministic without consuming training RNG state.
        with torch.random.fork_rng(devices=[acc.device.index] if acc.device.type=='cuda' else []),torch.no_grad():
            torch.manual_seed(int(tr.seed)+10000)
            for i,batch in enumerate(valid):
                if i>=int(tr.validation_batches):break
                values.append(acc.gather(loss_for(batch).detach().float().reshape(1)))
        if not values:raise ValueError('Empty validation loader')
        acc.print(f'[Valid] step={step} loss={torch.cat(values).mean().item():.6f}')
        model.train()
    model.train();opt.zero_grad()
    for epoch in range(start_epoch,int(tr.epochs)):
        iterator=acc.skip_first_batches(train,start_batch) if epoch==start_epoch and start_batch else train
        offset=start_batch if epoch==start_epoch else 0
        for index,batch in enumerate(iterator,start=offset):
            with acc.accumulate(model):
                loss=loss_for(batch)
                finite=acc.reduce(torch.isfinite(loss.detach()).int(),reduction='sum')
                if finite.item()!=acc.num_processes:raise FloatingPointError('Nonfinite flow loss')
                acc.backward(loss)
                if acc.sync_gradients:
                    grad_norm=acc.clip_grad_norm_(model.parameters(),float(tr.max_grad_norm))
                    grad_ok=acc.reduce(torch.isfinite(grad_norm).int(),reduction='sum')
                    if grad_ok.item()!=acc.num_processes:raise FloatingPointError('Nonfinite gradient norm')
                opt.step();opt.zero_grad()
            if acc.sync_gradients:
                step+=1
                if step==1 or step%20==0:acc.print(f'[Train] step={step} epoch={epoch+1} loss={loss.item():.6f}')
                if step%int(tr.validate_every)==0:validate()
                if step%int(tr.save_every)==0:save(epoch,index+1)
                if args.max_steps and step>=args.max_steps:
                    save(epoch,index+1);return
        start_batch=0;save(epoch+1,0)
    acc.print('FISH_S2TOK_TRAINING_COMPLETE')

if __name__=='__main__':main()
