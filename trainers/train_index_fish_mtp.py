"""Train Index-to-Fish discrete codebooks with soft time alignment and MTP."""
import argparse,json,sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from semantic2any.data.index_fish_codes import IndexFishDataset, collate_index_fish, prepare_index_batch
from semantic2any.models.index_fish_mtp import build_index_fish_model
from semantic2any.data.block_sampler import BlockShuffleSampler
from semantic2any.utils.semantic_codecs import IndexTTS25CodeDecoder


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--resume');p.add_argument('--max-steps',type=int)
    p.add_argument('--compile-blocks',action='store_true',help='Compile Transformer blocks in place; checkpoint keys and training config stay compatible')
    args=p.parse_args()
    cfg=OmegaConf.load(args.config);tr=cfg.train
    if cfg.model.source_mode == 'decoded' and (not cfg.index.sha256 or len(str(cfg.index.sha256)) != 64):raise ValueError('Pin the Index decoder SHA256 before training')
    if int(tr.save_every)<=0 or int(tr.validate_every)<=0:raise ValueError('Checkpoint/validation intervals must be positive')
    acc=Accelerator(gradient_accumulation_steps=int(tr.gradient_accumulation_steps),mixed_precision=tr.mixed_precision,kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)])
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
    model=build_index_fish_model(OmegaConf.to_container(cfg.model,resolve=True),cfg.get("architecture","cross_attention"))
    if args.compile_blocks:
        blocks=list(model.condition)+list(model.shared)+list(model.residual)
        for block in blocks:
            block.compile(fullgraph=True,dynamic=True)
        acc.print(f'[Compile] enabled {len(blocks)} Transformer blocks; fullgraph=True dynamic=True')
    decoder=None
    if model.source_mode == 'decoded':
        decoder=IndexTTS25CodeDecoder(cfg.index.bundle,expected_sha256=cfg.index.sha256).to(acc.device).eval()
        if decoder.codebook_size != model.index_embedding.num_embeddings or decoder.semantic_dim != model.index_projection.in_features:
            raise ValueError('Index decoder/model dimensions differ')
    opt=torch.optim.AdamW(model.parameters(),lr=float(tr.learning_rate),weight_decay=float(tr.weight_decay))
    def loader(path,shuffle):
        ds=IndexFishDataset(path,float(cfg.data.max_seconds),int(cfg.data.max_reference_frames))
        block_size=int(tr.get('shuffle_block_size',0))
        sampler=BlockShuffleSampler(ds,block_size,int(tr.seed)) if shuffle and block_size else None
        return DataLoader(ds,batch_size=int(tr.batch_size),shuffle=False,sampler=sampler,num_workers=int(cfg.data.num_workers),collate_fn=collate_index_fish,pin_memory=True)
    train=loader(cfg.data.train_manifest,True);valid=loader(cfg.data.valid_manifest,False)
    model,opt,train,valid=acc.prepare(model,opt,train,valid)
    acc.print(f'[Model] parameters={sum(p.numel() for p in model.parameters())}')
    manifests={str(Path(path).resolve()): {'size':Path(path).stat().st_size,'mtime_ns':Path(path).stat().st_mtime_ns} for path in (cfg.data.train_manifest,cfg.data.valid_manifest)}
    step=start_epoch=start_batch=0
    if args.resume:
        info=json.loads((Path(args.resume)/'trainer.json').read_text())
        if info.get('world_size',1)!=acc.num_processes:raise ValueError('Resume requires unchanged distributed world size')
        if info['config']!=OmegaConf.to_container(cfg,resolve=True):raise ValueError('Resume config differs from checkpoint')
        if info.get('manifests') != manifests:raise ValueError('Manifest files changed since checkpoint')
        acc.load_state(args.resume);step=info['step'];start_epoch=info['epoch'];start_batch=info['next_batch']
    def loss_for(batch):
        source, ids, lengths = prepare_index_batch(cfg.model.source_mode,decoder,batch['index_codes'],batch['index_lengths'])
        if hasattr(acc.unwrap_model(model), "validate_alignment_lengths"):
            acc.unwrap_model(model).validate_alignment_lengths(lengths,batch["target_lengths"])
        with acc.autocast():
            logits=model(source,ids,lengths,batch['reference_codes'],batch['reference_lengths'],batch['target_codes'],batch['target_lengths'])
        loss, per_book=acc.unwrap_model(model).loss(logits,batch['target_codes'],batch['target_lengths'])
        loss_for.last_books=per_book.detach()
        return loss
    def save(epoch,next_batch):
        dest=out/f'checkpoint-{step}'
        acc.save_state(str(dest))
        if acc.is_main_process:
            info={'world_size':acc.num_processes,'step':step,'epoch':epoch,'next_batch':next_batch,'config':OmegaConf.to_container(cfg,resolve=True),'manifests':manifests}
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
        if hasattr(train,'set_epoch'):train.set_epoch(epoch)
        iterator=acc.skip_first_batches(train,start_batch) if epoch==start_epoch and start_batch else train
        offset=start_batch if epoch==start_epoch else 0
        for index,batch in enumerate(iterator,start=offset):
            with acc.accumulate(model):
                loss=loss_for(batch)
                finite=acc.reduce(torch.isfinite(loss.detach()).int(),reduction='sum')
                if finite.item()!=acc.num_processes:raise FloatingPointError('Nonfinite codebook CE')
                acc.backward(loss)
                if acc.sync_gradients:
                    grad_norm=acc.clip_grad_norm_(model.parameters(),float(tr.max_grad_norm))
                    grad_ok=acc.reduce(torch.isfinite(grad_norm).int(),reduction='sum')
                    if grad_ok.item()!=acc.num_processes:raise FloatingPointError('Nonfinite gradient norm')
                opt.step();opt.zero_grad()
            if acc.sync_gradients:
                step+=1
                if step==1 or step%20==0:
                    metrics=acc.reduce(loss_for.last_books.float(),reduction='mean')
                    acc.print(f'[Train] step={step} epoch={epoch+1} loss={metrics.mean().item():.6f} codebook_ce={metrics.tolist()}')
                if step%int(tr.validate_every)==0:validate()
                if step%int(tr.save_every)==0:save(epoch,index+1)
                if args.max_steps and step>=args.max_steps:
                    save(epoch,index+1);return
        start_batch=0;save(epoch+1,0)
    acc.print('INDEX_FISH_MTP_TRAINING_COMPLETE')

if __name__=='__main__':main()
