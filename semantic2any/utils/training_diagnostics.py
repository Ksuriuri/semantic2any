"""Opt-in S2VAE observation and first-failure capture; does not change finite updates."""
import gzip
import json
import math
import random
from pathlib import Path
import torch
import torch.distributed as dist


def cpu_tree(x):
    if torch.is_tensor(x): return x.detach().cpu().clone()
    if isinstance(x, dict): return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return type(x)(cpu_tree(v) for v in x)
    return x


def clone_tree(x):
    if torch.is_tensor(x):return x.detach().clone()
    if isinstance(x,dict):return {k:clone_tree(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return type(x)(clone_tree(v) for v in x)
    return x


def compact_records(records):
    keys={'id','dataset','source_dataset','speaker_id','audio_path','semantic_code_path','semantic_code_offset','semantic_code_length','vae_latent_path','vae_latent_offset','vae_latent_length','duration','duration_seconds','num_frames'}
    def one(r):
        if not isinstance(r,dict):return str(r)
        return {k:one(v) if k in ('target','prompt') else v for k,v in r.items() if k in keys or k in ('target','prompt')}
    return [one(r) for r in (records or [])]


def tensor_summary(x):
    x=x.detach()
    if not x.numel(): return dict(shape=list(x.shape),nonfinite=0,amax=0.,rms=0.)
    # FP64 reductions distinguish actual nonfinite values from FP32 norm overflow.
    vals=torch.stack(((~torch.isfinite(x)).sum().double(),x.abs().amax().double(),torch.linalg.vector_norm(x,dtype=torch.float64)/math.sqrt(x.numel()))).cpu().tolist()
    return dict(shape=list(x.shape),nonfinite=int(vals[0]),amax=vals[1],rms=vals[2])


def collection_summary(items):
    names=[];values=[];sizes=[]
    for name,x in items:
        if x is None: continue
        x=x.detach();names.append(name);sizes.append(x.numel())
        values.append(torch.stack((torch.linalg.vector_norm(x,dtype=torch.float64),x.abs().amax().double(),(~torch.isfinite(x)).sum().double())))
    if not values:return dict(norm64=0.,nonfinite=0,top=[],modules={})
    rows=torch.stack(values).cpu().tolist();groups={}
    for name,(norm,amax,bad) in zip(names,rows):
        group='.'.join(name.split('.')[:4]);groups[group]=groups.get(group,0.)+norm*norm
    return dict(norm64=math.sqrt(sum(v[0]*v[0] for v in rows)),nonfinite=sum(int(v[2]) for v in rows),
                modules={k:math.sqrt(v) for k,v in groups.items()},
                top=[dict(name=names[i],norm64=rows[i][0],amax=rows[i][1],nonfinite=int(rows[i][2])) for i in sorted(range(len(rows)),key=lambda i: (rows[i][2]>0,rows[i][0]),reverse=True)[:10]])


class TrainingDiagnostics:
    def __init__(self,model,optimizer,accelerator,output_dir,interval=20,probe_interval=200,max_finite_bundles=3):
        self.model=model;self.optimizer=optimizer;self.acc=accelerator
        self.root=Path(output_dir)/'diagnostics';self.root.mkdir(parents=True,exist_ok=True)
        self.rank=int(accelerator.process_index);self.log=gzip.open(self.root/f'rank{self.rank}.jsonl.gz','at',encoding='utf-8')
        self.interval=interval;self.probe_interval=probe_interval;self.max_bundles=max_finite_bundles
        self.bundles=0;self.last_bundle_step=-10000;self.window=[];self.window_step=None
        self.reported_start=False;self.active=False;self.fixed=None;self.pre=None;self.last_metrics={};self.flow_data={};self.activations={}
        self.cfm=model.models['cfm'];self.cfm._diagnostic_sink=self.capture_flow
        self.handles=[]
        for name,mod in model.named_modules():
            if name=='models.length_regulator' or name.endswith(('.wavenet','.conv2')) or ('.transformer.layers.' in name and name.rsplit('.',1)[-1].isdigit()):
                self.handles.append(mod.register_forward_hook(self.hook(name)))

    def hook(self,name):
        def observe(module,args,output):
            if not self.active:return
            x=output[0] if isinstance(output,(list,tuple)) else output
            if torch.is_tensor(x):self.activations[name]=tensor_summary(x)
        return observe

    def emit(self,kind,**data):
        self.log.write(json.dumps(dict(kind=kind,rank=self.rank,step=self.step,epoch=self.epoch,microbatch=self.micro,**data),allow_nan=True,default=str)+'\n')
        if kind in ('anomaly','pause') or self.step%self.interval==0:self.log.flush()

    def flags(self,bad,spike=False):
        flags=torch.tensor([int(bad),int(spike)],device=self.acc.device,dtype=torch.int32)
        if self.acc.num_processes>1:dist.all_reduce(flags,op=dist.ReduceOp.MAX)
        return tuple(bool(v) for v in flags.cpu().tolist())

    def begin(self,batch,step,epoch,micro):
        self.step,self.epoch,self.micro=int(step),int(epoch),int(micro)
        if self.window_step!=step:self.window=[];self.window_step=step
        self.active=True;self.activations={};self.flow_data={}
        rng={'torch':torch.get_rng_state(),'python':random.getstate()}
        if self.acc.device.type=='cuda':rng['cuda']=torch.cuda.get_rng_state(self.acc.device)
        self.current=dict(batch=batch,rng=rng,step=step,epoch=epoch,microbatch=micro)
        self.window.append(self.current)
        if self.fixed is None and batch is not None:
            self.fixed=cpu_tree({k:v[:1] for k,v in batch.items() if torch.is_tensor(v) and v.ndim>0})
            torch.save(self.fixed,self.root/f'fixed-probe-rank{self.rank}.pt')
        missing,_=self.flags(batch is None)
        if missing:self.stop('missing_batch')

    def capture_flow(self,**tensors):
        if not self.active:return
        self.flow_data={k:v.detach() for k,v in tensors.items()}
        self.current['flow_random']={k:self.flow_data[k] for k in ('time','noise')}
        self.current['class_dropout']=bool(getattr(self.cfm.estimator,'_diagnostic_class_dropout',False))

    def capture(self,stage,force=False):
        # Global trigger and global cap: all ranks take exactly the same path.
        if not force and (self.bundles>=self.max_bundles or self.step-self.last_bundle_step<200):return
        dest=self.root/f'event-step{self.step}-micro{self.micro}-{stage}';dest.mkdir(exist_ok=True)
        torch.save(dict(stage=stage,window=cpu_tree(self.window),flow=cpu_tree({k:v for k,v in self.flow_data.items() if k in ('per_sample','time','denominators','valid_samples')}),
                        gradients={n:p.grad.detach().cpu() for n,p in self.model.named_parameters() if p.grad is not None}),dest/f'rank{self.rank}.pt')
        if self.rank==0:
            torch.save({'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),'step_before_update':self.step,
                        'pre_update_model':self.pre if stage=='postupdate_nonfinite' else None,
                        'pre_update_optimizer':getattr(self,'pre_optimizer',None) if stage=='postupdate_nonfinite' else None},dest/'state.pt')
        if self.acc.num_processes>1:dist.barrier()
        self.bundles+=1;self.last_bundle_step=self.step
        self.emit('anomaly',stage=stage,bundle=str(dest))
        if self.rank==0:print(f'[Diagnostics] step={self.step} stage={stage} bundle={dest}',flush=True)

    def stop(self,stage):
        self.capture(stage,force=True);self.emit('pause',stage=stage);self.log.flush()
        raise FloatingPointError(f'Diagnostic synchronized pause at {stage}; evidence in {self.root}')

    def after_forward(self,loss):
        finite=bool(torch.isfinite(loss.detach()));lv=float(loss.detach())
        f=self.flow_data
        summaries={k:tensor_summary(v) for k,v in f.items() if k in ('target','condition','prediction','velocity','noise')}
        bad=(not finite) or any(v['nonfinite'] for v in summaries.values()) or any(v['nonfinite'] for v in self.activations.values())
        per=f.get('per_sample',torch.tensor([lv],device=self.acc.device)).float().cpu().tolist()
        self.last_metrics={'loss_max':max(per),'loss_mean':lv}
        times=f['time'].flatten().cpu().tolist() if 'time' in f else []
        groups={}
        records=compact_records((self.current['batch'] or {}).get('records'))
        lengths=(self.current['batch'] or {}).get('mel_lens')
        lengths=lengths.cpu().tolist() if torch.is_tensor(lengths) else []
        for i,value in enumerate(per):
            row=records[i] if i<len(records) else {}
            row=row.get('target',row) if isinstance(row,dict) else {}
            source=row.get('dataset',row.get('source_dataset','unknown'))
            keys=['source:'+str(source)]
            if i<len(times):keys.append('time_bin:'+str(min(4,int(times[i]*5))))
            if i<len(lengths):keys.append('length_frames_250_bin:'+str(lengths[i]//250))
            for key in keys:groups.setdefault(key,[]).append(value)
        bins={key:dict(count=len(vals),mean=sum(vals)/len(vals),max=max(vals)) for key,vals in groups.items()}
        self.emit('forward',loss=lv,per_sample_loss=per,time=times,bins=bins,
            denominators=f['denominators'].cpu().tolist() if 'denominators' in f else [],
            valid_samples=f['valid_samples'].cpu().tolist() if 'valid_samples' in f else [],
            class_dropout=self.current.get('class_dropout'),summaries=summaries,activations=self.activations,
            records=compact_records((self.current['batch'] or {}).get('records')),
            target_paths=(self.current['batch'] or {}).get('target_audio_paths'),prompt_paths=(self.current['batch'] or {}).get('prompt_audio_paths'))
        bad,spike=self.flags(bad,finite and (lv>1.5 or max(per)>5.))
        if bad:self.stop('forward_nonfinite')
        if spike:self.capture('loss_spike')

    def after_backward(self,sync):
        stats=collection_summary((n,p.grad) for n,p in self.model.named_parameters())
        self.emit('gradient_accumulated',sync=bool(sync),**stats)
        self.last_metrics['grad_norm64']=stats['norm64']
        bad,spike=self.flags(stats['nonfinite']>0,stats['norm64']>10.)
        if bad:self.stop('gradient_nonfinite')
        if spike:self.capture('gradient_spike')
        self.pre_grad=stats

    def after_clip(self,native_norm,clip):
        post=collection_summary((n,p.grad) for n,p in self.model.named_parameters())
        self.emit('clip',native_norm=float(native_norm),pre_norm64=self.pre_grad['norm64'],post_norm64=post['norm64'],
                  expected_coefficient=min(1.,float(clip)/(self.pre_grad['norm64']+1e-6)),post_nonfinite=post['nonfinite'])
        # Preserve baseline clipping. If its scalar norm overflows, stop and retain
        # both the finite FP64 norm and native result rather than changing updates.
        bad,_=self.flags(post['nonfinite']>0 or not math.isfinite(float(native_norm)))
        if bad:self.stop('clip_nonfinite_or_norm_overflow')

    def optimizer_stats(self):
        return collection_summary((f'{name}.{key}',state[key]) for name,p in self.model.named_parameters()
            for state in [self.optimizer.state.get(p,{})] for key in ('exp_avg','exp_avg_sq') if key in state)

    def before_step(self):
        stats=collection_summary(self.model.named_parameters());opt=self.optimizer_stats()
        self.emit('before_update',parameters=stats,optimizer=opt,lr=[g['lr'] for g in self.optimizer.param_groups])
        bad,_=self.flags(stats['nonfinite']>0 or opt['nonfinite']>0)
        if bad:self.stop('preupdate_nonfinite')
        self.pre={n:p.detach().clone() for n,p in self.model.named_parameters()}
        self.pre_optimizer=clone_tree(self.optimizer.state_dict()) if self.rank==0 else None
        self.pre_parameter_norm=stats['norm64']

    def after_step(self):
        stats=collection_summary(self.model.named_parameters());opt=self.optimizer_stats()
        delta=collection_summary((n,p.detach()-self.pre[n]) for n,p in self.model.named_parameters()) if (self.step+1)%self.interval==0 else None
        ratio=(delta['norm64']/max(self.pre_parameter_norm,1e-30)) if delta else None
        self.emit('after_update',parameters=stats,optimizer=opt,update=delta,update_to_weight=ratio)
        bad,_=self.flags(stats['nonfinite']>0 or opt['nonfinite']>0)
        if bad:self.stop('postupdate_nonfinite')
        if ratio is not None:self.last_metrics['update_to_weight']=ratio
        if not self.reported_start and self.rank==0:
            print(f'[DiagnosticUpdate] step={self.step+1} parameters_and_Adam_finite=True grad_norm64={self.pre_grad["norm64"]:.6g}',flush=True)
        self.reported_start=True
        self.pre=None;self.pre_optimizer=None;self.active=False

    def metrics(self):
        # Max across ranks makes isolated spikes visible in the compact curve.
        keys=sorted(self.last_metrics);v=torch.tensor([self.last_metrics[k] for k in keys],device=self.acc.device)
        if self.acc.num_processes>1:dist.all_reduce(v,op=dist.ReduceOp.MAX)
        return {'diagnostic/max_rank_'+k:float(x) for k,x in zip(keys,v.cpu().tolist())}

    @staticmethod
    def trim_probe(batch):
        # A singleton no longer shares the original batch's maximum lengths.
        batch=dict(batch)
        batch['mel']=batch['mel'][..., :int(batch['mel_lens'].max())]
        if 'semantic_lens' in batch:
            batch['semantic']=batch['semantic'][:, :int(batch['semantic_lens'].max())]
        return batch

    @torch.no_grad()
    def probe(self,forward_loss):
        if self.fixed is None:return
        self.active=False;batch=self.trim_probe({k:v.to(self.acc.device) for k,v in self.fixed.items()})
        modes={m:m.training for m in self.model.modules()};py=random.getstate()
        try:
            devices=[self.acc.device.index] if self.acc.device.type=='cuda' else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(99173+self.rank);self.model.eval()
                loss=forward_loss(self.model,batch)
        finally:
            random.setstate(py)
            for m,mode in modes.items():m.training=mode
        values=self.acc.gather(loss.detach().float().reshape(1));mean=float(values.mean())
        self.emit('fixed_probe',loss=float(loss),mean_all_ranks=mean)
        self.acc.log({'diagnostic/fixed_probe_loss':mean},step=self.step+1)
        if self.rank==0:print(f'[DiagnosticProbe] step={self.step+1} loss={mean:.6f}',flush=True)
