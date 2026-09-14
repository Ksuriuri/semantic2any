import torch
from semantic2any.utils.dots_audiovae import DotsAudioVAE,tiny_audio_vae_config
from semantic2any.losses.s2vae_decoder_feature import S2VAEDecoderFeatureLoss,decoder_features

def test_full_targets_all_samples_gradients_and_metrics():
 torch.set_num_threads(2);torch.manual_seed(42)
 v=DotsAudioVAE.from_config(tiny_audio_vae_config())
 lm=S2VAEDecoderFeatureLoss(layers=(0,1),weights=(.4,.6),max_samples=1,chunk_frames=1,full_batch_full_length=True)
 w=torch.randn(8,8,requires_grad=True);x=torch.randn(3,8,12);gt=torch.randn_like(x,requires_grad=True)
 lengths=torch.tensor([12,10,8]);prompts=torch.tensor([4,3,2]);times=torch.tensor([0.,.5,1.])
 pred=torch.einsum('cd,bdt->bct',w,x)
 actual=lm(pred,gt,lengths,prompts,times,v)
 gp,gw,ggt=torch.autograd.grad(actual,(pred,w,gt),allow_unused=True,retain_graph=True)
 assert ggt is None
 for i,(p,l) in enumerate(zip(prompts,lengths)):
  assert gp[i,:,:p].abs().sum()==0 and gp[i,:,l:].abs().sum()==0
  assert gp[i,:,p:l].norm()>0
 assert lm.last_indices.tolist()==[0,1,2]
 assert float(lm.last_components['samples'])==3
 assert float(lm.last_components['supervised_frames'])==21
 assert float(lm.last_components['frames'])==7
 assert float(lm.last_components['frame_coverage'])==1
 assert float(lm.last_components['sample_coverage'])==1
 refs=[]
 for i,(p,l) in enumerate(zip(prompts,lengths)):
  a=decoder_features(v,pred[i:i+1,:,p:l],(0,1))
  with torch.no_grad():b=decoder_features(v,gt[i:i+1,:,p:l],(0,1))
  refs.append(sum(weight*((aa-bb).abs().mean(-1)/bb.abs().mean(-1).clamp_min(1e-4)).mean() for weight,aa,bb in zip((.4,.6),a,b)))
 reference=sum(refs)/3
 gr=torch.autograd.grad(reference,w)[0]
 assert torch.allclose(actual,reference,rtol=1e-5,atol=1e-6)
 assert torch.allclose(gw,gr,rtol=1e-5,atol=1e-6)
 with torch.no_grad():evaluated=lm(pred,gt,lengths,prompts,times,v)
 assert torch.allclose(evaluated,reference,rtol=1e-5,atol=1e-6)
 assert all(not p.requires_grad and p.grad is None for p in v.parameters())
 from trainers.train_s2mel_zipformer import _vae_hf_metrics_for_log
 class IdentityGather:
  device=torch.device('cpu')
  def gather(self,packed):return packed
 lm.last_flow=torch.tensor(.5)
 logged=_vae_hf_metrics_for_log(IdentityGather(),lm,prefix='s2vae_feature',weight=.5)
 assert logged['train/s2vae_feature/samples']==3
 assert logged['train/s2vae_feature/frame_coverage']==1
 assert logged['train/s2vae_feature/sample_coverage']==1
 assert logged['train/s2vae_feature/supervised_frames']==21
 assert logged['train/s2vae_feature/target_frames']==21
 bad=pred.detach().clone()
 for i,(p,l) in enumerate(zip(prompts,lengths)):
  bad[i,:,:p]=float('nan');bad[i,:,l:]=float('nan')
 bad.requires_grad_()
 masked_loss=lm(bad,gt,lengths,prompts,times,v)
 masked_grad=torch.autograd.grad(masked_loss,bad)[0]
 assert torch.isfinite(masked_loss) and torch.isfinite(masked_grad).all()
 assert torch.allclose(masked_loss,reference,rtol=1e-5,atol=1e-6)
