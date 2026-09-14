import pytest
import torch
from semantic2any.utils.dots_audiovae import DotsAudioVAE, tiny_audio_vae_config
from semantic2any.losses.s2vae_decoder_feature import S2VAEDecoderFeatureLoss, decoder_features

@pytest.mark.parametrize('use_checkpoint,micro', [(False,1),(False,3),(True,3)])
def test_crop_all_samples_context_and_parameter_gradient(use_checkpoint,micro):
    torch.set_num_threads(2); torch.manual_seed(42)
    vae=DotsAudioVAE.from_config(tiny_audio_vae_config())
    lm=S2VAEDecoderFeatureLoss(layers=(0,1),weights=(.4,.6),chunk_frames=5,max_samples=1,
        all_batch_random_crop=True,context_frames=2,micro_batch_size=micro,use_checkpoint=use_checkpoint)
    w=torch.randn(8,8,requires_grad=True)
    x=torch.randn(6,8,16); gt=torch.randn_like(x,requires_grad=True)
    prompts=torch.tensor([2,2,2,3,4,2]); lengths=torch.tensor([14,14,14,12,7,12])
    pred=torch.einsum('cd,bdt->bct',w,x)
    torch.manual_seed(7)
    actual=lm(pred,gt,lengths,prompts,torch.linspace(0,1,6),vae)
    gp,gw,ggt=torch.autograd.grad(actual,(pred,w,gt),allow_unused=True,retain_graph=True)
    bounds=list(lm.last_bounds)
    refs=[]
    for i,left,right,start,end in bounds:
        assert prompts[i]<=left<=start<end<=right<=lengths[i]
        assert end-start==min(5,int(lengths[i]-prompts[i]))
        assert start-left<=2 and right-end<=2
        feats=decoder_features(vae,pred[i:i+1,:,left:right],(0,1))
        with torch.no_grad(): features=decoder_features(vae,gt[i:i+1,:,left:right],(0,1))
        loss=0
        for weight,a,b in zip((.4,.6),feats,features):
            scale=a.size(-1)//(right-left)
            lo,hi=(start-left)*scale,(end-left)*scale
            a,b=a[...,lo:hi],b[...,lo:hi]
            loss+=weight*((a-b).abs().mean(-1)/b.abs().mean(-1).clamp_min(1e-4)).mean()
        refs.append(loss)
        assert gp[i,:,:left].abs().sum()==0 and gp[i,:,right:].abs().sum()==0
        assert gp[i,:,start:end].norm()>0
    reference=sum(refs)/len(refs)
    expected_grad=torch.autograd.grad(reference,w)[0]
    assert torch.allclose(actual,reference,rtol=2e-5,atol=2e-6)
    assert torch.allclose(gw,expected_grad,rtol=2e-5,atol=2e-6)
    assert ggt is None
    assert lm.last_indices.tolist()==list(range(6))
    assert float(lm.last_components['samples'])==6
    assert float(lm.last_components['supervised_frames'])==28
    assert float(lm.last_components['target_frames'])==58
    assert float(lm.last_components['frame_coverage'])==pytest.approx(28/58)
    assert float(lm.last_components['sample_coverage'])==1
    assert float(lm.last_components['decoded_frames'])==sum(b-a for _,a,b,_,_ in bounds)
    assert all(not p.requires_grad and p.grad is None for p in vae.parameters())
    with torch.no_grad():
        torch.manual_seed(7)
        evaluated=lm(pred,gt,lengths,prompts,None,vae)
    assert torch.allclose(evaluated,reference,rtol=2e-5,atol=2e-6)
    bad=pred.detach().clone()
    for i,(p,l) in enumerate(zip(prompts,lengths)):
        bad[i,:,:p]=float('nan');bad[i,:,l:]=float('nan')
    bad.requires_grad_();torch.manual_seed(7)
    masked=lm(bad,gt,lengths,prompts,None,vae)
    grad=torch.autograd.grad(masked,bad)[0]
    assert torch.isfinite(masked) and torch.isfinite(grad).all()
    assert torch.allclose(masked,reference,rtol=2e-5,atol=2e-6)


def test_crop_options_invalid():
    with pytest.raises(ValueError):
        S2VAEDecoderFeatureLoss(full_batch_full_length=True,all_batch_random_crop=True)
    with pytest.raises(ValueError):
        S2VAEDecoderFeatureLoss(context_frames=-1)
    with pytest.raises(ValueError):
        S2VAEDecoderFeatureLoss(micro_batch_size=0)
