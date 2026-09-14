import unittest
import torch
from semantic2any.utils.dots_audiovae import DotsAudioVAE,tiny_audio_vae_config
from semantic2any.losses.s2vae_decoder_feature import S2VAEDecoderFeatureLoss,decoder_features

class DecoderFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        cls.vae=DotsAudioVAE.from_config(tiny_audio_vae_config())

    def loss(self,checkpoint=True):
        return S2VAEDecoderFeatureLoss(layers=(0,1),weights=(.4,.6),chunk_frames=4,max_samples=4,use_checkpoint=checkpoint)

    def test_decoder_equivalence(self):
        x=torch.randn(2,8,8)
        with torch.no_grad():
            last=decoder_features(self.vae,x,(0,1))[-1]
            d=self.vae.vocoder.decoder
            y=d.conv_post(d.activation_post(last)).clamp(-1,1)
            self.assertTrue(torch.equal(y,self.vae.decode(x)))

    def test_prompt_and_padding_excluded_even_at_zero_t(self):
        gt=torch.randn(4,8,12);pred=gt.clone();pred[:,:,:4]+=100;pred[:,:,8:]-=100
        pred.requires_grad_();loss=self.loss()(pred,gt,torch.full((4,),8),torch.full((4,),4),torch.zeros(4),self.vae)
        self.assertLess(loss.item(),1e-6)
        g=torch.autograd.grad(loss,pred)[0]
        self.assertTrue(torch.equal(g[:,:,:4],torch.zeros_like(g[:,:,:4])))
        self.assertTrue(torch.equal(g[:,:,8:],torch.zeros_like(g[:,:,8:])))

    def test_checkpoint_gradients_and_frozen_target(self):
        x=torch.randn(4,8,12,requires_grad=True);gt=torch.randn_like(x,requires_grad=True)
        grads=[];values=[]
        for enabled in (False,True):
            torch.manual_seed(3)
            loss=self.loss(enabled)(x,gt,torch.full((4,),12),torch.full((4,),4),torch.tensor([0.,.1,.8,1.]),self.vae)
            gx,ggt=torch.autograd.grad(loss,(x,gt),allow_unused=True)
            self.assertIsNone(ggt);self.assertTrue(torch.isfinite(gx).all());self.assertGreater(gx.norm().item(),0)
            grads.append(gx);values.append(loss.item())
        self.assertEqual(values[0],values[1]);self.assertTrue(torch.equal(grads[0],grads[1]))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in self.vae.parameters()))

    def test_silent_target_is_finite(self):
        gt=torch.zeros(4,8,8);x=torch.randn_like(gt,requires_grad=True)
        loss=self.loss()(x,gt,torch.full((4,),8),torch.zeros(4,dtype=torch.long),torch.zeros(4),self.vae)
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(torch.autograd.grad(loss,x)[0]).all())

if __name__=='__main__':unittest.main()
