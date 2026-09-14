from types import SimpleNamespace
import torch
from trainers.train_s2mel_zipformer import _vae_hf_metrics_for_log

class TwoRankAccelerator:
    device=torch.device('cpu')
    def gather(self,packed):
        return torch.cat((packed,packed+torch.tensor([[2.,4.,4.,0.]])),dim=0)

def module():
    return SimpleNamespace(component_keys=('total','stage2','samples'),last_flow=torch.tensor(2.),last_components={'total':torch.tensor(3.),'stage2':torch.tensor(10.),'samples':torch.tensor(2.)})

def test_weighted_feature_uses_cross_rank_mean():
    metrics=_vae_hf_metrics_for_log(TwoRankAccelerator(),module(),prefix='s2vae_feature',weight=.05)
    assert metrics['train/flow']==3.
    assert metrics['train/s2vae_feature/total']==5.
    assert metrics['train/s2vae_feature/stage2']==12.
    assert metrics['train/s2vae_feature/weight']==.05
    assert metrics['train/s2vae_feature/weighted']==.25

def test_existing_hf_logging_keeps_its_keys():
    metrics=_vae_hf_metrics_for_log(TwoRankAccelerator(),module())
    assert metrics=={'train/flow':3.,'train/s2vae_hf/total':5.,'train/s2vae_hf/stage2':12.,'train/s2vae_hf/samples':2.}

def test_disabled_loss_has_no_collective():
    class NoCollective:
        def gather(self,*args):raise AssertionError('disabled loss must not gather')
    assert _vae_hf_metrics_for_log(NoCollective(),None,prefix='s2vae_feature',weight=.05)=={}
