import json
from pathlib import Path
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from semantic2any.data.fish_codes import read_codes,FishPairedDataset,collate_fish_pairs
from semantic2any.models.fish_s2tok import FishS2Tok
from semantic2any.utils.fish_tokenizer import FishCodeFeatures

def config():
    c=OmegaConf.load(Path(__file__).parents[1]/'configs/fish_s2tok_dit.yaml').model
    c.DiT.hidden_dim=64;c.DiT.num_heads=4;c.DiT.depth=2;c.DiT.block_size=64;c.DiT.content_dim=32;c.DiT.class_dropout_prob=0
    c.wavenet.hidden_dim=64;c.wavenet.num_layers=2;c.wavenet.p_dropout=0
    return c

def test_binary_ranges_and_pairing(tmp_path):
    a=np.arange(60,dtype=np.uint16).reshape(6,10);(tmp_path/'a.bin').write_bytes(a.tobytes())
    row={'codes':'a.bin','num_frames':3,'offset_bytes':20,'num_codebooks':10}
    assert np.array_equal(read_codes(row,tmp_path).numpy(),a[1:4])
    with pytest.raises(ValueError):read_codes(dict(row,offset_bytes=1),tmp_path)
    with pytest.raises(ValueError):read_codes(dict(row,num_frames=100),tmp_path)
    pair={'target':dict(row,id='t'),'reference':dict(row,id='r',offset_bytes=0,num_frames=2)}
    (tmp_path/'pairs.jsonl').write_text(json.dumps(pair)+'\n');ds=FishPairedDataset(tmp_path/'pairs.jsonl');b=collate_fish_pairs([ds[0]])
    assert b['target_codes'].shape==(1,3,10) and b['reference_lengths'].tolist()==[2]

def test_frozen_sum():
    tables=[torch.randn(16,1024) for _ in range(10)];f=FishCodeFeatures(tables);codes=torch.randint(0,16,(2,5,10));sem,s=f(codes)
    assert torch.equal(sem,tables[0][codes[:,:,0]])
    expected=tables[1][codes[:,:,1]].clone()
    for i in range(2,10):expected+=tables[i][codes[:,:,i]]
    expected=tables[0][codes[:,:,0]]+expected
    torch.testing.assert_close(s,expected,rtol=0,atol=0)
    assert not any(p.requires_grad for p in f.parameters())
    with pytest.raises(ValueError):f(codes.fill_(16))

def test_prompt_padding_mask_and_gradient():
    torch.manual_seed(7);m=FishS2Tok(config());m.cfm.setup_estimator_caches(4,16)
    sem=torch.randn(2,5,1024);target=torch.randn_like(sem);ref=torch.randn(2,3,1024);tl=torch.tensor([5,3]);rl=torch.tensor([2,3])
    x,mu,lens=m.pack(sem,target,ref,tl,rl)
    assert lens.tolist()==[7,6]
    torch.testing.assert_close(x[0,:,:2].T,ref[0,:2]);assert (mu[0,:2]==0).all()
    torch.manual_seed(9);loss=m(sem,target,ref,tl,rl)
    assert torch.isfinite(loss);loss.backward();assert m.condition.weight.grad is not None
    target[1,3:]=9999;ref[0,2:]=9999;sem[1,3:]=9999
    torch.manual_seed(9);loss2=m(sem,target,ref,tl,rl)
    torch.testing.assert_close(loss,loss2)
    m.eval();out=m.generate(sem[:1],ref[:1,:2],steps=2,cfg=0)
    assert out.shape==(1,5,1024) and torch.isfinite(out).all()
