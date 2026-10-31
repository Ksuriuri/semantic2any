import json, math
import numpy as np
import pytest
import torch
from semantic2any.models.index_fish_mtp import IndexFishMTP
from semantic2any.data.index_fish_codes import IndexFishDataset, decode_index_batch, fish_frames_for_duration

torch.set_num_threads(1)
def example():
    torch.manual_seed(5)
    m=IndexFishMTP(dim=16,heads=2,condition_layers=1,shared_layers=1,index_dim=8,index_vocab=16,codebook_sizes=[8]*10)
    data=[torch.randn(2,8,8),torch.randint(16,(2,8)),torch.tensor([8,6]),torch.randint(8,(2,3,10)),torch.tensor([3,2]),torch.randint(8,(2,4,10)),torch.tensor([4,3])]
    return m,data

def test_no_current_or_future_label_leakage():
    m,a=example(); m.eval(); expected=m(*a)
    for k in range(10):
        changed=[x.clone() for x in a]
        changed[5][:,2:,k:]=(changed[5][:,2:,k:]+1)%8
        actual=m(*changed)
        torch.testing.assert_close(expected[k][:,:3],actual[k][:,:3])

def test_padding_invariance_and_full_source_access():
    m,a=example(); m.eval(); expected=m(*a)
    changed=[x.clone() for x in a]
    changed[0][1,6:]=float('nan');changed[1][1,6:]=999
    changed[3][1,2:]=999;changed[5][1,3:]=999
    actual=m(*changed)
    for x,y in zip(expected,actual):torch.testing.assert_close(x[1,:3],y[1,:3])
    a[0].requires_grad_();out=m(*a);out[0][0,0].sum().backward()
    assert torch.all(a[0].grad[0].abs().sum(-1)>0)

def test_loss_gradients_and_overfit():
    m,a=example(); opt=torch.optim.Adam(m.parameters(),lr=.01)
    first=m.loss(m(*a),a[5],a[6])[0].item()
    for _ in range(60):
        opt.zero_grad();loss,books=m.loss(m(*a),a[5],a[6]);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        opt.step()
    assert loss.item()<first*.35
    m.eval();result=m.generate(*a[:5],frames=4)
    assert result.shape==(2,4,10) and result.min()>=0 and result.max()<8
    assert (result[0]==a[5][0]).float().mean()>.8

def test_batched_teacher_forcing_matches_prefix():
    m,a=example();m.eval();expected=m(*a)
    for t in range(4):
        actual=m(*(a[:5]+[a[5][:,:t+1],a[6].clamp_max(t+1)]))
        for x,y in zip(expected,actual):torch.testing.assert_close(x[0,t],y[0,-1],atol=2e-6,rtol=2e-5)

def test_duration_uses_fixed_clock():
    assert fish_frames_for_duration(60)==math.ceil(60*44100/2048)
    assert fish_frames_for_duration(1)==22
    with pytest.raises(ValueError):fish_frames_for_duration(float('nan'))

def test_individual_decode_without_padding():
    class Decoder:
        frames_per_code=2;semantic_dim=3
        def decode_sequences(self,seq):
            assert [len(x) for x in seq]==[4,2]
            return [x.repeat_interleave(2).float()[:,None].expand(-1,3) for x in seq]
    f,ids,lengths=decode_index_batch(Decoder(),torch.tensor([[1,2,3,4],[5,6,999,999]]),torch.tensor([4,2]))
    assert f.shape==(2,8,3) and lengths.tolist()==[8,4]
    assert ids[1,:4].tolist()==[5,5,6,6]

def test_manifest_identity_duration_and_ranges(tmp_path):
    np.zeros((10,10),dtype='<u2').tofile(tmp_path/'fish.bin')
    np.arange(5,dtype='<u2').tofile(tmp_path/'index.bin')
    row={'duration_seconds':.2,'target':{'id':'a','codes':'fish.bin','num_frames':5},'reference':{'id':'b','codes':'fish.bin','num_frames':5,'offset_bytes':100},'index':{'id':'a','codes':'index.bin','num_frames':5}}
    p=tmp_path/'data.jsonl'
    def load():
        p.write_text(json.dumps(row)+'\n');return IndexFishDataset(p)[0]
    assert len(load()['index'])==5
    row['index']['id']='other'
    with pytest.raises(ValueError,match='same audio'):load()
    row['index']['id']='a';row['duration_seconds']=1
    with pytest.raises(ValueError,match='duration mismatch'):load()

def test_streaming_future_invariance_and_chunk_equivalence():
    from semantic2any.models.index_fish_mtp import RawCodeStream
    torch.manual_seed(7)
    m=IndexFishMTP(dim=16,heads=2,condition_layers=1,shared_layers=1,index_vocab=16,source_mode='raw',lookahead_seconds=.08,codebook_sizes=[8]*10).eval()
    ids=torch.randint(16,(1,10)); source=torch.zeros(1,10,1)
    ref=torch.randint(8,(1,3,10));rl=torch.tensor([3]); sl=torch.tensor([10])
    target=torch.randint(8,(1,8,10));tl=torch.tensor([8])
    a=m(source,ids,sl,ref,rl,target,tl)
    altered=ids.clone();altered[:,3:]=(altered[:,3:]+1)%16
    b=m(source,altered,sl,ref,rl,target,tl)
    # First Fish deadline=.04644+.08, only first three Index tokens available.
    for x,y in zip(a,b):torch.testing.assert_close(x[:,0],y[:,0],atol=1e-6,rtol=1e-5)
    frames=fish_frames_for_duration(.4)
    offline=m.generate(source,ids,sl,ref,rl,frames)
    stream=RawCodeStream(m,ref,rl)
    first=stream.push(ids[0,:5]);assert first.shape[1]>0
    second=stream.push(ids[0,5:],final=True,duration_seconds=.4)
    torch.testing.assert_close(torch.cat((first,second),1),offline)
    with pytest.raises(ValueError):stream.push(ids[0,:1])
    with pytest.raises(ValueError):IndexFishMTP(source_mode='decoded',lookahead_seconds=.08)
