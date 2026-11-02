import math
import pytest
import torch
from semantic2any.models.index_fish_mtp import build_index_fish_model, RawCodeStream
from semantic2any.models.index_fish_prefix import IndexFishPrefixMTP, index_overlap_slots, validate_clock_coverage

torch.set_num_threads(1)
def example():
    torch.manual_seed(12)
    m=IndexFishPrefixMTP(dim=16,heads=2,condition_layers=1,shared_layers=1,index_vocab=16,codebook_sizes=[8]*10)
    args=[torch.zeros(2,8,1),torch.randint(16,(2,8)),torch.tensor([8,6]),torch.randint(8,(2,3,10)),torch.tensor([3,2]),torch.randint(8,(2,7,10)),torch.tensor([7,6])]
    return m,args

@pytest.mark.parametrize('count',[1,2,3,25,1500,15000])
def test_integer_clock_complete_coverage(count):
    t=math.ceil(count*1764/2048)
    idx,valid,meta=index_overlap_slots(t,torch.tensor([count]))
    assert torch.unique(idx[valid[0]]).tolist()==list(range(count))
    # Exactly partition each non-tail Fish interval by Index overlaps.
    full=torch.arange(t)*2048+2048<=count*1764
    torch.testing.assert_close((meta[...,1]*valid[0]).sum(-1)[full],torch.ones(int(full.sum())))
    assert valid.sum(-1).max()<=3
    for i in range(count):
        assert ((idx==i)&valid[0]).sum() in (1,2)
    validate_clock_coverage(torch.tensor([count]),torch.tensor([t]))

def test_tail_reject_and_lookahead_contract():
    with pytest.raises(ValueError,match='outside Fish'):
        validate_clock_coverage(torch.tensor([10]),torch.tensor([4]))
    with pytest.raises(ValueError,match='40 ms'):
        IndexFishPrefixMTP(lookahead_seconds=0)

def test_all_heads_no_current_or_future_target_leakage():
    m,a=example();m.eval();out=m(*a)
    for k in range(10):
        changed=[x.clone() for x in a];changed[5][:,2:,k:]=(changed[5][:,2:,k:]+1)%8
        actual=m(*changed)
        torch.testing.assert_close(out[k][:,:3],actual[k][:,:3],atol=2e-6,rtol=2e-5)

def test_padding_prefix_length_and_index_future_invariance():
    m,a=example();m.eval();out=m(*a)
    c=[x.clone() for x in a];c[1][1,6:]=999;c[3][1,2:]=999;c[5][1,6:]=999
    for x,y in zip(out,m(*c)):torch.testing.assert_close(x[1,:6],y[1,:6],atol=2e-6,rtol=2e-5)
    c=[x.clone() for x in a];c[3]=torch.cat((c[3],torch.full((2,2,10),999)),1)
    for x,y in zip(out,m(*c)):torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-5)
    c=[x.clone() for x in a];c[1][:,3:]=(c[1][:,3:]+1)%16
    for x,y in zip(out,m(*c)):torch.testing.assert_close(x[:,0],y[:,0],atol=2e-6,rtol=2e-5)

def test_reference_order_is_observable():
    m,a=example();m.eval();out=m(*a);c=[x.clone() for x in a];c[3][0]=c[3][0,torch.tensor([2,0,1])]
    assert max((x[0]-y[0]).abs().max().item() for x,y in zip(out,m(*c)))>1e-5

def test_teacher_forced_prefix_and_streaming_match():
    m,a=example();m.eval();a=[x[:1] for x in a];out=m(*a)
    for t in range(7):
        c=a[:5]+[a[5][:,:t+1],a[6].clamp_max(t+1)]
        for x,y in zip(out,m(*c)):torch.testing.assert_close(x[:,t],y[:,-1],atol=2e-6,rtol=2e-5)
    offline=m.generate(*a[:5],frames=7)
    stream=RawCodeStream(m,a[3],a[4]);first=stream.push(a[1][0,:4]);second=stream.push(a[1][0,4:],final=True,duration_seconds=.32)
    torch.testing.assert_close(torch.cat((first,second),1),offline)

def test_trainable_all_parameters_and_overfit():
    m,a=example();opt=torch.optim.Adam(m.parameters(),lr=.01);first=m.loss(m(*a),a[5],a[6])[0].item()
    for _ in range(60):
        opt.zero_grad();loss,_=m.loss(m(*a),a[5],a[6]);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        opt.step()
    assert loss.item()<first*.4
    m.eval();pred=m.generate(*[x[:1] for x in a[:5]],frames=7)
    assert (pred==a[5][:1]).float().mean()>.7

def test_architecture_checkpoint_roundtrip():
    m,a=example();cfg=dict(dim=16,heads=2,condition_layers=1,shared_layers=1,index_vocab=16,codebook_sizes=[8]*10)
    new=build_index_fish_model(cfg,'prefix_aligned');new.load_state_dict(m.state_dict(),strict=True)
    assert not any('attention' in n or 'cross_norm' in n for n,_ in new.named_parameters())
    with pytest.raises(ValueError):build_index_fish_model(cfg,'typo')
    assert build_index_fish_model(cfg).__class__.__name__=='IndexFishMTP'
