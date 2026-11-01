from semantic2any.data.block_sampler import BlockShuffleSampler

def test_coverage_and_replay():
 s=BlockShuffleSampler(range(53),8,1234);a=list(s)
 assert sorted(a)==list(range(53)) and a!=list(range(53))
 assert list(s)==a
 assert list(s)[17:]==a[17:]
 s.set_epoch(1);b=list(s)
 assert sorted(b)==list(range(53)) and b!=a
 s.set_epoch(0);assert list(s)==a

def test_tiny_and_empty():
 assert list(BlockShuffleSampler([],8))==[]
 assert list(BlockShuffleSampler([1],8))==[0]
