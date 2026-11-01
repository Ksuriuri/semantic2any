"""Deterministic full-coverage shuffle without a dataset-sized permutation."""
import numpy as np
from torch.utils.data import Sampler

class BlockShuffleSampler(Sampler):
    def __init__(self, data_source, block_size=4096, seed=1234):
        self.size=len(data_source);self.block_size=int(block_size);self.seed=int(seed);self.epoch=0
        if self.block_size<1 or self.seed<0:raise ValueError('Invalid block shuffle settings')
    def __len__(self):return self.size
    def set_epoch(self, epoch):self.epoch=int(epoch)
    def __iter__(self):
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch]))
        blocks=(self.size+self.block_size-1)//self.block_size
        for block in rng.permutation(blocks):
            start=int(block)*self.block_size
            for row in rng.permutation(min(self.block_size,self.size-start)):
                yield start+int(row)
