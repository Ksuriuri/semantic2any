"""Process-local, byte-bounded cache of immutable feature-file blocks."""
import os
import time
from collections import OrderedDict
import numpy as np

class FeatureBlockCache:
    def __init__(self, max_bytes=0, block_bytes=262144):
        self.max_bytes = int(max_bytes)
        self.block_bytes = int(block_bytes)
        if self.max_bytes < 0 or self.block_bytes <= 0:
            raise ValueError('invalid cache limits')
        self.pid = os.getpid()
        self.blocks = OrderedDict()
        self.used = self.hits = self.misses = self.read_bytes = 0
        self.calls = 0
        self.last_report = time.monotonic()

    def read(self, path, mmap, start, end, dtype):
        if not 0 <= start <= end <= mmap.size:
            raise ValueError('feature span out of bounds')
        if self.pid != os.getpid():
            self.blocks.clear(); self.used = 0; self.pid = os.getpid()
        if not self.max_bytes or not end-start:
            return np.array(mmap[start:end], dtype=dtype, copy=True)
        width = max(1, self.block_bytes // mmap.dtype.itemsize)
        out = np.empty(end-start, dtype=dtype)
        for base in range(start//width*width, end, width):
            stop = min(base+width, mmap.size)
            key = (str(path), mmap.dtype.str, mmap.size, base)
            block = self.blocks.pop(key, None)
            if block is None:
                self.misses += 1
                block = np.array(mmap[base:stop], copy=True)
                self.read_bytes += block.nbytes
                if block.nbytes <= self.max_bytes:
                    while self.used + block.nbytes > self.max_bytes:
                        _, old = self.blocks.popitem(last=False); self.used -= old.nbytes
                    self.blocks[key] = block; self.used += block.nbytes
            else:
                self.hits += 1; self.blocks[key] = block
            lo, hi = max(start,base), min(end,stop)
            out[lo-start:hi-start] = block[lo-base:hi-base]
        self.calls += 1
        if self.calls % 1000 == 0 and time.monotonic() - self.last_report >= 60:
            total = self.hits + self.misses
            print(f"[FeatureCache] pid={self.pid} used_MiB={self.used/2**20:.1f} "
                  f"limit_MiB={self.max_bytes/2**20:.1f} hits={self.hits} misses={self.misses} "
                  f"hit_rate={self.hits/max(1,total):.4f} read_MiB={self.read_bytes/2**20:.1f}", flush=True)
            self.last_report = time.monotonic()
        return out

_CACHE = FeatureBlockCache(int(os.environ.get('S2VAE_FEATURE_CACHE_BYTES', '0')),
                           int(os.environ.get('S2VAE_FEATURE_CACHE_BLOCK_BYTES', '262144')))
def read_feature_span(path, mmap, start, end, dtype):
    return _CACHE.read(path, mmap, start, end, dtype)

def disable_worker_cache(worker_id):
    global _CACHE
    _CACHE = FeatureBlockCache(0)
