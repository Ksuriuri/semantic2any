"""Explicit paired manifests referencing local/GCS uint16 [T,10] code ranges."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


def read_codes(record, base=None, max_frames=None):
    if record.get('num_codebooks',10) != 10 or record.get('dtype','uint16') != 'uint16' or record.get('layout','T,K') != 'T,K':
        raise ValueError('Expected uint16 T,K ten-codebook format')
    frames = int(record['num_frames']); offset = int(record.get('offset_bytes',0))
    if frames <= 0 or offset < 0 or offset % 20:
        raise ValueError('Invalid code range')
    # Deterministic prefix crops; target semantic and target S always share this crop.
    frames = min(frames, int(max_frames)) if max_frames else frames
    uri = record.get('codes', record.get('fish_code_path'))
    if not uri:
        raise ValueError('Missing codes/fish_code_path')
    if base and not uri.startswith(('gs://','/')):
        uri = str(base).rstrip('/')+'/'+uri
    size = frames*10*2
    if uri.startswith('gs://'):
        from google.cloud import storage
        bucket,key = uri[5:].split('/',1)
        generation = record.get('generation')
        if generation is not None:
            generation = int(generation)
        raw = storage.Client().bucket(bucket).blob(key, generation=generation).download_as_bytes(
            start=offset,end=offset+size-1,if_generation_match=generation)
    else:
        with open(uri,'rb') as f:
            f.seek(offset);raw=f.read(size)
    if len(raw)!=size:raise ValueError('Truncated code range')
    codes=np.frombuffer(raw,dtype='<u2').reshape(frames,10).astype(np.int64)
    if (codes[:,0]>=4096).any() or (codes[:,1:]>=1024).any():raise ValueError('Code out of range')
    return torch.from_numpy(codes)

class FishPairedDataset(Dataset):
    """JSONL: {target:{...}, reference:{...}}; references must be selected explicitly.

    No inference from filename/speaker identity: pairing/filtering is a data-preparation decision.
    Index stores byte offsets only, not all feature rows.
    """
    def __init__(self, manifest, max_target_frames=512, max_reference_frames=172):
        self.path=Path(manifest).resolve();self.target_max=max_target_frames;self.reference_max=max_reference_frames
        if min(self.target_max,self.reference_max)<=0:raise ValueError('Crop sizes must be positive')
        self.offsets=[]
        with self.path.open('rb') as f:
            while True:
                offset=f.tell();line=f.readline()
                if not line:break
                if line.strip():self.offsets.append(offset)
        if not self.offsets:raise ValueError('Empty paired manifest')
    def __len__(self):return len(self.offsets)
    def __getitem__(self,i):
        with self.path.open('rb') as f:f.seek(self.offsets[i]);row=json.loads(f.readline())
        t,r=row['target'],row['reference']
        if t==r:raise ValueError('Reference must be a different record from target')
        if t.get('id') and t.get('id')==r.get('id'):raise ValueError('Reference and target IDs coincide')
        if (t.get('codes',t.get('fish_code_path')),int(t.get('offset_bytes',0))) == (r.get('codes',r.get('fish_code_path')),int(r.get('offset_bytes',0))):
            raise ValueError('Reference and target point at the same code range')
        base=row.get('codes_root',str(self.path.parent))
        return read_codes(t,base,self.target_max),read_codes(r,base,self.reference_max)

def collate_fish_pairs(items):
    tl=torch.tensor([len(t) for t,r in items]);rl=torch.tensor([len(r) for t,r in items])
    target=torch.zeros(len(items),int(tl.max()),10,dtype=torch.long)
    ref=torch.zeros(len(items),int(rl.max()),10,dtype=torch.long)
    for i,(t,r) in enumerate(items):target[i,:len(t)]=t;ref[i,:len(r)]=r
    return {'target_codes':target,'reference_codes':ref,'target_lengths':tl,'reference_lengths':rl}
