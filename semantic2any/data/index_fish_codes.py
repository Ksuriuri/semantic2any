"""Strict paired same-segment Index/Fish manifests; no independent time crops."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from semantic2any.data.fish_codes import read_codes

FISH_HZ = 44100 / 2048

def fish_frames_for_duration(seconds):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Duration must be positive and finite')
    return math.ceil(seconds * FISH_HZ)

def read_index(record, base):
    path = record['codes']
    if not path.startswith(('/', 'gs://')):
        path = str(base).rstrip('/') + '/' + path
    count, offset = int(record['num_frames']), int(record.get('offset_bytes', 0))
    if count <= 0 or offset < 0 or offset % 2 or record.get('dtype', 'uint16') != 'uint16':
        raise ValueError('Invalid Index uint16 range')
    if path.startswith('gs://'):
        from google.cloud import storage
        bucket, key = path[5:].split('/', 1)
        generation = record.get('generation')
        generation = int(generation) if generation is not None else None
        raw = storage.Client().bucket(bucket).blob(key, generation=generation).download_as_bytes(start=offset, end=offset+2*count-1, if_generation_match=generation)
    else:
        with open(path, 'rb') as f:
            f.seek(offset)
            raw = f.read(2 * count)
    if len(raw) != 2 * count:
        raise ValueError('Truncated Index codes')
    return torch.from_numpy(np.frombuffer(raw, dtype='<u2').astype(np.int64))

class IndexFishDataset(Dataset):
    def __init__(self, manifest, max_seconds=60, max_reference_frames=646):
        self.path = Path(manifest).resolve()
        self.max_seconds, self.max_reference_frames = max_seconds, max_reference_frames
        index_root = Path(str(self.path) + '.idx')
        meta_path = index_root / 'meta.json'
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            stat = self.path.stat()
            expected = dict(path=str(self.path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
            if meta.get('version') != 1 or meta.get('source') != expected:
                raise ValueError('Stale manifest index; rebuild before training')
            offsets_path = index_root / 'offsets.u64'
            if meta.get('dtype') != '<u8' or offsets_path.stat().st_size != int(meta['rows']) * 8:
                raise ValueError('Invalid manifest offset index')
            self.offsets = np.memmap(offsets_path, dtype='<u8', mode='r')
        else:
            if self.path.stat().st_size > 1024**3:
                raise ValueError('Large manifest requires a prebuilt shared .idx index')
            self.offsets = []
            with self.path.open('rb') as f:
                while True:
                    offset, line = f.tell(), f.readline()
                    if not line:
                        break
                    if line.strip():
                        self.offsets.append(offset)
        if len(self.offsets) == 0:
            raise ValueError('Empty manifest')
    def __len__(self):
        return len(self.offsets)
    def __getitem__(self, i):
        with self.path.open('rb') as f:
            f.seek(self.offsets[i])
            row = json.loads(f.readline())
        target, reference, index = row['target'], row['reference'], row['index']
        # IDs identify the exact audio segment, not merely its speaker/filename.
        if not target.get('id') or target['id'] != index.get('id'):
            raise ValueError('Index and Fish must identify the same audio segment')
        if not reference.get('id') or reference['id'] == target['id']:
            raise ValueError('Reference must be a separate identified segment')
        if (target.get('codes', target.get('fish_code_path')), target.get('offset_bytes', 0)) == (reference.get('codes', reference.get('fish_code_path')), reference.get('offset_bytes', 0)):
            raise ValueError('Target/reference share code range')
        duration = float(row['duration_seconds'])
        if not math.isfinite(duration) or not 0 < duration <= self.max_seconds:
            raise ValueError('Invalid duration or exceeds max_seconds; prepare matched segments upstream')
        if float(index.get('start_seconds', 0)) != float(target.get('start_seconds', 0)):
            raise ValueError('Index/Fish segment start differs')
        # Independently check sample-clock durations, never normalize each stream
        # to [0,1] (that would silently stretch mismatched segments).
        if abs(int(index['num_frames']) / 25 - duration) > .081:
            raise ValueError('Index duration mismatch')
        if abs(int(target['num_frames']) / FISH_HZ - duration) > 2 / FISH_HZ:
            raise ValueError('Fish duration mismatch')
        base = row.get('codes_root', str(self.path.parent))
        return {'index': read_index(index, base), 'target': read_codes(target, base),
                'reference': read_codes(reference, base, self.max_reference_frames),
                'duration': duration}

def collate_index_fish(items):
    batch = {}
    for name in ('index', 'target', 'reference'):
        batch[name + '_lengths'] = torch.tensor([len(x[name]) for x in items])
        batch[name + '_codes'] = pad_sequence([x[name] for x in items], batch_first=True)
    batch['duration_seconds'] = torch.tensor([x['duration'] for x in items])
    return batch

@torch.no_grad()
def decode_index_batch(decoder, codes, lengths):
    # Decode each unpadded sequence separately: convolution near the right edge
    # is not invariant to zero-padding, even when outputs are masked afterwards.
    sequences = [codes[i, :int(n)] for i, n in enumerate(lengths)]
    decoded = decoder.decode_sequences(sequences)
    features, ids = [], []
    if decoder.frames_per_code != 2:
        raise ValueError('Expected 25 Hz codes decoded to 50 Hz features')
    for seq, feature in zip(sequences, decoded):
        # RepCodec returns [T,C]. Make this contract explicit, not inferred.
        if feature.shape != (len(seq) * 2, decoder.semantic_dim):
            raise ValueError(f'Unexpected decoded feature shape {feature.shape}')
        features.append(feature.float())
        ids.append(seq.repeat_interleave(2))
    return (pad_sequence(features, batch_first=True), pad_sequence(ids, batch_first=True), lengths * 2)


def prepare_index_batch(mode, decoder, codes, lengths):
    if mode == 'raw':
        return torch.zeros(*codes.shape, 1, device=codes.device), codes, lengths
    return decode_index_batch(decoder, codes, lengths)
