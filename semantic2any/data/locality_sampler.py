"""Bounded, deterministic storage-local reordering with an immutable resume anchor.

This changes future batch composition, not target coverage or batch lengths.
No feature I/O is performed here. Only fixed-size base batches are supported.
"""
from collections import defaultdict
from itertools import islice
import hashlib
import json
from pathlib import Path

VERSION = 'locality-window-v1'

class LocalityBatchSampler:
    def __init__(self, base, dataset, window_batches=1024):
        if window_batches < 1:
            raise ValueError('window_batches must be positive')
        if getattr(base, 'world_size', 1) != 1 or not base.drop_last:
            raise ValueError('requires rank-local, drop_last base sampler')
        self.base = base
        self.dataset = dataset
        self.window_batches = int(window_batches)
        self.epoch = 0
        self.anchor_epoch = 0
        self.anchor_batch = 0
        self.configured = False
        self.reorder_stop = len(base)

    def __len__(self):
        return len(self.base)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self.base.set_epoch(epoch)

    def configure(self, anchor_epoch, anchor_batch, reorder_stop=None):
        if anchor_epoch < 0 or anchor_batch < 0 or anchor_batch > len(self.base):
            raise ValueError('invalid locality anchor')
        self.anchor_epoch = int(anchor_epoch)
        self.anchor_batch = int(anchor_batch)
        self.reorder_stop = len(self.base) if reorder_stop is None else int(reorder_stop)
        if not 0 <= self.reorder_stop <= len(self.base):
            raise ValueError("invalid synchronized epoch boundary")
        self.configured = True

    def storage_key(self, index):
        ds = self.dataset
        record = ds.target_records[ds.target_indices[index]]
        # Both codes and latent shards derive from semantic_code_path.
        # Primary sort favors the much larger target latent reads. Prompt pairing
        # is unchanged and may still require a different shard.
        path = record.get('semantic_code_path')
        if not isinstance(path, str) or not path:
            raise ValueError('locality requires stored semantic_code_path')
        return path, int(record.get('vae_latent_offset', record.get('semantic_code_offset', 0))), index

    def reorder_window(self, batches):
        groups = defaultdict(list)
        slots = defaultdict(list)
        for position, batch in enumerate(batches):
            if len(batch) != self.base.batch_size:
                raise ValueError('locality requires full base batches')
            bucket = self.base._bucket_id(self.base.lengths[batch[0]])
            if any(self.base._bucket_id(self.base.lengths[i]) != bucket for i in batch):
                raise ValueError('base batch mixes length buckets')
            slots[bucket].append(position)
            groups[bucket].extend(batch)
        result = [None] * len(batches)
        for bucket, indices in groups.items():
            indices.sort(key=self.storage_key)
            size = self.base.batch_size
            for n, position in enumerate(slots[bucket]):
                result[position] = indices[n * size:(n + 1) * size]
        return result

    def __iter__(self):
        if not self.configured:
            raise RuntimeError('configure locality resume contract before iteration')
        source = iter(self.base)
        # Regenerate the original consumed prefix exactly. Accelerate skips it
        # without reading samples. On later resumes this anchor MUST NOT move.
        prefix = len(self.base) if self.epoch < self.anchor_epoch else (
            self.anchor_batch if self.epoch == self.anchor_epoch else 0)
        yield from islice(source, prefix)
        remaining = max(0, self.reorder_stop - prefix)
        while remaining:
            window = list(islice(source, min(self.window_batches, remaining)))
            if not window:
                break
            remaining -= len(window)
            yield from self.reorder_window(window)
        # Existing training ends when the shortest rank exhausts. Never move
        # examples across that boundary into or out of the consumed population.
        yield from source


def recipe_signature(config, world_size):
    data = dict(config['data'])
    data.pop('locality_window_batches', None)
    source = Path(data['train_jsonl']).resolve()
    stat = source.stat()
    recipe = dict(data=data, seed=config['seed'], world_size=world_size,
                  batch_size=config['train']['batch_size'],
                  grad_accumulation=config['train']['grad_accumulation'],
                  manifest=dict(path=str(source), size=stat.st_size, mtime_ns=stat.st_mtime_ns))
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()


def resume_contract(config, world_size, resume_path, epoch, epoch_step):
    signature = recipe_signature(config, world_size)
    window = int(config['data']['locality_window_batches'])
    saved = None
    if resume_path:
        checkpoint = Path(resume_path)
        if not checkpoint.is_dir():
            raise ValueError('locality resume requires full Accelerator checkpoint directory')
        state = json.loads((checkpoint / 'trainer_state.json').read_text())
        if state['epoch'] != epoch or state['epoch_step'] != epoch_step:
            raise ValueError('resume cursor overrides are unsafe with locality')
        sidecar = checkpoint / 'locality_state.json'
        if sidecar.exists():
            saved = json.loads(sidecar.read_text())
        else:
            old = checkpoint.parent / 'config.resolved.yaml'
            from omegaconf import OmegaConf
            old_config = OmegaConf.to_container(OmegaConf.load(old), resolve=True)
            if int(old_config['data'].get('locality_window_batches', 0)):
                raise ValueError('locality checkpoint missing its immutable anchor')
            if recipe_signature(old_config, world_size) != signature:
                raise ValueError('baseline data recipe changed; refusing index-only resume')
            # Each RNG file records one rank in the full checkpoint.
            if len(list(checkpoint.glob('random_states_*.pkl'))) != world_size:
                raise ValueError('checkpoint world size differs from requested world size')
    if saved is not None:
        if (saved['version'] != VERSION or saved['signature'] != signature
                or saved['window_batches'] != window):
            raise ValueError('locality contract mismatch; refusing silent reshuffle')
        if (epoch, epoch_step * int(config['train']['grad_accumulation'])) < (
                saved['anchor_epoch'], saved['anchor_batch']):
            raise ValueError('resume cursor precedes locality anchor')
        return saved
    return dict(version=VERSION, signature=signature, window_batches=window,
                anchor_epoch=int(epoch),
                anchor_batch=int(epoch_step) * int(config['train']['grad_accumulation']))
