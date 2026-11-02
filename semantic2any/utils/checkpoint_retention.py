"""Retain complete rolling checkpoints plus periodic archives."""
import copy
import json
import re
import shutil
from pathlib import Path

POLICY_KEYS = ('save_every', 'keep_last_checkpoints', 'keep_every_steps')


def check_resume_config(saved, current):
    """Only checkpoint scheduling/retention may change on exact-state resume."""
    a, b = copy.deepcopy(saved), copy.deepcopy(current)
    changed = {}
    for key in POLICY_KEYS:
        old, new = a.get('train', {}).pop(key, None), b.get('train', {}).pop(key, None)
        if old != new:
            changed[key] = {'old': old, 'new': new}
    if a != b:
        raise ValueError('Resume config differs outside checkpoint save/retention policy')
    return changed


def complete_step(path):
    path = Path(path)
    match = re.fullmatch(r'checkpoint-(\d+)', path.name)
    if not match or path.is_symlink() or not path.is_dir():
        return None
    try:
        info = json.loads((path / 'trainer.json').read_text())
        step, world = int(match[1]), int(info['world_size'])
        if info['step'] != step or world <= 0:
            return None
        names = ['model.safetensors', 'optimizer.bin', 'inference.pt', 'trainer.json']
        names += [f'random_states_{i}.pkl' for i in range(world)]
        if not all((path / n).is_file() and not (path / n).is_symlink()
                   and (path / n).stat().st_size > 0 for n in names):
            return None
        return step
    except (OSError, ValueError, KeyError, TypeError):
        return None


def retention_plan(root, keep_last, keep_every):
    if keep_last < 1 or keep_every < 1:
        raise ValueError('Retention requires positive keep_last and keep_every')
    root = Path(root)
    complete = sorted(s for p in root.glob('checkpoint-*') if (s := complete_step(p)) is not None)
    keep = set(complete[-keep_last:]) | {s for s in complete if s % keep_every == 0}
    return {'keep': sorted(keep), 'delete': [s for s in complete if s not in keep]}


def prune_checkpoints(root, keep_last, keep_every):
    """Call only after save's distributed barrier, or with the writer stopped."""
    root = Path(root)
    plan = retention_plan(root, keep_last, keep_every)
    deleted = []
    for step in plan['delete']:
        path = root / f'checkpoint-{step}'
        if complete_step(path) != step:
            raise RuntimeError(f'Checkpoint changed during retention: {path}')
        shutil.rmtree(path)
        deleted.append(step)
    if deleted:
        with (root / 'retention-audit.jsonl').open('a') as f:
            f.write(json.dumps({'keep_last': keep_last, 'keep_every': keep_every,
                                'kept': plan['keep'], 'deleted': deleted}) + '\n')
    return deleted
