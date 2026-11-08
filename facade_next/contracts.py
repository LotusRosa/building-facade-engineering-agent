import hashlib
import json
import math
from pathlib import Path

CLASSES = ['hollow', 'spalling', 'crack']
BACKBONES = ['resnet50', 'convnext_tiny', 'swin_t']
METHODS = ['standard', 'asl', 'sam', 'distill', 'hnm', 'hem', 'head_only', 'ewc_replay']
SCHEMA = 'facade_next_v1'
PREPROCESSING = 'full1280_letterbox114_gpu_v1'
ROOT = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def portable_roles_digest(roles):
    """Content/labels/groups identity, independent of Windows/Linux root paths."""
    from .data import union
    normalized={}
    pools={'train':union(roles.get('train',[]),roles.get('history',[])),
           'val':roles.get('val',[]),'evaluation':roles.get('evaluation',[])}
    for role,rows in pools.items():
        normalized[role]=[{k:r.get(k,'') for k in ('image_id','sha256','labels','scene','group')} for r in sorted(rows,key=lambda r:r['image_id'])]
    return digest(normalized)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    import os
    import uuid
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.' + uuid.uuid4().hex + '.partial')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def protocol(kind, overrides=None):
    if kind not in ('initial', 'update'):
        raise ValueError('kind must be initial/update')
    p = dict(id='engineering_full1280_v1', kind=kind, classes=CLASSES,
             image_size=1280, preprocessing=PREPROCESSING,
             training_precision='bf16', evaluation_precision='fp16',
             epochs=40 if kind == 'initial' else 5,
             patience=8, steps_per_epoch=38, effective_batch=16, micro_batch=8,
             target_draws=304, history_draws=304, prediction_batch=2,
             lr=1e-4 if kind == 'initial' else 1e-5, weight_decay=.05, minimum_lr=1e-6,
             checkpoint_selection='best_val_macro_AP' if kind == 'initial' else 'last_epoch',
             threshold_grid={'start':.05, 'stop':.95, 'step':.005, 'tie':'lower'},
             automatic_deployment=False, automatic_oom_retry=False)
    allowed = {'epochs','patience','steps_per_epoch','micro_batch','prediction_batch','lr','weight_decay','minimum_lr'}
    changes = overrides or {}
    if set(changes) - allowed:
        raise ValueError('Unsupported protocol override: ' + ','.join(sorted(set(changes)-allowed)))
    p.update(changes)
    for k in ('epochs','patience','steps_per_epoch','micro_batch','prediction_batch'):
        if type(p[k]) is not int or not 1 <= p[k] <= 10000:
            raise ValueError('Invalid integer ' + k)
    if 16 % p['micro_batch']:
        raise ValueError('micro_batch must divide 16')
    for k in ('lr','weight_decay','minimum_lr'):
        if type(p[k]) not in (float,int) or not math.isfinite(p[k]) or p[k] < 0:
            raise ValueError('Invalid finite number ' + k)
    if p['lr'] <= 0 or p['minimum_lr'] > p['lr']:
        raise ValueError('Require 0 <= minimum_lr <= lr and lr>0')
    p['target_draws'] = p['history_draws'] = p['steps_per_epoch'] * 8
    p['modified_preset'] = bool(changes)
    return p


def methods():
    return read(ROOT/'presets.json')['methods']


def check_contract(c):
    if c['schema'] != SCHEMA or c['backbone'] not in BACKBONES or c['classes'] != CLASSES:
        raise ValueError('Checkpoint schema/backbone/class order mismatch')
    if c['preprocessing'] != PREPROCESSING or c['image_size'] != 1280:
        raise ValueError('Checkpoint preprocessing mismatch; old crop checkpoints need a separate protocol')
    for key in ('thresholds','pos_weight'):
        vals = c[key]
        if not isinstance(vals, list) or len(vals) != 3 or any(type(v) not in (int,float) or not math.isfinite(v) or v <= 0 for v in vals):
            raise ValueError('Invalid checkpoint ' + key)
    if any(t >= 1 for t in c['thresholds']) or c.get('threshold_status') != 'frozen_on_val':
        raise ValueError('Missing valid Val-frozen thresholds')
    if not isinstance(c.get('checkpoint_sha256'), str) or len(c['checkpoint_sha256']) != 64:
        raise ValueError('Missing checkpoint SHA256')
    return c
