"""Server-only adapter. Import exclusively after worker.server_guard.

Original all-known losses/update transaction/Fisher/model/pixels are source-pinned.
The application orchestrator and partial-label adapter require GPU verification.
"""
import copy
import math
import os
import random
import time
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image, ImageOps
from .contracts import SCHEMA, PREPROCESSING, CLASSES, check_contract, digest, portable_roles_digest, sha, write, read
from .data import union
from .metrics import calibrate, scene_metrics, evaluate
from .planning import batches, mining, initial_pos_weight
from .vendor.legacy_engine import FullImageClassifier, _pixels
from .vendor.toolkit_methods import supervised_loss, configure_trainable, enforce_train_mode, historical_kl, run_update_step
from .vendor.ewc import model_fisher, EWCConstraint, capture_rng, restore_rng


def atomic_checkpoint(path,payload):
    tmp = path.with_suffix('.partial')
    torch.save(payload,tmp)
    os.replace(tmp,path)


def hardware(device):
    dev = torch.device(device)
    if dev.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU/precision fallback')
    torch.cuda.set_device(dev)
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError('Native BF16 required')
    return dev


def seed_all(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def cached_initialization(model,backbone):
    from torchvision.models import ConvNeXt_Tiny_Weights, ResNet50_Weights, Swin_T_Weights
    weights = {'convnext_tiny':ConvNeXt_Tiny_Weights.DEFAULT,'resnet50':ResNet50_Weights.DEFAULT,'swin_t':Swin_T_Weights.DEFAULT}[backbone]
    filename = Path(urlparse(weights.url).path).name
    path = Path(torch.hub.get_dir())/'checkpoints'/filename
    if not path.is_file():
        raise FileNotFoundError('No automatic download. Supply existing ImageNet cache: '+str(path))
    prefix = filename.rsplit('-',1)[1].split('.')[0]
    if not sha(path).startswith(prefix):
        raise ValueError('ImageNet cache checksum mismatch')
    state = torch.load(path,map_location='cpu',weights_only=True)
    head = {'resnet50':'fc.','convnext_tiny':'classifier.2.','swin_t':'head.'}[backbone]
    model.encoder.load_state_dict({k:v for k,v in state.items() if not k.startswith(head)},strict=True)
    return {'kind':'cached_imagenet','path':str(path),'sha256':sha(path)}


class Photos:
    def __init__(self,rows,cache):
        self.rows = rows
        self.cache = Path(cache)
        self.cache.mkdir(parents=True,exist_ok=True)

    def one(self,row):
        key = digest({'sha256':row['sha256'],'preprocessing':PREPROCESSING})
        p = self.cache/(key+'.npy')
        meta = self.cache/(key+'.json')
        if not p.exists() or not meta.exists():
            with Image.open(row['path']) as opened:
                image = ImageOps.exif_transpose(opened).convert('RGB')
                scale = min(1280/image.width,1280/image.height)
                w,h = max(1,min(1280,round(image.width*scale))),max(1,min(1280,round(image.height*scale)))
                image = image.resize((w,h),Image.Resampling.BILINEAR)
                canvas = Image.new('RGB',(1280,1280),(114,114,114))
                canvas.paste(image,((1280-w)//2,(1280-h)//2))
                a = np.asarray(canvas,dtype=np.uint8).transpose(2,0,1).copy()
            tmp = p.with_name(p.name+'.'+__import__('uuid').uuid4().hex+'.partial')
            with tmp.open('wb') as f:
                np.save(f,a,allow_pickle=False)
            os.replace(tmp,p)
            write(meta,{'sha256':sha(p),'source_sha256':row['sha256']})
        a = np.load(p,allow_pickle=False)
        if a.shape != (3,1280,1280) or a.dtype != np.uint8:
            raise ValueError('Invalid image cache')
        return torch.from_numpy(a.copy())

    def prepare(self):
        for r in self.rows:
            self.one(r)
            key = digest({'sha256':r['sha256'],'preprocessing':PREPROCESSING})
            p = self.cache/(key+'.npy')
            if sha(p) != read(p.with_suffix('.json'))['sha256']:
                raise ValueError('Image cache content changed')

    def batch(self,rows,device,training):
        pixels = torch.stack([self.one(r) for r in rows])
        images = _pixels(pixels,device,{'channels_last':True},training)
        labels = torch.tensor([[0 if x is None else x for x in r['labels']] for r in rows],device=device,dtype=torch.float32)
        mask = torch.tensor([[x is not None for x in r['labels']] for r in rows],device=device,dtype=torch.bool)
        return images,labels,mask


def masked_loss(logits,labels,mask,method,pw):
    if bool(mask.all()):
        return supervised_loss(logits,labels,method,pw)
    # Per-image mean over known labels, then mean over images. Training rows
    # with no known labels are rejected before GPU work. No unknown becomes 0.
    if bool((mask.sum(1)==0).any()):
        raise ValueError('Training image has zero known labels')
    values = logits.float()
    if method['loss']['name']=='bce':
        loss = F.binary_cross_entropy_with_logits(values,labels,pos_weight=pw,reduction='none')
    else:
        cfg = method['loss']
        lp,ln = F.logsigmoid(values),F.logsigmoid(-values)
        if cfg['clip']:
            ln = torch.logaddexp(ln,torch.full_like(values,math.log(cfg['clip']))).clamp(max=0)
        probability = labels*lp.exp()+(1-labels)*ln.exp()
        gamma = labels*cfg['gamma_pos']+(1-labels)*cfg['gamma_neg']
        base = torch.where(gamma==0,torch.ones_like(values),(1-probability).clamp(0,1))
        focal = base.pow(gamma)
        if cfg['detach_focal_weight']:
            focal = focal.detach()
        loss = -(focal*(labels*lp+(1-labels)*ln))
    result = ((loss*mask).sum(1)/mask.sum(1)).mean()
    if not bool(torch.isfinite(result)):
        raise FloatingPointError('Nonfinite masked loss')
    return result


@torch.no_grad()
def predict(model,rows,photos,device,batch_size):
    model.eval()
    result = []
    for i in range(0,len(rows),batch_size):
        images,_,_ = photos.batch(rows[i:i+batch_size],device,False)
        with torch.autocast('cuda',dtype=torch.float16):
            logits,_ = model(images)
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError('Nonfinite evaluation logits')
        result.extend(logits.float().sigmoid().cpu().tolist())
    return result


def load_version(version,device):
    contract = check_contract(version['contract'])
    if sha(version['checkpoint']) != contract['checkpoint_sha256']:
        raise ValueError('Weight digest mismatch')
    payload = torch.load(version['checkpoint'],map_location='cpu',weights_only=True)
    if payload['schema'] != SCHEMA:
        raise ValueError('Unsupported weight schema; use explicit import command')
    metadata = {k:v for k,v in contract.items() if k != 'checkpoint_sha256'}
    if payload['metadata'] != metadata:
        raise ValueError('Embedded checkpoint metadata differs from the registered contract')
    model = FullImageClassifier(contract['backbone'],pretrained=False)
    model.load_state_dict(payload['model'],strict=True)
    return model.to(device=device,dtype=torch.float32,memory_format=torch.channels_last)


def convert_checkpoint(source,output,roles):
    """Explicit CPU weight inspection on Linux server only; never called by UI.

    Legacy experiment files contain Python/NumPy RNG objects. weights_only=False
    is limited to the user's explicitly selected trusted experiment checkpoint.
    """
    payload = torch.load(source,map_location='cpu',weights_only=False)
    schema = payload.get('schema')
    metadata = payload.get('metadata',{})
    if schema == SCHEMA:
        c = dict(metadata,checkpoint_sha256=sha(source))
        check_contract(c)
        if metadata.get('portable_roles_digest') != portable_roles_digest(roles):
            raise ValueError('Native checkpoint content/label/role provenance differs or lacks portable proof; retain its original runtime state')
    elif schema == 'aic_capture_v2_full_image_v1':
        if (metadata.get('classes') != CLASSES or metadata.get('image_size')!=1280 or
            metadata.get('preprocessing')!=PREPROCESSING or metadata.get('threshold_status')!='frozen_on_val'):
            raise ValueError('Legacy class/preprocessing/calibration mismatch')
        val_ids = [r['image_id'] for r in roles['val']]
        val_sha = __import__('hashlib').sha256('\n'.join(val_ids).encode()).hexdigest()
        if metadata.get('val_ids_sha256') != val_sha:
            raise ValueError('Val IDs/order differs from original checkpoint calibration provenance')
        binding = metadata.get('role_bindings')
        if binding:
            expected = set(binding['history_ids']) | set(binding['diagnosis_ids'])
            if {r['image_id'] for r in roles['train']} != expected:
                raise ValueError('Historical training manifest differs from original role bindings')
        else:
            ids = [r['image_id'] for r in roles['train']]
            source_ids = metadata.get('pos_weight_source',{}).get('ids_sha256')
            if not source_ids or source_ids != __import__('hashlib').sha256('\n'.join(ids).encode()).hexdigest():
                raise ValueError('Core historical training IDs/order differ from original checkpoint')
        c = dict(schema=SCHEMA,backbone=metadata['backbone'],classes=CLASSES,image_size=1280,
                 preprocessing=PREPROCESSING,threshold_status='frozen_on_val',thresholds=metadata['thresholds'],
                 pos_weight=metadata['pos_weight'],original_schema=schema,original_checkpoint_sha256=sha(source),
                 original_metadata=metadata,import_roles_digest=digest(roles),
                 import_provenance='engineer_verified_original_manifests_and_protected_evaluation')
    else:
        raise ValueError('Unsupported checkpoint. Old 768 crop format is not compatible with full1280')
    c.pop('checkpoint_sha256',None)
    c['import_roles_digest'] = digest(roles)
    c['portable_roles_digest'] = portable_roles_digest(roles)
    model = FullImageClassifier(c['backbone'],pretrained=False)
    model.load_state_dict(payload['model'],strict=True)
    atomic_checkpoint(output/'model.pt',{'schema':SCHEMA,'model':model.state_dict(),'metadata':c})
    c['checkpoint_sha256'] = sha(output/'model.pt')
    check_contract(c)
    write(output/'model-contract.json',c)
    write(output/'role-bundle.json',roles)
    return c


def run(job,device):
    spec,p = job['spec'],job['spec']['protocol']
    if not spec['parent'] and spec.get('initialization')!='cached_imagenet':
        raise ValueError('Initial modeling requires ImageNet pretrained weights; resubmit this old task')
    device = hardware(device)
    seed_all(spec['seed'])
    out = Path(job['output'])
    rows = union(*spec['roles'].values())
    photos = Photos(rows,out.parent.parent/'image_cache')
    photos.prepare()
    train,hist,val,ev = [spec['roles'][k] for k in ('train','history','val','evaluation')]
    if any(all(y is None for y in r['labels']) for r in train+hist):
        raise ValueError('Each training/replay photo needs at least one known label')
    parent = spec['parent']
    method = copy.deepcopy(spec['method'])
    method['optimizer'].update(lr=p['lr'],weight_decay=p['weight_decay'],minimum_lr=p['minimum_lr'])
    parent_eval = parent_probs = None
    initialization = None
    started = time.perf_counter()
    if parent:
        model = load_version(parent,device)
        pw = parent['contract']['pos_weight']
        parent_probs = predict(model,ev,photos,device,p['prediction_batch'])
        parent_eval = scene_metrics(ev,parent_probs,parent['contract']['thresholds'])
    else:
        model = FullImageClassifier(spec['backbone'],pretrained=False)
        initialization = cached_initialization(model,spec['backbone'])
        model = model.to(device=device,dtype=torch.float32,memory_format=torch.channels_last)
        pw = initial_pos_weight(train)
    selected,weights = list(range(len(train))),None
    if method['id'] in ('hnm','hem'):
        target_probs = predict(model,train,photos,device,p['prediction_batch'])
        selected,weights = mining(train,target_probs,parent['contract']['thresholds'],pw,method['id'])
        write(out/'mining.json',{'ids':[train[i]['image_id'] for i in selected], 'weights':weights,'source_role':'train','parent_sha256':parent['contract']['checkpoint_sha256']})
        if not selected:
            return dict(status='skip_empty_pool',job_digest=digest(spec),evaluation_digest=digest(ev),reason='No known-label false positives in new training pool',comparison={'parent':parent_eval,'candidate':None},artifacts={'job.json':sha(out/'job.json'),'mining.json':sha(out/'mining.json')})
    teacher = copy.deepcopy(model).requires_grad_(False).eval() if method['teacher']['weight'] else None
    configure_trainable(model,method['trainable']['scope'])
    constraint = None
    fisher_seconds = 0.
    if method['id']=='ewc_replay':
        t = time.perf_counter()
        diagonal = model_fisher(model,(photos.batch([r],device,False)[0] for r in hist),len(hist),device,bf16=True)
        torch.cuda.synchronize(device)
        fisher_seconds = time.perf_counter()-t
        atomic_checkpoint(out/'fisher.pt',{'diagonal':{n:v.cpu() for n,v in diagonal.items()},'parent_sha256':parent['contract']['checkpoint_sha256'],'history_digest':digest(hist)})
        constraint = EWCConstraint(model,diagonal,method['ewc']['lambda'],{})
    pos_weight = torch.tensor(pw,device=device,dtype=torch.float32)
    optimizer = torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=p['lr'],weight_decay=p['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=p['epochs'],eta_min=p['minimum_lr'])
    all_train = train+hist
    hset = set(range(len(train),len(all_train)))
    history,work = [],[]
    best_score,best_epoch,stale = -1.,0,0
    torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()
    for epoch in range(p['epochs']):
        if spec['kind']=='initial':
            order = list(range(len(train)))
            random.Random(spec['seed']*1000003+epoch*1009+17).shuffle(order)
            schedule = [order[i:i+16] for i in range(0,len(order),16)]
        else:
            schedule = batches(selected,sorted(hset),spec['seed'],epoch,p['steps_per_epoch'],weights)
        write(out/f'schedule_{epoch+1:02d}.json',{'indices':schedule,'image_ids':[[all_train[i]['image_id'] for i in b] for b in schedule]})
        enforce_train_mode(model,method['trainable']['scope'])
        for step,batch in enumerate(schedule):
            # Decode before update snapshots. Augmentation runs inside callback so
            # SAM repeats original RNG and retains original running-buffer policy.
            records = [all_train[i] for i in batch]
            n = len(batch)
            hc = sum(i in hset for i in batch)
            physical = {'micro_calls':0,'student_draws':0,'teacher_draws':0}
            def backward():
                stats = dict(loss_sum=0.,draws=n,supervised_loss_sum=0.,distillation_loss_sum=0.,teacher_draws=0)
                for left in range(0,n,p['micro_batch']):
                    sub = records[left:left+p['micro_batch']]
                    images,labels,known = photos.batch(sub,device,True)
                    hmask = torch.tensor([i in hset for i in batch[left:left+len(sub)]],device=device,dtype=torch.bool)
                    physical['student_draws'] += len(sub)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits,_ = model(images)
                        tl = None
                        if teacher is not None and bool(hmask.any()):
                            with torch.no_grad():
                                tl,_ = teacher(images[hmask])
                            physical['teacher_draws'] += int(hmask.sum())
                    sup = masked_loss(logits,labels,known,method,pos_weight)*len(sub)
                    kl = logits.float()[:0].sum()
                    if tl is not None:
                        nh = int(hmask.sum())
                        kl = historical_kl(logits[hmask].float(),tl.float(),torch.ones(nh,device=device,dtype=torch.bool))*nh*method['teacher']['weight']*(n/hc)
                        stats['teacher_draws'] += nh
                    ((sup+kl)/n).backward()
                    physical['micro_calls'] += 1
                    stats['supervised_loss_sum'] += float(sup.detach())
                    stats['distillation_loss_sum'] += float(kl.detach())
                stats['loss_sum'] = stats['supervised_loss_sum']+stats['distillation_loss_sum']
                if constraint:
                    penalty = constraint.penalty(model)
                    penalty.backward()
                    stats['loss_sum'] += float(penalty.detach())*n
                return stats
            t = time.perf_counter()
            try:
                record = run_update_step(model,optimizer,backward,capture_rng,restore_rng,sam_rho=method['optimizer'].get('rho'))
            except BaseException as e:
                write(out/f'failed_epoch_{epoch+1}_step_{step+1}.json',{'physical_attempts':physical,'error':str(e),'finite_step':getattr(e,'update_record',{})})
                raise
            torch.cuda.synchronize(device)
            record.update(epoch=epoch+1,step=step+1,seconds=time.perf_counter()-t,physical_attempts=physical)
            write(out/f'work_{epoch+1:02d}_{step+1:04d}.json',record)
            work.append(record)
        scheduler.step()
        vprobs = predict(model,val,photos,device,p['prediction_batch'])
        score = evaluate(val,vprobs,[.5]*3)['macro_AP']
        history.append({'epoch':epoch+1,'val_macro_AP':score,'steps':len(schedule)})
        write(out/'history.json',history)
        if spec['kind']=='initial':
            if score is not None and score > best_score:
                best_score,best_epoch,stale = score,epoch+1,0
                atomic_checkpoint(out/'best-state.pt',model.state_dict())
            else:
                stale += 1
            if stale >= p['patience']:
                break
        print({'epoch':epoch+1,'val_macro_AP':score,'method':method['id']},flush=True)
    torch.cuda.synchronize(device)
    train_seconds = time.perf_counter()-training_started
    if spec['kind']=='initial':
        model.load_state_dict(torch.load(out/'best-state.pt',map_location=device,weights_only=True),strict=True)
    else:
        best_epoch = p['epochs']
    vprobs = predict(model,val,photos,device,p['prediction_batch'])
    thresholds,status = calibrate(val,vprobs,parent['contract']['thresholds'] if parent else None)
    write(out/'validation_predictions.json',{'rows':val,'probs':vprobs,'thresholds':thresholds,'threshold_status':status})
    contract = dict(schema=SCHEMA,backbone=spec['backbone'],classes=CLASSES,image_size=1280,preprocessing=PREPROCESSING,
                    thresholds=thresholds,threshold_status='frozen_on_val',threshold_calibration=status,
                    pos_weight=pw,pos_weight_policy='inherited_parent' if parent else 'clip_1_20_negative_over_positive_known_train',
                    job_digest=digest(spec),roles_digest=spec['roles_digest'],portable_roles_digest=portable_roles_digest(spec['roles']),val_digest=digest(val),seed=spec['seed'],
                    parent_sha256=parent['contract']['checkpoint_sha256'] if parent else None,
                    method=method,protocol=p,initialization=initialization,selected_epoch=best_epoch,
                    source_hashes=spec['source_hashes'],training_precision='bf16',evaluation_precision='fp16',
                    masked_loss_policy='per_image_known_label_mean_then_image_mean_v1',distillation_policy='history_all_three_independent_outputs')
    atomic_checkpoint(out/'model.pt',{'schema':SCHEMA,'model':model.state_dict(),'metadata':contract})
    contract['checkpoint_sha256'] = sha(out/'model.pt')
    write(out/'model-contract.json',contract)
    eprobs = predict(model,ev,photos,device,p['prediction_batch'])
    write(out/'evaluation_predictions.json',{'rows':ev,'candidate_probs':eprobs,'parent_probs':parent_probs})
    comparison = {'candidate':scene_metrics(ev,eprobs,thresholds),'parent':parent_eval,
                  'historical_loss_is_advisory':True,'automatic_deployment':False}
    write(out/'comparison.json',comparison)
    artifacts = {p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.suffix in ('.json','.pt')}
    return dict(status='ready',job_digest=digest(spec),evaluation_digest=digest(ev),checkpoint=str(out/'model.pt'),contract=contract,
                comparison=comparison,artifacts=artifacts,history=history,
                cost={'training_and_val_seconds':train_seconds,'fisher_seconds':fisher_seconds,'total_seconds':time.perf_counter()-started,
                      'optimizer_steps':sum(x['optimizer_steps'] for x in work),'logical_draws':sum(x['draws'] for x in work),
                      'physical_draws':sum(x['physical_draws'] for x in work),'teacher_draws':sum(x['teacher_draws'] for x in work),
                      'peak_memory_bytes':int(torch.cuda.max_memory_allocated(device)), 'human_active_seconds':None},
                environment={'python':__import__('platform').python_version(),'torch':torch.__version__,
                             'torchvision':__import__('torchvision').__version__,'gpu':torch.cuda.get_device_name(device)})
