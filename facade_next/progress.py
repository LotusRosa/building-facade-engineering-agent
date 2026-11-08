"""Read existing worker artifacts for progress; never imports or runs a model."""
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import re
from .contracts import digest, read, PREPROCESSING


def small_json(path, default):
    try:
        if path.stat().st_size > 1024*1024:
            return default
        return read(path)
    except (OSError,ValueError):
        return default


def file_names(path, cache):
    key=str(path)
    if key not in cache:
        try:
            with os.scandir(path) as entries:
                cache[key]={e.name for e in entries if e.is_file()}
        except OSError:
            cache[key]=set()
    return cache[key]


def training_ended(history,spec,maximum):
    if not history:
        return False
    if maximum and len(history)>=maximum:
        return True
    if spec.get('kind')!='initial':
        return False
    best,stale=-1.,0
    for epoch in history:
        score=epoch.get('val_macro_AP')
        if isinstance(score,(int,float)) and math.isfinite(score) and score>best:
            best,stale=score,0
        else:
            stale+=1
    return stale>=int(spec.get('protocol',{}).get('patience',8))


def job_progress(job,directory_cache=None):
    directory_cache=directory_cache if directory_cache is not None else {}
    status=job['status']
    result=dict(stage=status, fraction=None, elapsed_seconds=None)
    try:
        started=datetime.fromisoformat(job['started'])
        ended=datetime.fromisoformat(job.get('finished') or job.get('ended')) if job.get('finished') or job.get('ended') else datetime.now(timezone.utc)
        result['elapsed_seconds']=max(0,int((ended-started).total_seconds()))
    except (ValueError,TypeError,KeyError):
        pass
    if status in ('queued','failed','cancelled'):
        return result
    out=Path(job['output']);spec=job.get('spec') or {};protocol=spec.get('protocol') or {}
    max_epochs=int(protocol.get('epochs',0))
    history=(job.get('result') or {}).get('history') or small_json(out/'history.json',[])
    if not isinstance(history,list):history=[]
    if not history and status=='succeeded':history=(job.get('result') or {}).get('history') or []
    result.update(completed_epochs=len(history),max_epochs=max_epochs)
    if status=='succeeded':
        result.update(stage='skipped' if (job.get('result') or {}).get('status')=='skip_empty_pool' else 'completed',fraction=1)
        return result
    names=file_names(out,directory_cache)
    if 'evaluation_predictions.json' in names or 'comparison.json' in names:
        result['stage']='finalizing'
        return result
    if 'validation_predictions.json' in names:
        result['stage']='evaluation'
        return result
    if training_ended(history,spec,max_epochs):
        result['stage']='calibrating'
        return result
    works=[(int(m[1]),int(m[2])) for name in names if (m:=re.fullmatch(r'work_(\d+)_(\d+)\.json',name))]
    schedules=[int(m[1]) for name in names if (m:=re.fullmatch(r'schedule_(\d+)\.json',name))]
    roles=spec.get('roles') or {}
    if works or schedules:
        epoch=max([x[0] for x in works]+schedules)
        steps=math.ceil(len(roles.get('train',[]))/max(1,int(protocol.get('effective_batch',16)))) if spec.get('kind')=='initial' else int(protocol.get('steps_per_epoch',0))
        planned=max_epochs*steps
        done=len(works)
        result.update(stage='training',epoch=epoch,step=sum(1 for e,_ in works if e==epoch),steps_per_epoch=steps,
                      completed_steps=done,planned_steps=planned,fraction=min(1,done/planned) if planned else None)
        if result['step']>=steps and steps and len(history)<epoch:
            result.update(stage='validation',fraction=None)
        return result
    rows={r.get('image_id',r['sha256']):r for group in roles.values() for r in group}
    keys={digest(dict(sha256=r['sha256'],preprocessing=PREPROCESSING)) for r in rows.values()}
    cache=out.parent.parent/'image_cache'
    cached=file_names(cache,directory_cache)
    done=sum(key+'.npy' in cached and key+'.json' in cached for key in keys)
    result.update(stage='preparing' if done<len(keys) else 'initializing',cached_images=done,total_images=len(keys),
                  fraction=done/len(keys) if keys and done<len(keys) else None)
    return result


def decorate_jobs(snapshot):
    directory_cache={}
    for job in snapshot.get('jobs',[]):
        try:
            job['progress']=job_progress(job,directory_cache)
        except (OSError,ValueError,KeyError,TypeError):
            job['progress']=dict(stage=job.get('status','unknown'),fraction=None,elapsed_seconds=None)
    return snapshot
