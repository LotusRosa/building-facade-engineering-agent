"""Explicit Windows/Ubuntu GPU entry; safe to import without torch."""
import os
import platform
import traceback
import contextlib
from pathlib import Path
from .contracts import digest, sha, write
from .service import Service


def server_guard(execute):
    from .platform_support import supported_host
    if not execute or not supported_host() or os.environ.get('FACADE_DISABLE_MODEL_EXECUTION')=='1':
        raise RuntimeError('Model execution requires explicit --execute on supported 64-bit Windows/Linux; model execution may be disabled by host policy')


def verify_job(spec):
    if digest(spec['roles']) != spec['roles_digest']:
        raise ValueError('Job role snapshot changed')
    root = Path(__file__).parent
    actual = {p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.suffix in ('.py','.json')}
    if actual != spec['source_hashes']:
        raise ValueError('Job source/config changed; create a new round')
    for rows in spec['roles'].values():
        for r in rows:
            if sha(r['path']) != r['sha256']:
                raise ValueError('Photo changed: '+r['image_id'])
    if spec['parent'] and sha(spec['parent']['checkpoint']) != spec['parent']['contract']['checkpoint_sha256']:
        raise ValueError('Active parent weight changed')


def run_one(state,device,execute=False,job_id=None):
    server_guard(execute)
    service = Service(state)
    job = service.claim(device,job_id=job_id)
    if not job:
        return False
    output = Path(job['output'])
    output.mkdir(parents=True,exist_ok=True)
    try:
        verify_job(job['spec'])
        write(output/'job.json',job['spec'])
        # Heavy imports occur only after platform/explicit-execution/data checks.
        with (output/'worker.log').open('a',encoding='utf-8',buffering=1) as log:
            with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                from .gpu import run
                result = run(job,device)
        verify_job(job['spec'])
        write(output/'receipt.json',result)
        result.update(receipt_path=str(output/'receipt.json'),receipt_sha256=sha(output/'receipt.json'))
        service.finish(job['id'],result)
    except BaseException:
        error = traceback.format_exc()
        write(output/'failure.json',{'error':error,'job_digest':digest(job['spec'])})
        service.finish(job['id'],error=error)
        print(error,flush=True)
    return True
