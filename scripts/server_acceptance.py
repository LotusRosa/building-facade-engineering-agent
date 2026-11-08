"""Explicit one-model compatibility task; default is a printed plan, no training."""
import argparse
import json
import sys
import uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from facade_next.service import Service
from facade_next.worker import server_guard, run_one
from facade_next.contracts import METHODS, write

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state',default='runtime_validation')
    for name in ('project','train','evaluation','backbone'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--val',default='')
    p.add_argument('--tools',default='standard')
    p.add_argument('--seed',type=int,default=20260921)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--steps',type=int,default=1)
    p.add_argument('--execute',action='store_true')
    a = p.parse_args()
    request = dict(project=a.project,train=a.train.split(','),evaluation=a.evaluation.split(','),
                   val=a.val.split(',') if a.val else [],tools=METHODS if a.tools=='all' else a.tools.split(','),
                   backbone=a.backbone,seed=a.seed,parameters={'epochs':a.epochs,'steps_per_epoch':a.steps})
    print(json.dumps({'engineering_compatibility_task':request,'execute':a.execute,'automatic_adoption':False},indent=2))
    if not a.execute:
        return
    server_guard(True)
    s = Service(a.state)
    if any(j['status'] in ('running','queued') for j in s.snapshot()['jobs']):
        raise ValueError('Use an idle validation state to avoid consuming another queued task')
    submitted = s.dispatch('submit_task',request,uuid.uuid4().hex)
    rnd = {'id':submitted['round']}
    jobs = submitted['jobs']
    for job_id in jobs:
        if not run_one(a.state,a.device,True,job_id=job_id):
            raise RuntimeError('Requested validation job could not claim its GPU slot; inspect queue before continuing')
    records = [j for j in s.snapshot()['jobs'] if j['id'] in jobs]
    write(Path(a.state)/'server_acceptance_report.json',{'round':rnd,'jobs':records,'engineer_decision_required':True})
    print(json.dumps({'jobs':[(j['id'],j['status']) for j in records],'next':'Inspect comparison in UI; engineer decides'},indent=2))
    if any(j['status']!='succeeded' for j in records):
        raise SystemExit(1)

if __name__=='__main__':
    main()
