"""Read-only identification of a completed operation-time comparison case."""
import argparse
import json
from pathlib import Path
import sqlite3


def inspect_case(state,round_id=None):
    path=Path(state).expanduser().resolve()/'audit.sqlite3'
    if not path.is_file():raise ValueError('Existing audit.sqlite3 not found: '+str(path))
    entities={key:[] for key in ('project','round','job','version')}
    db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=10)
    try:
        db.execute('PRAGMA query_only=ON');db.execute('BEGIN')
        for kind,data in db.execute('SELECT kind,data FROM entities ORDER BY rowid'):
            if kind in entities:entities[kind].append(json.loads(data))
    finally:db.close()
    projects={p['id']:p for p in entities['project']};rounds={r['id']:r for r in entities['round']}
    complete=[]
    for job in entities['job']:
        rnd=rounds.get(job.get('round'));result=job.get('result') or {}
        if not rnd or job.get('project')!=rnd.get('project') or job.get('status')!='succeeded' or not result.get('comparison'):continue
        spec=job.get('spec') or rnd
        if spec.get('kind')!='update':continue
        roles=spec.get('roles') or {};parent=spec.get('parent');comparison=result['comparison']
        complete.append(dict(project=job['project'],project_name=projects.get(job['project'],{}).get('name'),round=rnd['id'],round_number=rnd.get('number'),job=job['id'],
            backbone=spec.get('backbone'),seed=spec.get('seed'),method=job.get('method'),parent=parent.get('id') if isinstance(parent,dict) else parent,
            role_counts={key:len(roles[key]) if isinstance(roles.get(key),list) else None for key in ('train','history','val','evaluation')},
            parent_macro_AP=((comparison.get('parent') or {}).get('merged') or {}).get('macro_AP'),candidate_macro_AP=((comparison.get('candidate') or {}).get('merged') or {}).get('macro_AP'),
            started=job.get('started'),finished=job.get('finished'),receipt_cost=result.get('cost'),
            decision=rnd.get('decision'),reason=rnd.get('reason'),effective_version=rnd.get('active'),decided=rnd.get('decided')))
    matches=[item for item in complete if item['backbone']=='convnext_tiny' and item['seed']==20260921 and item['method']=='standard' and (round_id is None or item['round']==round_id)]
    return dict(status='matched' if matches else 'not_found',read_only=True,source_database=str(path),matches=matches,available_completed_update_records=complete)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--state',required=True);parser.add_argument('--round')
    args=parser.parse_args()
    try:
        report=inspect_case(args.state,args.round);print(json.dumps(report,ensure_ascii=False,indent=2))
        if report['status']!='matched':raise SystemExit(2)
    except (ValueError,OSError,sqlite3.Error) as error:parser.exit(1,str(error)+'\n')

if __name__=='__main__':main()
