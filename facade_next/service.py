"""Transactional workflow and human event receipts; never imports model code."""
import json
import time
import uuid
import threading
from datetime import datetime, timezone
from pathlib import Path
from .store import Store
from .contracts import BACKBONES, METHODS, CLASSES, digest, protocol, sha, check_contract, methods
from .data import load_manifest, normalize, snapshot, union, validate_roles, scan_folder
from .names import WORKFLOW_METHODS, entry_tools, engine_tools


def now():
    return datetime.now(timezone.utc).isoformat()


def uid(prefix):
    return prefix+'_'+uuid.uuid4().hex[:16]


class Service:
    def __init__(self,root,llm=None):
        self.store = Store(root)
        self.root = self.store.root
        self.session = uuid.uuid4().hex
        from .llm import LLMClient
        self.llm=llm if llm is not None else LLMClient()
        self._llm_chat_lock=threading.RLock()

    def verify_result(self,job,result):
        if result['job_digest'] != digest(job['spec']):
            raise ValueError('Result belongs to another job')
        output = Path(job['output']).resolve()
        for relative,expected in result.get('artifacts',{}).items():
            path = (output/relative).resolve()
            if not path.is_relative_to(output) or sha(path) != expected:
                raise ValueError('Candidate artifact escaped output or changed: '+relative)
        if result['status']=='ready':
            contract = check_contract(result['contract'])
            path = Path(result['checkpoint']).resolve()
            if not path.is_relative_to(output):
                raise ValueError('Checkpoint outside candidate output')
            parent = job['spec']['parent']
            if (contract.get('job_digest') != digest(job['spec']) or
                contract.get('roles_digest') != job['spec']['roles_digest'] or
                contract.get('parent_sha256') != (parent['contract']['checkpoint_sha256'] if parent else None) or
                contract['backbone'] != job['spec']['backbone'] or sha(path) != contract['checkpoint_sha256']):
                raise ValueError('Result checkpoint changed or provenance mismatch')
            if parent and sha(parent['checkpoint']) != parent['contract']['checkpoint_sha256']:
                raise ValueError('Parent checkpoint changed')
        if result.get('receipt_path'):
            receipt = Path(result['receipt_path']).resolve()
            if not receipt.is_relative_to(output) or sha(receipt) != result['receipt_sha256']:
                raise ValueError('Completion receipt changed')
            from .contracts import read
            if read(receipt) != {k:v for k,v in result.items() if k not in ('receipt_path','receipt_sha256')}:
                raise ValueError('Completion receipt differs from database result')

    @staticmethod
    def evaluation_complete(job):
        result=job.get('result') or {}
        expected=job['spec']['roles']['evaluation']
        if job['status']!='succeeded' or result.get('evaluation_digest')!=digest(expected):
            return False
        comparison=result.get('comparison') or {}
        key='candidate' if result.get('status')=='ready' else 'parent'
        if (comparison.get(key) or {}).get('merged',{}).get('images')!=len(expected):
            return False
        if job['spec']['parent'] and (comparison.get('parent') or {}).get('merged',{}).get('images')!=len(expected):
            return False
        return True

    def dispatch(self,action,payload,submission_id):
        from .llm_chat import wants_llm,prepare
        if action=='chat' and self.llm.status()['enabled'] and wants_llm(payload):
            # Paid requests must not hold SQLite locks or repeat on HTTP retries.
            with self._llm_chat_lock:
                self._validate_submission(submission_id)
                with self.store.transaction() as db:
                    old=self._previous(db,action,payload,submission_id)
                if old is not None:return old
                prepared=prepare(self,payload)
                return self._dispatch(action,payload,submission_id,prepared)
        return self._dispatch(action,payload,submission_id)

    @staticmethod
    def _validate_submission(submission_id):
        if not isinstance(submission_id,str) or not 1 <= len(submission_id) <= 100:
            raise ValueError('A stable submission_id is required')

    def _previous(self,db,action,payload,submission_id):
        old=db.execute('SELECT * FROM submissions WHERE id=?',(submission_id,)).fetchone()
        if old:
            if old['fingerprint']!=digest({'action':action,'payload':payload}):
                raise ValueError('submission_id collision: different content')
            return json.loads(old['response'])

    def _dispatch(self,action,payload,submission_id,prepared=None):
        self._validate_submission(submission_id)
        fingerprint = digest({'action':action,'payload':payload})
        with self.store.transaction() as db:
            old = db.execute('SELECT * FROM submissions WHERE id=?',(submission_id,)).fetchone()
            if old:
                if old['fingerprint'] != fingerprint:
                    raise ValueError('submission_id collision: different content')
                return json.loads(old['response'])
            handler = getattr(self,'action_'+action,None)
            if handler is None:
                raise ValueError('Unknown action')
            response = self.action_chat(db,payload,prepared) if action=='chat' else handler(db,payload)
            event_project = payload.get('project')
            if action=='chat':
                event_project=response.get('project')
            if action=='create_project':
                event_project = response['id']
            elif payload.get('round'):
                event_project = self.store.get(db,'round',payload['round'])['project']
            elif payload.get('job'):
                event_project = self.store.get(db,'job',payload['job'])['project']
            event = dict(source='human',submission_id=submission_id, action=action,
                         project=event_project,round=payload.get('round'),utc=now(),
                         O=1,K=len(payload),payload_digest=digest(payload))
            db.execute('INSERT INTO events(data) VALUES(?)',(json.dumps(event),))
            db.execute('INSERT INTO submissions VALUES(?,?,?)',(submission_id,fingerprint,json.dumps(response,ensure_ascii=False,allow_nan=False)))
            return response

    def snapshot(self):
        with self.store.transaction() as db:
            value = {kind+'s':self.store.all(db,kind) for kind in ('project','dataset','round','job','version','rollback','timer')}
            value['conversations']=self.store.all(db,'conversation')
            value['batches']=self.store.all(db,'batch')
            for ds in value['datasets']:
                ds.pop('rows',None)
            value['events'] = [json.loads(r[0]) for r in db.execute('SELECT data FROM events ORDER BY seq')]
            value.update(classes=CLASSES,backbones=BACKBONES,tools=WORKFLOW_METHODS,engine_methods=METHODS,method_presets=methods(),
                         initial_protocol=protocol('initial'),update_protocol=protocol('update'))
            return value

    def action_chat(self,db,p,prepared=None):
        from .conversation import handle
        return handle(self,db,p,prepared)

    def action_create_project(self,db,p):
        name = str(p.get('name','')).strip()
        if not name:
            raise ValueError('Project name is required')
        project = dict(id=uid('project'),name=name,active=None,train_pool=[],val_pool=[],eval_pool=[],created=now())
        self.store.put(db,'project',project)
        return {'id':project['id']}

    def action_scan_folder(self,db,p):
        self.store.get(db,'project',p['project'])
        rows = scan_folder(p['root'])
        if not rows:
            raise ValueError('No supported photos in folder')
        draft = dict(id=uid('draft'),project=p['project'],root=str(Path(p['root']).resolve()),rows=rows)
        self.store.put(db,'draft',draft)
        return draft

    def action_import_dataset(self,db,p):
        self.store.get(db,'project',p['project'])
        rows = load_manifest(p['manifest'],p['root']) if p.get('manifest') else normalize(p['rows'],p['root'])
        if not any(v is not None for r in rows for v in r['labels']):
            raise ValueError('No known labels; annotate first')
        source = {'manifest':str(Path(p['manifest']).resolve()),'sha256':sha(p['manifest'])} if p.get('manifest') else {'kind':'manual_annotation','digest':digest(p['rows'])}
        for row in rows:
            if row['scene']=='default':
                row['scene'] = str(p.get('name') or 'default')
        ds = dict(id=uid('dataset'),project=p['project'],name=str(p.get('name') or 'dataset'),created=now(),source=source,**snapshot(rows))
        self.store.put(db,'dataset',ds)
        return {'id':ds['id'],'count':len(rows),'digest':ds['digest']}

    def _datasets(self,db,project,ids):
        if not isinstance(ids,list) or len(ids) != len(set(ids)):
            raise ValueError('Dataset IDs must be a distinct list')
        pools = []
        for id in ids:
            ds = self.store.get(db,'dataset',id)
            if ds['project'] != project:
                raise ValueError('Dataset belongs to another project')
            pools.append(ds['rows'])
        return union(*pools)

    def action_create_round(self,db,p):
        project = self.store.get(db,'project',p['project'])
        if any(r['status'] not in ('decided','cancelled') for r in self.store.all(db,'round',project['id'])):
            raise ValueError('Finish or cancel the pending round first')
        parent = self.store.get(db,'version',project['active']) if project['active'] else None
        kind = 'update' if parent else 'initial'
        backbone = p.get('backbone','convnext_tiny')
        if backbone not in BACKBONES or parent and parent['contract']['backbone'] != backbone:
            raise ValueError('Maintenance backbone must match actual active parent')
        entries = entry_tools(p.get('tools'),kind)
        tools = engine_tools(entries,kind)
        train = self._datasets(db,project['id'],p.get('train',[]))
        val = union(project['val_pool'], self._datasets(db,project['id'],p.get('val',[])))
        incoming_evaluation=self._datasets(db,project['id'],p.get('evaluation',[]))
        old_evaluation={row['sha256'] for row in project['eval_pool']}
        if parent and not any(row['sha256'] not in old_evaluation for row in incoming_evaluation):
            raise ValueError('Maintenance requires a new independent Current Gate; historical Core Safety is reused automatically')
        evaluation = union(project['eval_pool'],incoming_evaluation)
        history = project['train_pool'] if parent else []
        roles = dict(train=train,history=history,val=val,evaluation=evaluation)
        if not train or not val or not evaluation or kind=='update' and not history:
            raise ValueError('Require new train, Val, evaluation and (for maintenance) historical training pool')
        validate_roles(roles)
        if any(all(y is None for y in r['labels']) for r in train+history):
            raise ValueError('Each training/replay photo needs at least one known label')
        if kind=='initial':
            from .planning import initial_pos_weight
            initial_pos_weight(train)
        # Includes cancelled rounds: inspected evaluation/Val remains protected forever.
        rounds = self.store.all(db,'round',project['id'])
        protected_val = union(*(r['roles']['val'] for r in rounds),val)
        protected_eval = union(*(r['roles']['evaluation'] for r in rounds),evaluation)
        ever_train = union(*(union(r['roles']['train'],r['roles']['history']) for r in rounds),train,history)
        validate_roles({'all_train':ever_train,'all_val':protected_val,'all_evaluation':protected_eval})
        for j in range(3):
            known = {r['labels'][j] for r in val if r['labels'][j] is not None}
            if kind == 'initial' and known != {0,1}:
                raise ValueError('Initial Val needs positive and negative labels for each class')
        seed = p.get('seed',20260921)
        if type(seed) is not int or not 0 <= seed <= 2**32-1:
            raise ValueError('Invalid seed')
        init = p.get('initialization','cached_imagenet')
        if init!='cached_imagenet':
            raise ValueError('Initial modeling requires ImageNet pretrained weights; random initialization is not supported')
        config = protocol(kind,p.get('parameters',{}))
        rnd = dict(id=uid('round'),project=project['id'],number=len(rounds)+1,status='draft',kind=kind,
                   parent=parent,backbone=backbone,tools=tools,entry_tools=entries,seed=seed,initialization=init,
                   roles=roles,roles_digest=digest(roles),datasets={k:p.get(k,[]) for k in ('train','val','evaluation')},
                   protocol=config,created=now(),jobs=[])
        self.store.put(db,'round',rnd)
        return {'id':rnd['id'],'kind':kind,'roles_digest':rnd['roles_digest']}

    def action_import_active(self,db,p):
        from .contracts import read
        project = self.store.get(db,'project',p['project'])
        if project['active'] or self.store.all(db,'round',project['id']):
            raise ValueError('Import into a new empty project before creating rounds')
        contract = check_contract(read(p['contract']))
        if sha(p['checkpoint']) != contract['checkpoint_sha256']:
            raise ValueError('Imported weight changed')
        roles = {k:self._datasets(db,project['id'],p.get(k,[])) for k in ('train','val','evaluation')}
        if any(not r for r in roles.values()):
            raise ValueError('Imported active version requires verified train, Val and protected evaluation manifests')
        validate_roles(roles)
        if contract.get('import_roles_digest') != digest(roles) or not str(p.get('reason','')).strip():
            raise ValueError('Role bundle differs from server-validated import contract, or reason missing')
        version = dict(id=uid('version'),project=project['id'],parent=None,round=None,job=None,
                       checkpoint=str(Path(p['checkpoint']).resolve()),contract=contract,
                       roles_digest=digest(roles),reason=p['reason'],adopted=now(),source='explicit_server_import')
        self.store.put(db,'version',version)
        project.update(active=version['id'],train_pool=roles['train'],val_pool=roles['val'],eval_pool=roles['evaluation'])
        self.store.put(db,'project',project)
        return {'active':version['id']}

    def action_start_round(self,db,p):
        if getattr(self,'before_start',None):
            self.before_start()
        rnd = self.store.get(db,'round',p['round'])
        if rnd['status'] != 'draft':
            raise ValueError('Round has already been submitted')
        project = self.store.get(db,'project',rnd['project'])
        if project['active'] != (rnd['parent']['id'] if rnd['parent'] else None):
            raise ValueError('Active version changed; create a new round')
        source = {path.relative_to(Path(__file__).parent).as_posix():sha(path) for path in sorted(Path(__file__).parent.rglob('*')) if path.suffix in ('.py','.json')}
        for method in rnd['tools']:
            id = uid('job')
            spec = dict(round_id=rnd['id'],project=rnd['project'],kind=rnd['kind'],parent=rnd['parent'],
                        backbone=rnd['backbone'],seed=rnd['seed'],initialization=rnd['initialization'],
                        roles=rnd['roles'],roles_digest=rnd['roles_digest'],protocol=rnd['protocol'],
                        method=methods()[method],source_hashes=source)
            job = dict(id=id,project=rnd['project'],round=rnd['id'],method=method,status='queued',
                       spec=spec,output=str(self.root/'jobs'/id),created=now(),result=None,error=None)
            self.store.put(db,'job',job)
            rnd['jobs'].append(id)
        rnd['status'] = 'submitted'
        self.store.put(db,'round',rnd)
        return {'jobs':rnd['jobs']}

    def action_submit_task(self,db,p):
        # One explicit CLI submission; internal orchestration emits no extra O.
        rnd = self.action_create_round(db,p)
        started = self.action_start_round(db,{'round':rnd['id']})
        return {'round':rnd['id'],'jobs':started['jobs']}

    def action_decide(self,db,p):
        rnd = self.store.get(db,'round',p['round'])
        project = self.store.get(db,'project',rnd['project'])
        if rnd['status'] in ('decided','cancelled'):
            raise ValueError('Round is already closed')
        if any(j['status'] in ('queued','running') for j in self.store.all(db,'job',rnd['project']) if j['round']==rnd['id']):
            raise ValueError('Wait for queued/running jobs before deciding')
        if project['active'] != (rnd['parent']['id'] if rnd['parent'] else None):
            raise ValueError('Stale round parent')
        reason = str(p.get('reason','')).strip()
        if not reason:
            raise ValueError('Record the engineer decision reason')
        evidence=[self.store.get(db,'job',id) for id in rnd['jobs']]
        completed=[j for j in evidence if self.evaluation_complete(j)]
        if not completed:
            raise ValueError('Round evaluation is incomplete; Gate cannot accumulate before evaluated adoption/Hold. Cancel or finish evaluation first')
        for evaluated in completed:
            self.verify_result(evaluated,evaluated['result'])
        chosen = p.get('job')
        if chosen:
            if chosen not in rnd['jobs']:
                raise ValueError('Candidate does not belong to this round')
            job = self.store.get(db,'job',chosen)
            if job['status'] != 'succeeded' or job['result']['status'] != 'ready':
                raise ValueError('Candidate is not ready')
            result = job['result']
            if not self.evaluation_complete(job):
                raise ValueError('Candidate evaluation is incomplete')
            self.verify_result(job,result)
            check_contract(result['contract'])
            if sha(result['checkpoint']) != result['contract']['checkpoint_sha256']:
                raise ValueError('Checkpoint changed after evaluation')
            version = dict(id=uid('version'),project=project['id'],parent=project['active'],round=rnd['id'],
                           checkpoint=result['checkpoint'],contract=result['contract'],job=chosen,
                           result_digest=digest(result),roles_digest=rnd['roles_digest'],reason=reason,adopted=now())
            self.store.put(db,'version',version)
            project['active'] = version['id']
        elif not project['active']:
            raise ValueError('No active initial model to Hold; cancel this round instead')
        project['train_pool'] = union(project['train_pool'],rnd['roles']['train'])
        project['val_pool'] = union(project['val_pool'],rnd['roles']['val'])
        project['eval_pool'] = union(project['eval_pool'],rnd['roles']['evaluation'])
        rnd.update(status='decided',decision='adopt' if chosen else 'hold',reason=reason,
                   selected_job=chosen,active=project['active'],decided=now())
        self.store.put(db,'project',project)
        self.store.put(db,'round',rnd)
        return {'active':project['active'],'decision':rnd['decision']}

    def action_cancel_round(self,db,p):
        rnd = self.store.get(db,'round',p['round'])
        if rnd['status'] in ('decided','cancelled'):
            raise ValueError('Round already closed')
        for job in self.store.all(db,'job',rnd['project']):
            if job['round'] != rnd['id']:
                continue
            if job['status'] == 'running':
                raise ValueError('Running worker must finish or be marked interrupted before cancellation')
            if job['status'] == 'queued':
                job['status'] = 'cancelled'
                self.store.put(db,'job',job)
        rnd.update(status='cancelled',reason=str(p.get('reason') or 'engineer cancelled'),closed=now())
        self.store.put(db,'round',rnd)
        return {'id':rnd['id'],'status':'cancelled'}

    def action_rollback(self,db,p):
        project = self.store.get(db,'project',p['project'])
        if 'expected_active' in p and p['expected_active']!=project['active']:
            raise ValueError('Rollback confirmation is stale because the actual active version changed')
        if any(r['status'] not in ('decided','cancelled') for r in self.store.all(db,'round',project['id'])):
            raise ValueError('Close pending round before rollback')
        version = self.store.get(db,'version',p['version'])
        if version['project'] != project['id'] or not str(p.get('reason','')).strip():
            raise ValueError('Choose a previously adopted version and give a rollback reason')
        if sha(version['checkpoint']) != version['contract']['checkpoint_sha256']:
            raise ValueError('Rollback checkpoint changed')
        receipt = dict(id=uid('rollback'),project=project['id'],from_version=project['active'],to_version=version['id'],reason=p['reason'],utc=now())
        self.store.put(db,'rollback',receipt)
        project['active'] = version['id']
        self.store.put(db,'project',project)
        return receipt

    def action_timer(self,db,p):
        project = self.store.get(db,'project',p['project'])
        timer = project.get('timer',{'state':'off','active_seconds':0.,'segments':[]})
        command = p['command']
        if timer.get('host_session') != self.session and timer['state'] in ('active','waiting'):
            if command not in ('start','end'):
                raise ValueError('Timer interrupted by service restart; end or start a new session')
            timer['segments'].append({'command':'interrupted_by_restart','utc':now()})
            timer['state'] = 'interrupted'
        before = timer['state']
        allowed = {'start':('off','ended','interrupted'),'wait':('active',),'resume':('waiting',),'end':('active','waiting','interrupted')}
        if command not in allowed or before not in allowed[command]:
            raise ValueError('Invalid explicit timer transition')
        clock = time.monotonic()
        if command == 'start':
            timer = {'state':'active','active_seconds':0.,'segments':[],'session':uid('timer')}
        elif before == 'active':
            # Timer clock is valid only during this host boot/session.
            timer['active_seconds'] += max(0.,clock-timer['since'])
        timer['state'] = {'start':'active','wait':'waiting','resume':'active','end':'ended'}[command]
        timer['since'] = clock
        timer['host_session'] = self.session
        timer['segments'].append({'command':command,'utc':now(),'monotonic':clock})
        project['timer'] = timer
        self.store.put(db,'timer',dict(timer,id=timer['session'],project=project['id']))
        self.store.put(db,'project',project)
        return timer

    def action_inspect(self,db,p):
        project = self.store.get(db,'project',p['project'])
        return {'active':project['active'],'utc':now()}

    def action_export_audit(self,db,p):
        return {'utc':now(),'purpose':'explicit_audit_export'}

    def action_mark_interrupted(self,db,p):
        job = self.store.get(db,'job',p['job'])
        if job['status'] != 'running' or not str(p.get('reason','')).strip():
            raise ValueError('Only interrupted running jobs with a reason can be recovered')
        if job.get('pid'):
            from .platform_support import process_alive
            if process_alive(job['pid']):
                raise ValueError('Worker PID still exists; stop/inspect it before marking interrupted')
        # Only after operator verifies process stopped. No automatic retry.
        job.update(status='failed',error='Engineer marked interrupted: '+p['reason'],finished=now())
        self.store.put(db,'job',job)
        return {'id':job['id'],'status':'failed'}

    def claim(self,device,job_id=None):
        with self.store.transaction() as db:
            if any(j['status']=='running' and j.get('device')==device for j in self.store.all(db,'job')):
                return None
            for job in self.store.all(db,'job'):
                if job_id is not None and job['id'] != job_id:
                    continue
                if job['status']=='queued':
                    job.update(status='running',device=device,started=now(),pid=__import__('os').getpid())
                    self.store.put(db,'job',job)
                    return job
        return None

    def finish(self,id,result=None,error=None):
        with self.store.transaction() as db:
            job = self.store.get(db,'job',id)
            if job['status'] != 'running':
                raise ValueError('Job is no longer running')
            if error is None:
                if result['job_digest'] != digest(job['spec']):
                    raise ValueError('Result belongs to another job')
                if result['status'] not in ('ready','skip_empty_pool'):
                    raise ValueError('Invalid result status')
                if result['status']=='ready':
                    check_contract(result['contract'])
                    if result['contract']['backbone'] != job['spec']['backbone'] or sha(result['checkpoint']) != result['contract']['checkpoint_sha256']:
                        raise ValueError('Result checkpoint binding mismatch')
                self.verify_result(job,result)
            job.update(status='failed' if error else 'succeeded',result=result,error=error,finished=now())
            self.store.put(db,'job',job)
