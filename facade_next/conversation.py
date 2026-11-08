"""Stateful conversation controller with an optional validated LLM proposal.

Only explicit supported intents or task-card submissions invoke typed service
actions. Replies are grounded in the same SQLite state as the auxiliary panels.
"""
import copy
import re
from .contracts import BACKBONES, METHODS, CLASSES
from .intake import prepare_folder
from .names import method_name, mentioned_methods, entry_tools


def words(zh,en):
    return {'zh':zh,'en':en}


def replacement_decision(text,event,prepared,context):
    if event:
        return event=='propose_decision'
    if (prepared or {}).get('result',{}).get('intent')=='propose_decision' and context.get('pending',{}).get('kind') in ('adopt','hold'):
        return True
    prefix=r"(?:(?:please[\s,，]+)|(?:(?:do\s+not|don't|never)\s+)|(?:请|帮我|我想|我要|我决定|我现在决定|现在|这次|决定|不要|不想|先别)\s*)*"
    return bool(re.match(r'^'+prefix+r'(?:采用|启用|保留|adopt\b|promote\b|hold\b|retain\b)',text,re.I))


def decision_reason(text):
    return re.search(r'(?:因为|理由\s*[:：]?|原因\s*[:：]?|\bbecause\b\s*|\breason\s*:)(.+)',text,re.I)


def adoption_candidate(text):
    """Bind one explicit candidate before the engineer's reason, or clarify."""
    reason=decision_reason(text)
    clause=text[:reason.start()] if reason else text
    methods=mentioned_methods(clause)
    jobs=list(dict.fromkeys(re.findall(r'(?<![a-z0-9_])job_[a-z0-9_]+(?![a-z0-9_])',clause,re.I)))
    negated=re.search(r'\b(?:not|never|instead\s+of)\b|\b(?:do|does|did)n[\'’]t\b|不(?:要|用|选|采用|启用|是)|勿|别',clause,re.I)
    alternatives=re.search(r'\bor\b|或者|或',clause,re.I)
    if negated or alternatives or len(methods)>1 or len(jobs)>1:
        return None
    return jobs[0] if jobs else methods[0] if methods else None


def folder_request(text,default_role=None):
    """Parse an absolute local path before intent keywords; ignore its name."""
    found=next((m for m in re.finditer(r'[“\"]([^”\"]+)[”\"]',text)
                if re.match(r'^(?:[A-Za-z]:[\\/]|/)',m.group(1))),None)
    if found:
        path=found.group(1)
        outside=text[:found.start()]+text[found.end():]
    else:
        found=re.search(r'(?:^|[\s:：]|在)((?:[A-Za-z]:[\\/]|/)[^\n]+)',text)
        if not found:
            return None
        path=found.group(1).strip()
        outside=text[:found.start(1)]+text[found.end(1):]
    lower=outside.lower()
    role='val' if re.search(r'\bval\b|验证|校准',lower) else 'evaluation' if re.search(r'gate|safety|评价|评估|评测',lower) else 'train' if re.search(r'train|训练|更新数据',lower) else default_role
    return dict(root=path,role=role)


def handle(service,db,p,prepared=None):
    from .service import uid,now
    store=service.store
    if p.get('conversation'):
        c=store.get(db,'conversation',p['conversation'])
        if p.get('project') and c.get('project') and c['project']!=p['project']:
            raise ValueError('Conversation belongs to another project')
    else:
        c=dict(id=uid('chat'),project=p.get('project'),messages=[],context={'selected':{},'options':{}},created=now())
        if c['project']:
            store.get(db,'project',c['project'])
    text=str(p.get('text') or '').strip()
    if len(text)>8000:
        raise ValueError('Chat message too long')
    event=p.get('event')
    if not text and not event:
        raise ValueError('Enter a message or submit a task card')
    if replacement_decision(text,event,prepared,c['context']):
        ctx=c['context']
        superseded=ctx.get('pending',{}).get('kind') in ('adopt','hold')
        if superseded:
            ctx.pop('pending',None)
        if superseded or ctx.get('await_reason'):
            ctx.pop('await_reason',None)
            _,c['card']=comparison(service,db,c)
        # Restore this superseded state on action/provider validation failure:
        # the engineer has explicitly moved on from the previous choice.
    c['messages'].append(dict(role='user',text=text or '',event=event,utc=now()))
    before=copy.deepcopy(c['context'])
    project_before=c.get('project')
    db.execute('SAVEPOINT conversation_action')
    try:
        if prepared is None:
            reply,card=route(service,db,c,text,event,p.get('inputs') or {})
        else:
            from .llm_chat import apply
            reply,card=apply(service,db,c,prepared,text,p)
        db.execute('RELEASE SAVEPOINT conversation_action')
    except (ValueError,KeyError,TypeError,OSError) as e:
        db.execute('ROLLBACK TO SAVEPOINT conversation_action')
        db.execute('RELEASE SAVEPOINT conversation_action')
        c['context']=before
        c['project']=project_before
        reply=words('这一步未执行：'+str(e)+'。请修正后再提交；我保留了当前任务。',
                    'This step was not executed: '+str(e)+'. Correct it and resubmit; the current task is retained.')
        card=c.get('card')
    c['card']=card
    c['messages'].append(dict(role='assistant',reply=reply,utc=now()))
    store.put(db,'conversation',c)
    return dict(conversation=c['id'],project=c.get('project'),message_count=len(c['messages']),reply=reply,card=card)


def current(service,db,c):
    p=service.store.get(db,'project',c['project'])
    rounds=service.store.all(db,'round',p['id'])
    open_round=next((r for r in reversed(rounds) if r['status'] not in ('decided','cancelled')),None)
    return p,rounds,open_round


def sync_closed_round(service,db,c):
    """A panel decision must advance the chat's data-selection context too."""
    if not c.get('project'):
        return
    closed=[r for r in service.store.all(db,'round',c['project']) if r['status'] in ('decided','cancelled')]
    if closed and c['context'].get('last_closed_round')!=closed[-1]['id']:
        ctx=c['context']
        ctx['selected']={}
        ctx.pop('prepared_round',None)
        ctx.pop('await_reason',None)
        if ctx.get('pending',{}).get('kind')!='rollback':
            ctx.pop('pending',None)
        ctx['last_closed_round']=closed[-1]['id']
    project=service.store.get(db,'project',c['project'])
    options=c['context'].get('options',{})
    if project['active'] and options.get('tools')==['core']:
        options['tools']=['standard']


def pending(c,kind,arguments,summary):
    from .service import uid
    proposal=dict(kind=kind,token=uid('confirm'),arguments=arguments,summary=summary)
    c['context']['pending']=proposal
    return {**proposal,'kind':'confirm_'+kind,'action':kind}


def next_step(service,db,c):
    sync_closed_round(service,db,c)
    if not c.get('project'):
        return words('先给项目起个名字，例如“创建项目 外立面维护”。之后我会逐步组织数据、训练和人工选择。',
                     'Name the project first, for example “create project Facade maintenance”. I will guide data intake, training and engineer decisions.'),{'kind':'create_project'}
    p,rounds,rnd=current(service,db,c)
    if c['context'].get('pending',{}).get('kind')=='batch':
        from .batch_flow import card
        batch=service.store.get(db,'batch',c['context']['pending']['arguments']['batch'])
        # Status requests retain the exact confirmation token.
        value={**c['context']['pending'],'kind':'confirm_batch','action':'batch','parts':batch['parts'],'stage':batch['stage'],'name':batch['name']}
        return batch['summary'],value
    old=c['context'].get('pending')
    if old and old['kind']=='rollback' and old['arguments'].get('expected_active')!=p['active']:
        c['context'].pop('pending',None)
        old=None
    if old and old['kind'] in ('cancel','rollback','adopt','hold'):
        return old['summary'],{**old,'kind':'confirm_'+old['kind'],'action':old['kind']}
    if c['context'].get('draft') and not rnd:
        draft=service.store.get(db,'draft',c['context']['draft'])
        return words('当前这批照片还在标注中，继续编辑后整批保存即可。也可以明确保留未知标签导入，或说“取消标注”。',
                     'This photo batch is still being annotated. Continue editing and save once, explicitly retain unknown labels, or say “cancel annotation”.'),{'kind':'annotation','draft':draft}
    if not rnd and not c['context'].get('draft') and c['context'].get('batch_id'):
        from .batch_flow import advance
        value=advance(service,db,c)
        if value:return value
    if rnd:
        if rnd['status']=='draft':
            stage='初始建模' if rnd['kind']=='initial' else '模型维护'
            stage_en='Initial modeling' if rnd['kind']=='initial' else 'Model maintenance'
            summary=words(f'本轮为{stage}，主干{rnd["backbone"]}，工具{", ".join(method_name(t,kind=rnd["kind"]) for t in rnd["tools"])}；新训练{len(rnd["roles"]["train"])}、历史训练{len(rnd["roles"]["history"])}、Val {len(rnd["roles"]["val"])}、独立评价{len(rnd["roles"]["evaluation"])}张。确认后只启动这些候选，完成后仍由你决定采用。',
                          f'Round {rnd["number"]}: {stage_en}, {rnd["backbone"]}, tools {", ".join(method_name(t, "en",kind=rnd["kind"]) for t in rnd["tools"])}; new train {len(rnd["roles"]["train"])}, replay {len(rnd["roles"]["history"])}, Val {len(rnd["roles"]["val"])}, evaluation {len(rnd["roles"]["evaluation"])}. Confirm to start these candidates; adoption remains your decision.')
            old=c['context'].get('pending')
            if old and old['kind']=='start' and old['arguments']=={'round':rnd['id']}:
                card={**old,'kind':'confirm_start','action':'start'}
            else:
                card=pending(c,'start',{'round':rnd['id']},summary)
            return summary,card
        jobs=[service.store.get(db,'job',j) for j in rnd['jobs']]
        if any(j['status'] in ('queued','running') for j in jobs):
            state=', '.join(method_name(j['method'],kind=rnd['kind'])+': '+j['status'] for j in jobs)
            state_en=', '.join(method_name(j['method'],'en',kind=rnd['kind'])+': '+j['status'] for j in jobs)
            return words('实际任务状态：'+state+'。等待自动任务完成即可；可随时说“查看状态”。界面模式不执行GPU，需服务器Worker。',
                         'Actual job status: '+state_en+'. Wait for automatic work or ask “status”. UI-only mode needs an explicitly enabled server Worker.'),{'kind':'status','round':rnd['id'],'jobs':[{k:j.get(k) for k in ('id','method','status','device','error')} for j in jobs]}
        reason=c['context'].get('await_reason')
        if reason:
            if reason.get('round')==rnd['id']:
                return words('结果已完成；你尚未填写本次采用/保留理由。查询状态不会替你填写或确认理由。',
                             'Results are complete; your adoption/Hold reason is still pending. Status queries do not fill or confirm it.'),{'kind':'reason',**reason}
            c['context'].pop('await_reason',None)
        return comparison(service,db,c,rnd)
    selected=c['context'].setdefault('selected',{})
    if rounds and rounds[-1]['id']==c['context'].get('prepared_round'):
        selected.clear()
        c['context'].pop('prepared_round',None)
        c['context'].pop('pending',None)
        c['context'].pop('await_reason',None)
    need=['train']+([] if p['val_pool'] else ['val'])+['evaluation']
    missing=[role for role in need if not selected.get(role)]
    if missing:
        role=missing[0]
        c['context']['await_role']=role
        if not any(selected.values()):
            stage='update' if p['active'] else 'initial'
            return words(('当前准备新一轮维护。请把本轮已划分好的train、gate/test放在一个批次文件夹中；val可交，也可沿用历史Val。' if p['active'] else
                          '当前准备初始建模。请把已划分好的train、val、safety/test放在一个批次文件夹中。')+
                         '在对话中告诉我这批资料的名称和文件夹路径即可，我会核对已有子目录并展示用途，确认后分别导入。后续说“开始新一轮维护”，再提交一个新批次文件夹即可持续维护。也可逐份导入子集。',
                         ('Preparing another maintenance round: put pre-split train and gate/test in one batch folder; val is optional and existing Val can be reused. ' if p['active'] else
                          'Preparing initial modeling: put pre-split train, val and safety/test in one batch folder. ')+
                         'Tell me its name and folder path in chat. I will inspect existing subsets and show their roles for confirmation. Say “start a new maintenance round” for each subsequent batch, without a fixed round limit. Individual subset import also remains available.'),{'kind':'folder','role':role,'batch':True,'stage':stage}
        names={'train':('本轮Train训练/更新数据','new Train/update data'),'val':('独立Val校准数据（首次指定，之后沿用）','independent Val calibration data (supply once, then reuse)'),
               'evaluation':('本轮验收数据（Current Gate）' if p['active'] else '初始验收数据（S0）','this round’s independent Current Gate' if p['active'] else 'initial independent acceptance data (S0)')}
        zh,en=names[role]
        guide=words('后续每轮请预先划分两份：新增Train用于更新，本轮验收集Current Gate只做独立评价。已有Val、历史训练和历史验收由系统沿用，不必重复提交；可按需另交独立的新Val。A、B、C只是连续批次，不限制轮数。',
                    'For each maintenance round, submit two pre-split subsets: new Train for updating and an independent Current Gate for acceptance. Existing Val, training history and acceptance history are reused; no repeat submission is needed. An additional independent Val is optional. A, B and C are successive batches, not a fixed round limit.') if p['active'] else words(
                    '初始建模请预先划分三份：Train用于学习权重，Val用于选择检查点和校准阈值，初始验收集 S0只做独立评价。每次提交已经划分好的对应子集，不需要另外准备final test。',
                    'Before initial modeling, prepare three separate subsets: Train learns weights, Val selects the checkpoint and calibrates thresholds, and initial acceptance S0 is reserved for independent evaluation. Submit each already-split subset; no additional final-test dataset is required.')
        prefix=guide if role=='train' else words('','')
        return words(prefix['zh']+f'\n下一步需要{zh}。请发送该子集文件夹的路径，不要发送包含多个角色的总目录；也可在下面选择角色并填写。系统不自动划分或重排数据，会查找labels.xlsx/labels.csv。完整标注直接校验导入，缺失标注可在聊天内补齐。',
                     prefix['en']+f'\nNext I need {en}. Send that subset’s folder, not a parent containing multiple roles, or use the card below. The system does not split or reshuffle data. It detects labels.xlsx/labels.csv, validates complete labels and offers annotation for missing labels.'),{'kind':'folder','role':role}
    return words('数据角色已就绪。首轮使用 Core 初始建模；后续维护默认 Standard，也可选择其他维护方法。准备后我会列出实际任务，等你确认启动。',
                 'The data roles are ready. Initial modeling uses Core; maintenance defaults to Standard and supports other maintenance methods. I will summarize the actual task before you confirm starting it.'),{'kind':'configure','selected':selected,'options':c['context'].get('options',{})}


def import_folder(service,db,c,inputs):
    if not inputs.get('_batch'):
        if c['context'].get('pending',{}).get('kind')=='batch':
            raise ValueError('Confirm or cancel the pending batch mapping before another import')
        from .batch_intake import role_folders
        if role_folders(inputs['root']):
            if inputs.get('manifest'):raise ValueError('For a batch root, put each annotation table in its role subfolder or import subsets individually')
            from .batch_flow import propose
            return propose(service,db,c,inputs)
        from pathlib import Path
        root=Path(inputs['root']).resolve()
        nested=[p for p in root.iterdir() if p.is_dir() and p.resolve().is_relative_to(root) and role_folders(p)]
        if nested:
            raise ValueError('This folder contains pre-split batches; submit one batch folder: '+', '.join(str(p) for p in nested))
    if current(service,db,c)[2]:
        raise ValueError('Finish or cancel the current round before attaching another folder through chat')
    if c['context'].get('draft'):
        raise ValueError('Save or explicitly cancel the current annotation draft before importing another folder')
    role=inputs.get('role') or c['context'].get('await_role')
    if role not in ('train','val','evaluation'):
        raise ValueError('Specify Train, Val or independent evaluation role')
    bundle=prepare_folder(inputs['root'],inputs.get('manifest'))
    name=str(inputs.get('name') or {'train':'Train','val':'Val','evaluation':'Gate'}[role])
    from .service import uid
    draft=dict(id=uid('draft'),project=c['project'],role=role,name=name,**bundle)
    if inputs.get('_batch'):
        if inputs['_batch']!=c['context'].get('batch_id'):raise ValueError('Batch import context changed')
        draft['batch']=inputs['_batch']
    service.store.put(db,'draft',draft)
    c['context']['draft']=draft['id']
    if bundle['unknown_images'] or bundle['partial_images']:
        c['context']['draft']=draft['id']
        return words(f'{name}：找到{bundle["count"]}张：完整标注{bundle["complete_images"]}、部分标注{bundle["partial_images"]}、未标注{bundle["unknown_images"]}。请在这里补标；确需保留部分未知标签时可明确选择“按未知标签导入”。未标注不会当阴性。',
                     f'{name}: found {bundle["count"]} photos: {bundle["complete_images"]} complete, {bundle["partial_images"]} partial, {bundle["unknown_images"]} unannotated. Annotate here, or explicitly retain unknown labels. Unannotated never means negative.'),{'kind':'annotation','draft':draft}
    save_draft(service,db,c,draft['id'],draft['rows'])
    reply,card=next_step(service,db,c)
    if inputs.get('_batch'):return reply,card
    return words(f'{bundle["count"]}张照片标注完整，已校验并导入，无需重复标注。'+reply['zh'],
                 f'All {bundle["count"]} photos had complete labels and were validated/imported; no repeat annotation is needed. '+reply['en']),card


def save_draft(service,db,c,draft_id,rows):
    draft=service.store.get(db,'draft',draft_id)
    if draft['project']!=c['project'] or c['context'].get('draft')!=draft_id:
        raise ValueError('Annotation draft belongs to another task')
    if draft.get('batch'):
        batch=service.store.get(db,'batch',draft['batch'])
        if c['context'].get('batch_id')!=batch['id'] or batch['status']!='importing':
            raise ValueError('Batch annotation card is stale or cancelled')
    # Source images are fixed by the draft; submitted labels/scene/group may change.
    if len(rows)!=len(draft['rows']) or any(a['path']!=b['path'] or a['image_id']!=b['image_id'] for a,b in zip(rows,draft['rows'])):
        raise ValueError('Annotation photo identity changed')
    from pathlib import Path
    from .contracts import sha
    for original in draft['rows']:
        if original.get('source_sha256') and sha(Path(draft['root'])/original['path'])!=original['source_sha256']:
            raise ValueError('Photo changed during annotation; cancel and re-import this folder')
    result=service.action_import_dataset(db,dict(project=c['project'],name=draft['name'],root=draft['root'],rows=rows))
    if draft.get('batch'):
        batch=service.store.get(db,'batch',draft['batch'])
        ds=service.store.get(db,'dataset',result['id'])
        ds.update(batch=batch['id'],batch_name=batch['name'],batch_stage=batch['stage'],batch_role=draft['role'])
        service.store.put(db,'dataset',ds)
        batch['datasets'][draft['role']]=ds['id'];service.store.put(db,'batch',batch)
    c['context'].setdefault('selected',{}).setdefault(draft['role'],[]).append(result['id'])
    c['context'].pop('draft',None)
    return result


def configure(service,db,c,inputs):
    project,_,rnd=current(service,db,c)
    if rnd:
        return next_step(service,db,c)
    if 'tools' in inputs:
        inputs=dict(inputs,tools=entry_tools(inputs['tools'],'update' if project['active'] else 'initial'))
    selected=c['context'].setdefault('selected',{})
    for role in ('train','val','evaluation'):
        if role in inputs:
            selected[role]=inputs[role]
    if not selected.get('train') or not selected.get('evaluation') or not (selected.get('val') or project['val_pool']):
        return next_step(service,db,c)
    opts=c['context'].setdefault('options',{})
    opts.update({k:inputs[k] for k in ('backbone','tools','seed','initialization','parameters') if k in inputs})
    if project['active'] and 'tools' not in inputs and opts.get('tools')==['core']:
        opts['tools']=['standard']
    parent=service.store.get(db,'version',project['active']) if project['active'] else None
    result=service.action_create_round(db,dict(project=c['project'],**selected,
        backbone=parent['contract']['backbone'] if parent else opts.get('backbone','convnext_tiny'),tools=opts.get('tools',['standard']),
        seed=opts.get('seed',20260921),initialization=opts.get('initialization','cached_imagenet'),parameters=opts.get('parameters',{})))
    c['context']['prepared_round']=result['id']
    return next_step(service,db,c)


def comparison(service,db,c,rnd=None):
    p,rounds,op=current(service,db,c)
    rnd=rnd or op or (rounds[-1] if rounds else None)
    if not rnd:
        return next_step(service,db,c)
    jobs=[service.store.get(db,'job',j) for j in rnd['jobs']]
    complete=[j for j in jobs if j['status']=='succeeded']
    if not complete:
        return words('当前没有完成的评价结果。可取消本轮后重新配置；只有完成评价的Gate才能在人工决定后累计。',
                     'No evaluation result is complete. Cancel and reconfigure; only an evaluated Gate can accumulate after your decision.'),{'kind':'comparison','round':rnd['id'],'jobs':[],'can_hold':False,'closed':rnd['status'] in ('decided','cancelled')}
    if rnd.get('kind')=='initial':
        return words('下面是初始模型在 S0 验收集上的真实评估结果。初始轮没有父模型，不做新旧模型对比。请核对各类 AP、Precision、Recall、F1 和 FP/FN，决定采用时需填写理由并确认。',
                     'These are the initial model evaluation results on the S0 acceptance set. The initial round has no parent model to compare. Review per-class AP, precision, recall, F1 and FP/FN; adoption requires your reason and confirmation.'),{'kind':'comparison','round':rnd['id'],'jobs':[{k:j.get(k) for k in ('id','method','status','result')} for j in complete],'can_hold':False,'closed':rnd['status'] in ('decided','cancelled')}
    return words('下面是真实结果：合并及各场景的逐类AP、precision、recall、F1、FP/FN。历史退化只是提示。我不会替你选模型；可说“采用标准训练／更新候选，因为……”或“保留，因为……”。',
                 'These are actual merged/per-scene AP, precision, recall, F1 and FP/FN results. Historical loss is advisory. I do not select a model for you; say “adopt standard because …” or “hold because …”.'),{'kind':'comparison','round':rnd['id'],'jobs':[{k:j.get(k) for k in ('id','method','status','result')} for j in complete],'can_hold':bool(rnd['parent']),'closed':rnd['status'] in ('decided','cancelled')}


def propose_decision(service,db,c,inputs):
    project,_,rnd=current(service,db,c)
    if not rnd:
        raise ValueError('No undecided round')
    if inputs.get('round') and inputs['round']!=rnd['id']:
        raise ValueError('Decision card is stale and belongs to another round')
    jobs=[service.store.get(db,'job',j) for j in rnd['jobs']]
    if any(j['status'] in ('queued','running') for j in jobs):
        raise ValueError('Wait for queued/running tasks to finish before deciding')
    if not any(service.evaluation_complete(j) for j in jobs):
        raise ValueError('Complete the round evaluation before confirming adoption/Hold; unfinished Gate data does not accumulate')
    choice=inputs.get('choice','adopt')
    reason=str(inputs.get('reason') or '').strip()
    job_id=inputs.get('job')
    if job_id=='core':
        if rnd['kind']!='initial':
            raise ValueError('Core is the initial candidate only; choose a maintenance candidate')
        job_id='standard'
    # A new choice supersedes its previous confirmation, including while the
    # engineer is still supplying a new reason or clarifying the candidate.
    c['context'].pop('pending',None)
    c['context'].pop('await_reason',None)
    if choice=='adopt':
        candidates=[j for j in jobs if j['status']=='succeeded' and j['result']['status']=='ready' and (j['id']==job_id or j['method']==job_id)]
        if len(candidates)!=1:
            return words('请明确选择一个已完成的候选工具。我不会根据“最好”自行采用。',
                         'Choose one completed candidate explicitly; I will not infer adoption from “best”.'),comparison(service,db,c,rnd)[1]
        job_id=candidates[0]['id']
    elif choice=='hold':
        if not project['active']:
            raise ValueError('There is no active initial model to Hold')
        job_id=None
    else:
        raise ValueError('Decision must be adopt/hold')
    if not reason:
        c['context']['await_reason']={'choice':choice,'job':job_id,'round':rnd['id']}
        return words('请补充这次采用/保留的理由；之后我会列出确切版本，请你确认。',
                     'Give the adoption/Hold reason. I will then show the exact version for confirmation.'),{'kind':'reason','choice':choice,'job':job_id,'round':rnd['id']}
    version=next((j for j in jobs if j['id']==job_id),None)
    contract=version['result']['contract'] if version else rnd['parent']['contract']
    summary=words(f'准备{ "采用候选" if job_id else "Hold保留父模型" }。权重SHA：{contract["checkpoint_sha256"]}；阈值：{contract["thresholds"]}；理由：{reason}。这由你决定，确认后本轮Gate才并入历史评价。',
                  f'Ready to {"adopt the candidate" if job_id else "Hold the parent"}. Weight SHA: {contract["checkpoint_sha256"]}; thresholds: {contract["thresholds"]}; reason: {reason}. Confirm your decision; only then will this round’s Gate join historical evaluation.')
    c['context'].pop('await_reason',None)
    return summary,pending(c,'adopt' if job_id else 'hold',dict(round=rnd['id'],job=job_id,reason=reason),summary)


def confirm(service,db,c,inputs,text):
    proposal=c['context'].get('pending')
    if not proposal:
        if inputs.get('token'):
            raise ValueError('Confirmation card is stale; no current operation is awaiting it')
        return words('当前没有待确认的操作。先说“下一步”，我会依据实际状态准备任务。',
                     'No action is awaiting confirmation. Ask “next” and I will prepare the actual next step.'),None
    if inputs.get('token') and inputs['token']!=proposal['token']:
        raise ValueError('This confirmation card is stale; ask next for the current task')
    kind=proposal['kind']
    named={'确认启动':'start','confirm start':'start','确认采用':'adopt','confirm adopt':'adopt','确认保留':'hold','confirm hold':'hold','确认取消':'cancel','confirm cancel':'cancel','确认回滚':'rollback','confirm rollback':'rollback','确认导入':'batch','confirm import':'batch'}
    if text.lower() in named and named[text.lower()]!=kind:
        raise ValueError('Confirmation does not match the pending action')
    if kind=='batch':
        from .batch_flow import confirm as confirm_batch
        return confirm_batch(service,db,c,proposal)
    fn={'start':'start_round','adopt':'decide','hold':'decide','cancel':'cancel_round','rollback':'rollback'}[kind]
    getattr(service,'action_'+fn)(db,proposal['arguments'])
    c['context'].pop('pending',None)
    reply,card=next_step(service,db,c)
    prefix=words('已按你确认的操作执行。','Your confirmed action has been executed. ')
    return words(prefix['zh']+reply['zh'],prefix['en']+reply['en']),card


def status_reply(service,db,c):
    reply,card=next_step(service,db,c)
    p,rounds,_=current(service,db,c)
    summary=words(f'当前在役版本：{p["active"] or "尚未建立"}；历史训练{len(p["train_pool"])}、累计Val {len(p["val_pool"])}、历史验收 {len(p["eval_pool"])}张。',
                  f'Actual active version: {p["active"] or "none"}; historical train {len(p["train_pool"])}, Val {len(p["val_pool"])}, historical acceptance {len(p["eval_pool"])}.')
    return words(summary['zh']+reply['zh'],summary['en']+' '+reply['en']),card


def route(service,db,c,text,event,inputs):
    sync_closed_round(service,db,c)
    ctx=c['context']
    lower=text.lower().strip('。.!！')
    from .llm_chat import direct_query
    query=direct_query(text) if not event else None
    if event in ('compare','status') or query:
        if not c.get('project'):
            return next_step(service,db,c)
        if event=='compare' or query=='compare':
            requested=inputs.get('round')
            rnd=service.store.get(db,'round',requested) if requested else None
            if rnd and rnd['project']!=c['project']:
                raise ValueError('Result round belongs to another project')
            return comparison(service,db,c,rnd)
        return status_reply(service,db,c)
    if event=='begin_round' or (not event and not folder_request(text) and re.search(r'新一轮(?:维护)?(?:开始|训练)?|开始(?:新|下一|一)轮维护|下一轮维护|start (?:a )?new maintenance round',lower)):
        if not c.get('project'):return next_step(service,db,c)
        if inputs.get('stage') and inputs['stage']!=('update' if current(service,db,c)[0]['active'] else 'initial'):
            raise ValueError('Requested phase conflicts with actual active project version')
        return next_step(service,db,c)
    if event=='cancel_batch' or lower in ('取消本次导入','取消批次导入','cancel batch import'):
        from .batch_flow import cancel
        return cancel(service,db,c,inputs)
    if not event and c.get('project') and not ctx.get('await_reason') and not re.match(r'^(?:采用|启用|保留|回滚|adopt|promote|hold|retain|rollback)',lower):
        request=folder_request(text,ctx.get('await_role'))
        if request:
            return import_folder(service,db,c,request)
    if ctx.get('await_name') and not event and lower in ('下一步','继续','查看状态','帮助','next','continue','status','help'):
        return words('还需要你填写项目名称；“下一步”和“查看状态”不会作为项目名称。',
                     'Please provide the project name; next/status commands are not used as names.'),{'kind':'create_project'}
    if event=='create_project' or ctx.get('await_name') or re.search(r'(?:创建|新建)(?:一个)?(?:项目|工程)|create (?:a )?(?:new )?project',text,re.I):
        match=re.search(r'(?:创建|新建)(?:一个)?(?:项目|工程)\s*[:：]?\s*(.*)|create (?:a )?(?:new )?project\s*[:：]?\s*(.*)',text,re.I)
        name=inputs.get('name') or (text if ctx.get('await_name') else next((g for g in match.groups() if g),''))
        name=str(name).strip(' \"“”')
        if not name:
            ctx['await_name']=True
            return words('这个项目叫什么名字？','What is the project name?'),{'kind':'create_project'}
        if c.get('project'):
            raise ValueError('Use New conversation / project to create a separate project')
        c['project']=service.action_create_project(db,{'name':name})['id']
        ctx.pop('await_name',None)
        reply,card=next_step(service,db,c)
        return words('项目已创建。'+reply['zh'],'Project created. '+reply['en']),card
    if not c.get('project'):
        return next_step(service,db,c)
    if event=='confirm' or lower in ('确认','确认启动','确认采用','确认保留','确认取消','确认回滚','确认导入','confirm','confirm start','confirm adopt','confirm hold','confirm cancel','confirm rollback','confirm import'):
        return confirm(service,db,c,inputs,lower)
    if event=='import_folder':
        return import_folder(service,db,c,inputs)
    if event=='import_batch':
        from .batch_flow import propose
        return propose(service,db,c,inputs)
    if event=='template':
        reply,card=next_step(service,db,c)
        return words('已准备标注模板：filename、三类缺陷列和no_defect。空白保持未标注，no_defect=1时三类必须明确为0。'+reply['zh'],
                     'Annotation template prepared: filename, three defect columns and no_defect. Blank stays unannotated; no_defect=1 requires all three labels explicitly0. '+reply['en']),card
    if event=='annotation_save':
        save_draft(service,db,c,inputs['draft'],inputs['rows'])
        return next_step(service,db,c)
    if lower in ('取消标注','cancel annotation'):
        if ctx.get('batch_id'):
            from .batch_flow import cancel
            return cancel(service,db,c,{})
        ctx.pop('draft',None)
        return next_step(service,db,c)
    if lower in ('按未知标签导入','保留未知标签导入','import with unknown labels'):
        draft=service.store.get(db,'draft',ctx['draft'])
        save_draft(service,db,c,draft['id'],draft['rows'])
        return next_step(service,db,c)
    if event=='configure':
        return configure(service,db,c,inputs)
    if event=='propose_decision':
        return propose_decision(service,db,c,inputs)
    if event=='reason':
        if not ctx.get('await_reason') or inputs.get('round') and inputs['round']!=ctx['await_reason'].get('round'):
            raise ValueError('Reason card is stale; choose a current candidate again')
        return propose_decision(service,db,c,dict(ctx.get('await_reason',{}),reason=inputs.get('reason') or text))
    interrupt=lower in ('下一步','继续','帮助','next','continue','help','取消本轮','cancel round') or re.match(r'^(?:查看|查询|显示|比较|状态|进度|采用|启用|保留|回滚|status\b|progress\b|show\b|check\b|compare\b|results\b|adopt\b|promote\b|hold\b|retain\b|rollback\b)',lower)
    if ctx.get('await_reason') and not event and not interrupt:
        return propose_decision(service,db,c,dict(ctx['await_reason'],reason=text))
    if event=='rollback' or lower.startswith(('回滚','rollback')):
        p=service.store.get(db,'project',c['project'])
        versions=service.store.all(db,'version',p['id'])
        if not inputs.get('version') or not inputs.get('reason'):
            return words('请选择一个曾采用版本并说明回滚理由；Hold只保留本轮父模型，与回滚分开。',
                         'Choose a previously adopted version and give a rollback reason. Hold only retains the round parent; rollback is separate.'),{'kind':'rollback','versions':[{k:v.get(k) for k in ('id','reason','adopted')} for v in versions]}
        version=service.store.get(db,'version',inputs['version'])
        if version['project']!=p['id']:
            raise ValueError('Rollback version belongs to another project')
        summary=words('准备回滚至'+version['id']+'；理由：'+inputs['reason']+'。确认后才修改实际在役指针。',
                      'Ready to roll back to '+version['id']+'; reason: '+inputs['reason']+'. Confirm to change the actual active version.')
        return summary,pending(c,'rollback',dict(project=p['id'],version=version['id'],reason=inputs['reason'],expected_active=p['active']),summary)
    if lower in ('取消本轮','cancel round'):
        _,_,rnd=current(service,db,c)
        if not rnd:
            raise ValueError('No open round to cancel')
        summary=words('准备取消本轮的未运行任务；在运行的Worker不会被聊天强行终止。确认取消后不累计本轮Gate。',
                      'Ready to cancel queued tasks in this round. A running Worker will not be forcibly stopped. Cancelled Gate data is not added to cumulative evaluation.')
        return summary,pending(c,'cancel',dict(round=rnd['id'],reason='Engineer cancelled through conversation'),summary)
    if re.match(r'^(采用|启用|adopt|promote)',lower) or re.match(r'^(保留|hold|retain)',lower):
        choice='hold' if re.match(r'^(保留|hold|retain)',lower) else 'adopt'
        method=adoption_candidate(text) if choice=='adopt' else None
        reason=decision_reason(text)
        return propose_decision(service,db,c,dict(choice=choice,job=method,reason=reason.group(1).strip() if reason else ''))
    if any(x in lower for x in ('比较','结果','compare','results','哪个最好')):
        return comparison(service,db,c)
    if any(x in lower for x in ('状态','进度','好了','status','progress')):
        return status_reply(service,db,c)
    aliases={'resnet50':('resnet50','resnet-50'),'convnext_tiny':('convnext-tiny','convnext_tiny'),'swin_t':('swin-t','swin_t')}
    backbone=next((b for b,names in aliases.items() if any(x in lower for x in names)),None)
    tools=mentioned_methods(text)
    if backbone or tools or lower in ('开始训练','准备训练','开始维护','维护模型','初始建模','train','start training','maintain model'):
        opts={}
        if backbone:
            opts['backbone']=backbone
        if tools:
            opts['tools']=entry_tools(tools,'update' if service.store.get(db,'project',c['project'])['active'] else 'initial')
        ctx.setdefault('options',{}).update(opts)
        selected=ctx.get('selected',{})
        if selected.get('train') and selected.get('evaluation'):
            return configure(service,db,c,opts)
        return next_step(service,db,c)
    if lower in ('下一步','继续','帮助','你好','next','continue','help','hello'):
        return next_step(service,db,c)
    reply,card=next_step(service,db,c)
    return words('我目前通过受控任务流程处理创建项目、导入、标注、训练、状态、比较和人工采用/保留。刚才的表述没有执行任何模型操作。'+reply['zh'],
                 'I use a controlled workflow for projects, intake, annotation, training, status, comparisons and engineer adoption/Hold. That message did not execute a model action. '+reply['en']),card
