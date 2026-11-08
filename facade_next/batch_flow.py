"""Chat orchestration of one pre-split batch, including partial annotations."""
from .contracts import digest
from .batch_intake import prepare,inspect_part


def card(c,batch):
    from .conversation import pending
    value=pending(c,'batch',dict(batch=batch['id'],expected_active=batch['expected_active'],selected_digest=batch['selected_digest']),batch['summary'])
    return dict(value,parts=batch['parts'],stage=batch['stage'],name=batch['name'])


def propose(service,db,c,inputs):
    from .conversation import current,words
    from .service import uid
    p,_,rnd=current(service,db,c)
    ctx=c['context']
    if rnd or ctx.get('draft') or any(ctx.get('selected',{}).values()) or ctx.get('batch_queue'):
        raise ValueError('Finish the current round/intake before submitting a whole new batch')
    if ctx.get('pending'):
        raise ValueError('Confirm or cancel the current proposal before submitting another batch')
    stage='update' if p['active'] else 'initial'
    if inputs.get('stage') and inputs['stage']!=stage:
        raise ValueError('Batch stage conflicts with the project active version; do not infer initial/update from a folder name')
    plan=prepare(inputs['root'],stage)
    name=str(inputs.get('name') or plan['name']).strip()
    names={'train':'Train','val':'Val','evaluation':'Gate' if stage=='update' else 'S0'}
    details='；'.join(names[x['role']]+' ('+str(x['count'])+')' for x in plan['parts'])
    summary=words(f'这批“{name}”按项目状态识别为{"维护轮" if stage=="update" else "初始建模"}数据。已有划分：{details}。请核对后确认分别导入；缺失标注会在聊天中依次补充。不会自动划分或启动训练。',
                  f'Batch “{name}” is for {stage} modeling from actual project state. Existing subsets: {details}. Confirm the mapping to import each subset; partial labels continue in chat. No splitting or training is started.')
    batch=dict(plan,id=uid('batch'),project=c['project'],name=name,status='awaiting_confirmation',datasets={},
               expected_active=p['active'],selected_digest=digest(ctx.get('selected',{})),summary=summary)
    service.store.put(db,'batch',batch)
    return summary,card(c,batch)


def confirm(service,db,c,proposal):
    from .conversation import current,next_step
    args=proposal['arguments'];batch=service.store.get(db,'batch',args['batch'])
    p,_,rnd=current(service,db,c)
    if batch['project']!=c['project'] or batch['status']!='awaiting_confirmation' or rnd:
        raise ValueError('Batch proposal is stale or belongs to another task')
    if p['active']!=args['expected_active'] or digest(c['context'].get('selected',{}))!=args['selected_digest']:
        raise ValueError('Batch task/active version changed; submit this batch again')
    if prepare(batch['root'],batch['stage'])['signature']!=batch['signature']:
        raise ValueError('Batch photos, labels or folder mapping changed after preview; cancel and resubmit')
    c['context'].pop('pending',None)
    c['context']['batch_id']=batch['id'];c['context']['batch_queue']=batch['parts'][:]
    batch['status']='importing';service.store.put(db,'batch',batch)
    return next_step(service,db,c)


def advance(service,db,c):
    from .conversation import import_folder,next_step,words
    ctx=c['context'];batch=service.store.get(db,'batch',ctx['batch_id'])
    p=service.store.get(db,'project',c['project'])
    if p['active']!=batch['expected_active']:raise ValueError('Active version changed during batch intake')
    queue=ctx.get('batch_queue',[])
    if not queue:
        batch['status']='imported';service.store.put(db,'batch',batch)
        ctx.pop('batch_id',None);ctx.pop('batch_queue',None)
        reply,card=next_step(service,db,c)
        counts=', '.join(x['role']+' '+str(x['count']) for x in batch['parts'])
        return words('批次“'+batch['name']+'”已按确认的划分导入：'+counts+'。'+reply['zh'],
                     'Batch “'+batch['name']+'” imported using the confirmed roles: '+counts+'. '+reply['en']),card
    part=queue[0]
    if inspect_part(part['folder'],part['role'])[0]['signature']!=part['signature']:
        raise ValueError('Pending batch subset changed; cancel this intake and resubmit')
    ctx['batch_queue']=queue[1:]
    return import_folder(service,db,c,dict(root=part['folder'],role=part['role'],name=batch['name']+' · '+part['role'],_batch=batch['id']))


def cancel(service,db,c,inputs):
    from .conversation import next_step
    ctx=c['context'];proposal=ctx.get('pending',{})
    if inputs.get('token') and inputs['token']!=proposal.get('token'):
        raise ValueError('Batch cancellation card is stale')
    id=proposal.get('arguments',{}).get('batch') if proposal.get('kind')=='batch' else ctx.get('batch_id')
    if not id:raise ValueError('No batch intake is pending')
    batch=service.store.get(db,'batch',id)
    if batch['project']!=c['project']:raise ValueError('Batch belongs to another project')
    selected=ctx.setdefault('selected',{})
    for role,dataset in batch['datasets'].items():
        selected[role]=[x for x in selected.get(role,[]) if x!=dataset]
        if not selected[role]:selected.pop(role,None)
    batch['status']='cancelled';service.store.put(db,'batch',batch)
    if proposal.get('kind')=='batch':ctx.pop('pending',None)
    for key in ('batch_id','batch_queue','draft'):ctx.pop(key,None)
    return next_step(service,db,c)
