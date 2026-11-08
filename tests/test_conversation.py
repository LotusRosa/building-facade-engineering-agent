import unittest
from pathlib import Path
from test_logic import local_tmp
from facade_next.service import Service
from facade_next.intake import prepare_folder
from facade_next.contracts import digest,sha
from facade_next.metrics import scene_metrics


class IntakeTests(unittest.TestCase):
    def test_legacy_filename_no_defect_and_partial_folder(self):
        with local_tmp() as d:
            root=Path(d)
            (root/'images').mkdir()
            (root/'images/a.jpg').write_bytes(b'a')
            (root/'images/b.jpg').write_bytes(b'b')
            (root/'labels.csv').write_text('filename,hollow,spalling,crack,no_defect\na.jpg,0,0,0,1\n',encoding='utf-8')
            x=prepare_folder(root)
            self.assertEqual(x['count'],2)
            self.assertEqual(x['unknown_images'],1)
            self.assertEqual(x['rows'][0]['hollow'],0)
            self.assertIsNone(x['rows'][1]['hollow'])

    def test_no_defect_conflict_rejected(self):
        with local_tmp() as d:
            root=Path(d)
            (root/'a.jpg').write_bytes(b'a')
            (root/'labels.csv').write_text('filename,hollow,spalling,crack,no_defect\na.jpg,1,0,0,1\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'no_defect'):
                prepare_folder(root)


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=local_tmp()
        self.root=Path(self.tmp.__enter__())
        self.s=Service(self.root/'state')
        self.conversation=None
        self.project=None
        self.n=0

    def tearDown(self):
        self.tmp.__exit__(None,None,None)

    def chat(self,text='',event=None,inputs=None):
        self.n+=1
        result=self.s.dispatch('chat',dict(conversation=self.conversation,project=self.project,text=text,event=event,inputs=inputs or {},language='zh'),str(self.n))
        self.conversation=result['conversation']
        self.project=result['project']
        return result

    def folder(self,name):
        p=self.root/name
        p.mkdir()
        for i in range(2):
            (p/(str(i)+'.jpg')).write_bytes((name+str(i)).encode())
        (p/'labels.csv').write_text('filename,hollow,spalling,crack,no_defect\n0.jpg,0,0,0,1\n1.jpg,1,1,1,0\n',encoding='utf-8')
        return str(p)

    def test_chat_prepares_and_confirms_real_job_without_double_count(self):
        r=self.chat('创建项目 外立面维护')
        self.assertTrue(self.project)
        for role in ('train','val','evaluation'):
            r=self.chat(event='import_folder',inputs=dict(root=self.folder(role),role=role,name=role))
        self.assertEqual(r['card']['kind'],'configure')
        r=self.chat(event='configure',inputs=dict(backbone='resnet50',tools=['standard'],seed=7))
        self.assertEqual(r['card']['kind'],'confirm_start')
        self.assertEqual(self.s.snapshot()['jobs'],[])
        before=len(self.s.snapshot()['events'])
        self.chat('确认启动')
        self.assertEqual(self.s.snapshot()['jobs'][0]['status'],'queued')
        self.assertEqual(len(self.s.snapshot()['events']),before+1)
        self.assertEqual(self.s.snapshot()['rounds'][0]['backbone'],'resnet50')

    def test_unknown_text_does_not_execute_and_chat_is_persisted(self):
        self.chat('创建项目 Chat project')
        r=self.chat('随便把最好的模型上线')
        self.assertFalse(self.s.snapshot()['versions'])
        self.assertTrue(r['reply']['zh'])
        with self.s.store.transaction() as db:
            c=self.s.store.get(db,'conversation',self.conversation)
        self.assertEqual(len(c['messages']),4)
        self.assertEqual(self.s.snapshot()['events'][-1]['project'],self.project)

    def test_chat_submission_id_is_idempotent(self):
        payload=dict(text='create project once',language='en',inputs={})
        a=self.s.dispatch('chat',payload,'same-chat')
        b=self.s.dispatch('chat',payload,'same-chat')
        self.assertEqual(a,b)
        self.assertEqual(len(self.s.snapshot()['projects']),1)
        self.assertEqual(len(self.s.snapshot()['events']),1)

    def test_status_resumes_pending_annotation(self):
        self.chat('创建项目 pending annotation')
        root=Path(self.folder('partial'))
        (root/'labels.csv').write_text('filename,hollow,spalling,crack,no_defect\n0.jpg,,0,0,\n1.jpg,1,1,1,0\n',encoding='utf-8')
        r=self.chat(event='import_folder',inputs=dict(root=str(root),role='train'))
        self.assertEqual(r['card']['kind'],'annotation')
        draft_id=r['card']['draft']['id']
        r=self.chat('查看状态')
        self.assertEqual(r['card']['kind'],'annotation')
        self.assertEqual(r['card']['draft']['id'],draft_id)

    def finish_fixture(self):
        j=self.s.claim('cuda:0')
        out=Path(j['output']);out.mkdir(parents=True)
        cp=out/'model.pt';cp.write_bytes(('inert chat test fixture '+j['id']).encode())
        parent=j['spec']['parent']
        contract=dict(schema='facade_next_v1',backbone=j['spec']['backbone'],classes=['hollow','spalling','crack'],image_size=1280,preprocessing='full1280_letterbox114_gpu_v1',thresholds=[.5]*3,pos_weight=[1]*3,threshold_status='frozen_on_val',checkpoint_sha256=sha(cp),job_digest=digest(j['spec']),roles_digest=j['spec']['roles_digest'],parent_sha256=parent['contract']['checkpoint_sha256'] if parent else None)
        ev=j['spec']['roles']['evaluation']
        metrics=scene_metrics(ev,[[.8 if y else .2 for y in r['labels']] for r in ev],[.5]*3)
        self.s.finish(j['id'],dict(status='ready',job_digest=digest(j['spec']),evaluation_digest=digest(ev),checkpoint=str(cp),contract=contract,artifacts={'model.pt':sha(cp)},comparison={'candidate':metrics,'parent':metrics if parent else None}))

    def test_full_chat_adopt_then_hold_uses_actual_parent(self):
        self.chat('创建项目 chat lineage')
        for role in ('train','val','evaluation'):
            self.chat(event='import_folder',inputs=dict(root=self.folder(role),role=role))
        self.chat(event='configure',inputs=dict(backbone='resnet50',tools=['standard']))
        self.chat('确认启动');self.finish_fixture()
        self.assertEqual(self.chat('查看结果')['card']['kind'],'comparison')
        self.assertEqual(self.chat('采用 standard，因为检查了各类结果')['card']['kind'],'confirm_adopt')
        self.assertIsNone(self.s.snapshot()['projects'][0]['active'])
        self.chat('确认采用')
        active=self.s.snapshot()['projects'][0]['active']
        for role in ('train','evaluation'):
            self.chat(event='import_folder',inputs=dict(root=self.folder('new_'+role),role=role))
        self.chat('使用 head_only')
        rnd=self.s.snapshot()['rounds'][-1]
        self.assertEqual(rnd['parent']['id'],active)
        self.assertEqual(rnd['backbone'],'resnet50')
        self.chat('确认启动');self.finish_fixture()
        self.assertEqual(self.chat('保留，因为优先历史场景')['card']['kind'],'confirm_hold')
        self.chat('确认保留')
        project=self.s.snapshot()['projects'][0]
        self.assertEqual(project['active'],active)
        self.assertEqual(len(project['eval_pool']),4)
        self.assertTrue(all(e['action']=='chat' and e['O']==1 for e in self.s.snapshot()['events']))

    def test_stale_confirmation_after_panel_change_is_not_executed(self):
        self.chat('创建项目 stale chat')
        for role in ('train','val','evaluation'):
            self.chat(event='import_folder',inputs=dict(root=self.folder(role),role=role))
        proposed=self.chat(event='configure',inputs=dict(backbone='resnet50',tools=['standard']))
        rnd=self.s.snapshot()['rounds'][0]
        self.s.dispatch('cancel_round',{'round':rnd['id'],'reason':'panel cancelled'},'panel-once')
        result=self.chat(event='confirm',inputs={'token':proposed['card']['token']})
        self.assertIn('未执行',result['reply']['zh'])
        self.assertEqual(self.s.snapshot()['jobs'],[])

    def test_changed_photo_cannot_keep_old_annotation(self):
        self.chat('创建项目 photo integrity')
        root=Path(self.folder('partial'))
        (root/'labels.csv').write_text('filename,hollow,spalling,crack,no_defect\n0.jpg,,0,0,\n1.jpg,1,1,1,0\n',encoding='utf-8')
        result=self.chat(event='import_folder',inputs=dict(root=str(root),role='train'))
        draft=result['card']['draft']
        (root/'0.jpg').write_bytes(b'changed after preview')
        result=self.chat(event='annotation_save',inputs=dict(draft=draft['id'],rows=draft['rows']))
        self.assertIn('Photo changed',result['reply']['zh'])
        self.assertEqual(self.s.snapshot()['datasets'],[])

    def ready_initial_result(self):
        self.chat('创建项目 reaudit')
        for role in ('train','val','evaluation'):
            self.chat(event='import_folder',inputs=dict(root=self.folder(role),role=role))
        self.chat(event='configure',inputs=dict(backbone='resnet50',tools=['standard']))
        self.chat('确认启动');self.finish_fixture()

    def test_paths_keywords_do_not_override_explicit_train_role(self):
        self.chat('创建项目 路径核对')
        path=self.folder('results_val_safety_train')
        self.chat('训练数据在 "'+path+'"')
        datasets=self.s.snapshot()['datasets']
        self.assertEqual(len(datasets),1)
        with self.s.store.transaction() as db:
            c=self.s.store.get(db,'conversation',self.conversation)
        self.assertTrue(c['context']['selected'].get('train'))
        self.assertFalse(c['context']['selected'].get('evaluation'))

    def test_status_is_not_used_as_pending_engineer_reason(self):
        self.ready_initial_result()
        self.assertEqual(self.chat('采用 standard')['card']['kind'],'reason')
        result=self.chat('查看状态')
        self.assertNotEqual(result['card']['kind'],'confirm_adopt')
        with self.s.store.transaction() as db:
            c=self.s.store.get(db,'conversation',self.conversation)
        self.assertNotIn('pending',c['context'])
        self.assertIn('await_reason',c['context'])

    def test_next_while_awaiting_name_does_not_create_project_called_next(self):
        self.chat('创建项目')
        self.chat('下一步')
        self.assertEqual(self.s.snapshot()['projects'],[])

    def test_old_hold_card_cannot_prepare_decision_for_a_new_round(self):
        self.ready_initial_result()
        old=self.s.snapshot()['rounds'][0]['id']
        self.chat('采用 standard，因为初始确认');self.chat('确认采用')
        for role in ('train','evaluation'):
            self.chat(event='import_folder',inputs=dict(root=self.folder('new_'+role),role=role))
        self.chat('使用 standard');self.chat('确认启动');self.finish_fixture()
        result=self.chat(event='propose_decision',inputs=dict(round=old,choice='hold',reason='from old card'))
        self.assertIn('stale',result['reply']['en'].lower())
        self.assertNotEqual(result['card']['kind'],'confirm_hold')

    def test_pending_rollback_does_not_override_a_later_panel_adoption(self):
        self.ready_initial_result()
        self.chat('采用 standard，因为初始化');self.chat('确认采用')
        first=self.s.snapshot()['projects'][0]['active']
        for role in ('train','evaluation'):
            self.chat(event='import_folder',inputs=dict(root=self.folder('new_'+role),role=role))
        proposed=self.chat(event='rollback',inputs=dict(version=first,reason='prepared before panel change'))
        with self.s.store.transaction() as db:
            selected=self.s.store.get(db,'conversation',self.conversation)['context']['selected']
        rnd=self.s.dispatch('create_round',dict(project=self.project,**selected,backbone='resnet50',tools=['standard']), 'panel-create')['id']
        job=self.s.dispatch('start_round',{'round':rnd},'panel-start')['jobs'][0]
        self.finish_fixture()
        second=self.s.dispatch('decide',dict(round=rnd,job=job,reason='later panel adoption'),'panel-decide')['active']
        result=self.chat(event='confirm',inputs={'token':proposed['card']['token']})
        self.assertIn('stale',result['reply']['en'].lower())
        self.assertEqual(self.s.snapshot()['projects'][0]['active'],second)
        resumed=self.chat('下一步')
        self.assertEqual(resumed['card']['kind'],'folder')
        self.assertEqual(resumed['card']['role'],'train')

    def test_three_backbone_parents_can_prepare_all_eight_tools_without_model_execution(self):
        from facade_next.contracts import BACKBONES,METHODS
        rounds=[]
        for backbone in BACKBONES:
            self.conversation=None;self.project=None
            self.chat('创建项目 '+backbone)
            for role in ('train','val','evaluation'):
                self.chat(event='import_folder',inputs=dict(root=self.folder(backbone+'_'+role),role=role))
            self.chat(event='configure',inputs=dict(backbone=backbone,tools=['standard']))
            self.chat('确认启动');self.finish_fixture()
            self.chat('采用 standard，因为协议测试');self.chat('确认采用')
            for role in ('train','evaluation'):
                self.chat(event='import_folder',inputs=dict(root=self.folder('new_'+backbone+'_'+role),role=role))
            self.chat(event='configure',inputs=dict(backbone=backbone,tools=METHODS))
            rnd=self.s.snapshot()['rounds'][-1]
            rounds.append((backbone,rnd['id']))
            self.assertEqual(rnd['parent']['contract']['backbone'],backbone)
        for backbone,rid in rounds:
            jobs=self.s.dispatch('start_round',{'round':rid},'plan_all_'+backbone)['jobs']
            self.assertEqual(len(jobs),8)
            actual=[j for j in self.s.snapshot()['jobs'] if j['id'] in jobs]
            self.assertEqual({j['method'] for j in actual},set(METHODS))
            self.assertTrue(all(j['status']=='queued' and j['spec']['parent']['contract']['backbone']==backbone for j in actual))


class DecisionSelectionTests(unittest.TestCase):
    """Decision routing fixtures contain metadata only, without model files."""
    chat=ConversationTests.chat
    tearDown=ConversationTests.tearDown

    def setUp(self):
        ConversationTests.setUp(self)
        self.chat('create project decision selection')
        self.parent='version_parent'
        self.round='round_selection'
        self.contract=dict(checkpoint_sha256='metadata-only-parent',thresholds=[.5]*3)
        roles=dict(train=[],history=[],val=[],evaluation=[dict(image_id='evaluation_photo')])
        parent=dict(id=self.parent,contract=self.contract)
        with self.s.store.transaction() as db:
            project=self.s.store.get(db,'project',self.project)
            project['active']=self.parent
            self.s.store.put(db,'project',project)
            self.s.store.put(db,'round',dict(id=self.round,project=self.project,number=1,
                kind='update',status='submitted',parent=parent,backbone='resnet50',roles=roles,tools=['standard','sam'],
                jobs=['job_standard','job_sam']))
            for method in ('standard','sam'):
                self.s.store.put(db,'job',dict(id='job_'+method,project=self.project,
                    round=self.round,method=method,status='succeeded',spec=dict(roles=roles,parent=parent),
                    result=dict(status='ready',evaluation_digest=digest(roles['evaluation']),
                        contract=dict(self.contract,checkpoint_sha256='metadata-only-'+method),
                        comparison=dict(candidate={'merged':{'images':1}},parent={'merged':{'images':1}}))))

    def context(self):
        with self.s.store.transaction() as db:
            return self.s.store.get(db,'conversation',self.conversation)['context']

    def test_reselecting_candidate_invalidates_previous_confirmation_until_new_reason(self):
        old=self.chat('adopt standard because historical results are stable')
        result=self.chat('adopt SAM')
        self.assertEqual(result['card']['kind'],'reason')
        self.assertEqual(result['card']['round'],self.round)
        self.assertNotIn('pending',self.context())
        result=self.chat('confirm')
        self.assertIn('No action is awaiting confirmation',result['reply']['en'])
        self.assertEqual(self.s.snapshot()['projects'][0]['active'],self.parent)
        result=self.chat(event='confirm',inputs={'token':old['card']['token']})
        self.assertIn('stale',result['reply']['en'])
        result=self.chat('new scene performance is acceptable')
        self.assertEqual(result['card']['kind'],'confirm_adopt')
        self.assertEqual(result['card']['arguments']['job'],'job_sam')
        self.assertEqual(result['card']['arguments']['reason'],'new scene performance is acceptable')

    def test_switching_to_hold_invalidates_the_previous_adoption_confirmation(self):
        self.chat('adopt standard because historical results are stable')
        result=self.chat('hold')
        self.assertEqual(result['card']['kind'],'reason')
        self.assertNotIn('pending',self.context())
        result=self.chat('confirm')
        self.assertIn('No action is awaiting confirmation',result['reply']['en'])
        result=self.chat('historical performance matters')
        self.assertEqual(result['card']['kind'],'confirm_hold')
        self.assertIsNone(result['card']['arguments']['job'])

    def test_rejected_initial_hold_cannot_restore_an_earlier_adoption_confirmation(self):
        with self.s.store.transaction() as db:
            project=self.s.store.get(db,'project',self.project)
            project['active']=None
            self.s.store.put(db,'project',project)
            rnd=self.s.store.get(db,'round',self.round)
            rnd.update(kind='initial',parent=None,tools=['standard'],jobs=['job_standard'])
            self.s.store.put(db,'round',rnd)
            job=self.s.store.get(db,'job','job_standard')
            job['spec']['parent']=None
            self.s.store.put(db,'job',job)
        old=self.chat('adopt standard because initial results are acceptable')
        result=self.chat('保留，因为需要重新检查')
        self.assertIn('no active initial model',result['reply']['en'])
        self.assertNotIn('pending',self.context())
        self.assertEqual(result['card']['kind'],'comparison')
        result=self.chat('confirm')
        self.assertIn('No action is awaiting confirmation',result['reply']['en'])
        result=self.chat(event='confirm',inputs={'token':old['card']['token']})
        self.assertIn('stale',result['reply']['en'])
        self.assertIsNone(self.s.snapshot()['projects'][0]['active'])
        old=self.chat('adopt standard because initial results are acceptable')
        result=self.chat(event='propose_decision',inputs=dict(choice='hold',reason='recheck',round=self.round))
        self.assertIn('no active initial model',result['reply']['en'])
        self.assertNotIn('pending',self.context())
        result=self.chat(event='confirm',inputs={'token':old['card']['token']})
        self.assertIn('stale',result['reply']['en'])

    def test_ambiguous_or_negated_method_adoption_never_chooses_by_alias_order(self):
        messages=['adopt SAM, not standard because historical performance matters',
                  'adopt standard or SAM because both are ready',
                  'adopt job_sam or standard because both are ready',
                  '采用 job_sam 或 标准训练，因为都已完成',
                  'adopt not SAM because its old scene regressed',
                  '采用 SAM，不采用标准更新，因为历史场景优先',
                  '采用标准训练或者锐度感知优化，因为都已完成']
        for text in messages:
            with self.subTest(text=text):
                self.chat('adopt standard because the old choice was acceptable')
                result=self.chat(text)
                self.assertEqual(result['card']['kind'],'comparison')
                self.assertNotIn('pending',self.context())
                self.assertIsNone(self.context().get('await_reason'))

    def test_llm_proposal_cannot_choose_a_method_from_ambiguous_adoption_text(self):
        from facade_next.llm import LLMClient
        from test_llm import tool
        old=self.chat('adopt standard because the earlier choice was acceptable')
        self.s.llm=LLMClient(environ={},transport=lambda *args:tool('propose_decision',
            choice='adopt',job='standard',reason='both are ready'))
        self.s.llm.configure(dict(provider='openai',model='fake',api_key='fake-key'))
        result=self.chat('adopt SAM or standard because both are ready')
        self.assertIn('not executed',result['reply']['en'])
        self.assertNotIn('pending',self.context())
        self.assertEqual(result['card']['kind'],'comparison')
        result=self.chat(event='confirm',inputs={'token':old['card']['token']})
        self.assertIn('stale',result['reply']['en'])
        self.assertEqual(self.s.snapshot()['projects'][0]['active'],self.parent)

    def test_names_in_the_reason_do_not_change_the_requested_candidate(self):
        result=self.chat('adopt SAM because standard regressed')
        self.assertEqual(result['card']['kind'],'confirm_adopt')
        self.assertEqual(result['card']['arguments']['job'],'job_sam')
        self.assertEqual(result['card']['arguments']['reason'],'standard regressed')

    def test_new_llm_choice_invalidates_prior_confirmation_even_on_provider_failure(self):
        from facade_next.llm import LLMClient
        def fail(*args):raise TimeoutError('offline fake transport')
        self.s.llm=LLMClient(environ={},transport=fail)
        self.s.llm.configure(dict(provider='deepseek',model='fake',api_key='fake-key'))
        for text in ('hold','please hold','请帮我保留父模型','我决定保留父模型','不要采用standard'):
            with self.subTest(text=text):
                old=self.chat(event='propose_decision',inputs=dict(choice='adopt',job='job_standard',reason='earlier choice',round=self.round))
                result=self.chat(text)
                self.assertIn('not executed',result['reply']['en'])
                self.assertNotIn('pending',self.context())
                self.assertEqual(result['card']['kind'],'comparison')
                result=self.chat(event='confirm',inputs={'token':old['card']['token']})
                self.assertIn('stale',result['reply']['en'])

    def test_llm_reason_only_reply_keeps_the_selected_candidate(self):
        from facade_next.llm import LLMClient
        from test_llm import tool
        result=self.chat('adopt SAM')
        self.assertEqual(result['card']['kind'],'reason')
        self.s.llm=LLMClient(environ={},transport=lambda *args:tool('propose_decision',choice='adopt',job='job_sam',reason='historical performance matters'))
        self.s.llm.configure(dict(provider='deepseek',model='fake',api_key='fake-key'))
        result=self.chat('historical performance matters')
        self.assertEqual(result['card']['kind'],'confirm_adopt')
        self.assertEqual(result['card']['arguments']['job'],'job_sam')
        self.assertEqual(result['card']['arguments']['reason'],'historical performance matters')

    def test_local_queries_and_unrelated_llm_errors_preserve_the_current_decision(self):
        from facade_next.llm import LLMClient
        old=self.chat('adopt standard because the earlier choice was acceptable')
        def fail(*args):raise TimeoutError('offline fake transport')
        self.s.llm=LLMClient(environ={},transport=fail)
        self.s.llm.configure(dict(provider='openai',model='fake',api_key='fake-key'))
        for text in ('status','compare results','hello'):
            with self.subTest(text=text):
                previous=next(c for c in self.s.snapshot()['conversations'] if c['id']==self.conversation)['card']
                result=self.chat(text)
                if text=='hello':
                    self.assertIn('not executed',result['reply']['en'])
                    self.assertEqual(result['card'],previous)
                else:
                    self.assertNotIn('not executed',result['reply']['en'])
                if text=='status':self.assertEqual(result['card']['token'],old['card']['token'])
                self.assertEqual(self.context()['pending']['token'],old['card']['token'])

    def test_exact_candidate_card_identity_and_unambiguous_aliases_remain_supported(self):
        result=self.chat(event='propose_decision',inputs=dict(
            choice='adopt',job='job_sam',reason='reviewed this exact candidate',round=self.round))
        self.assertEqual(result['card']['arguments']['job'],'job_sam')
        result=self.chat('adopt Sharpness-aware optimization (SAM) because suitable')
        self.assertEqual(result['card']['kind'],'confirm_adopt')
        self.assertEqual(result['card']['arguments']['job'],'job_sam')
        result=self.chat('adopt job_sam (SAM) because reviewed this exact candidate')
        self.assertEqual(result['card']['kind'],'confirm_adopt')
        self.assertEqual(result['card']['arguments']['job'],'job_sam')
