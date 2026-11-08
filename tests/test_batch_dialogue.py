import json
import unittest
from pathlib import Path
from test_logic import local_tmp
from test_llm import tool
from facade_next.service import Service
from facade_next.llm import LLMClient


class BatchDialogueTests(unittest.TestCase):
    def setUp(self):
        self.tmp=local_tmp();self.root=Path(self.tmp.__enter__());self.s=Service(self.root/'state')
        self.n=0;self.chat_id=None;self.project=None
        self.chat(event='create_project',inputs={'name':'批次项目'})

    def tearDown(self):self.tmp.__exit__(None,None,None)

    def chat(self,text='',event=None,inputs=None,sid=None):
        self.n+=1
        r=self.s.dispatch('chat',dict(text=text,event=event,inputs=inputs or {},conversation=self.chat_id,project=self.project,language='zh'),sid or str(self.n))
        self.chat_id=r['conversation'];self.project=r['project'];return r

    def batch(self,name='首轮',roles=('train','val','safety'),partial=False):
        root=self.root/name;root.mkdir()
        for role in roles:
            p=root/role;p.mkdir()
            for i in range(2):(p/(str(i)+'.jpg')).write_bytes((name+role+str(i)).encode())
            label=',0,0,' if partial and role=='train' else '0,0,0,1'
            (p/'labels.csv').write_text('filename,hollow,spalling,crack,no_defect\n0.jpg,'+label+'\n1.jpg,1,1,1,0\n',encoding='utf-8')
        return root

    def test_initial_batch_has_explicit_mapping_before_any_import(self):
        root=self.batch()
        r=self.chat('这是初始数据，放在“'+str(root)+'”')
        self.assertEqual(r['card']['kind'],'confirm_batch')
        self.assertEqual(r['card']['stage'],'initial')
        self.assertEqual({x['role'] for x in r['card']['parts']},{'train','val','evaluation'})
        self.assertEqual(self.s.snapshot()['datasets'],[])
        count=len(self.s.snapshot()['events'])
        self.chat(event='confirm',inputs={'token':r['card']['token']})
        snap=self.s.snapshot();self.assertEqual(len(snap['datasets']),3);self.assertEqual(snap['jobs'],[])
        self.assertEqual(len(snap['events']),count+1)
        self.assertTrue(all(d['batch_name']=='首轮' and d['batch_stage']=='initial' for d in snap['datasets']))

    def test_parent_via_single_folder_card_cannot_be_mixed_into_train(self):
        r=self.chat(event='import_folder',inputs=dict(root=str(self.batch()),role='train'))
        self.assertEqual(r['card']['kind'],'confirm_batch');self.assertEqual(self.s.snapshot()['datasets'],[])

    def test_missing_or_ambiguous_subsets_do_not_scan_as_one_dataset(self):
        for i,roles in enumerate([('train','safety'),('train','training','val','safety')]):
            r=self.chat('初始数据在“'+str(self.batch('missing'+str(i),roles))+'”')
            self.assertIn('未执行',r['reply']['zh']);self.assertEqual(self.s.snapshot()['datasets'],[])

    def test_changed_source_or_changed_parent_reject_confirmation(self):
        root=self.batch();r=self.chat(event='import_batch',inputs={'root':str(root)})
        (root/'train/0.jpg').write_bytes(b'changed')
        r=self.chat(event='confirm',inputs={'token':r['card']['token']})
        self.assertIn('未执行',r['reply']['zh']);self.assertEqual(self.s.snapshot()['datasets'],[])
        self.chat(event='cancel_batch')
        r=self.chat(event='import_batch',inputs={'root':str(root)})
        with self.s.store.transaction() as db:
            p=self.s.store.get(db,'project',self.project);p['active']='new-active-pointer';self.s.store.put(db,'project',p)
        r=self.chat(event='confirm',inputs={'token':r['card']['token']})
        self.assertIn('未执行',r['reply']['zh']);self.assertEqual(self.s.snapshot()['datasets'],[])

    def test_partial_labels_pause_and_continue_the_batch(self):
        r=self.chat(event='import_batch',inputs={'root':str(self.batch(partial=True))})
        r=self.chat(event='confirm',inputs={'token':r['card']['token']})
        self.assertEqual(r['card']['kind'],'annotation')
        draft=r['card']['draft'];rows=draft['rows'];rows[0]['hollow']=0
        r=self.chat(event='annotation_save',inputs={'draft':draft['id'],'rows':rows})
        self.assertEqual(len(self.s.snapshot()['datasets']),3);self.assertEqual(r['card']['kind'],'configure')

    def test_maintenance_uses_project_state_not_the_batch_name_and_val_is_optional(self):
        with self.s.store.transaction() as db:
            p=self.s.store.get(db,'project',self.project);p['active']='adopted-version';p['val_pool']=[{'image_id':'old'}];self.s.store.put(db,'project',p)
        r=self.chat('新一轮维护开始')
        self.assertTrue(r['card']['batch']);self.assertEqual(r['card']['stage'],'update')
        r=self.chat(event='import_batch',inputs={'root':str(self.batch('也叫首轮',('train','gate')))})
        self.assertEqual(r['card']['stage'],'update');self.assertEqual(len(r['card']['parts']),2)
        self.chat(event='cancel_batch')
        r=self.chat(event='import_batch',inputs={'root':str(self.batch('下一批',('train','val','test'))),'stage':'update'})
        self.assertEqual(len(r['card']['parts']),3)

    def test_cross_role_duplicate_photos_are_rejected(self):
        root=self.batch();(root/'val/0.jpg').write_bytes((root/'train/0.jpg').read_bytes())
        r=self.chat(event='import_batch',inputs={'root':str(root)})
        self.assertIn('Role overlap',r['reply']['zh']);self.assertEqual(self.s.snapshot()['datasets'],[])

    def test_cancel_and_status_preserve_confirmation_identity(self):
        r=self.chat(event='import_batch',inputs={'root':str(self.batch())})
        token=r['card']['token'];r=self.chat('查看状态')
        self.assertEqual(r['card']['token'],token)
        r=self.chat(event='cancel_batch',inputs={'token':token})
        self.assertEqual(self.s.snapshot()['datasets'],[])
        r=self.chat(event='confirm',inputs={'token':token})
        self.assertIn('stale',r['reply']['en'])

    def test_llm_proposes_one_batch_and_retains_its_phase_in_context(self):
        root=self.batch('第一批资料');calls=[]
        def transport(*args):calls.append(args);return tool('import_batch',root=str(root),stage='initial')
        client=LLMClient(environ={},transport=transport);client.configure(dict(provider='openai',model='fake',api_key='fake-batch-key'));self.s.llm=client
        r=self.chat('这份是初始建模的数据：“'+str(root)+'”')
        self.assertEqual(r['card']['kind'],'confirm_batch')
        self.chat(event='confirm',inputs={'token':r['card']['token']})
        from facade_next.llm_chat import context
        with self.s.store.transaction() as db:
            summary=context(self.s,db,{'conversation':self.chat_id,'project':self.project})[1]
        self.assertEqual(summary['batches'][-1]['stage'],'initial');self.assertEqual(summary['batches'][-1]['name'],'第一批资料')
        self.assertEqual(len(calls),1)

    def test_llm_can_chat_without_repeating_the_intake_checklist(self):
        client=LLMClient(environ={},transport=lambda *args:{'choices':[{'finish_reason':'stop','message':{'content':'你好，我们可以先聊聊这个项目。'}}]})
        client.configure(dict(provider='openai',model='fake',api_key='fake-chat-key'));self.s.llm=client
        r=self.chat('先聊两句吧')
        self.assertIn('你好',r['reply']['zh']);self.assertNotIn('实际任务状态',r['reply']['zh'])
        self.assertEqual(self.s.snapshot()['jobs'],[]);self.assertEqual(self.s.snapshot()['datasets'],[])

    def test_multiple_maintenance_batches_keep_actual_parent_and_optional_val(self):
        import test_conversation as fixtures
        for i in range(4):
            if i:
                r=self.chat('开始新一轮维护');self.assertEqual(r['card']['stage'],'update')
            roles=('train','val','safety') if i==0 else ('train','val','gate') if i==2 else ('train','gate')
            root=self.batch('采集'+str(i),roles)
            r=self.chat(event='import_batch',inputs={'root':str(root)})
            self.chat(event='confirm',inputs={'token':r['card']['token']})
            r=self.chat(event='configure',inputs={'backbone':'resnet50','tools':['standard']})
            self.assertEqual(r['card']['kind'],'confirm_start')
            self.chat('确认启动');fixtures.ConversationTests.finish_fixture(self)
            r=self.chat('采用 standard，因为检查了本轮和历史结果' if i in (0,2) else '保留，因为本轮先保留父模型')
            self.chat('确认采用' if i in (0,2) else '确认保留')
        snap=self.s.snapshot();self.assertEqual(len(snap['rounds']),4)
        self.assertEqual(len(snap['projects'][0]['eval_pool']),8)
        self.assertEqual(len(snap['projects'][0]['val_pool']),4)
        self.assertTrue(all(b['status']=='imported' for b in snap['batches']))
        self.assertEqual(snap['rounds'][2]['parent']['id'],snap['rounds'][1]['parent']['id'])

    def test_nested_batches_and_pending_mapping_do_not_allow_mixed_single_import(self):
        root=self.batch()
        r=self.chat('数据在“'+str(self.root)+'”')
        self.assertIn('未执行',r['reply']['zh']);self.assertEqual(self.s.snapshot()['drafts'] if 'drafts' in self.s.snapshot() else self.s.snapshot()['datasets'],[])
        r=self.chat(event='import_batch',inputs={'root':str(root)})
        r=self.chat(event='import_folder',inputs={'root':str(root/'train'),'role':'train'})
        self.assertIn('未执行',r['reply']['zh']);self.assertEqual(self.s.snapshot()['datasets'],[])

    def test_cancel_during_annotation_stops_remaining_imports(self):
        r=self.chat(event='import_batch',inputs={'root':str(self.batch(partial=True))})
        self.chat(event='confirm',inputs={'token':r['card']['token']})
        r=self.chat('取消标注')
        self.assertEqual(self.s.snapshot()['datasets'],[])
        self.assertEqual(self.s.snapshot()['batches'][0]['status'],'cancelled')
        self.assertEqual(r['card']['kind'],'folder')

    def test_cancel_after_train_import_allows_a_replacement_batch(self):
        root=self.batch()
        (root/'val/labels.csv').write_text('filename,hollow,spalling,crack,no_defect\n0.jpg,,0,0,\n1.jpg,1,1,1,0\n',encoding='utf-8')
        r=self.chat(event='import_batch',inputs={'root':str(root)})
        r=self.chat(event='confirm',inputs={'token':r['card']['token']})
        self.assertEqual(r['card']['draft']['role'],'val');self.assertEqual(len(self.s.snapshot()['datasets']),1)
        r=self.chat('取消标注');self.assertTrue(r['card']['batch'])
        r=self.chat(event='import_batch',inputs={'root':str(self.batch('替换资料'))})
        self.assertEqual(r['card']['kind'],'confirm_batch')

    def test_cancelled_annotation_card_cannot_resurrect_a_batch(self):
        r=self.chat(event='import_batch',inputs={'root':str(self.batch(partial=True))})
        r=self.chat(event='confirm',inputs={'token':r['card']['token']});draft=r['card']['draft']
        self.chat('取消标注');draft['rows'][0]['hollow']=0
        r=self.chat(event='annotation_save',inputs={'draft':draft['id'],'rows':draft['rows']})
        self.assertIn('未执行',r['reply']['zh']);self.assertEqual(self.s.snapshot()['datasets'],[])

    def test_new_round_phrases_do_not_intercept_paths_or_explicit_llm_tools(self):
        root=self.batch('新一轮')
        r=self.chat('这是初始数据：“'+str(root)+'”');self.assertEqual(r['card']['kind'],'confirm_batch')
        self.chat(event='cancel_batch')
        with self.s.store.transaction() as db:
            p=self.s.store.get(db,'project',self.project);p['active']='adopted';p['val_pool']=[{'image_id':'old'}];self.s.store.put(db,'project',p)
        root=self.batch('新一轮维护',('train','gate'))
        client=LLMClient(environ={},transport=lambda *args:tool('import_batch',root=str(root),stage='update'))
        client.configure(dict(provider='openai',model='fake',api_key='fake-key'));self.s.llm=client
        r=self.chat('start a new maintenance round; batch path "'+str(root)+'"')
        self.assertEqual(r['card']['kind'],'confirm_batch');self.assertEqual(r['card']['stage'],'update')


if __name__=='__main__':unittest.main()
