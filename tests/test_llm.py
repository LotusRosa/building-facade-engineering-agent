import json
import sqlite3
from contextlib import closing
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from test_logic import local_tmp
from facade_next.llm import LLMClient
from facade_next.service import Service


def tool(intent, **arguments):
    return {'choices':[{'finish_reason':'tool_calls','message':{'content':None,'tool_calls':[
        {'type':'function','function':{'name':'propose_task','arguments':json.dumps({'intent':intent,'arguments':arguments})}}
    ]}}]}


class LLMTests(unittest.TestCase):
    def test_exact_result_and_status_queries_use_local_state_without_a_provider(self):
        from facade_next.llm_chat import wants_llm
        for text in ['Compare results','View evaluation results','Please show results.','查看评估结果','status','查看训练进度']:
            self.assertFalse(wants_llm(dict(text=text)),text)
        for text in ['Do not compare results','Compare results and adopt standard','why is recall low?']:
            self.assertTrue(wants_llm(dict(text=text)),text)

    def client(self, transport):
        client=LLMClient(environ={},transport=transport)
        client.configure(dict(provider='deepseek',model='deepseek-flash',api_key='test-secret'))
        return client

    def test_disabled_does_not_borrow_an_unselected_provider_key(self):
        client=LLMClient(environ={'OPENAI_API_KEY':'test-secret'})
        self.assertFalse(client.status()['enabled'])
        self.assertNotIn('test-secret',json.dumps(client.status()))

    def test_no_model_suite_does_not_inherit_live_llm_settings(self):
        import os
        with patch.dict(os.environ,{'FACADE_LLM_PROVIDER':'openai','FACADE_LLM_MODEL':'account-model','OPENAI_API_KEY':'fake-key'}):
            with local_tmp() as folder:
                self.assertFalse(Service(folder).llm.status()['enabled'])

    def test_request_contract_and_secret_free_status(self):
        seen=[]
        client=self.client(lambda *args: seen.append(args) or tool('status'))
        self.assertEqual(client.complete([{'role':'user','content':'progress?'}])['intent'],'status')
        url,body,key,timeout=seen[0]
        self.assertEqual(url,'https://api.deepseek.com/chat/completions')
        self.assertEqual(key,'test-secret')
        self.assertEqual(body['model'],'deepseek-flash')
        self.assertFalse(body['stream'])
        self.assertLessEqual(body['max_tokens'],2048)
        self.assertEqual(body['tools'][0]['function']['name'],'propose_task')
        self.assertLessEqual(timeout,45)
        self.assertNotIn('test-secret',json.dumps(client.status()))

    def test_environment_selection_and_key_echo_redaction(self):
        client=LLMClient(environ={'FACADE_LLM_PROVIDER':'openai','FACADE_LLM_MODEL':'account-model','OPENAI_API_KEY':'fake-key'},
                         transport=lambda *args:tool('create_project',name='fake-key'))
        self.assertTrue(client.status()['enabled'])
        with self.assertRaises(ValueError):client.complete([])
        client.transport=lambda *args:{'choices':[{'finish_reason':'stop','message':{'content':'Key: fake-key'}}]}
        self.assertNotIn('fake-key',client.complete([])['text'])

    def test_endpoint_change_requires_a_new_key_and_bad_url_is_atomic(self):
        client=self.client(lambda *args:tool('status'))
        with self.assertRaises(ValueError):
            client.configure(dict(provider='compatible',base_url='http://remote.example/v1',model='m',api_key='new'))
        self.assertEqual(client.status()['provider'],'deepseek')
        with self.assertRaises(ValueError):
            client.configure(dict(provider='compatible',base_url='https://elsewhere.example/v1',model='m'))
        self.assertEqual(client.status()['provider'],'deepseek')
        client.configure(dict(provider='openai',model='account-model',api_key='new'))
        self.assertEqual(client.status()['base_url'],'https://api.openai.com/v1')
        client.configure({'provider':'off'})
        self.assertFalse(client.status()['enabled'])

    def test_bad_reply_and_raw_error_do_not_leak_key(self):
        for response in [tool('confirm'),tool('run_shell',command='anything'),{'choices':[]},
                         {'choices':[{'finish_reason':'length','message':{'content':'partial'}}]}]:
            client=self.client(lambda *args:response)
            with self.assertRaises(ValueError) as exc:client.complete([])
            self.assertNotIn('test-secret',str(exc.exception))
        def fail(*args):raise OSError('test-secret https://private.example')
        with self.assertRaises(ValueError) as exc:self.client(fail).complete([])
        self.assertNotIn('test-secret',str(exc.exception))

    def test_connection_errors_identify_the_layer_without_exposing_secrets(self):
        import socket,ssl,urllib.error
        failures=[
            (urllib.error.URLError(ConnectionRefusedError(10061,'test-secret')),'连接被拒绝'),
            (urllib.error.URLError(TimeoutError('test-secret')),'超时'),
            (TimeoutError('test-secret'),'超时'),
            (urllib.error.URLError(socket.gaierror(-2,'test-secret')),'DNS'),
            (urllib.error.URLError(ssl.SSLCertVerificationError('test-secret')),'证书'),
            (OSError('test-secret https://private.example'),'网络请求失败'),
        ]
        for failure,expected in failures:
            with self.subTest(expected=expected):
                calls=[]
                def fail(*args):calls.append(args);raise failure
                with self.assertRaises(ValueError) as exc:self.client(fail).complete([])
                message=str(exc.exception)
                self.assertIn(expected,message)
                self.assertNotIn('test-secret',message)
                self.assertNotIn('private.example',message)
                self.assertEqual(len(calls),1)

    def test_http_status_explains_auth_balance_and_rate_limit_safely(self):
        import urllib.error
        for code,expected in [(401,'API Key'),(402,'余额'),(429,'限流')]:
            with self.subTest(code=code):
                def fail(*args):raise urllib.error.HTTPError('https://private.example',code,'test-secret',{},None)
                with self.assertRaises(ValueError) as exc:self.client(fail).complete([])
                self.assertIn('HTTP '+str(code),str(exc.exception))
                self.assertIn(expected,str(exc.exception))
                self.assertNotIn('test-secret',str(exc.exception))
                self.assertNotIn('private.example',str(exc.exception))

    def test_user_facing_method_names_keep_canonical_tool_identity(self):
        from facade_next.llm_chat import validate
        from facade_next.names import METHOD_LABELS
        import re
        app=(Path(__file__).resolve().parents[1]/'facade_next/static/app.js').read_text(encoding='utf-8')
        self.assertEqual(json.loads(re.search(r'const METHOD_LABELS=(\{.*\});',app).group(1)),METHOD_LABELS)
        names={'standard':'标准训练／更新','asl':'非对称损失','sam':'锐度感知优化','distill':'历史知识蒸馏',
               'hnm':'困难负样本挖掘','hem':'困难样本重加权','head_only':'仅更新分类头','ewc_replay':'抗遗忘约束与历史回放'}
        for method,label in names.items():
            with self.subTest(method=method):
                proposal=dict(intent='configure',arguments=dict(tools=[method]))
                self.assertEqual(validate(proposal,'使用'+label)[1]['tools'],[method])
                self.assertEqual(validate(proposal,'use '+method)[1]['tools'],[method])
                self.assertEqual(validate(proposal,'use '+METHOD_LABELS[method]['en'])[1]['tools'],[method])
        with self.assertRaises(ValueError):validate(dict(intent='configure',arguments=dict(tools=['sam'])),'使用标准训练')

    def test_rule_chat_accepts_localized_method_name_without_running_a_job(self):
        with local_tmp() as folder:
            service=Service(folder)
            first=service.dispatch('chat',dict(text='创建项目 名称测试',language='zh'),'name-project')
            result=service.dispatch('chat',dict(conversation=first['conversation'],project=first['project'],text='使用标准训练',language='zh'),'name-method')
            chat=next(c for c in service.snapshot()['conversations'] if c['id']==result['conversation'])
            self.assertEqual(chat['context']['options']['tools'],['core'])
            self.assertEqual(service.snapshot()['jobs'],[])
            from facade_next.names import METHOD_LABELS
            for method,label in METHOD_LABELS.items():
                result=service.dispatch('chat',dict(conversation=first['conversation'],project=first['project'],text='use '+label['en'],language='en'),'english-'+method)
                chat=next(c for c in service.snapshot()['conversations'] if c['id']==result['conversation'])
                self.assertEqual(chat['context']['options']['tools'],['core'])
                if method not in ('core','standard'):
                    self.assertIn('Core',chat['messages'][-1]['reply']['en'])

    def test_localized_adoption_still_requires_the_explicit_candidate(self):
        from facade_next.llm_chat import validate
        for method,label in [('distill','历史知识蒸馏'),('hnm','Hard negative mining')]:
            args=dict(choice='adopt',job=method,reason='合适')
            self.assertEqual(validate(dict(intent='propose_decision',arguments=args),'采用'+label+'，因为合适')[1]['job'],method)
        for candidate in ['sam','job_other']:
            with self.assertRaises(ValueError):validate(dict(intent='propose_decision',arguments=dict(choice='adopt',job=candidate)),'采用标准训练')

    def test_adoption_candidate_is_bound_before_the_reason_and_cannot_be_ambiguous(self):
        from facade_next.llm_chat import validate
        for text in ['adopt SAM because standard regressed','采用 SAM，因为标准训练在旧场景退化']:
            with self.subTest(text=text):
                result=dict(intent='propose_decision',arguments=dict(choice='adopt',job='sam'))
                self.assertEqual(validate(result,text)[1]['job'],'sam')
                result['arguments']['job']='standard'
                with self.assertRaises(ValueError):validate(result,text)
        for text in ['adopt SAM or standard because both are ready',
                     'adopt SAM, not standard because safer',
                     'adopt not SAM because unsafe','采用 SAM，不采用标准训练，因为稳妥']:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    validate(dict(intent='propose_decision',arguments=dict(choice='adopt',job='sam')),text)

    def test_adoption_keeps_exact_candidate_ids_binding(self):
        from facade_next.llm_chat import validate
        proposal=dict(intent='propose_decision',arguments=dict(choice='adopt',job='job_abc'))
        self.assertEqual(validate(proposal,'adopt job_abc because SAM regressed')[1]['job'],'job_abc')
        self.assertEqual(validate(proposal,'adopt job_abc (SAM) because suitable')[1]['job'],'job_abc')
        for text in ['adopt job_abc or job_other because both are ready',
                     'adopt job_abc or standard because both are ready',
                     '采用 job_abc 或 标准训练，因为都已完成',
                     'adopt SAM because job_abc regressed']:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):validate(proposal,text)


class LLMWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=local_tmp(); self.root=Path(self.tmp.__enter__())
        self.calls=[]; self.response=tool('create_project',name='Natural chat project')
        self.client=LLMClient(environ={},transport=self.transport)
        self.client.configure(dict(provider='openai',model='account-model',api_key='fake-key'))
        self.s=Service(self.root/'state',llm=self.client)

    def tearDown(self):self.tmp.__exit__(None,None,None)

    def transport(self,*args):
        self.calls.append(args)
        with closing(sqlite3.connect(self.s.store.path,timeout=0)) as db:
            db.execute('BEGIN IMMEDIATE');db.rollback()
        return self.response

    def chat(self,text,sid='one',**extra):
        return self.s.dispatch('chat',dict(text=text,language='zh',**extra),sid)

    def test_natural_request_is_grounded_deduplicated_and_not_a_worker(self):
        a=self.chat('帮我开一个外立面模型项目')
        b=self.chat('帮我开一个外立面模型项目')
        self.assertEqual(a,b);self.assertEqual(len(self.calls),1)
        snap=self.s.snapshot()
        self.assertEqual(len(snap['projects']),1)
        self.assertEqual(snap['jobs'],[])
        self.assertEqual(len(snap['events']),1)
        self.assertNotIn('fake-key',json.dumps(snap))
        self.assertNotIn(b'fake-key',self.s.store.path.read_bytes())

    def test_concurrent_duplicate_has_one_api_call(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            a,b=list(pool.map(lambda _:self.chat('新项目'),range(2)))
        self.assertEqual(a,b);self.assertEqual(len(self.calls),1)

    def test_folder_import_and_configuration_use_existing_guards(self):
        a=self.chat('建立工程')
        for role in ('train','val','evaluation'):
            folder=self.root/role;folder.mkdir()
            for i in range(2):(folder/(str(i)+'.jpg')).write_bytes((role+str(i)).encode())
            (folder/'labels.csv').write_text('filename,hollow,spalling,crack,no_defect\n0.jpg,0,0,0,1\n1.jpg,1,1,1,0\n',encoding='utf-8')
            self.response=tool('import_folder',root=str(folder),role=role)
            r=self.chat('把这批数据用于'+role+'：'+str(folder),role,conversation=a['conversation'],project=a['project'])
        self.assertEqual(r['card']['kind'],'configure')
        self.response=tool('configure',backbone='resnet50',tools=['standard'])
        r=self.chat('这次先用残差网络和基础方法','configure',conversation=a['conversation'],project=a['project'])
        self.assertEqual(r['card']['kind'],'confirm_start')
        self.assertEqual(self.s.snapshot()['jobs'],[])
        count=len(self.calls)
        self.chat('确认启动','start',conversation=a['conversation'],project=a['project'])
        self.assertEqual(len(self.calls),count)
        self.assertEqual(self.s.snapshot()['jobs'][0]['status'],'queued')

    def test_invented_engineer_reasons_and_candidates_are_rejected(self):
        from facade_next.llm_chat import validate
        for result in [dict(intent='propose_decision',arguments=dict(choice='hold',reason='made up')),
                       dict(intent='propose_decision',arguments=dict(choice='adopt',job='standard',reason='')),
                       dict(intent='rollback',arguments=dict(version='version-made-up',reason=''))]:
            with self.assertRaises(ValueError):validate(result,'帮我挑最好的')

    def test_model_cannot_confirm_or_invent_a_folder(self):
        a=self.chat('帮我开项目')
        for i,response in enumerate([tool('confirm'),tool('import_folder',root=str(self.root),role='train')]):
            self.response=response
            r=self.chat('请帮我处理下一步','bad'+str(i),conversation=a['conversation'],project=a['project'])
            self.assertIn('未执行',r['reply']['zh'])
            self.assertEqual(self.s.snapshot()['jobs'],[])
            self.assertEqual(self.s.snapshot()['datasets'],[])

    def test_explicit_cards_and_confirmation_do_not_call_the_api(self):
        a=self.chat('创建项目 Direct',event='create_project',inputs={'name':'Direct'})
        self.assertEqual(self.calls,[])
        self.chat('确认启动','confirm',conversation=a['conversation'],project=a['project'])
        self.assertEqual(self.calls,[])

    def test_provider_failure_retains_project_and_task(self):
        a=self.chat('新建项目')
        def fail(*args):raise TimeoutError('fake-key')
        self.client.transport=fail
        r=self.chat('现在有什么可以做的','failed',conversation=a['conversation'],project=a['project'])
        self.assertEqual(r['project'],a['project'])
        self.assertIn('未执行',r['reply']['zh'])
        self.assertNotIn('fake-key',json.dumps(r))
        self.assertEqual(r['card']['kind'],'folder')

    def test_protocol_disconnect_is_recorded_and_not_retried(self):
        from http.client import IncompleteRead
        calls=[]
        def disconnected(*args):calls.append(args);raise IncompleteRead(b'fake-key',100)
        self.client.transport=disconnected
        a=self.chat('我想处理下一步')
        b=self.chat('我想处理下一步')
        self.assertEqual(a,b);self.assertEqual(len(calls),1)
        self.assertIn('未执行',a['reply']['zh'])
        self.assertNotIn('fake-key',json.dumps(a))

    def test_exact_folder_role_and_explicit_model_options(self):
        from facade_next.llm_chat import validate
        for root in ('/','/home','/home/test'):
            with self.assertRaises(ValueError):
                validate(dict(intent='import_folder',arguments=dict(root=root,role='train')),'训练照片在 /home/test/photos')
        with self.assertRaises(ValueError):
            validate(dict(intent='import_folder',arguments=dict(root='/home/test/photos',role='evaluation')),'训练照片在 /home/test/photos')
        with self.assertRaises(ValueError):
            validate(dict(intent='configure',arguments=dict(backbone='swin_t',tools=['sam','distill'])),'现在状态怎样？')
        with self.assertRaises(ValueError):
            validate(dict(intent='configure',arguments=dict(backbone='resnet50',tools=['sam'])),'准备训练')
        proposal=dict(intent='import_folder',arguments=dict(root='/home/test/my photos',role='val'))
        self.assertEqual(validate(proposal,'验证照片放在“/home/test/my photos”')[0],'import_folder')
        self.assertEqual(validate(dict(intent='configure',arguments=dict(backbone='resnet50',tools=['standard'])),'这次先用残差网络和基础方法')[0],'configure')
        with self.assertRaises(ValueError):
            validate(dict(intent='import_folder',arguments=dict(root='/photos',role='evaluation')),'使用 /photos',default_role='train')
        self.assertEqual(validate(dict(intent='propose_decision',arguments=dict(choice='adopt',job='job_abc',reason='表现合适')),
                                 '表现合适',await_reason={'choice':'adopt','job':'job_abc'})[0],'propose_decision')

    def test_panel_update_during_api_call_invalidates_proposal(self):
        a=self.chat('开始一个项目')
        def changed(*args):
            self.s.dispatch('chat',dict(conversation=a['conversation'],project=a['project'],event='template',inputs={}), 'panel-change')
            return tool('configure',backbone='resnet50',tools=['standard'])
        self.client.transport=changed
        r=self.chat('请准备配置','stale',conversation=a['conversation'],project=a['project'])
        self.assertIn('状态已变化',r['reply']['zh'])
        self.assertEqual(self.s.snapshot()['rounds'],[])


if __name__=='__main__':unittest.main()
