import unittest
from facade_next.llm import LLMClient
import test_workflow
import test_conversation


class LocalResultsTests(unittest.TestCase):
    def test_status_retains_cancellation_instead_of_replacing_it_with_start(self):
        f=test_conversation.ConversationTests('test_chat_submission_id_is_idempotent')
        f.setUp();self.addCleanup(f.tearDown)
        f.chat('create project cancellation check')
        for role in ('train','val','evaluation'):
            f.chat(event='import_folder',inputs=dict(root=f.folder(role),role=role))
        f.chat(event='configure',inputs=dict(backbone='resnet50',tools=['standard']))
        cancel=f.chat('cancel round')
        self.assertEqual(cancel['card']['kind'],'confirm_cancel')
        status=f.chat('status')
        self.assertEqual(status['card']['kind'],'confirm_cancel')
        self.assertEqual(status['card']['token'],cancel['card']['token'])
        f.chat('confirm',inputs=dict(token=cancel['card']['token']))
        self.assertEqual(f.s.snapshot()['jobs'],[])
        self.assertEqual(f.s.snapshot()['rounds'][0]['status'],'cancelled')

    def test_initial_results_are_local_without_adoption_or_new_training(self):
        f=test_workflow.WorkflowTests('test_idempotent_submissions_and_collision')
        f.setUp();self.addCleanup(f.tearDown)
        rid=f.round();job=f.fake_complete(rid)
        calls=[]
        def forbidden(*args):
            calls.append(args)
            raise ValueError('Provider must not be called to inspect stored results')
        client=LLMClient(environ={},transport=forbidden)
        client.configure(dict(provider='openai',model='fixture',api_key='fake-key'))
        f.s.llm=client
        created=f.call('chat',project=f.p,text='status',language='en')
        with f.s.store.transaction() as db:
            chat=f.s.store.get(db,'conversation',created['conversation'])
            chat['context']['await_reason']=dict(choice='adopt',job=job,round=rid)
            f.s.store.put(db,'conversation',chat)
        result=f.call('chat',project=f.p,conversation=created['conversation'],text='View evaluation results',language='en')
        self.assertEqual(calls,[])
        self.assertEqual(result['card']['kind'],'comparison')
        self.assertIn('initial model evaluation',result['reply']['en'])
        self.assertFalse(result['card']['can_hold'])
        self.assertIsNone(f.s.snapshot()['projects'][0]['active'])
        self.assertEqual([j['id'] for j in f.s.snapshot()['jobs']],[job])
        saved=next(c for c in f.s.snapshot()['conversations'] if c['id']==created['conversation'])
        self.assertEqual(saved['context']['await_reason']['job'],job)
