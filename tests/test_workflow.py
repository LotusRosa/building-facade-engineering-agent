import unittest
from pathlib import Path
from test_logic import local_tmp
from facade_next.service import Service
from facade_next.contracts import digest, sha, write


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = local_tmp()
        self.root = Path(self.tmp.__enter__())
        self.s = Service(self.root/'state')
        self.n = 0
        self.p = self.call('create_project', name='demo')['id']

    def tearDown(self):
        self.tmp.__exit__(None,None,None)

    def call(self, action, **payload):
        self.n += 1
        return self.s.dispatch(action,payload,str(self.n))

    def dataset(self,name,scene='A'):
        rows = []
        for i in range(2):
            p = self.root/(name+str(i)+'.jpg')
            p.write_bytes((name+str(i)).encode())
            rows.append(dict(image_id=name+str(i),path=p.name,hollow=i,spalling=i,crack=i,scene=scene))
        return self.call('import_dataset', project=self.p,name=name,root=str(self.root),rows=rows)['id']

    def round(self):
        return self.call('create_round',project=self.p,train=[self.dataset('train')],val=[self.dataset('val')],evaluation=[self.dataset('test')],backbone='resnet50',tools=['standard'],seed=7,initialization='cached_imagenet')['id']

    def fake_complete(self,rid):
        jobs = self.call('start_round',round=rid)['jobs']
        job = self.s.claim('cuda:0')
        checkpoint = Path(job['output'])/'model.pt'
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        checkpoint.write_bytes(('test fixture, never model loaded '+job['id']).encode())
        contract = dict(schema='facade_next_v1',backbone='resnet50',classes=['hollow','spalling','crack'],preprocessing='full1280_letterbox114_gpu_v1',image_size=1280,threshold_status='frozen_on_val',thresholds=[.3,.4,.5],pos_weight=[1,1,1],checkpoint_sha256=sha(checkpoint))
        parent = job['spec']['parent']
        contract.update(job_digest=digest(job['spec']),roles_digest=job['spec']['roles_digest'],parent_sha256=parent['contract']['checkpoint_sha256'] if parent else None)
        from facade_next.metrics import scene_metrics
        ev=job['spec']['roles']['evaluation']
        fixture_metrics=scene_metrics(ev,[[.8 if y else .2 for y in r['labels']] for r in ev],contract['thresholds'])
        result = dict(job_digest=digest(job['spec']),evaluation_digest=digest(ev),status='ready',checkpoint=str(checkpoint),contract=contract,comparison={'candidate':fixture_metrics,'parent':fixture_metrics if parent else None},artifacts={'model.pt':sha(checkpoint)})
        self.s.finish(job['id'],result)
        return jobs[0]

    def test_idempotent_submissions_and_collision(self):
        a = self.s.dispatch('inspect',{'project':self.p},'unique')
        self.assertEqual(a,self.s.dispatch('inspect',{'project':self.p},'unique'))
        with self.assertRaises(ValueError):
            self.s.dispatch('inspect',{},'unique')
        events = self.s.snapshot()['events']
        self.assertEqual(sum(e['submission_id']=='unique' for e in events),1)
        self.assertEqual(events[0]['project'],self.p)

    def test_adopt_binds_exact_checkpoint_and_next_round_parent(self):
        r = self.round()
        j = self.fake_complete(r)
        v = self.call('decide',round=r,job=j,reason='reviewed class tradeoffs')['active']
        state = self.s.snapshot()
        self.assertEqual(state['projects'][0]['active'],v)
        r2 = self.call('create_round',project=self.p,train=[self.dataset('next')],val=[],evaluation=[self.dataset('next_test')],tools=['standard','head_only'],backbone='resnet50',seed=7)['id']
        rnd = next(x for x in self.s.snapshot()['rounds'] if x['id']==r2)
        self.assertEqual(rnd['parent']['id'],v)
        self.assertEqual(len(rnd['roles']['history']),2)
        self.assertEqual(len(rnd['roles']['evaluation']),4)
        with self.assertRaises(ValueError):
            self.call('decide',round=r,job=j,reason='stale')

    def test_hold_and_protected_evaluation(self):
        r = self.round()
        self.fake_complete(r)
        self.call('decide',round=r,job=self.s.snapshot()['jobs'][0]['id'],reason='initial')
        r2 = self.call('create_round',project=self.p,train=[self.dataset('new')],val=[],evaluation=[self.dataset('newtest')],tools=['hnm'],backbone='resnet50',seed=7)['id']
        before = self.s.snapshot()['projects'][0]['active']
        self.fake_complete(r2)
        self.call('decide',round=r2,job=None,reason='retain')
        self.assertEqual(self.s.snapshot()['projects'][0]['active'],before)
        test_id = next(d['id'] for d in self.s.snapshot()['datasets'] if d['name']=='newtest')
        with self.assertRaisesRegex(ValueError,'overlap'):
            self.call('create_round',project=self.p,train=[test_id],val=[],evaluation=[self.dataset('thirdtest')],tools=['standard'],backbone='resnet50',seed=7)

    def test_changed_checkpoint_rejected_and_timer_explicit(self):
        r = self.round()
        j = self.fake_complete(r)
        model = Path(self.s.snapshot()['jobs'][0]['result']['checkpoint'])
        model.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'changed'):
            self.call('decide',round=r,job=j,reason='bad')
        self.call('timer',project=self.p,command='start')
        self.call('timer',project=self.p,command='wait')
        with self.assertRaises(ValueError):
            self.call('timer',project=self.p,command='wait')
        self.call('timer',project=self.p,command='resume')
        self.call('timer',project=self.p,command='end')
        self.assertEqual(self.s.snapshot()['projects'][0]['timer']['state'],'ended')

    def test_artifact_escape_and_timer_restart(self):
        self.call('timer',project=self.p,command='start')
        other = Service(self.root/'state')
        with self.assertRaisesRegex(ValueError,'restart'):
            other.dispatch('timer',{'project':self.p,'command':'resume'},'restart-resume')
        timer = other.dispatch('timer',{'project':self.p,'command':'end'},'restart-end')
        self.assertEqual(timer['active_seconds'],0.)

    def test_finish_rejects_artifacts_outside_candidate(self):
        r = self.round()
        self.call('start_round',round=r)
        job = self.s.claim('cuda:0')
        outside = self.root/'outside.pt'
        outside.write_bytes(b'outside')
        contract = dict(schema='facade_next_v1',backbone='resnet50',classes=['hollow','spalling','crack'],preprocessing='full1280_letterbox114_gpu_v1',image_size=1280,threshold_status='frozen_on_val',thresholds=[.3,.4,.5],pos_weight=[1,1,1],checkpoint_sha256=sha(outside))
        with self.assertRaises(ValueError):
            self.s.finish(job['id'],dict(job_digest=digest(job['spec']),status='ready',checkpoint=str(outside),contract=contract,comparison={}))

    def test_rollback_is_separate_and_keeps_protected_history(self):
        r = self.round()
        j = self.fake_complete(r)
        first = self.call('decide',round=r,job=j,reason='first')['active']
        r2 = self.call('create_round',project=self.p,train=[self.dataset('next')],val=[],evaluation=[self.dataset('nexttest')],tools=['standard'],backbone='resnet50',seed=7)['id']
        j2 = self.fake_complete(r2)
        second = self.call('decide',round=r2,job=j2,reason='second')['active']
        self.assertNotEqual(first,second)
        self.call('rollback',project=self.p,version=first,reason='observed issue')
        p = self.s.snapshot()['projects'][0]
        self.assertEqual(p['active'],first)
        self.assertEqual(len(p['eval_pool']),4)
        self.assertEqual(len(self.s.snapshot()['versions']),2)
        self.assertEqual(self.s.snapshot()['rollbacks'][0]['to_version'],first)

    def test_concurrent_duplicate_request_has_one_mutation(self):
        from concurrent.futures import ThreadPoolExecutor
        def submit(_):
            return self.s.dispatch('create_project',{'name':'concurrent'},'same-submission')
        with ThreadPoolExecutor(max_workers=2) as pool:
            a,b = list(pool.map(submit,range(2)))
        self.assertEqual(a,b)
        self.assertEqual(sum(p['name']=='concurrent' for p in self.s.snapshot()['projects']),1)
        self.assertEqual(sum(e['submission_id']=='same-submission' for e in self.s.snapshot()['events']),1)

    def test_worker_photo_checks_and_failed_jobs_have_no_human_events(self):
        from facade_next.worker import verify_job,server_guard
        r = self.round()
        self.call('start_round',round=r)
        count = len(self.s.snapshot()['events'])
        job = self.s.claim('cuda:0')
        verify_job(job['spec'])
        Path(job['spec']['roles']['train'][0]['path']).write_bytes(b'changed photo')
        with self.assertRaisesRegex(ValueError,'Photo changed'):
            verify_job(job['spec'])
        self.s.finish(job['id'],error='simulated process error; no model run')
        self.assertEqual(len(self.s.snapshot()['events']),count)
        with self.assertRaises(RuntimeError):
            server_guard(False)

    def test_projects_have_independent_pending_rounds(self):
        first_round = self.round()
        first_project = self.p
        self.p = self.call('create_project',name='other project')['id']
        second_round = self.round()
        with self.s.store.transaction() as db:
            a = self.s.store.all(db,'round',first_project)
            b = self.s.store.all(db,'round',self.p)
        self.assertEqual([r['id'] for r in a],[first_round])
        self.assertEqual([r['id'] for r in b],[second_round])

    def test_explicit_acceptance_claim_cannot_consume_another_job(self):
        r1 = self.round()
        first = self.call('start_round',round=r1)['jobs'][0]
        self.p = self.call('create_project',name='other')['id']
        r2 = self.round()
        second = self.call('start_round',round=r2)['jobs'][0]
        job = self.s.claim('cuda:0',job_id=second)
        self.assertEqual(job['id'],second)
        self.s.finish(second,error='test-only simulated stop')
        self.assertEqual(self.s.claim('cuda:0',job_id=first)['id'],first)

    def test_single_cli_submission_does_not_count_automatic_subtasks(self):
        p = dict(project=self.p,train=[self.dataset('train')],val=[self.dataset('val')],evaluation=[self.dataset('eval')],tools=['standard'],backbone='resnet50',seed=7)
        before = len(self.s.snapshot()['events'])
        result = self.s.dispatch('submit_task',p,'one-cli-command')
        self.assertEqual(len(result['jobs']),1)
        self.assertEqual(len(self.s.snapshot()['events']),before+1)

    def test_hold_cannot_accumulate_a_gate_without_completed_evaluation(self):
        r=self.round()
        self.call('decide',round=r,job=self.fake_complete(r),reason='initial')
        r2=self.call('create_round',project=self.p,train=[self.dataset('new')],val=[],evaluation=[self.dataset('gate')],tools=['standard'],backbone='resnet50',seed=7)['id']
        before=len(self.s.snapshot()['projects'][0]['eval_pool'])
        with self.assertRaisesRegex(ValueError,'evaluation'):
            self.call('decide',round=r2,job=None,reason='premature Hold')
        self.assertEqual(len(self.s.snapshot()['projects'][0]['eval_pool']),before)

    def test_maintenance_cannot_silently_reuse_only_old_safety_without_new_gate(self):
        r=self.round()
        self.call('decide',round=r,job=self.fake_complete(r),reason='initial')
        with self.assertRaisesRegex(ValueError,'Current Gate'):
            self.call('create_round',project=self.p,train=[self.dataset('new')],val=[],evaluation=[],tools=['standard'],backbone='resnet50',seed=7)


if __name__ == '__main__':
    unittest.main()
