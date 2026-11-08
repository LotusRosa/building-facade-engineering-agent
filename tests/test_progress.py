import importlib.util
import unittest
from pathlib import Path
from test_logic import local_tmp
from facade_next.contracts import digest,write,PREPROCESSING


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('facade_next.progress'),'Read-only training progress is not implemented')
        from facade_next.progress import job_progress
        self.progress=job_progress
        self.tmp=local_tmp();self.root=Path(self.tmp.__enter__())
        self.addCleanup(lambda:self.tmp.__exit__(None,None,None))
        self.output=self.root/'jobs'/'fixture';self.output.mkdir(parents=True)
        self.rows=[dict(image_id=str(i),sha256=str(i)*64) for i in range(4)]
        self.job=dict(status='running',output=str(self.output),started='2026-11-02T04:00:00+00:00',
            spec=dict(kind='initial',roles=dict(train=self.rows,val=[],history=[],evaluation=[]),protocol=dict(epochs=40,effective_batch=16,steps_per_epoch=38)))

    def test_cache_preparation_counts_actual_pairs_and_not_temporary_files(self):
        cache=self.root/'image_cache';cache.mkdir()
        key=digest(dict(sha256=self.rows[0]['sha256'],preprocessing=PREPROCESSING))
        (cache/(key+'.npy')).write_bytes(b'fixture; never decoded')
        (cache/(key+'.json')).write_text('{}',encoding='utf-8')
        (cache/'incomplete.npy').write_bytes(b'fixture')
        result=self.progress(self.job)
        self.assertEqual(result['stage'],'preparing')
        self.assertEqual(result['cached_images'],1)
        self.assertEqual(result['total_images'],4)
        self.assertAlmostEqual(result['fraction'],.25)

    def test_step_progress_is_actual_records_over_planned_maximum(self):
        self.job['spec']['roles']['train']=[dict(image_id=str(i),sha256=str(i).zfill(64)) for i in range(32)]
        write(self.output/'schedule_01.json',dict(indices=[[0]*16,[0]*16]))
        write(self.output/'work_01_0001.json',dict(epoch=1,step=1,seconds=3))
        (self.output/'work_01_0002.json.partial').write_text('not complete')
        result=self.progress(self.job)
        self.assertEqual(result['stage'],'training')
        self.assertEqual(result['completed_steps'],1)
        self.assertEqual(result['planned_steps'],80)
        self.assertEqual(result['epoch'],1)
        self.assertAlmostEqual(result['fraction'],1/80)

    def test_evaluation_is_indeterminate_until_the_job_succeeds(self):
        write(self.output/'validation_predictions.json',{})
        result=self.progress(self.job)
        self.assertEqual(result['stage'],'evaluation')
        self.assertIsNone(result['fraction'])
        write(self.output/'evaluation_predictions.json',{})
        self.assertEqual(self.progress(self.job)['stage'],'finalizing')

    def test_early_stop_can_complete_without_faking_all_planned_epochs(self):
        write(self.output/'history.json',[dict(epoch=i) for i in range(1,12)])
        self.job.update(status='succeeded',finished='2026-11-02T04:10:00+00:00')
        result=self.progress(self.job)
        self.assertEqual(result['stage'],'completed')
        self.assertEqual(result['fraction'],1)
        self.assertEqual(result['completed_epochs'],11)
        self.assertEqual(result['max_epochs'],40)

    def test_empty_hnm_pool_does_not_claim_update_training_completed(self):
        self.job.update(status='succeeded',result=dict(status='skip_empty_pool'))
        self.assertEqual(self.progress(self.job)['stage'],'skipped')

    def test_completed_training_history_advances_to_calibration(self):
        write(self.output/'history.json',[dict(epoch=i,val_macro_AP=.5) for i in range(1,10)])
        self.job['spec']['protocol']['patience']=8
        self.assertEqual(self.progress(self.job)['stage'],'calibrating')

    def test_queued_failed_and_missing_artifacts_do_not_report_completion(self):
        for status in ['queued','failed','cancelled']:
            self.job['status']=status
            result=self.progress(self.job)
            self.assertEqual(result['stage'],status)
            self.assertIsNone(result['fraction'])

    def test_observing_progress_does_not_change_job_or_files(self):
        import copy
        frozen=copy.deepcopy(self.job)
        before=set(self.output.iterdir())
        self.progress(self.job)
        self.assertEqual(self.job,frozen)
        self.assertEqual(set(self.output.iterdir()),before)
