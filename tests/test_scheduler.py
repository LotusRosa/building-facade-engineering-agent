import os
import subprocess
import unittest
from unittest.mock import patch,Mock
from pathlib import Path
from test_logic import local_tmp
from facade_next.service import Service
from facade_next.server import Scheduler


class SchedulerTests(unittest.TestCase):
    def test_selected_interpreter_and_weight_cache_are_used_by_workers(self):
        with local_tmp() as root:
            service=Service(root)
            with service.store.transaction() as db:
                service.store.put(db,'job',dict(id='queued_shared',project='test',status='queued'))
            scheduler=Scheduler(service,'1',python='chosen_env/python',torch_home='chosen_cache')
            def spawned(*args,**kwargs):
                scheduler.stop.set()
                return Mock(pid=54321)
            with patch.dict(os.environ,{'PYTHONHOME':'wrong','PYTHONPATH':'wrong'}), patch('facade_next.server.subprocess.Popen',side_effect=spawned) as popen:
                scheduler.loop()
            try:
                self.assertEqual(popen.call_args.args[0][0],'chosen_env/python')
                self.assertEqual(popen.call_args.kwargs['env']['TORCH_HOME'],'chosen_cache')
                self.assertFalse('PYTHONHOME' in popen.call_args.kwargs['env'])
                self.assertFalse('PYTHONPATH' in popen.call_args.kwargs['env'])
            finally:
                for _,log in scheduler.running.values():log.close()

    def test_worker_has_independent_terminal_session_without_running_a_worker(self):
        with local_tmp() as root:
            service=Service(root)
            with service.store.transaction() as db:
                service.store.put(db,'job',dict(id='queued_fixture',project='test',status='queued'))
            scheduler=Scheduler(service,'0')
            child=Mock(pid=12345)
            def spawned(*args,**kwargs):
                scheduler.stop.set()
                return child
            with patch('facade_next.server.subprocess.Popen',side_effect=spawned) as popen:
                scheduler.loop()
            try:
                kwargs=popen.call_args.kwargs
                self.assertEqual(kwargs.get('stdin'),subprocess.DEVNULL)
                self.assertEqual(kwargs.get('start_new_session'),os.name!='nt')
            finally:
                for process,log in scheduler.running.values():
                    log.close()
