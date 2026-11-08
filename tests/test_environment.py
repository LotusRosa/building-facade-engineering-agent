import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import unittest
import uuid


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('facade_next.environment'),
                             'First-run environment manager is not implemented')
        from facade_next.environment import EnvironmentManager
        self.Manager = EnvironmentManager
        self.base = Path(__file__).resolve().parents[2] / '.env_tests' / uuid.uuid4().hex[:8]
        self.base.mkdir(parents=True)
        def cleanup():
            assert self.base.resolve().is_relative_to(Path(__file__).resolve().parents[2])
            shutil.rmtree(self.base)
        self.addCleanup(cleanup)
        self.calls = []
        self.activated = []
        self.config = self.base / 'host' / 'environment.json'

    def fake_run(self, args, env, log):
        self.calls.append(list(args))
        if args[1:3] == ['-m', 'venv']:
            p = self.python(Path(args[-1]))
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b'fake interpreter, never executed')
        log('test operation output')
        if any(str(x).endswith('environment_probe.py') for x in args):
            return json.dumps(dict(ok=True, python='3.12.3', executable=str(args[0]),
                packages=dict(torch='2.11.0+cu128', torchvision='0.26.0+cu128', numpy='2.5.4', Pillow='13.0.0'),
                cuda=True, devices=[dict(index=0, name='fixture', native_bf16=True),
                                    dict(index=1, name='fixture', native_bf16=True)],
                torch_home=env.get('TORCH_HOME', str(self.base / 'cache')), weights=[]))
        return ''

    @staticmethod
    def python(folder):
        return folder / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')

    def manager(self, **kwargs):
        manager=self.Manager(self.config, runner=kwargs.pop('runner', self.fake_run),
                             activate=self.activated.append, **kwargs)
        self.addCleanup(lambda: manager.thread.join(5) if manager.thread else None)
        return manager

    def existing(self):
        folder = self.base / 'existing'
        python = self.python(folder)
        python.parent.mkdir(parents=True)
        python.write_bytes(b'fake existing interpreter, never executed')
        return folder

    def request(self, folder, mode='existing'):
        return dict(folder=str(folder), mode=mode, gpus='1', weights=[], enable_workers=True)

    def finish(self, manager):
        manager.thread.join(5)
        self.assertFalse(manager.thread.is_alive())
        return manager.snapshot()

    def test_existing_environment_is_saved_once_and_restored_for_other_projects(self):
        folder = self.existing()
        manager = self.manager()
        manager.start(self.request(folder))
        status = self.finish(manager)
        self.assertEqual(status['operation']['state'], 'succeeded')
        self.assertTrue(status['ready'])
        self.assertEqual(status['profile']['python'], str(self.python(folder)))
        self.assertFalse(any('pip' in a or 'venv' in a for a in self.calls))
        other = self.manager()
        self.assertEqual(other.snapshot()['profile'], status['profile'])
        self.assertEqual(self.activated[0]['gpus'], '1')

    def test_new_folder_gets_managed_environment_and_all_method_dependencies(self):
        folder = self.base / 'managed'
        manager = self.manager()
        manager.start(self.request(folder, 'create'))
        status = self.finish(manager)
        self.assertTrue(status['ready'], status)
        self.assertTrue((folder / '.facade-environment.json').is_file())
        self.assertEqual(status['profile']['python'], str(self.python(folder / '.venv')))
        installs = [a for a in self.calls if 'pip' in a]
        self.assertTrue(any('torch==2.11.0' in a and 'torchvision==0.26.0' in a for a in installs))
        self.assertTrue(any('numpy>=1.26' in a and 'Pillow>=10.4' in a for a in installs))
        self.assertEqual(status['profile']['torch_home'], str(folder / 'torch'))
        calls=len(self.calls)
        with self.assertRaises(ValueError): self.manager().start(self.request(folder,'create'))
        self.assertEqual(len(self.calls),calls)

    def test_setup_continues_with_bundled_pip_without_network_upgrade(self):
        def offline_upgrade(args, env, log):
            if 'pip' in args and '--upgrade' in args:
                raise RuntimeError("ReadTimeoutError: HTTPSConnectionPool(host='files.pythonhosted.org', port=443): Read timed out.")
            return self.fake_run(args, env, log)
        manager = self.manager(runner=offline_upgrade)
        manager.start(self.request(self.base / 'bundled_pip', 'create'))
        status = self.finish(manager)
        self.assertEqual(status['operation']['state'], 'succeeded', status['operation'])
        self.assertTrue(status['ready'])

    def test_interrupted_dependency_download_retries_and_keeps_installation_in_venv(self):
        interrupted = []
        def interrupted_download(args, env, log):
            if 'torch==2.11.0' in args:
                interrupted.append(list(args))
                if len(interrupted) == 1:
                    raise RuntimeError("pip._vendor.urllib3.exceptions.ReadTimeoutError: HTTPSConnectionPool(host='download.pytorch.org', port=443): Read timed out.")
            return self.fake_run(args, env, log)
        folder = self.base / 'retry_download'
        manager = self.manager(runner=interrupted_download)
        manager.start(self.request(folder, 'create'))
        status = self.finish(manager)
        self.assertEqual(status['operation']['state'], 'succeeded', status['operation'])
        self.assertEqual(len(interrupted), 2)
        self.assertTrue(all(a[0] == str(self.python(folder / '.venv')) for a in interrupted))
        self.assertIn('Retrying', status['operation']['log'])
        installs = [a for a in self.calls if 'install' in a and 'pip' in a]
        self.assertTrue(all('--timeout' in a and int(a[a.index('--timeout') + 1]) >= 120 for a in installs))
        self.assertTrue(all('--retries' in a and '--disable-pip-version-check' in a for a in installs))

    def test_persistent_download_timeout_is_bounded_and_preserves_existing_profile(self):
        manager = self.manager()
        manager.start(self.request(self.existing()))
        self.finish(manager)
        original = self.config.read_bytes()
        interrupted = []
        def always_timeout(args, env, log):
            if 'torch==2.11.0' in args:
                interrupted.append(list(args))
                raise RuntimeError("ReadTimeoutError: HTTPSConnectionPool(host='download.pytorch.org', port=443): Read timed out.")
            return self.fake_run(args, env, log)
        manager = self.manager(runner=always_timeout)
        manager.start(self.request(self.base / 'persistent_timeout', 'create'))
        status = self.finish(manager)
        self.assertEqual(status['operation']['state'], 'failed')
        self.assertEqual(len(interrupted), 3)
        self.assertEqual(self.config.read_bytes(), original)
        self.assertIn('download.pytorch.org', status['operation']['error'])

    def test_dependency_conflict_is_not_retried_as_a_network_timeout(self):
        conflicts = []
        def incompatible(args, env, log):
            if 'torch==2.11.0' in args:
                conflicts.append(list(args))
                raise RuntimeError('ERROR: ResolutionImpossible: incompatible dependencies')
            return self.fake_run(args, env, log)
        manager = self.manager(runner=incompatible)
        manager.start(self.request(self.base / 'conflict', 'create'))
        status = self.finish(manager)
        self.assertEqual(status['operation']['state'], 'failed')
        self.assertEqual(len(conflicts), 1)
        self.assertIn('ResolutionImpossible', status['operation']['error'])

    def test_shutdown_between_download_attempts_prevents_retry(self):
        attempts = []
        manager = self.manager()
        def cancelled_download(args, env, log):
            if 'torch==2.11.0' in args:
                attempts.append(list(args))
                manager.close()
                raise RuntimeError('ReadTimeoutError: Read timed out.')
            return self.fake_run(args, env, log)
        manager.runner = cancelled_download
        manager.start(self.request(self.base / 'cancel_download', 'create'))
        self.assertEqual(self.finish(manager)['operation']['state'], 'failed')
        self.assertEqual(len(attempts), 1)
        self.assertEqual(self.activated, [])
        self.assertFalse(self.config.exists())

    def test_dependency_ssl_eof_is_retried_without_changing_install_target(self):
        attempts=[]
        def unstable(args,env,log):
            if 'torch==2.11.0' in args:
                attempts.append(args)
                if len(attempts)==1:raise RuntimeError('[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol')
            return self.fake_run(args,env,log)
        manager=self.manager(runner=unstable)
        manager.start(self.request(self.base/'ssl_retry','create'))
        self.assertEqual(self.finish(manager)['operation']['state'],'succeeded')
        self.assertEqual(len(attempts),2)
        self.assertEqual(attempts[0][0],attempts[1][0])

    def test_weight_progress_survives_polling_and_malformed_packets_are_ignored(self):
        manager=self.manager()
        packet=dict(backbone='convnext_tiny',status='downloading',received=1024,total=4096,attempt=2,verified=False)
        manager._log('FACADE_WEIGHT_PROGRESS '+json.dumps(packet))
        self.assertEqual(manager.snapshot()['operation'].get('download'),packet)
        manager._log('FACADE_WEIGHT_PROGRESS not-json')
        manager._log('FACADE_WEIGHT_PROGRESS '+json.dumps(dict(packet,received=-1)))
        self.assertEqual(manager.snapshot()['operation']['download'],packet)

    def test_unmanaged_nonempty_folder_and_relative_paths_are_rejected_without_commands(self):
        folder = self.base / 'photos'
        folder.mkdir()
        (folder / 'keep.txt').write_text('keep', encoding='utf-8')
        manager = self.manager()
        with self.assertRaises(ValueError): manager.start(self.request(folder, 'create'))
        with self.assertRaises(ValueError): manager.start(self.request(Path('relative'), 'create'))
        self.assertEqual(self.calls, [])
        self.assertEqual((folder / 'keep.txt').read_text(), 'keep')

    def test_reusing_managed_environment_keeps_its_weight_cache(self):
        folder=self.base/'managed_cache'
        manager=self.manager();manager.start(self.request(folder,'create'))
        first=self.finish(manager)
        again=self.manager();again.start(self.request(folder))
        self.assertEqual(self.finish(again)['profile']['torch_home'],first['profile']['torch_home'])

    def test_failed_install_keeps_previous_profile_and_can_retry_managed_folder(self):
        manager = self.manager()
        manager.start(self.request(self.existing()))
        self.finish(manager)
        original = self.config.read_bytes()
        def fail(args, env, log):
            if 'pip' in args: raise RuntimeError('fixture network failure')
            return self.fake_run(args, env, log)
        folder = self.base / 'partial'
        retry = self.manager(runner=fail)
        retry.start(self.request(folder, 'create'))
        status = self.finish(retry)
        self.assertEqual(status['operation']['state'], 'failed')
        self.assertIn('network failure', status['operation']['error'])
        self.assertEqual(self.config.read_bytes(), original)
        done = self.manager()
        done.start(self.request(folder, 'create'))
        self.assertEqual(self.finish(done)['operation']['state'], 'succeeded')

    def test_running_jobs_and_concurrent_setup_are_rejected(self):
        folder = self.existing()
        blocked = self.manager(busy=lambda: True)
        with self.assertRaises(ValueError): blocked.start(self.request(folder))
        gate = threading.Event()
        def slow(args, env, log):
            gate.wait(3)
            return self.fake_run(args, env, log)
        manager = self.manager(runner=slow)
        manager.start(self.request(folder))
        try:
            with self.assertRaises(ValueError): manager.start(self.request(folder))
            with self.assertRaises(ValueError): self.manager().start(self.request(folder))
        finally: gate.set()
        self.assertEqual(self.finish(manager)['operation']['state'], 'succeeded')

    def test_shutdown_cancels_setup_before_profile_commit_or_worker_activation(self):
        manager=self.manager()
        def cancelled(args,env,log):
            manager.close()
            return self.fake_run(args,env,log)
        manager.runner=cancelled
        manager.start(self.request(self.existing()))
        self.assertEqual(self.finish(manager)['operation']['state'],'failed')
        self.assertFalse(self.config.exists())
        self.assertEqual(self.activated,[])

    def test_invalid_marker_is_rejected_as_unmanaged_folder(self):
        folder=self.base/'bad_marker';folder.mkdir()
        (folder/'.facade-environment.json').write_text('[]',encoding='utf-8')
        with self.assertRaises(ValueError):self.manager().start(self.request(folder,'create'))

    @unittest.skipIf(os.name == 'nt', 'Linux venv interpreter symlink semantics')
    def test_existing_venv_python_symlink_is_not_resolved_to_system_interpreter(self):
        folder = self.base / 'linked'
        python = self.python(folder)
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        manager = self.manager()
        manager.start(self.request(folder))
        self.assertEqual(self.finish(manager)['profile']['python'], str(python))

    def test_selected_gpu_must_be_reported_with_native_bf16(self):
        folder = self.existing()
        def wrong_gpu(args, env, log):
            report = json.loads(self.fake_run(args, env, log))
            report['devices'] = [dict(index=0, native_bf16=True)]
            return json.dumps(report)
        manager = self.manager(runner=wrong_gpu)
        manager.start(self.request(folder))
        status = self.finish(manager)
        self.assertFalse(status['ready'])
        self.assertEqual(status['operation']['state'], 'failed')
        self.assertFalse(self.config.exists())


if __name__ == '__main__':
    unittest.main()
