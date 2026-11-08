import unittest
from unittest.mock import patch
from facade_next.worker import server_guard
from facade_next.platform_support import supported_host,process_alive


class PlatformTests(unittest.TestCase):
    def test_windows_and_linux_target_without_model_import(self):
        self.assertTrue(supported_host('Windows','AMD64'))
        self.assertTrue(supported_host('Linux','x86_64'))
        self.assertFalse(supported_host('Darwin','arm64'))
        for system in ('Windows','Linux'):
            with patch('facade_next.platform_support.platform.system',return_value=system),patch('facade_next.platform_support.platform.machine',return_value='AMD64'),patch.dict('os.environ',{'FACADE_DISABLE_MODEL_EXECUTION':'0'}):
                server_guard(True)
        with self.assertRaises(RuntimeError):
            server_guard(False)

    def test_current_pid_read_only(self):
        import os
        self.assertTrue(process_alive(os.getpid()))

    def test_portable_roles_bind_content_and_labels_but_not_os_paths(self):
        import copy
        from facade_next.contracts import portable_roles_digest
        row=dict(image_id='a',sha256='a'*64,labels=[1,0,None],scene='A',group='building',path='C:/photos/a.jpg')
        a={'train':[row],'history':[],'val':[],'evaluation':[]}
        b=copy.deepcopy(a);b['train'][0]['path']='/home/photos/a.jpg'
        self.assertEqual(portable_roles_digest(a),portable_roles_digest(b))
        b['train'][0]['labels'][1]=1
        self.assertNotEqual(portable_roles_digest(a),portable_roles_digest(b))
