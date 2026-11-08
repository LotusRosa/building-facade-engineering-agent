import unittest
from pathlib import Path
from test_logic import local_tmp
from facade_next.service import Service


class InitializationTests(unittest.TestCase):
    def test_initial_round_uses_imagenet_and_rejects_random_initialization(self):
        with local_tmp() as d:
            root=Path(d);s=Service(root/'state');n=0
            def call(action,payload):
                nonlocal n
                n+=1;return s.dispatch(action,payload,str(n))
            project=call('create_project',{'name':'ImageNet default'})['id'];parts={}
            for role in ('train','val','evaluation'):
                folder=root/role;folder.mkdir()
                for i in range(2):(folder/(str(i)+'.jpg')).write_bytes((role+str(i)).encode())
                rows=[dict(path=str(i)+'.jpg',hollow=i,spalling=i,crack=i) for i in range(2)]
                parts[role]=[call('import_dataset',dict(project=project,root=str(folder),rows=rows))['id']]
            payload=dict(project=project,**parts,backbone='resnet50',tools=['standard'])
            with self.assertRaisesRegex(ValueError,'ImageNet'):
                call('create_round',dict(payload,initialization='scratch'))
            result=call('create_round',payload)
            rnd=next(r for r in s.snapshot()['rounds'] if r['id']==result['id'])
            self.assertEqual(rnd['initialization'],'cached_imagenet')
            self.assertEqual(s.snapshot()['jobs'],[])
