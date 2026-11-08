import json
from pathlib import Path
import threading
import unittest
import urllib.error
import urllib.request
from test_logic import local_tmp
from facade_next.server import make_server


class EnvironmentHTTPTests(unittest.TestCase):
    def test_setup_requires_csrf_and_persists_host_profile(self):
        with local_tmp() as root:
            from facade_next.environment import python_in
            folder=Path(root)/'env';python=folder/'Scripts/python.exe'
            import os
            if os.name!='nt':python=folder/'bin/python'
            python.parent.mkdir(parents=True);python.write_bytes(b'fixture')
            calls=[]
            def run(args,env,log):
                calls.append(args)
                return json.dumps(dict(ok=True,cuda=True,torch_home=str(folder/'torch'),
                    devices=[dict(index=0,native_bf16=True)],packages={}))
            server=make_server(Path(root)/'state',0,environment_path=Path(root)/'shared.json',environment_runner=run)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            base='http://127.0.0.1:'+str(server.server_port)
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            def request(path,data=None,csrf=None):
                headers={'Content-Type':'application/json'}
                if csrf:headers['X-Facade-CSRF']=csrf
                return opener.open(urllib.request.Request(base+path,data=json.dumps(data).encode() if data is not None else None,headers=headers))
            try:
                with request('/api/state') as response:state=json.load(response)
                self.assertFalse(state['environment']['ready'])
                payload=dict(folder=str(folder),mode='existing',weights=[],gpus='0',enable_workers=False)
                with self.assertRaises(urllib.error.HTTPError) as failure:request('/api/environment/setup',payload)
                self.assertEqual(failure.exception.code,403)
                with request('/api/environment/setup',payload,state['csrf']) as response:self.assertEqual(response.status,200)
                server.environment.thread.join(5)
                with request('/api/environment') as response:status=json.load(response)
                self.assertTrue(status['ready'])
                self.assertEqual(status['profile']['python'],str(python_in(folder)))
                self.assertEqual(len(calls),1)
                self.assertFalse(server.workers_enabled)
                with request('/api/environment/folders?path='+urllib.parse.quote(str(root))) as response:
                    self.assertIn('env',[x['name'] for x in json.load(response)['folders']])
                with server.service.store.transaction() as db:
                    server.service.store.put(db,'job',dict(id='queued',status='queued',project='fixture'))
                with self.assertRaises(urllib.error.HTTPError) as failure:request('/api/environment/setup',payload,state['csrf'])
                self.assertEqual(failure.exception.code,400)
                self.assertEqual(len(calls),1)
            finally:
                server.shutdown();server.server_close();thread.join(3)

    def test_setup_reservation_blocks_round_submission(self):
        with local_tmp() as root:
            server=make_server(root,0,environment_path=Path(root)/'shared.json')
            try:
                server.environment.operation['state']='running'
                with server.service.store.transaction() as db:
                    with self.assertRaisesRegex(ValueError,'Environment setup'):
                        server.service.action_start_round(db,dict(round='missing_fixture'))
                server.environment.operation['state']='idle'
                with server.service.store.transaction() as db:
                    with self.assertRaisesRegex(ValueError,'Configure the shared'):
                        server.service.action_start_round(db,dict(round='missing_fixture'))
            finally:server.server_close()
