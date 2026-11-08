import json
import threading
import unittest
import urllib.request
import urllib.error
from test_logic import local_tmp
from facade_next.server import make_server


class HTTPTests(unittest.TestCase):
    def test_no_model_workflow_and_origin_protection(self):
        with local_tmp() as d:
            server = make_server(d,0)
            t = threading.Thread(target=server.serve_forever,daemon=True)
            t.start()
            base = 'http://127.0.0.1:'+str(server.server_port)
            try:
                with urllib.request.urlopen(base+'/') as r:
                    self.assertIn(b'building-facade-engineering-agent',r.read())
                with urllib.request.urlopen(base+'/api/state') as r:
                    state = json.load(r)
                payload = json.dumps({'action':'create_project','payload':{'name':'HTTP test'},'submission_id':'http-once'}).encode()
                req = urllib.request.Request(base+'/api/action',data=payload,headers={'Content-Type':'application/json'})
                with self.assertRaises(urllib.error.HTTPError) as e:
                    urllib.request.urlopen(req)
                self.assertEqual(e.exception.code,403)
                req.add_header('X-Facade-CSRF',state['csrf'])
                with urllib.request.urlopen(req) as r:
                    created = json.load(r)
                with urllib.request.urlopen(req) as r:
                    self.assertEqual(json.load(r),created)
                req.add_header('Origin','https://foreign.example')
                with self.assertRaises(urllib.error.HTTPError) as e:
                    urllib.request.urlopen(req)
                self.assertEqual(e.exception.code,403)
                self.assertEqual(len(server.service.snapshot()['events']),1)
            finally:
                server.shutdown()
                server.server_close()
                t.join(timeout=3)
