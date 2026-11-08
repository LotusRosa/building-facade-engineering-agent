import json
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from test_logic import local_tmp
from test_llm import tool
from facade_next.server import make_server
from facade_next.llm import LLMClient


class LLMHTTPTests(unittest.TestCase):
    def test_settings_are_csrf_protected_memory_only_and_test_is_explicit(self):
        with local_tmp() as d:
            server=make_server(d,0)
            calls=[]
            server.service.llm=LLMClient(environ={},transport=lambda *args:calls.append(args) or tool('status'))
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            base='http://127.0.0.1:'+str(server.server_port)
            def get(path):
                with urllib.request.urlopen(base+path) as r:return json.load(r)
            csrf=get('/api/state')['csrf']
            def post(path,data,token=csrf):
                req=urllib.request.Request(base+path,data=json.dumps(data).encode(),headers={'Content-Type':'application/json','X-Facade-CSRF':token})
                with urllib.request.urlopen(req) as r:return json.load(r)
            try:
                config=dict(provider='deepseek',model='deepseek-flash',api_key='only-in-memory-test-key')
                with self.assertRaises(urllib.error.HTTPError) as exc:post('/api/llm/config',config,'wrong')
                self.assertEqual(exc.exception.code,403)
                status=post('/api/llm/config',config)
                self.assertTrue(status['enabled']);self.assertEqual(calls,[])
                self.assertNotIn('only-in-memory-test-key',json.dumps(status))
                self.assertTrue(post('/api/llm/test',{})['ok']);self.assertEqual(len(calls),1)
                state=get('/api/state')
                self.assertTrue(state['llm']['enabled'])
                self.assertNotIn('only-in-memory-test-key',json.dumps(state))
                self.assertEqual(server.service.snapshot()['events'],[])
                self.assertNotIn(b'only-in-memory-test-key',(Path(d)/'audit.sqlite3').read_bytes())
                self.assertFalse(post('/api/llm/config',{'provider':'off'})['enabled'])
                with self.assertRaises(urllib.error.HTTPError):post('/api/llm/test',{})
            finally:
                server.shutdown();server.server_close();thread.join(timeout=3)

    def test_real_http_wire_and_redirect_never_forward_credentials(self):
        from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
        seen=[];redirect=[False]
        class Fake(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                seen.append((self.path,self.headers.get('Authorization'),body))
                if redirect[0]:
                    self.send_response(307);self.send_header('Location','/stolen');self.end_headers();return
                raw=json.dumps(tool('status')).encode()
                self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        server=ThreadingHTTPServer(('127.0.0.1',0),Fake)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            client=LLMClient(environ={})
            client.configure(dict(provider='compatible',base_url='http://127.0.0.1:'+str(server.server_port)+'/v1',model='fake',api_key='fake-wire-key'))
            self.assertEqual(client.complete([dict(role='user',content='status')])['intent'],'status')
            self.assertEqual(seen[0][0],'/v1/chat/completions')
            self.assertEqual(seen[0][1],'Bearer fake-wire-key')
            redirect[0]=True
            with self.assertRaisesRegex(ValueError,'HTTP 307'):client.complete([])
            self.assertEqual(len(seen),2)
            self.assertNotIn('/stolen',[x[0] for x in seen])
        finally:
            server.shutdown();server.server_close();thread.join(timeout=3)
