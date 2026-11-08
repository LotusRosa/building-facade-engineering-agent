import json
from pathlib import Path
import threading
import unittest
import urllib.request
from test_logic import local_tmp
from facade_next.server import make_server


class HistoryHTTPTests(unittest.TestCase):
    def test_original_agent_serves_integrated_history_without_workflow_submissions(self):
        with local_tmp() as root:
            server=make_server(Path(root)/'state',0,environment_path=Path(root)/'environment.json')
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            base='http://127.0.0.1:'+str(server.server_port)
            before=server.service.snapshot()
            try:
                for route,marker in [('/',b'id="chatInput"'),('/history.js',b'AgentHistory'),('/history.css',b'agentHistorySheet')]:
                    with opener.open(base+route,timeout=5) as response:
                        self.assertEqual(response.status,200);self.assertIn(marker,response.read())
                with opener.open(base+'/api/state',timeout=5) as response:self.assertEqual(json.load(response)['jobs'],[])
                self.assertEqual(server.service.snapshot(),before)
                self.assertFalse(server.workers_enabled)
            finally:
                server.shutdown();server.server_close();server.environment.close();thread.join(3)

if __name__=='__main__':unittest.main()
