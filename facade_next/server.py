"""Loopback-only local browser service and optional one-worker-per-GPU launcher."""
import json
import os
import mimetypes
import secrets
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from .service import Service
from .contracts import read
from .environment import EnvironmentManager, browse_folders, process_environment
from .progress import decorate_jobs

STATIC = Path(__file__).parent/'static'


def make_server(state,port=8765,environment_path=None,environment_runner=None):
    service = Service(state)
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass

        def send(self,status,value,content_type='application/json; charset=utf-8',filename=None):
            data = json.dumps(value,ensure_ascii=False,allow_nan=False).encode() if content_type.startswith('application/json') else value
            self.send_response(status)
            self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store')
            if filename:
                self.send_header('Content-Disposition','attachment; filename="'+filename+'"')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def trusted(self):
            allowed = {f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
            if self.headers.get('Host') not in allowed:
                return False
            origin = self.headers.get('Origin')
            if origin and origin not in {'http://'+host for host in allowed}:
                return False
            return self.headers.get('Sec-Fetch-Site') not in ('cross-site',)

        def do_GET(self):
            if not self.trusted():
                return self.send(403,{'error':'Untrusted origin/Host'})
            url = urlparse(self.path)
            if url.path=='/api/state':
                return self.send(200,dict(decorate_jobs(service.snapshot()),csrf=token,llm=service.llm.status(),workers_enabled=self.server.workers_enabled,environment=self.server.environment.snapshot()))
            if url.path=='/api/environment':
                return self.send(200,self.server.environment.snapshot())
            if url.path=='/api/environment/folders':
                try:
                    return self.send(200,browse_folders(parse_qs(url.query).get('path',[None])[0]))
                except (OSError,ValueError) as e:
                    return self.send(400,{'error':str(e)})
            if url.path=='/api/template':
                from .templates import template_bytes
                kind=parse_qs(url.query).get('format',['csv'])[0]
                if kind not in ('csv','xlsx'):
                    return self.send(400,{'error':'Template must be csv/xlsx'})
                mime='text/csv; charset=utf-8' if kind=='csv' else 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                return self.send(200,template_bytes(kind),mime,'labels.'+kind)
            if url.path=='/api/photo':
                query = parse_qs(url.query)
                try:
                    with service.store.transaction() as db:
                        draft = service.store.get(db,'draft',query['draft'][0])
                        row = draft['rows'][int(query['index'][0])]
                    path = (Path(draft['root'])/row['path']).resolve()
                    if not path.is_relative_to(Path(draft['root']).resolve()):
                        raise ValueError('Photo path escaped root')
                    return self.send(200,path.read_bytes(),mimetypes.guess_type(path)[0] or 'application/octet-stream')
                except (KeyError,ValueError,IndexError,OSError) as e:
                    return self.send(400,{'error':str(e)})
            if url.path=='/api/log':
                try:
                    id = parse_qs(url.query)['job'][0]
                    with service.store.transaction() as db:
                        job = service.store.get(db,'job',id)
                    p = Path(job['output'])/'worker.log'
                    with p.open('rb') as f:
                        f.seek(max(0,p.stat().st_size-32000))
                        return self.send(200,{'log':f.read().decode('utf-8',errors='replace')})
                except (KeyError,ValueError,OSError) as e:
                    return self.send(400,{'error':str(e)})
            names = {'/':'index.html','/app.js':'app.js','/chat.js':'chat.js','/workspace.js':'workspace.js','/llm.js':'llm.js','/environment.js':'environment.js','/style.css':'style.css','/history.js':'history.js','/history.css':'history.css'}
            if url.path in names:
                p = STATIC/names[url.path]
                return self.send(200,p.read_bytes(),mimetypes.guess_type(p)[0] or 'text/plain')
            self.send(404,{'error':'Not found'})

        def do_POST(self):
            if not self.trusted() or not secrets.compare_digest(self.headers.get('X-Facade-CSRF',''),token):
                return self.send(403,{'error':'CSRF/origin check failed'})
            if self.path not in ('/api/action','/api/llm/config','/api/llm/test','/api/environment/setup'):
                return self.send(404,{'error':'Not found'})
            try:
                size = int(self.headers.get('Content-Length','0'))
                if not 0 < size <= 32*1024*1024:
                    raise ValueError('Request must be between 1 byte and 32MiB')
                req = json.loads(self.rfile.read(size))
                if self.path=='/api/environment/setup':
                    if size>16384:
                        raise ValueError('Environment settings are too large.')
                    return self.send(200,self.server.environment.start(req))
                if self.path.startswith('/api/llm/'):
                    if size>16384 or not isinstance(req,dict):
                        raise ValueError('LLM settings must be a small JSON object')
                    if self.path=='/api/llm/config':
                        return self.send(200,service.llm.configure(req))
                    if req:
                        raise ValueError('Connection test takes no project data')
                    service.llm.complete([{'role':'system','content':'This is a connection test. Reply with one short greeting; do not call a tool.'},
                                          {'role':'user','content':'Hello'}])
                    return self.send(200,dict(ok=True,**service.llm.status()))
                with self.server.transition:
                    result = service.dispatch(req['action'],req['payload'],req['submission_id'])
                self.send(200,result)
            except (ValueError,KeyError,TypeError,OSError) as e:
                self.send(400,{'error':str(e)})

    server = ThreadingHTTPServer(('127.0.0.1',port),Handler)
    server.service = service
    server.daemon_threads = True
    server.transition = threading.RLock()
    server.workers_enabled = False
    server.controller = WorkerController(server)
    server.environment = EnvironmentManager(environment_path,runner=environment_runner,
        busy=server.controller.busy,activate=server.controller.configure,transition=server.transition)
    def before_start():
        if server.environment.operation['state']=='running':
            raise ValueError('Environment setup is in progress. Wait until it finishes before starting training.')
        if not server.workers_enabled:
            raise ValueError('Configure the shared training environment in Settings before starting training, or explicitly enable GPU workers at launch.')
    service.before_start = before_start
    return server


class Scheduler:
    def __init__(self,service,gpus,python=None,torch_home=None,transition=None):
        self.service = service
        self.devices = ['cuda:'+s for s in gpus.split(',')]
        if not all(s.removeprefix('cuda:').isdigit() for s in self.devices) or len(set(self.devices))!=len(self.devices):
            raise ValueError('GPU list must contain distinct indices, e.g. 0,1')
        self.running = {}
        self.stop = threading.Event()
        self.python = python or sys.executable
        self.torch_home = torch_home
        self.transition = transition or threading.RLock()

    def loop(self):
        while not self.stop.is_set():
            try:
                with self.transition:
                    if self.stop.is_set():
                        break
                    for device,(process,log) in list(self.running.items()):
                        if process.poll() is not None:
                            log.close()
                            with self.service.store.transaction() as db:
                                existing = self.service.store.all(db,'job')
                            for job in existing:
                                if job['status']=='running' and job.get('pid')==process.pid:
                                    self.service.finish(job['id'],error='Worker exited without completing receipt; inspect worker.log')
                            del self.running[device]
                    with self.service.store.transaction() as db:
                        jobs = self.service.store.all(db,'job')
                    for device in self.devices:
                        if device in self.running or any(j['status']=='running' and j.get('device')==device for j in jobs):
                            continue
                        if not any(j['status']=='queued' for j in jobs):
                            break
                        # Per-device launcher log; job-specific worker.log is created
                        # by worker after its transactional claim.
                        logs = self.service.root/'launcher_logs'
                        logs.mkdir(exist_ok=True)
                        log = (logs/(device.replace(':','_')+'.log')).open('ab')
                        process = subprocess.Popen([self.python,'-m','facade_next','worker','--state',str(self.service.root),'--device',device,'--execute'],
                                                   cwd=Path(__file__).resolve().parent.parent,stdout=log,stderr=subprocess.STDOUT,
                                                   env=process_environment(self.torch_home),
                                                   stdin=subprocess.DEVNULL,start_new_session=os.name!='nt',
                                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                        self.running[device] = process,log
            except Exception as e:
                print('Scheduler:',e,flush=True)
            self.stop.wait(2)


class WorkerController:
    def __init__(self,server):
        self.server=server
        self.scheduler=None

    def busy(self):
        with self.server.transition:
            if self.scheduler and self.scheduler.running:
                return True
            with self.server.service.store.transaction() as db:
                return any(j['status'] in ('queued','running') for j in self.server.service.store.all(db,'job'))

    def configure(self,profile,initial=False):
        with self.server.transition:
            if not initial and self.busy():
                raise ValueError('Wait for queued/running training before switching environments.')
            if profile.get('enable_workers'):
                from .worker import server_guard
                server_guard(True)
            if self.scheduler:
                self.scheduler.stop.set()
            self.scheduler=None
            self.server.workers_enabled=False
            if profile.get('enable_workers'):
                self.scheduler=Scheduler(self.server.service,profile['gpus'],python=profile['python'],
                    torch_home=profile.get('torch_home'),transition=self.server.transition)
                threading.Thread(target=self.scheduler.loop,daemon=True).start()
                self.server.workers_enabled=True

    def close(self):
        if self.scheduler:
            self.scheduler.stop.set()


def serve(state,port,execute_workers=False,gpus='0',environment_path=None,ui_only=False,open_browser=False):
    server=make_server(state,port,environment_path=environment_path)
    profile=server.environment.snapshot()
    if not ui_only:
        if profile['ready']:
            saved=dict(profile['profile'])
            saved['enable_workers']=saved.get('enable_workers') or execute_workers
            if saved['enable_workers']:
                server.controller.configure(saved,initial=True)
        elif execute_workers and not profile['configured']:
            server.controller.configure(dict(python=sys.executable,gpus=gpus,enable_workers=True,
                torch_home=os.environ.get('TORCH_HOME')),initial=True)
    print(f'building-facade-engineering-agent: http://127.0.0.1:{server.server_port}',flush=True)
    print('GPU workers enabled: '+str(server.workers_enabled),flush=True)
    if open_browser:
        import webbrowser
        timer=threading.Timer(.5,lambda:webbrowser.open('http://127.0.0.1:'+str(server.server_port)))
        timer.daemon=True
        timer.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.environment.close()
        server.controller.close()
        server.server_close()
