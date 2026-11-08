"""Host-wide training environment setup. Importing this module is stdlib-only."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid
from .contracts import BACKBONES, read, write
from .platform_support import process_alive


def config_path():
    return Path(os.environ.get('FACADE_ENV_CONFIG', str(Path.home()/'.facade-agent'/'environment.json'))).expanduser().absolute()


def process_environment(torch_home=None):
    env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1')
    for key in ('PYTHONHOME','PYTHONPATH','PIP_TARGET','PIP_PREFIX','PIP_USER','PYTHONUSERBASE'):
        env.pop(key,None)
    if torch_home:
        env['TORCH_HOME']=str(torch_home)
    return env


def python_in(folder):
    # Do not resolve the final symlink: Linux venv/bin/python must remain in its venv.
    for sub in ('Scripts/python.exe', 'bin/python', '.venv/Scripts/python.exe', '.venv/bin/python'):
        path = folder/sub
        if path.is_file():
            return path.absolute()
    raise ValueError('No Python interpreter in this folder. Choose the existing venv folder, or create a new environment.')


def gpu_list(value):
    values = str(value).split(',')
    if not values or any(not x.strip().isdigit() for x in values):
        raise ValueError('GPU indices must be numbers such as 0 or 0,1.')
    result = [str(int(x.strip())) for x in values]
    if len(result) != len(set(result)):
        raise ValueError('GPU indices must be distinct.')
    return ','.join(result)


def browse_folders(value=None):
    path = Path(value or Path.home()).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise ValueError('Choose a folder on the Agent host.')
    children = []
    for item in path.iterdir():
        try:
            if item.is_dir():
                children.append(dict(name=item.name, path=str(item)))
        except OSError:
            continue
    roots = [str(Path('/'))] if os.name != 'nt' else [letter+':\\' for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if Path(letter+':\\').is_dir()]
    return dict(path=str(path), parent=str(path.parent), roots=roots,
                folders=sorted(children, key=lambda x:x['name'].lower())[:300])


class EnvironmentManager:
    def __init__(self, path=None, runner=None, busy=None, activate=None, transition=None):
        self.path = Path(path).absolute() if path is not None else config_path()
        self.runner = runner or self._run
        self.native_runner = runner is None
        self.busy = busy or (lambda: False)
        self.activate = activate or (lambda profile: None)
        self.lock = threading.RLock()
        self.transition = transition or threading.RLock()
        self.thread = None
        self.process = None
        self.closed = threading.Event()
        self.started_at = None
        self.operation = dict(state='idle', phase='', error='', log='', download=None, activity='', elapsed_seconds=0)

    def profile(self):
        try:
            p = read(self.path)
            if p.get('schema') != 'facade_environment_v1':
                return None
            if any(not isinstance(p.get(k),str) or not p[k] for k in ('folder','python','torch_home','gpus')) or not isinstance(p.get('enable_workers'),bool):
                return None
            if not Path(p['folder']).is_absolute() or not Path(p['python']).is_absolute() or not Path(p['torch_home']).is_absolute():
                return None
            gpu_list(p['gpus'])
            return p
        except (OSError, ValueError, TypeError, AttributeError):
            return None

    def snapshot(self):
        with self.lock:
            p = self.profile()
            ready = bool(p and Path(p['python']).is_file())
            operation=dict(self.operation)
            if operation['state']=='running' and self.started_at is not None:
                operation['elapsed_seconds']=max(0,int(time.monotonic()-self.started_at))
            operation['process_running']=bool(self.process and self.process.poll() is None)
            return dict(configured=bool(p), ready=ready, profile=p,
                        operation=operation, config_path=str(self.path),
                        current_python=sys.executable, current_prefix=sys.prefix,
                        default_folder=str(Path.home()/'FacadeAgentEnvironment'))

    def _log(self, text):
        with self.lock:
            line=str(text).strip()
            if line.startswith('FACADE_WEIGHT_PROGRESS '):
                try:
                    item=json.loads(line.split(' ',1)[1])
                    valid=(isinstance(item,dict) and item.get('backbone') in BACKBONES
                        and item.get('status') in ('downloading','retrying','verifying','ready','cached')
                        and type(item.get('received')) is int and item['received']>=0
                        and (item.get('total') is None or type(item['total']) is int and item['total']>=item['received'] and item['total']>0)
                        and type(item.get('attempt')) is int and 0<=item['attempt']<=3
                        and type(item.get('verified')) is bool)
                    if valid:self.operation['download']={k:item[k] for k in ('backbone','status','received','total','attempt','verified')}
                except (ValueError,TypeError,KeyError):
                    pass
            elif line.startswith(('Downloading ','Using cached ','Collecting ','Installing collected packages','Requirement already satisfied:')):
                self.operation['activity']=line[:240]
            self.operation['log'] = (self.operation['log']+str(text)+'\n')[-16000:]

    def _phase(self, phase):
        if self.closed.is_set():
            raise ValueError('Environment setup was cancelled because the Agent service stopped.')
        with self.lock:
            self.operation['phase'] = phase
        self._log(phase)

    def _run(self, args, env, log):
        with self.transition, self.lock:
            if self.closed.is_set():
                raise ValueError('Environment setup was cancelled because the Agent service stopped.')
            process = subprocess.Popen([str(x) for x in args], env=env, cwd=Path(__file__).resolve().parent.parent,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                text=True, encoding='utf-8', errors='replace',
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            self.process = process
        timer = threading.Timer(3600, process.kill)
        timer.daemon = True
        timer.start()
        output = ''
        try:
            for line in process.stdout:
                log(line.rstrip())
                output = (output+line)[-64000:]
            code = process.wait()
            if code:
                raise RuntimeError('Environment command failed (exit '+str(code)+').\n'+output[-5000:])
            return output
        finally:
            timer.cancel()
            process.stdout.close()
            with self.lock:
                self.process = None

    def start(self, request):
        if self.native_runner and os.environ.get('FACADE_DISABLE_MODEL_EXECUTION')=='1':
            raise ValueError('Environment setup is disabled by this host session policy.')
        if not isinstance(request, dict) or set(request)-{'folder','mode','gpus','weights','enable_workers'}:
            raise ValueError('Unsupported environment setup option.')
        if request.get('mode') not in ('create','existing'):
            raise ValueError('Choose create or existing environment.')
        folder = Path(str(request.get('folder',''))).expanduser()
        if not folder.is_absolute():
            raise ValueError('Use an absolute folder path on the Agent host.')
        folder = folder.resolve()
        if folder.parent == folder or folder == Path.home().resolve():
            raise ValueError('Choose a dedicated environment folder, not a filesystem root or your home folder.')
        gpus = gpu_list(request.get('gpus','0'))
        weights = request.get('weights', [])
        if not isinstance(weights, list) or len(weights)!=len(set(weights)) or any(w not in BACKBONES for w in weights):
            raise ValueError('Choose supported backbone weights.')
        if not isinstance(request.get('enable_workers', True), bool):
            raise ValueError('enable_workers must be true or false.')
        if request['mode']=='existing':
            python_in(folder)
        elif folder.exists():
            if not folder.is_dir():
                raise ValueError('Environment location is not a folder.')
            marker = folder/'.facade-environment.json'
            try:
                marker_data=read(marker) if marker.is_file() else {}
            except (ValueError,OSError):
                marker_data={}
            if not isinstance(marker_data,dict):
                marker_data={}
            if any(folder.iterdir()) and marker_data.get('manager')!='facade-env-v1':
                raise ValueError('This folder is not empty. Choose a new folder, or use the existing-environment option.')
            saved=self.profile()
            if marker_data.get('completed') or (saved and Path(saved['folder']).resolve()==folder):
                raise ValueError('This environment is already configured. Choose existing environment to recheck it without reinstalling.')
            paths=[folder/x for x in ('.venv','.venv/bin','.venv/Scripts','.venv/Lib','.venv/lib','torch','torch/hub','torch/hub/checkpoints','pip_cache')]
            for path in paths:
                if path.is_symlink() or getattr(path,'is_junction',lambda:False)():
                    raise ValueError('Managed environment/cache directories must not be symlinks or junctions: '+str(path))
        with self.transition, self.lock:
            if self.closed.is_set():
                raise ValueError('The Agent service is shutting down.')
            if self.thread and self.thread.is_alive():
                raise ValueError('Environment setup is already running.')
            if self.busy():
                raise ValueError('Wait until queued/running training tasks finish before changing the shared environment.')
            self.path.parent.mkdir(parents=True, exist_ok=True)
            lockfile = self.path.with_suffix('.setup.lock')
            token = uuid.uuid4().hex
            try:
                fd = os.open(lockfile, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
            except FileExistsError:
                try:
                    old = read(lockfile)
                    if process_alive(old['pid']):
                        raise ValueError('Another Agent is configuring this shared environment.')
                    lockfile.unlink()
                    fd = os.open(lockfile, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
                except (OSError, KeyError, TypeError):
                    raise ValueError('Environment setup lock is unavailable; inspect '+str(lockfile)) from None
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(dict(pid=os.getpid(), token=token), f)
            self.started_at=time.monotonic()
            self.operation = dict(state='running', phase='Preparing', error='', log='', download=None, activity='', elapsed_seconds=0)
            options = dict(request, folder=str(folder), gpus=gpus, weights=list(weights))
            self.thread = threading.Thread(target=self._setup, args=(options,lockfile,token), daemon=True)
            self.thread.start()
        return self.snapshot()

    def _pip_install(self, python, requirements, env, index_url=None):
        # venv already supplies pip. Its network upgrade must not block setup.
        # pip 24 retries connections, but a read timeout mid-wheel can escape
        # those retries. Retry that command a bounded number of times as well.
        args = [str(python), '-m', 'pip', 'install', *requirements,
                '--timeout', '120', '--retries', '5',
                '--disable-pip-version-check', '--no-input', '--progress-bar', 'off']
        if index_url:
            args += ['--index-url', index_url]
        transient = ('readtimeout', 'read timed out', 'connecttimeout',
                     'connection reset by peer', 'connectionreseterror',
                     'connection broken: incomplete', 'incomplete read', 'incompleteread',
                     'unexpected_eof_while_reading', 'ssl eof', 'connection aborted')
        for attempt in range(1, 4):
            if self.closed.is_set():
                raise ValueError('Environment setup was cancelled because the Agent service stopped.')
            self._log('Dependency installation attempt '+str(attempt)+'/3')
            try:
                return self.runner(args, env, self._log)
            except RuntimeError as error:
                if self.closed.is_set():
                    raise ValueError('Environment setup was cancelled because the Agent service stopped.') from error
                message=str(error).lower()
                fatal=('certificate_verify_failed','certificate verify failed','resolutionimpossible',
                       'no matching distribution found','permission denied','no space left on device')
                if any(marker in message for marker in fatal) or not any(marker in message for marker in transient):
                    raise
                if attempt == 3:
                    raise RuntimeError('Dependency download failed after 3 attempts. The unfinished environment is retained; retry the same folder after checking the download connection.\n'+str(error)) from error
                self._log('Download interrupted. Retrying the same installation command ('+str(attempt+1)+'/3).')

    def _setup(self, options, lockfile, token):
        try:
            folder = Path(options['folder'])
            env = process_environment()
            if options['mode']=='create':
                folder.mkdir(parents=True, exist_ok=True)
                write(folder/'.facade-environment.json', dict(manager='facade-env-v1'))
                env['TORCH_HOME'] = str(folder/'torch')
                env['PIP_CACHE_DIR'] = str(folder/'pip_cache')
                env['PIP_CONFIG_FILE'] = os.devnull
                venv = folder/'.venv'
                self._phase('Creating the training environment')
                if not any((venv/p).is_file() for p in ('bin/python','Scripts/python.exe')):
                    self.runner([sys.executable,'-m','venv',str(venv)], env, self._log)
                python = python_in(venv)
                self._phase('Installing shared training dependencies')
                self._pip_install(python, ['torch==2.11.0','torchvision==0.26.0'], env,
                                  index_url='https://download.pytorch.org/whl/cu128')
                self._pip_install(python, ['numpy>=1.26','Pillow>=10.4'], env)
            else:
                python = python_in(folder)
                saved=self.profile()
                if saved and os.path.normcase(saved['python'])==os.path.normcase(str(python)):
                    env['TORCH_HOME']=saved['torch_home']
                else:
                    for managed in (folder,folder.parent):
                        marker=managed/'.facade-environment.json'
                        if marker.is_file() and read(marker).get('manager')=='facade-env-v1':
                            env['TORCH_HOME']=str(managed/'torch')
                            break
            self._phase('Checking Python, CUDA/BF16 and requested ImageNet weights')
            args = [str(python), str(Path(__file__).with_name('environment_probe.py')), '--execute', '--gpus', options['gpus']]
            if options['weights']:
                args += ['--weights', *options['weights']]
            output = self.runner(args, env, self._log)
            report = json.loads(output.strip().splitlines()[-1])
            if report.get('ok') is not True or report.get('cuda') is not True:
                raise ValueError('The selected environment did not pass CUDA validation.')
            devices = {str(d['index']):d for d in report['devices']}
            if any(i not in devices or not devices[i].get('native_bf16') for i in options['gpus'].split(',')):
                raise ValueError('Selected GPUs must be available and support native BF16.')
            profile = dict(schema='facade_environment_v1', folder=str(folder), python=str(python),
                torch_home=str(Path(report['torch_home']).expanduser().absolute()), gpus=options['gpus'],
                enable_workers=options.get('enable_workers', True), report=report,
                configured_utc=datetime.now(timezone.utc).isoformat())
            with self.transition:
                if self.closed.is_set():
                    raise ValueError('Environment setup was cancelled because the Agent service stopped.')
                if self.busy():
                    raise ValueError('Training tasks started during setup. Configuration was not switched; try again after they finish.')
                if options['mode']=='create':
                    write(folder/'.facade-environment.json', dict(manager='facade-env-v1', completed=True))
                old = self.path.read_bytes() if self.path.exists() else None
                write(self.path, profile)
                try:
                    self.activate(profile)
                except Exception:
                    if old is None:
                        self.path.unlink(missing_ok=True)
                    else:
                        write(self.path, json.loads(old))
                    raise
                with self.lock:
                    self.operation.update(state='succeeded', phase='Shared environment ready')
        except Exception as e:
            with self.lock:
                self.operation.update(state='failed', phase='Setup failed', error=str(e))
            self._log(str(e))
        finally:
            with self.lock:
                if self.started_at is not None:
                    self.operation['elapsed_seconds']=max(0,int(time.monotonic()-self.started_at))
            try:
                if read(lockfile).get('token')==token:
                    lockfile.unlink()
            except (OSError, ValueError):
                pass

    def close(self):
        with self.transition, self.lock:
            self.closed.set()
            if self.process and self.process.poll() is None:
                self.process.terminate()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(3)
