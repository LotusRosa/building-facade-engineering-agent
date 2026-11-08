"""Optional, dependency-free Chat Completions transport. Never persists keys."""
import json
import http.client
import os
import socket
import ssl
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from urllib.parse import urlsplit

PROVIDERS={'openai':'https://api.openai.com/v1','deepseek':'https://api.deepseek.com','compatible':''}
INTENTS=('create_project','begin_round','import_batch','import_folder','configure','status','compare','next','propose_decision','rollback','cancel_round')
TOOLS=[{'type':'function','function':{'name':'propose_task','description':
    'Propose ONE facade workflow step. The application validates it. Never confirms or executes training.',
    'parameters':{'type':'object','properties':{
        'intent':{'type':'string','enum':list(INTENTS)},
        'arguments':{'type':'object','properties':{
            'name':{'type':'string'},'root':{'type':'string'},
            'stage':{'type':'string','enum':['initial','update']},
            'role':{'type':'string','enum':['train','val','evaluation']},
            'backbone':{'type':'string','enum':['resnet50','convnext_tiny','swin_t']},
            'tools':{'type':'array','uniqueItems':True,'items':{'type':'string','enum':['core','standard','asl','sam','distill','hnm','hem','head_only','ewc_replay']}},
            'choice':{'type':'string','enum':['adopt','hold']},'job':{'type':'string'},
            'reason':{'type':'string'},'version':{'type':'string'}},'additionalProperties':False}},
        'required':['intent','arguments'],'additionalProperties':False}}}]


@dataclass(frozen=True)
class Config:
    provider:str='off'
    base_url:str=''
    model:str=''
    key:str=field(default='',repr=False)
    timeout:float=30


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):
        return None


def post(url,body,key,timeout):
    request=urllib.request.Request(url,data=json.dumps(body,ensure_ascii=False).encode('utf-8'),
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json','Accept':'application/json'})
    with urllib.request.build_opener(NoRedirect).open(request,timeout=timeout) as response:
        raw=response.read(1024*1024+1)
        if len(raw)>1024*1024:
            raise ValueError('LLM response exceeded the size limit')
        return json.loads(raw)


class LLMClient:
    def __init__(self,environ=None,transport=None):
        self._lock=threading.RLock()
        self._config=Config()
        self._error=''
        self.transport=transport or post
        env=os.environ if environ is None else environ
        provider=env.get('FACADE_LLM_PROVIDER','off').lower()
        if provider!='off':
            key=env.get('FACADE_LLM_API_KEY') or env.get({'openai':'OPENAI_API_KEY','deepseek':'DEEPSEEK_API_KEY'}.get(provider,''),'')
            try:
                self.configure(dict(provider=provider,base_url=env.get('FACADE_LLM_BASE_URL',''),
                    model=env.get('FACADE_LLM_MODEL',''),api_key=key))
            except ValueError as e:
                self._error=str(e)

    def configure(self,values):
        if not isinstance(values,dict) or set(values)-{'provider','base_url','model','api_key'}:
            raise ValueError('Unsupported LLM setting')
        provider=str(values.get('provider','off')).strip().lower()
        if provider=='off':
            with self._lock:self._config=Config();self._error=''
            return self.status()
        if provider not in PROVIDERS:
            raise ValueError('Choose openai, deepseek, compatible or off')
        base=str(values.get('base_url') or PROVIDERS[provider]).strip().rstrip('/')
        parsed=urlsplit(base)
        if (not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or
            (parsed.scheme!='https' and not (parsed.scheme=='http' and parsed.hostname in ('localhost','127.0.0.1','::1'))) or
            any(c.isspace() for c in base) or base.endswith(('/chat/completions','/responses'))):
            raise ValueError('Use an HTTPS API base URL (loopback HTTP is allowed), without credentials/query or endpoint suffix')
        model=str(values.get('model') or '').strip()
        if not model or len(model)>200 or any(ord(c)<32 for c in model):
            raise ValueError('Enter the model ID available in your API account')
        with self._lock:
            old=self._config
            key=str(values.get('api_key') or '').strip()
            if not key and (provider,base)==(old.provider,old.base_url):key=old.key
            if not key or len(key)>2048 or any(c.isspace() for c in key):
                raise ValueError('Enter an API key; changing provider or base URL requires a new key')
            self._config=Config(provider,base,model,key)
            self._error=''
        return self.status()

    def status(self):
        with self._lock:
            cfg=self._config
            return dict(enabled=cfg.provider!='off',provider=cfg.provider,base_url=cfg.base_url,
                        model=cfg.model,key_configured=bool(cfg.key),error=self._error)

    def complete(self,messages):
        with self._lock:cfg=self._config
        if cfg.provider=='off':raise ValueError('LLM is disabled')
        body=dict(model=cfg.model,messages=messages,tools=TOOLS,tool_choice='auto',stream=False)
        body['max_completion_tokens' if cfg.provider=='openai' else 'max_tokens']=2048
        # Avoid model-specific sampling and thinking parameters.
        try:
            data=self.transport(cfg.base_url+'/chat/completions',body,cfg.key,cfg.timeout)
            choices=data['choices']
            if len(choices)!=1:raise ValueError('Expected one response')
            choice=choices[0]
            if choice.get('finish_reason') not in ('stop','tool_calls'):raise ValueError('Incomplete response')
            message=choice['message']
            calls=message.get('tool_calls') or []
            if calls:
                if len(calls)!=1 or calls[0].get('type')!='function' or calls[0]['function']['name']!='propose_task':
                    raise ValueError('Unsupported tool call')
                proposal=json.loads(calls[0]['function']['arguments'])
                if not isinstance(proposal,dict) or set(proposal)!={'intent','arguments'} or proposal['intent'] not in INTENTS or not isinstance(proposal['arguments'],dict):
                    raise ValueError('Unsupported proposal')
                if cfg.key in json.dumps(proposal,ensure_ascii=False):
                    raise ValueError('Credential echoed in model proposal')
                return dict(proposal,provider=cfg.provider,model=cfg.model)
            text=message.get('content')
            if not isinstance(text,str) or not text.strip() or len(text)>8000:
                raise ValueError('Empty/oversized response')
            return dict(intent='explain',arguments={},text=text.replace(cfg.key,'[redacted]'),provider=cfg.provider,model=cfg.model)
        except urllib.error.HTTPError as e:
            hints={400:'请求格式被拒绝；请核对接口协议与参数',401:'API Key 验证失败；请核对该服务的密钥',
                   402:'账户余额不足；请检查服务商账户',403:'接口访问被拒绝；请检查账户权限',
                   404:'接口或模型不可用；请核对 API Base URL 和模型 ID',422:'请求参数不被服务商接受',
                   429:'请求被限流；请稍后手动重试',500:'服务商内部错误；请稍后手动重试',
                   503:'服务商繁忙；请稍后手动重试'}
            raise ValueError('LLM API returned HTTP '+str(e.code)+'；'+hints.get(e.code,'请检查接口地址、权限与服务状态')+'。未执行任务。') from None
        except OSError as e:
            # urllib wraps socket/SSL failures; classify by type without exposing
            # raw exception text, URLs, proxy credentials or the API key.
            cause=e.reason if isinstance(e,urllib.error.URLError) else e
            if isinstance(cause,TimeoutError):
                hint='请求超时（timeout）；请检查 Agent 主机的网络、代理或稍后手动重试'
            elif isinstance(cause,ssl.SSLError):
                hint='TLS/证书校验失败；请检查主机时间、证书与 HTTPS 代理'
            elif isinstance(cause,socket.gaierror):
                hint='DNS 域名解析失败；请检查 Agent 主机的 DNS 与代理配置'
            elif isinstance(cause,ConnectionRefusedError) or getattr(cause,'winerror',None)==10061:
                hint='网络连接被拒绝（connection refused）；请检查 Agent 服务进程的代理地址、端口和网络连接'
            else:
                hint='网络请求失败（network error）；请检查运行 Agent 的主机网络与代理'
            raise ValueError('LLM '+hint+'。未执行任务。') from None
        except http.client.HTTPException:
            raise ValueError('LLM 连接中断或响应未接收完整（connection interrupted）。未执行任务。') from None
        except (ValueError,KeyError,TypeError,IndexError,AttributeError,RecursionError):
            raise ValueError('LLM 返回无效或不完整响应（invalid/incomplete response）。未执行任务。') from None
