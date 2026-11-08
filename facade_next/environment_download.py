"""Verified official-weight downloads, independent of model dependencies."""
import errno
import hashlib
import http.client
import os
from pathlib import Path
import re
import ssl
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit
import uuid

if __package__:
    from .platform_support import process_alive
else:
    from platform_support import process_alive


class IncompleteDownload(RuntimeError):
    pass


class HTTPSRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,request,fp,code,msg,headers,newurl):
        target=urlsplit(newurl)
        if target.scheme!='https' or target.username or target.password:
            raise urllib.error.URLError('Official weight redirects must retain HTTPS without URL credentials.')
        return super().redirect_request(request,fp,code,msg,headers,newurl)


def retryable(error):
    if isinstance(error,urllib.error.HTTPError):
        return error.code in (408,429,500,502,503,504)
    if isinstance(error,urllib.error.URLError):
        return retryable(error.reason) if isinstance(error.reason,BaseException) else False
    if isinstance(error,ssl.SSLCertVerificationError):
        return False
    if isinstance(error,ssl.SSLError) and 'UNEXPECTED_EOF' in str(error).upper():
        return True
    if isinstance(error,(TimeoutError,ConnectionError,ssl.SSLEOFError,http.client.IncompleteRead,IncompleteDownload)):
        return True
    return isinstance(error,OSError) and error.errno in (
        errno.ETIMEDOUT,errno.ECONNRESET,errno.ECONNREFUSED,errno.ENETUNREACH,errno.EHOSTUNREACH,errno.EPIPE)


def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def reclaim_interrupted(path):
    """Only reclaim this downloader's files whose owner process has exited."""
    pattern=re.compile(re.escape(path.name)+r'\.facade-([0-9]+)-[0-9a-f]{32}\.partial')
    if not path.parent.is_dir():return
    for candidate in path.parent.iterdir():
        match=pattern.fullmatch(candidate.name)
        if match and not candidate.is_symlink() and candidate.is_file() and not process_alive(int(match[1])):
            candidate.unlink(missing_ok=True)


def download_verified(url,path,hash_prefix,*,opener=None,progress=None,sleep=time.sleep):
    """Retry whole downloads; expose only complete hash-verified cache files.

    The injected opener is used by no-network tests. Native calls preserve
    urllib's default TLS/certificate verification and host proxy settings.
    """
    if opener is None and os.environ.get('FACADE_DISABLE_MODEL_EXECUTION')=='1':
        raise ValueError('Weight download is disabled by this host session policy.')
    parsed=urlsplit(url)
    if (parsed.scheme!='https' or parsed.hostname!='download.pytorch.org'
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.port not in (None,443) or not parsed.path.startswith('/models/')):
        raise ValueError('Use an official HTTPS PyTorch model URL.')
    if not isinstance(hash_prefix,str) or not re.fullmatch('[0-9a-f]{8,64}',hash_prefix):
        raise ValueError('A valid official weight hash prefix is required.')
    path=Path(path);emit=progress or (lambda value:None)
    reclaim_interrupted(path)
    if path.is_file():
        digest=file_hash(path)
        if digest.startswith(hash_prefix):
            size=path.stat().st_size
            emit(dict(status='cached',received=size,total=size,attempt=0,verified=True))
            return digest
    path.parent.mkdir(parents=True,exist_ok=True)
    connect=opener or urllib.request.build_opener(HTTPSRedirects).open
    limit=1024**3
    for attempt in range(1,4):
        temporary=path.with_name(path.name+'.facade-'+str(os.getpid())+'-'+uuid.uuid4().hex+'.partial')
        received=0;total=None;last_reported=-1
        emit(dict(status='downloading',received=0,total=None,attempt=attempt,verified=False))
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'FacadeAgent environment setup','Accept-Encoding':'identity'})
            with connect(request,timeout=120) as response:
                total=int(response.headers['Content-Length']) if response.headers.get('Content-Length') is not None else None
                if total is not None and not 0<total<=limit:
                    raise ValueError('Unexpected official weight download size.')
                with temporary.open('xb') as handle:
                    while True:
                        chunk=response.read(1024*1024)
                        if not chunk:break
                        received+=len(chunk)
                        if received>limit:raise ValueError('Weight download exceeded the size limit.')
                        handle.write(chunk)
                        percent=int(100*received/total) if total else received//(5*1024*1024)
                        if percent>=last_reported+5 or total is not None and received>=total:
                            emit(dict(status='downloading',received=received,total=total,attempt=attempt,verified=False))
                            last_reported=percent
                if total is not None and received!=total:
                    raise IncompleteDownload('Weight download ended before the declared byte count.')
            emit(dict(status='verifying',received=received,total=total,attempt=attempt,verified=False))
            digest=file_hash(temporary)
            if not digest.startswith(hash_prefix):
                raise IncompleteDownload('Downloaded ImageNet weight hash mismatch.')
            os.replace(temporary,path)
            emit(dict(status='ready',received=received,total=received,attempt=attempt,verified=True))
            return digest
        except Exception as error:
            if not retryable(error):raise
            if attempt==3:
                raise RuntimeError('ImageNet weight download failed after 3 attempts. Installed dependencies are retained; retry the same environment when the connection is available. '+str(error)) from error
            emit(dict(status='retrying',received=received,total=total,attempt=attempt+1,verified=False))
            sleep(attempt)
        finally:
            temporary.unlink(missing_ok=True)
