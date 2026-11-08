"""Verified-download behavior using text fixtures and injected network streams."""
import hashlib
import importlib
import importlib.util
import io
import os
from pathlib import Path
import ssl
import unittest
import urllib.error
from unittest.mock import patch
from test_logic import local_tmp


class Response(io.BytesIO):
    def __init__(self,content,total=None):
        super().__init__(content)
        self.headers={'Content-Length':str(len(content) if total is None else total)}
        self.status=200


class VerifiedDownloadTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('facade_next.environment_download'),
                             'Verified retrying weight downloader is not implemented')
        self.download=importlib.import_module('facade_next.environment_download').download_verified
        self.tmp=local_tmp();self.root=Path(self.tmp.__enter__());self.addCleanup(lambda:self.tmp.__exit__(None,None,None))
        self.content=b'official weight text fixture, never model loaded'*40000
        self.digest=hashlib.sha256(self.content).hexdigest();self.path=self.root/'cache'/'fixture.pth'
        self.url='https://download.pytorch.org/models/fixture-'+self.digest[:8]+'.pth'
        self.events=[];self.calls=[]

    def run_download(self,opener):
        return self.download(self.url,self.path,self.digest[:8],opener=opener,
                             progress=self.events.append,sleep=lambda seconds:None)

    def test_ssl_eof_then_success_is_retried_and_published_only_after_hash_validation(self):
        test=self
        class Interrupted(Response):
            def read(self,n=-1):
                if self.tell():raise ssl.SSLEOFError('UNEXPECTED_EOF_WHILE_READING')
                test.assertFalse(test.path.exists())
                return super().read(n)
        def opener(request,timeout):
            self.calls.append((request,timeout))
            self.assertTrue(request.full_url.startswith('https://'))
            return Interrupted(self.content) if len(self.calls)==1 else Response(self.content)
        self.assertEqual(self.run_download(opener),self.digest)
        self.assertEqual(len(self.calls),2)
        self.assertEqual(self.path.read_bytes(),self.content)
        self.assertEqual(list(self.path.parent.glob('*.partial')),[])
        self.assertTrue(any(e['status']=='retrying' for e in self.events))
        self.assertTrue(any(e.get('received',0)>0 and e.get('total')==len(self.content) for e in self.events))
        self.assertTrue(self.events[-1]['verified'])

    def test_persistent_timeout_is_bounded_and_leaves_no_usable_partial_weight(self):
        def opener(request,timeout):
            self.calls.append(timeout);raise TimeoutError('read timed out')
        with self.assertRaisesRegex(RuntimeError,'3|attempt'):
            self.run_download(opener)
        self.assertEqual(len(self.calls),3)
        self.assertFalse(self.path.exists())
        self.assertEqual(list(self.path.parent.glob('*.partial')),[])

    def test_existing_verified_cache_does_not_connect(self):
        self.path.parent.mkdir();self.path.write_bytes(self.content)
        def opener(*args,**kwargs):self.fail('Verified cached weights must not reconnect')
        self.assertEqual(self.run_download(opener),self.digest)
        self.assertEqual(self.events[-1]['status'],'cached')

    def test_restart_reclaims_only_dead_owner_partials_for_this_weight(self):
        self.path.parent.mkdir()
        abandoned=self.path.with_name(self.path.name+'.facade-12345-'+('a'*32)+'.partial')
        live=self.path.with_name(self.path.name+'.facade-23456-'+('b'*32)+'.partial')
        unrelated=self.path.parent/'unrelated.partial'
        for p in (abandoned,live,unrelated):p.write_bytes(b'incomplete fixture')
        with patch('facade_next.environment_download.process_alive',side_effect=lambda pid:pid==23456,create=True):
            self.run_download(lambda request,timeout:Response(self.content))
        self.assertFalse(abandoned.exists(),'A terminated probe must not accumulate abandoned files')
        self.assertTrue(live.exists(),'Do not remove an active concurrent download')
        self.assertTrue(unrelated.exists(),'Only reclaim owned temporary files for this weight')

    def test_invalid_cache_is_replaced_only_after_successful_download(self):
        self.path.parent.mkdir();self.path.write_bytes(b'broken old cache')
        def opener(request,timeout):
            self.calls.append(timeout)
            self.assertEqual(self.path.read_bytes(),b'broken old cache')
            return Response(b'bad hash') if len(self.calls)<3 else Response(self.content)
        self.assertEqual(self.run_download(opener),self.digest)
        self.assertEqual(len(self.calls),3)
        self.assertEqual(self.path.read_bytes(),self.content)

    def test_short_response_is_retried_even_when_stream_ends_without_exception(self):
        def opener(request,timeout):
            self.calls.append(timeout)
            return Response(self.content[:10],len(self.content)) if len(self.calls)==1 else Response(self.content)
        self.assertEqual(self.run_download(opener),self.digest)
        self.assertEqual(len(self.calls),2)

    def test_certificate_error_is_not_hidden_by_retries_or_disabled_tls(self):
        def opener(request,timeout):
            self.calls.append(timeout)
            raise urllib.error.URLError(ssl.SSLCertVerificationError('certificate verify failed'))
        with self.assertRaises(urllib.error.URLError):self.run_download(opener)
        self.assertEqual(len(self.calls),1)
        self.assertFalse(self.path.exists())

    def test_insecure_or_untrusted_weight_urls_are_rejected_without_connection(self):
        def opener(*args,**kwargs):self.fail('Invalid weight URL must not connect')
        for url in ['http://download.pytorch.org/models/a.pth','https://untrusted.example/a.pth']:
            with self.subTest(url=url),self.assertRaises(ValueError):
                self.download(url,self.path,self.digest[:8],opener=opener)

    def test_native_download_is_blocked_by_host_session_policy(self):
        with patch.dict(os.environ,{'FACADE_DISABLE_MODEL_EXECUTION':'1'}):
            with self.assertRaises(ValueError):self.download(self.url,self.path,self.digest[:8])
        self.assertFalse(self.path.exists())

    def test_wrapped_ssl_eof_is_retried(self):
        def opener(request,timeout):
            self.calls.append(timeout)
            if len(self.calls)==1:raise urllib.error.URLError(ssl.SSLError('UNEXPECTED_EOF_WHILE_READING'))
            return Response(self.content)
        self.assertEqual(self.run_download(opener),self.digest)
        self.assertEqual(len(self.calls),2)

    def test_native_redirect_handler_rejects_https_downgrade(self):
        module=importlib.import_module('facade_next.environment_download')
        self.assertTrue(hasattr(module,'HTTPSRedirects'),'Native weight redirects must retain HTTPS')
        import urllib.request
        request=urllib.request.Request(self.url)
        with self.assertRaises(urllib.error.URLError):
            module.HTTPSRedirects().redirect_request(request,None,302,'redirect',{},'http://download.pytorch.org/models/a.pth')


if __name__=='__main__':unittest.main()
