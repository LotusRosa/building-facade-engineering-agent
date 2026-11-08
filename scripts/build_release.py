"""Allowlist-only lightweight source distribution, never packages runtime data."""
import hashlib
import json
import zipfile
import tomllib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root/'dist'
out.mkdir(exist_ok=True)
files = []
for directory,extensions in [('facade_next',{'.py','.json','.html','.js','.css'}),('docs',{'.md','.json'}),('tests',{'.py','.js'}),('templates',{'.csv','.xlsx'}),('protocols',{'.csv','.json','.md'}),('.github',{'.yml','.yaml'})]:
    files += [p for p in (root/directory).rglob('*') if p.is_file() and p.suffix in extensions and '__pycache__' not in p.parts and '.tmp' not in p.parts]
files += [root/p for p in ('README.md','README_EN.md','LICENSE','CITATION.cff','THIRD_PARTY_NOTICES.md','SUPPORTED_ENVIRONMENTS.md','ACCUMULATING_GATE_PROTOCOL.md','AGENT_COMPLETE_GPU_HANDOFF.md','requirements.worker.txt','pyproject.toml','start_ui.cmd','start_ui.sh','run_windows.ps1','run_linux.sh','setup_windows_gpu.ps1','setup_linux_gpu.sh','.gitignore','.gitattributes','llm.example.env')]
files += [root/'scripts'/p for p in ('server_acceptance.py','make_ui_fixture.py','build_release.py','check_release.py','check_data_archives.py','check_saved_case.py','prepare_bfd_ml_4k.py')]
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
mapping = {p.relative_to(root).as_posix():sha(p) for p in sorted(set(files))}
version=tomllib.loads((root/'pyproject.toml').read_text(encoding='utf-8'))['project']['version']
worker_manifest=json.loads((root/'facade_next/worker_manifest.json').read_text(encoding='utf-8'))
assert version==worker_manifest['version'], 'Worker and project release versions differ'
release_date=worker_manifest['release_date']
manifest = {'version':version,'date':release_date,'status':'development_preview_gpu_unverified','files':mapping}
(root/'RELEASE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
archive = out/'building-facade-engineering-agent.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for relative in mapping:
        if any(s in relative.lower() for s in ('.sqlite','.pt','.npy','runtime','ui_smoke_state')):
            raise ValueError('Unexpected runtime artifact '+relative)
        z.write(root/relative,arcname='building-facade-engineering-agent/'+relative)
    z.writestr('building-facade-engineering-agent/RELEASE_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
with zipfile.ZipFile(archive) as z:
    if z.testzip() is not None:
        raise ValueError('ZIP CRC failure')
    for relative,expected in mapping.items():
        if hashlib.sha256(z.read('building-facade-engineering-agent/'+relative)).hexdigest()!=expected:
            raise ValueError('ZIP content hash mismatch')
report = dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,files=len(mapping)+1,
              contains_weights=False,contains_dataset_images=False,contains_protocol_manifests=True,
              contains_runtime_database=False)
(out/'RELEASE_RECEIPT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
(out/('RELEASE_RECEIPT_'+version+'.json')).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
