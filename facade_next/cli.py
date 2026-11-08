import argparse
import importlib.metadata
import json
import platform
import sys
import uuid
from pathlib import Path
from .contracts import BACKBONES, digest, read, sha, write


def preflight():
    packages = {}
    for name in ('torch','torchvision','numpy','Pillow'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = 'MISSING (only needed by GPU Worker)'
    return dict(python=platform.python_version(),platform=platform.platform(),packages=packages,
                model_imported=False,automatic_install=False,
                worker_requires='64-bit Windows/Ubuntu, CUDA/BF16 environment; configure it from the first-run browser wizard')


def import_model(args):
    from .worker import server_guard
    server_guard(args.execute)
    if not args.acknowledge_role_provenance:
        raise ValueError('Confirm original training/Val/evaluation manifests using --acknowledge-role-provenance')
    from .service import Service, uid
    from .data import validate_roles
    service = Service(args.state)
    with service.store.transaction() as db:
        project = service.store.get(db,'project',args.project)
        if project['active'] or service.store.all(db,'round',args.project):
            raise ValueError('Use a new empty project')
        ids = {k:getattr(args,k).split(',') for k in ('train','val','evaluation')}
        roles = {k:service._datasets(db,args.project,v) for k,v in ids.items()}
    validate_roles(roles)
    from .gpu import convert_checkpoint
    output = service.root/'imports'/uid('import')
    output.mkdir(parents=True)
    contract = convert_checkpoint(Path(args.checkpoint).resolve(),output,roles)
    response = service.dispatch('import_active',dict(project=args.project,checkpoint=str(output/'model.pt'),
                                contract=str(output/'model-contract.json'),**ids,reason=args.reason),uuid.uuid4().hex)
    print(json.dumps({'result':response,'contract':contract},ensure_ascii=False,indent=2))


def main():
    parser = argparse.ArgumentParser(description='Lightweight facade engineering agent with optional LLM chat')
    sub = parser.add_subparsers(dest='command',required=True)
    serve = sub.add_parser('serve')
    serve.add_argument('--state',default='runtime')
    serve.add_argument('--port',type=int,default=8765)
    serve.add_argument('--execute-workers',action='store_true')
    serve.add_argument('--gpus',default='0')
    serve.add_argument('--environment-config',help='Override the host-wide environment profile path')
    serve.add_argument('--ui-only',action='store_true',help='Skip startup of saved GPU workers; setup remains available in the UI')
    serve.add_argument('--open-browser',action='store_true')
    sub.add_parser('check')
    worker = sub.add_parser('worker')
    worker.add_argument('--state',default='runtime')
    worker.add_argument('--device',default='cuda:0')
    worker.add_argument('--execute',action='store_true')
    worker.add_argument('--job',help='Only claim this queued job ID')
    gpucheck = sub.add_parser('gpu-check')
    gpucheck.add_argument('--execute',action='store_true')
    imp = sub.add_parser('import-model')
    imp.add_argument('--state',default='runtime')
    for name in ('project','checkpoint','train','val','evaluation','reason'):
        imp.add_argument('--'+name,required=True)
    imp.add_argument('--acknowledge-role-provenance',action='store_true')
    imp.add_argument('--execute',action='store_true')
    export = sub.add_parser('export')
    export.add_argument('--state',default='runtime')
    export.add_argument('--output',required=True)
    args = parser.parse_args()
    if args.command=='check':
        print(json.dumps(preflight(),indent=2))
    elif args.command=='serve':
        from .server import serve
        serve(args.state,args.port,args.execute_workers,args.gpus,environment_path=args.environment_config,ui_only=args.ui_only,open_browser=args.open_browser)
    elif args.command=='worker':
        from .worker import run_one
        run_one(args.state,args.device,args.execute,job_id=args.job)
    elif args.command=='gpu-check':
        from .worker import server_guard
        server_guard(args.execute)
        import torch
        import torchvision
        print(json.dumps({'torch':str(torch.__version__),'torchvision':str(torchvision.__version__),
                          'cuda':torch.cuda.is_available(),'devices':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                          'bf16':torch.cuda.is_bf16_supported(including_emulation=False)},indent=2))
    elif args.command=='import-model':
        import_model(args)
    elif args.command=='export':
        from .service import Service
        service = Service(args.state)
        # Explicit result retrieval counts once; automatic browser polling never does.
        service.dispatch('export_audit',{},uuid.uuid4().hex)
        write(args.output,service.snapshot())


if __name__=='__main__':
    main()
