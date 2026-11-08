"""Inspect existing role folders. Never creates or changes a data split."""
from pathlib import Path
from .contracts import digest
from .data import normalize,validate_roles
from .intake import prepare_folder

ALIASES={'train':{'train','training','训练'},'val':{'val','validation','验证','校准'},
         'evaluation':{'safety','s0','core_safety','gate','current_gate','test','evaluation','验收','初始验收','本轮验收'}}


def role_folders(root):
    root=Path(root).resolve()
    if not root.is_dir():raise ValueError('Batch folder does not exist: '+str(root))
    found={}
    for child in sorted(root.iterdir()):
        role=next((r for r,names in ALIASES.items() if child.name.casefold() in names),None)
        if role and child.is_dir():
            if not child.resolve().is_relative_to(root):raise ValueError('Batch role folder escapes the selected root')
            if role in found:raise ValueError('Ambiguous batch role '+role+': multiple matching folders')
            found[role]=child
    return found


def inspect_part(folder,role):
    folder=Path(folder).resolve()
    # Inspect link targets before reading or hashing any annotation/photo file.
    for p in folder.rglob('*'):
        if not p.resolve().is_relative_to(folder):raise ValueError('Batch file/link escapes its role folder')
    bundle=prepare_folder(folder)
    if not Path(bundle['root']).is_relative_to(folder):raise ValueError('Batch images escape the role folder')
    return dict(role=role,folder=str(folder),signature=digest(bundle),
                **{k:bundle[k] for k in ('count','complete_images','partial_images','unknown_images')}),normalize(bundle['rows'],bundle['root'])


def prepare(root,stage):
    if stage not in ('initial','update'):raise ValueError('Batch stage must be initial/update')
    root=Path(root).resolve();found=role_folders(root)
    required={'train','evaluation'}|({'val'} if stage=='initial' else set())
    missing=required-set(found)
    if missing:raise ValueError('Missing pre-split batch subsets: '+', '.join(sorted(missing)))
    parts=[];roles={}
    for role in ('train','val','evaluation'):
        if role in found:
            part,rows=inspect_part(found[role],role);parts.append(part);roles[role]=rows
    validate_roles(roles)
    plan=dict(root=str(root),stage=stage,name=root.name,parts=parts)
    plan['signature']=digest(plan)
    return plan
