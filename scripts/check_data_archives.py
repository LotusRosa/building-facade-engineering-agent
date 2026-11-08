"""Verify the two portable data ZIPs without extraction, model imports or training."""
import argparse
import csv
import hashlib
import io
import json
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from facade_next.data import CLASSES, label, validate_roles
from facade_next.planning import initial_pos_weight


def safe_name(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe archive path')
    return name


def check(initial, round_a):
    roles, reports = {}, []
    for path, batch, stage, required in [(initial, 'initial', 'initial', {'train','val','safety'}),
                                         (round_a, 'round_A', 'update', {'train','gate'})]:
        print('Checking '+str(path), flush=True)
        with zipfile.ZipFile(path) as z:
            info = json.loads(z.read(batch+'/DATASET_MANIFEST.json'))
            if (info['schema'],info['batch'],info['stage']) != ('facade_dataset_package_v1',batch,stage):
                raise ValueError('Unexpected data package identity')
            files = info['files']
            names = [entry.filename for entry in z.infolist()]
            if len(names)!=len(set(names)):
                raise ValueError('Duplicate archive entry')
            for entry in z.infolist():
                mode=stat.S_IFMT(entry.external_attr>>16)
                if entry.is_dir() or mode not in (0,stat.S_IFREG):
                    raise ValueError('Only regular archive files are supported')
            for name in names:safe_name(name)
            expected = {batch+'/'+safe_name(name) for name in files}|{batch+'/DATASET_MANIFEST.json'}
            if set(names)!=expected:
                raise ValueError('Unexpected or missing archive file')
            for n,(name,expected_hash) in enumerate(files.items(),1):
                h = hashlib.sha256()
                with z.open(batch+'/'+name) as f:
                    for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
                if h.hexdigest()!=expected_hash:
                    raise ValueError('Data file SHA256 mismatch: '+name)
                if n%500==0:print(batch+': verified '+str(n)+' files',flush=True)
            if set(info['subsets'])!=required:
                raise ValueError('Unexpected subset mapping')
            counts = {}
            for subset in sorted(required):
                spec = info['subsets'][subset]
                table = subset+'/labels.csv'
                if table not in files:
                    raise ValueError('Missing labels table')
                raw = list(csv.DictReader(io.StringIO(z.read(batch+'/'+table).decode('utf-8-sig'))))
                rows = []
                for row in raw:
                    filename = safe_name(row['path'])
                    if PurePosixPath(filename).name!=filename:
                        raise ValueError('Photo path must be relative to its images folder')
                    image = subset+'/images/'+filename
                    if image not in files:
                        raise ValueError('Labelled image missing from package')
                    values = [label(row.get(c)) for c in CLASSES]
                    if any(v is None for v in values):
                        raise ValueError('Delivery labels must be complete')
                    negative = label(row.get('no_defect'))
                    if negative != int(values==[0,0,0]):
                        raise ValueError('no_defect conflicts with defect labels')
                    if not row.get('image_id') or not row.get('group'):
                        raise ValueError('Delivery requires original image/group identity')
                    if any(row[key]!=row[key].strip() for key in ('image_id','group')):
                        raise ValueError('Whitespace around image/group identity')
                    rows.append(dict(image_id=row['image_id'],sha256=files[image],group=row['group'],labels=values))
                images = {name for name in files if name.startswith(subset+'/images/')}
                if len(rows)!=spec['count'] or len(images)!=len(rows):
                    raise ValueError('Photo/label count mismatch')
                for key in ('image_id','sha256'):
                    if len({row[key] for row in rows})!=len(rows):raise ValueError('Duplicate '+key)
                roles[batch+'/'+subset] = rows
                counts[subset] = dict(images=len(rows),positive_counts={c:sum(row['labels'][j]==1 for row in rows) for j,c in enumerate(CLASSES)})
            reports.append(dict(archive=str(Path(path).resolve()),batch=batch,subsets=counts,hashes_verified=True))
    validate_roles(roles)
    initial_pos_weight(roles['initial/train'])
    for j in range(3):
        if {r['labels'][j] for r in roles['initial/val']}!={0,1}:raise ValueError('Initial Val lacks positive/negative class coverage')
    return dict(status='passed',packages=reports,total_images=sum(len(rows) for rows in roles.values()),
                cross_subset_id_hash_group_overlap=False,model_imported=any(x in sys.modules for x in ('torch','torchvision')),
                extraction=False,training=False,pixel_decode=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial',required=True)
    parser.add_argument('--round-a',required=True)
    parser.add_argument('--output')
    args=parser.parse_args()
    result=check(args.initial,args.round_a)
    text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if args.output:Path(args.output).write_text(text,encoding='utf-8')
    print(text)


if __name__=='__main__':main()
