"""Convert the pinned BFD-ML-4K download using existing frozen role assignments.

Standard library only. No network, image decoding, model imports or random split.
Source files remain unchanged; only selected batches are copied to a NEW folder.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import uuid


BATCHES = {
    'initial': {'core_train':'train', 'validation':'val', 'test0':'safety'},
    'round_A': {'diagnosis_a':'train', 'test_a':'gate'},
    'round_B': {'diagnosis_b':'train', 'test_b':'gate'},
}
LABELS = ('hollow', 'spalling', 'crack', 'no_defect')


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def table(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def prepare(dataset_root, output, batches=None, protocol=None):
    source, output = Path(dataset_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Output already exists; choose a new folder: '+str(output))
    batches = list(dict.fromkeys(batches or ['initial', 'round_A']))
    if not batches or any(batch not in BATCHES for batch in batches):
        raise ValueError('Supported batches: '+', '.join(BATCHES))
    protocol = Path(protocol or Path(__file__).resolve().parents[1]/'protocols/bfd_ml_4k')
    spec = json.loads((protocol/'protocol.json').read_text(encoding='utf-8'))
    if sha(protocol/'roles.csv') != spec['roles_sha256']:
        raise ValueError('Frozen roles checksum mismatch')
    metadata = source/'metadata/all_samples.csv'
    if sha(metadata) != spec['upstream_metadata_sha256']:
        raise ValueError('HF metadata checksum mismatch; download the pinned dataset revision')
    rows, upstream = table(protocol/'roles.csv'), table(metadata)
    index = {row['image_id']:row for row in upstream}
    if len(index) != len(upstream) or set(index) != {row['image_id'] for row in rows}:
        raise ValueError('HF metadata image identities do not match the frozen protocol')
    if dict(Counter(row['role'] for row in rows)) != spec['role_counts']:
        raise ValueError('Frozen role counts do not match protocol.json')
    ids, hashes, groups = set(), set(), {}
    selected = {role:(batch, folder) for batch in batches for role,folder in BATCHES[batch].items()}
    planned = []
    destinations = set()
    for row in rows:
        if row['image_id'] in ids or row['sha256'] in hashes:
            raise ValueError('Duplicate frozen image identity/content')
        ids.add(row['image_id']); hashes.add(row['sha256'])
        if not row['group'] or groups.setdefault(row['group'], row['role']) != row['role']:
            raise ValueError('Missing or cross-role acquisition group')
        if any(row[key] not in ('0','1') for key in LABELS):
            raise ValueError('Frozen labels must be binary')
        if row['no_defect'] != str(int(not any(row[key]=='1' for key in LABELS[:3]))):
            raise ValueError('Inconsistent no_defect label')
        reference = index[row['image_id']]
        if any(row[key] != reference[key] for key in ('file_name','sha256',*LABELS)):
            raise ValueError('HF metadata and frozen image/label entry disagree')
        if row['role'] not in selected:
            continue
        relative = Path(row['file_name'])
        photo = (source/relative).resolve()
        if relative.is_absolute() or not photo.is_relative_to(source) or not photo.is_file():
            raise ValueError('Image path is missing or escapes dataset root: '+row['image_id'])
        if sha(photo) != row['sha256']:
            raise ValueError('Image SHA256 hash mismatch: '+row['image_id'])
        batch, folder = selected[row['role']]
        destination = (batch, folder, photo.name)
        if destination in destinations:
            raise ValueError('Duplicate output filename')
        destinations.add(destination)
        planned.append((row, photo, batch, folder))
    if not planned:
        raise ValueError('Selected batches contain no images')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name('.'+output.name+'.preparing_'+uuid.uuid4().hex)
    staging.mkdir()
    tables, counts = {}, Counter()
    try:
        for row, photo, batch, folder in planned:
            subset = staging/batch/folder
            (subset/'images').mkdir(parents=True, exist_ok=True)
            target = subset/'images'/photo.name
            shutil.copyfile(photo, target)
            if sha(target) != row['sha256']:
                raise ValueError('Copied image checksum mismatch: '+row['image_id'])
            item = dict(image_id=row['image_id'], filename=photo.name,
                        **{key:row[key] for key in LABELS}, scene=batch, group=row['group'])
            tables.setdefault(subset, []).append(item)
            counts[batch+'/'+folder] += 1
        for subset, items in tables.items():
            with (subset/'labels.csv').open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(items[0]))
                writer.writeheader(); writer.writerows(items)
        receipt = dict(schema='bfd_fixed_roles_preparation_v1',
                       upstream_revision=spec['upstream_revision'],
                       upstream_metadata_sha256=spec['upstream_metadata_sha256'],
                       frozen_roles_sha256=spec['roles_sha256'], counts=dict(counts),
                       random_split=False, source_modified=False,
                       final_test_included=False, image_checksums_verified=True)
        (staging/'preparation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
        if output.exists():
            raise FileExistsError('Output appeared during preparation; refusing to overwrite')
        staging.rename(output)
        return receipt
    except Exception as error:
        raise RuntimeError('Preparation incomplete. Inspect retained staging folder '+str(staging)+': '+str(error)) from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch', action='append', choices=tuple(BATCHES),
                        help='Repeat for several batches; default: initial and round_A')
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset_root,args.output,args.batch),indent=2))


if __name__=='__main__':
    main()
