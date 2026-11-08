"""Folder-first intake compatible with the published filename/no_defect tables."""
from pathlib import Path
from .contracts import CLASSES,sha
from .data import read_table,normalize,scan_folder


def prepare_folder(root,manifest=None):
    batch=Path(root).resolve()
    if not batch.is_dir():
        raise ValueError('Photo folder does not exist: '+str(batch))
    images=batch/'images' if (batch/'images').is_dir() else batch
    table=None
    if manifest:
        table=(batch/manifest).resolve()
    else:
        candidates=[p for p in batch.iterdir() if p.is_file() and p.stem.lower() in ('labels','annotations') and p.suffix.lower() in ('.csv','.xlsx')]
        if len(candidates)>1:
            raise ValueError('Multiple annotation tables found; specify the intended CSV/XLSX explicitly')
        table=candidates[0] if candidates else None
    rows=scan_folder(images)
    if not rows:
        raise ValueError('No supported photos in folder')
    for row in rows:
        row['source_sha256']=sha(images/row['path'])
    lookup={str((images/r['path']).resolve()):r for r in rows}
    if table:
        labelled=normalize(read_table(table),images)
        ordered=[]
        listed_paths=set()
        for item in labelled:
            if item['path'] not in lookup:
                raise ValueError('Annotation refers to a photo outside this folder')
            row=lookup[item['path']]
            row.update({c:item['labels'][j] for j,c in enumerate(CLASSES)})
            row.update(image_id=item['image_id'],scene=item['scene'],group=item['group'])
            ordered.append(row)
            listed_paths.add(row['path'])
        # Keep original manifest provenance; remaining scans still need annotation.
        rows=ordered+[row for row in rows if row['path'] not in listed_paths]
    unknown=sum(all(r[c] is None for c in CLASSES) for r in rows)
    partial=sum(any(r[c] is None for c in CLASSES) and any(r[c] is not None for c in CLASSES) for r in rows)
    return dict(root=str(images),batch_root=str(batch),manifest=str(table) if table else None,rows=rows,
                count=len(rows),unknown_images=unknown,partial_images=partial,complete_images=len(rows)-unknown-partial)
