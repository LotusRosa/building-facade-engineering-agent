"""Immutable annotated photo snapshots; stdlib CSV/XLSX intake."""
import csv
from pathlib import Path
from .contracts import CLASSES, digest, sha


def _xlsx(path):
    import zipfile
    import xml.etree.ElementTree as ET
    ns = {'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as z:
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            strings = [''.join(e.itertext()) for e in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',ns)]
        # Resolve first workbook sheet using relationships instead of assuming sheet1.
        book = ET.fromstring(z.read('xl/workbook.xml'))
        sid = book.find('s:sheets/s:sheet',ns).get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
        rels = ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        target = next(e.get('Target') for e in rels if e.get('Id') == sid)
        sheet = target.lstrip('/') if target.startswith('/') else 'xl/' + target
        matrix = []
        for row in ET.fromstring(z.read(sheet)).findall('s:sheetData/s:row',ns):
            cells = {}
            for c in row.findall('s:c',ns):
                letters = ''.join(x for x in c.get('r','') if x.isalpha())
                col = 0
                for x in letters:
                    col = col*26 + ord(x.upper())-64
                if c.find('s:f',ns) is not None:
                    raise ValueError('XLSX formulas are unsupported; paste values first')
                v = c.find('s:v',ns)
                text = '' if v is None else v.text or ''
                if c.get('t') == 's':
                    text = strings[int(text)]
                elif c.get('t') == 'inlineStr':
                    text = ''.join(c.find('s:is',ns).itertext())
                cells[col-1] = text
            matrix.append([cells.get(i,'') for i in range(max(cells,default=-1)+1)])
        if not matrix:
            raise ValueError('Empty XLSX')
        return [dict(zip(matrix[0], row+['']*len(matrix[0]))) for row in matrix[1:] if any(row)]


def label(value):
    if value is None or str(value).strip().lower() in ('','unknown','null','na','n/a','?'):
        return None
    if str(value).strip() in ('0','0.0'):
        return 0
    if str(value).strip() in ('1','1.0'):
        return 1
    raise ValueError('Labels must be 0, 1 or blank/unknown: ' + str(value))


def load_manifest(path, root):
    return normalize(read_table(path),root)


def read_table(path):
    path = Path(path).resolve()
    if path.suffix.lower() == '.xlsx':
        raw = _xlsx(path)
    elif path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as f:
            raw = list(csv.DictReader(f))
    else:
        raise ValueError('Use CSV or XLSX')
    return raw


def normalize(raw, root):
    base = Path(root).resolve()
    result = []
    for n, r in enumerate(raw):
        value=r.get('path') or r.get('filename')
        if not value:
            raise ValueError('A path or filename column is required')
        p = (base / value).resolve()
        if not p.is_file() and r.get('filename') and not r.get('path'):
            matches=[x for x in base.rglob(str(value)) if x.is_file()]
            if len(matches)!=1:
                raise ValueError('filename is missing or ambiguous: '+str(value))
            p=matches[0].resolve()
        if not p.is_relative_to(base) or not p.is_file():
            raise ValueError('Missing photo or path outside image root: ' + str(p))
        if p.suffix.lower() not in ('.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'):
            raise ValueError('Unsupported image extension: ' + str(p))
        labels = [label(r.get(c)) for c in CLASSES]
        negative=label(r.get('no_defect'))
        if negative==1 and labels!=[0,0,0]:
            raise ValueError('no_defect=1 requires every defect label to be explicitly 0')
        if negative==0 and labels==[0,0,0]:
            raise ValueError('no_defect=0 conflicts with a fully negative image')
        fingerprint=sha(p)
        key = str(r.get('image_id') or 'photo_'+fingerprint).strip()
        if not key or len(key) > 300:
            raise ValueError('Invalid image_id at row ' + str(n+2))
        result.append(dict(image_id=key, path=str(p), labels=labels, sha256=fingerprint,
                           scene=str(r.get('scene') or 'default').strip(), group=str(r.get('group') or '').strip()))
    if not result:
        raise ValueError('Dataset is empty')
    for key in ('image_id','sha256'):
        if len({r[key] for r in result}) != len(result):
            raise ValueError('Duplicate ' + key + ' within dataset')
    return result


def scan_folder(root):
    root = Path(root).resolve()
    return [{'image_id':'photo_'+sha(p), 'path':p.relative_to(root).as_posix(),
             **{c:None for c in CLASSES}, 'scene':'default', 'group':''}
            for p in sorted(root.rglob('*')) if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff')]


def snapshot(rows):
    return {'rows':rows, 'digest':digest(rows), 'count':len(rows)}


def validate_roles(roles):
    names = list(roles)
    for i, left in enumerate(names):
        for right in names[i+1:]:
            for key in ('image_id','sha256','group'):
                a = {r[key] for r in roles[left] if r.get(key)}
                b = {r[key] for r in roles[right] if r.get(key)}
                if a & b:
                    raise ValueError(f'Role overlap by {key}: {left}/{right}: {next(iter(a & b))}')


def union(*pools):
    result, by_id, by_hash = [], {}, {}
    for rows in pools:
        for r in rows:
            old = by_id.get(r['image_id']) or by_hash.get(r['sha256'])
            if old:
                if old != r:
                    raise ValueError('Conflicting repeated image identity/labels: ' + r['image_id'])
                continue
            result.append(r)
            by_id[r['image_id']] = by_hash[r['sha256']] = r
    return result
