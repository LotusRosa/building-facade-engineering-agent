"""Tiny standard-library CSV/XLSX annotation templates."""
import csv
import io
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape
from .contracts import CLASSES

SAMPLE=[['filename',*CLASSES,'no_defect'],['image_001.jpg',1,0,0,0],['image_002.jpg',0,1,1,0],['image_003.jpg',0,0,0,1],['image_004.jpg','','','','']]


def template_bytes(kind):
    if kind=='csv':
        f=io.StringIO(newline='')
        csv.writer(f).writerows(SAMPLE)
        return ('\ufeff'+f.getvalue()).encode('utf-8')
    if kind!='xlsx':
        raise ValueError('Template format must be csv/xlsx')
    f=io.BytesIO()
    with zipfile.ZipFile(f,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml','<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>')
        z.writestr('_rels/.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr('xl/workbook.xml','<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="labels" sheetId="1" r:id="rId1"/></sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>')
        rows=[]
        for n,row in enumerate(SAMPLE,1):
            cells=[]
            for col,value in enumerate(row):
                ref=chr(65+col)+str(n)
                if type(value) is int:
                    cells.append(f'<c r="{ref}"><v>{value}</v></c>')
                else:
                    cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(value)}</t></is></c>')
            rows.append(f'<row r="{n}">'+''.join(cells)+'</row>')
        z.writestr('xl/worksheets/sheet1.xml','<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'+''.join(rows)+'</sheetData></worksheet>')
    return f.getvalue()


def write_templates():
    root=Path(__file__).resolve().parent.parent/'templates'
    root.mkdir(exist_ok=True)
    for kind in ('csv','xlsx'):
        (root/('labels.'+kind)).write_bytes(template_bytes(kind))
