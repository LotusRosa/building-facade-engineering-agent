"""Generate tiny synthetic PNGs/CSV for model-free UI intake checks only."""
import csv
import struct
import zlib
from pathlib import Path
root = Path(__file__).resolve().parents[1]/'ui_smoke_state'/'fixtures'
root.mkdir(parents=True,exist_ok=True)
def chunk(name,data):
    return struct.pack('>I',len(data))+name+data+struct.pack('>I',zlib.crc32(name+data)&0xffffffff)
for n,role in enumerate(('train','val','evaluation')):
    rows = []
    for i in range(2):
        name=f'{role}_{i}.png'
        color=bytes([30+n*60,70+i*100,90+n*30])
        pixels=b''.join(b'\x00'+color*48 for _ in range(32))
        image=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',48,32,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(pixels))+chunk(b'IEND',b'')
        (root/name).write_bytes(image)
        rows.append(dict(image_id=f'{role}_{i}',path=name,hollow='' if role=='evaluation' else i,spalling=i,crack=i,scene=role,group=role))
    with (root/(role+'.csv')).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    batch=root/('chat_'+role)
    (batch/'images').mkdir(parents=True,exist_ok=True)
    legacy=[]
    for row in rows:
        (batch/'images'/row['path']).write_bytes((root/row['path']).read_bytes())
        legacy.append(dict(filename=row['path'],hollow=row['hollow'],spalling=row['spalling'],crack=row['crack'],
                           no_defect=1 if row['hollow']==0 and row['spalling']==0 and row['crack']==0 else 0 if row['hollow']!='' else '',scene=role,group=role))
    with (batch/'labels.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(legacy[0]))
        w.writeheader()
        w.writerows(legacy)
print(root)
