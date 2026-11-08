import csv
import hashlib
import io
import json
import stat
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from test_logic import local_tmp
from scripts.check_data_archives import check


def fixture(root,batch,*,tamper=False,symlink=False,whitespace=False):
    subsets=['train','val','safety'] if batch=='initial' else ['train','gate']
    data={};specs={}
    for subset in subsets:
        table=io.StringIO();writer=csv.DictWriter(table,fieldnames=['image_id','path','hollow','spalling','crack','no_defect','scene','group'])
        writer.writeheader()
        for i in range(2):
            key=batch+'_'+subset+'_'+str(i);name=str(i)+'.jpg'
            data[subset+'/images/'+name]=key.encode()
            writer.writerow(dict(image_id=key,path=name,hollow=i,spalling=i,crack=i,no_defect=1-i,scene=batch,
                                 group=(' shared ' if whitespace and subset=='gate' and i==0 else key)))
        data[subset+'/labels.csv']=table.getvalue().encode();specs[subset]={'count':2}
    manifest=dict(schema='facade_dataset_package_v1',batch=batch,stage='initial' if batch=='initial' else 'update',
                  subsets=specs,files={k:hashlib.sha256(v).hexdigest() for k,v in data.items()})
    data['DATASET_MANIFEST.json']=json.dumps(manifest).encode()
    if tamper:data['train/images/0.jpg']=b'changed photo'
    path=Path(root)/(batch+'.zip')
    with zipfile.ZipFile(path,'w') as z:
        for name,value in data.items():
            if symlink and name=='train/images/0.jpg':
                entry=zipfile.ZipInfo(batch+'/'+name);entry.create_system=3;entry.external_attr=(stat.S_IFLNK|0o777)<<16;z.writestr(entry,value)
            else:z.writestr(batch+'/'+name,value)
    return path


class DataArchiveTests(unittest.TestCase):
    def run_check(self,*args):
        with redirect_stdout(io.StringIO()):return check(*args)

    def test_portable_pair_has_all_roles_and_no_models(self):
        with local_tmp() as root:
            result=self.run_check(fixture(root,'initial'),fixture(root,'round_A'))
            self.assertEqual(result['total_images'],10)
            self.assertFalse(result['model_imported'])
            self.assertFalse(result['cross_subset_id_hash_group_overlap'])

    def test_tampered_image_is_rejected(self):
        with local_tmp() as root:
            with self.assertRaisesRegex(ValueError,'SHA256'):
                self.run_check(fixture(root,'initial',tamper=True),fixture(root,'round_A'))

    def test_symlink_photo_is_rejected_before_portability_is_claimed(self):
        with local_tmp() as root:
            with self.assertRaisesRegex(ValueError,'regular'):
                self.run_check(fixture(root,'initial',symlink=True),fixture(root,'round_A'))

    def test_identity_whitespace_is_rejected_like_runtime_normalization(self):
        with local_tmp() as root:
            with self.assertRaisesRegex(ValueError,'Whitespace'):
                self.run_check(fixture(root,'initial'),fixture(root,'round_A',whitespace=True))


if __name__=='__main__':unittest.main()
