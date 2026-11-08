import hashlib
import json
from pathlib import Path
import sqlite3
import unittest
from test_logic import local_tmp


class SavedCaseProbeTests(unittest.TestCase):
    def test_identifies_only_the_requested_completed_case_without_writes(self):
        from scripts.check_saved_case import inspect_case
        with local_tmp() as root:
            path=Path(root)/'audit.sqlite3';db=sqlite3.connect(path)
            db.execute('CREATE TABLE entities(kind TEXT,data TEXT)')
            for kind,value in [('project',{'id':'P','name':'Saved project'}),
                ('round',{'id':'R','project':'P','kind':'update','number':2,'decision':'hold','active':'V0','reason':'Original reason'}),
                ('job',{'id':'J','project':'P','round':'R','method':'standard','status':'succeeded','spec':{'kind':'update','seed':20260921,'backbone':'convnext_tiny','parent':{'id':'V0'},'roles':{'train':[{}]*3}},'result':{'status':'ready','comparison':{'candidate':{'merged':{'macro_AP':.72}},'parent':{'merged':{'macro_AP':.74}}},'cost':{'total_seconds':211}}})]:
                db.execute('INSERT INTO entities VALUES(?,?)',(kind,json.dumps(value)))
            db.commit();db.close();before=hashlib.sha256(path.read_bytes()).hexdigest()
            value=inspect_case(root,'R');self.assertEqual(value['status'],'matched');self.assertEqual(value['matches'][0]['job'],'J')
            self.assertEqual(value['matches'][0]['decision'],'hold');self.assertEqual(value['matches'][0]['effective_version'],'V0')
            self.assertEqual(value['matches'][0]['role_counts']['train'],3)
            self.assertEqual(inspect_case(root,'not-this-round')['status'],'not_found')
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),before)

    def test_missing_runtime_is_not_created(self):
        from scripts.check_saved_case import inspect_case
        with local_tmp() as root:
            missing=Path(root)/'missing'
            with self.assertRaises(ValueError):inspect_case(missing)
            self.assertFalse(missing.exists())
