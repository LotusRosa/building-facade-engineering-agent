import tempfile
import unittest
from pathlib import Path
from contextlib import contextmanager
import uuid
import shutil
import zipfile
import os
from unittest.mock import patch
tempfile.tempdir = str(Path(__file__).resolve().parent / '.tmp')
Path(tempfile.tempdir).mkdir(exist_ok=True)

@contextmanager
def local_tmp():
    base = Path(tempfile.tempdir).resolve()
    p = base / uuid.uuid4().hex
    p.mkdir()
    try:
        # Running the no-model suite on a configured host must never bill its API.
        with patch.dict(os.environ,{'FACADE_LLM_PROVIDER':'off'}):
            yield str(p)
    finally:
        assert p.resolve().is_relative_to(base)
        shutil.rmtree(p)
from facade_next.data import load_manifest, snapshot, validate_roles
from facade_next.metrics import evaluate, calibrate
from facade_next.contracts import protocol
from facade_next.planning import initial_pos_weight


class LogicTests(unittest.TestCase):
    def test_unknown_and_negative_are_distinct(self):
        with local_tmp() as d:
            p = Path(d)
            (p/'a.jpg').write_bytes(b'photo')
            (p/'data.csv').write_text('image_id,path,hollow,spalling,crack,scene\na,a.jpg,,0,1,A\n', encoding='utf-8')
            rows = load_manifest(p/'data.csv', p)
            self.assertEqual(rows[0]['labels'], [None, 0, 1])
            self.assertEqual(len(snapshot(rows)['digest']), 64)

    def test_content_and_group_leakage(self):
        a = dict(image_id='a', path='/a', sha256='1'*64, labels=[1,0,0], scene='A', group='')
        b = {**a, 'image_id':'b', 'path':'/b'}
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_roles({'train':[a], 'val':[b]})
        b.update(sha256='2'*64, group='building1')
        a['group'] = 'building1'
        with self.assertRaisesRegex(ValueError, 'group'):
            validate_roles({'train':[a], 'evaluation':[b]})

    def test_masked_metrics_ties_and_undefined(self):
        rows = [{'labels':[1,None,0]}, {'labels':[0,None,0]}]
        m = evaluate(rows, [[.5,.9,.1],[.5,.2,.2]], [.5]*3)
        self.assertEqual(m['classes']['hollow']['AP'], .5)
        self.assertIsNone(m['classes']['spalling']['AP'])
        self.assertIsNone(m['classes']['crack']['AP'])
        self.assertEqual(m['classes']['hollow']['FP'], 1)

    def test_calibration_inherits_only_uncalibratable_classes(self):
        rows = [{'labels':[1,None,0]}, {'labels':[0,None,0]}]
        t, status = calibrate(rows, [[.8,.9,.1],[.2,.1,.2]], [.4,.6,.7])
        self.assertAlmostEqual(t[0], .205)
        self.assertEqual(t[1:], [.6,.7])
        self.assertEqual(status[1], 'inherited_no_positive_or_negative')
        with self.assertRaises(ValueError):
            calibrate(rows, [[.8,.9,.1],[.2,.1,.2]])

    def test_protocol_rejects_silent_precision_changes(self):
        self.assertEqual(protocol('update')['steps_per_epoch'], 38)
        with self.assertRaises(ValueError):
            protocol('update', {'image_size':768})
        with self.assertRaises(ValueError):
            protocol('update', {'micro_batch':3})

    def test_initial_weight_uses_known_training_only_and_source_clip(self):
        rows = [{'labels':[1,1,1]}]+[{'labels':[0,0,0]} for _ in range(30)]+[{'labels':[None,None,None]}]
        self.assertEqual(initial_pos_weight(rows),[20.,20.,20.])
        with self.assertRaises(ValueError):
            initial_pos_weight([{'labels':[1,None,0]}])

    def test_xlsx_first_sheet_sparse_cells_and_formula_rejection(self):
        with local_tmp() as d:
            root = Path(d)
            (root/'a.png').write_bytes(b'fixture')
            def book(formula=False):
                with zipfile.ZipFile(root/'data.xlsx','w') as z:
                    z.writestr('xl/workbook.xml','<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="data" sheetId="3" r:id="rId7"/></sheets></workbook>')
                    z.writestr('xl/_rels/workbook.xml.rels','<Relationships><Relationship Id="rId7" Target="worksheets/sheet3.xml"/></Relationships>')
                    z.writestr('xl/worksheets/sheet3.xml','<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row><c r="A1" t="inlineStr"><is><t>path</t></is></c><c r="B1" t="inlineStr"><is><t>crack</t></is></c></row><row><c r="A2" t="inlineStr"><is><t>a.png</t></is></c><c r="B2">'+('<f>1+0</f>' if formula else '')+'<v>1</v></c></row></sheetData></worksheet>')
            book()
            self.assertEqual(load_manifest(root/'data.xlsx',root)[0]['labels'],[None,None,1])
            book(True)
            with self.assertRaisesRegex(ValueError,'formulas'):
                load_manifest(root/'data.xlsx',root)

    def test_shipped_xlsx_template_roundtrips_without_dependencies(self):
        from facade_next.templates import template_bytes
        from facade_next.data import read_table
        with local_tmp() as d:
            p=Path(d)/'labels.xlsx'
            p.write_bytes(template_bytes('xlsx'))
            rows=read_table(p)
            self.assertEqual(rows[2]['no_defect'],'1')
            self.assertEqual(rows[3]['hollow'],'')


if __name__ == '__main__':
    unittest.main()
