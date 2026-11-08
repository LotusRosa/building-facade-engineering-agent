import ast
import sys
import unittest
from pathlib import Path
from facade_next.planning import batches, mining
from facade_next.contracts import read,sha


class StaticTests(unittest.TestCase):
    def test_ui_has_no_model_imports(self):
        import facade_next.service
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('torchvision',sys.modules)

    def test_schedules_and_mining_are_role_limited(self):
        b = batches(list(range(10)),list(range(20,30)),7,0,38)
        self.assertEqual(len(b),38)
        self.assertTrue(all(sum(i<10 for i in x)==8 and len(x)==16 for x in b))
        self.assertEqual(b,batches(list(range(10)),list(range(20,30)),7,0,38))
        rows = [{'labels':[None,0,1]},{'labels':[None,1,0]}]
        ids,_ = mining(rows,[[.99,.1,.8],[.99,.8,.7]],[.5]*3,[1]*3,'hnm')
        self.assertEqual(ids,[1])

    def test_vendor_integrity_and_python_syntax(self):
        root = Path(__file__).resolve().parents[1]
        for name,receipt in read(root/'docs/SOURCE_PROVENANCE.json').items():
            self.assertEqual(sha(root/'facade_next/vendor'/name),receipt['output_sha256'])
        for p in root.rglob('*.py'):
            ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))

    def test_extracted_rng_wrapper_binds_numpy_without_importing_it(self):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root/'facade_next/vendor/ewc.py').read_text(encoding='utf-8'))
        imports = {a.asname or a.name.split('.')[0] for n in tree.body if isinstance(n,ast.Import) for a in n.names}
        self.assertIn('np',imports)
