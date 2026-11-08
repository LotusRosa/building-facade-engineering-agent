import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

from test_logic import local_tmp


class DatasetBridgeTests(unittest.TestCase):
    def bridge(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/prepare_bfd_ml_4k.py'
        self.assertTrue(script.is_file(), 'The published HF dataset needs a fixed-role import bridge')
        spec = importlib.util.spec_from_file_location('dataset_bridge', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def fixture(self, base, shared_group=False):
        source = base / 'download'
        source.mkdir()
        rows = []
        for index, role in enumerate(('core_train', 'validation', 'test0', 'diagnosis_a', 'test_a', 'final_test')):
            file_name = 'data/legacy_split/DFWI_%05d.jpg' % index
            photo = source / file_name
            photo.parent.mkdir(parents=True, exist_ok=True)
            photo.write_bytes(('synthetic-test-content-%s' % index).encode())
            rows.append(dict(image_id='DFWI_%05d' % index, file_name=file_name,
                             hollow='0', spalling='1', crack='0', no_defect='0',
                             sha256=hashlib.sha256(photo.read_bytes()).hexdigest(),
                             role=role, group='G0' if shared_group else 'G%s' % index))
        def table(path, items):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(items[0]))
                writer.writeheader()
                writer.writerows(items)
        table(source / 'metadata/all_samples.csv', rows)
        protocol = base / 'protocol'
        table(protocol / 'roles.csv', rows)
        metadata = dict(role_counts={r['role']:1 for r in rows},
                        roles_sha256=hashlib.sha256((protocol/'roles.csv').read_bytes()).hexdigest(),
                        upstream_metadata_sha256=hashlib.sha256((source/'metadata/all_samples.csv').read_bytes()).hexdigest(),
                        upstream_revision='pinned-test-revision')
        (protocol/'protocol.json').write_text(json.dumps(metadata), encoding='utf-8')
        return source, protocol, rows

    def test_existing_fixed_roles_become_complete_importable_batches(self):
        bridge = self.bridge()
        with local_tmp() as temporary:
            base = Path(temporary)
            source, protocol, _ = self.fixture(base)
            output = base/'prepared'
            result = bridge.prepare(source, output, ['initial', 'round_A'], protocol)
            self.assertEqual(result['counts'], {'initial/train':1, 'initial/val':1,
                                              'initial/safety':1, 'round_A/train':1, 'round_A/gate':1})
            self.assertFalse((output/'final_test').exists())
            with (output/'round_A/train/labels.csv').open(encoding='utf-8', newline='') as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row['image_id'], 'DFWI_00003')
            self.assertEqual(row['group'], 'G3')
            self.assertEqual(row['scene'], 'round_A')
            self.assertEqual(row['filename'], 'DFWI_00003.jpg')
            self.assertEqual((output/'round_A/train/images'/row['filename']).read_bytes(),
                             (source/'data/legacy_split/DFWI_00003.jpg').read_bytes())
            self.assertTrue((output/'preparation_receipt.json').is_file())

    def test_changed_photo_fails_before_publishing_output(self):
        bridge = self.bridge()
        with local_tmp() as temporary:
            base = Path(temporary)
            source, protocol, rows = self.fixture(base)
            (source/rows[0]['file_name']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash|SHA|checksum'):
                bridge.prepare(source, base/'prepared', ['initial'], protocol)
            self.assertFalse((base/'prepared').exists())

    def test_wrong_metadata_and_cross_role_groups_are_rejected(self):
        bridge = self.bridge()
        with local_tmp() as temporary:
            base = Path(temporary)
            source, protocol, _ = self.fixture(base, shared_group=True)
            with self.assertRaisesRegex(ValueError, 'group'):
                bridge.prepare(source, base/'prepared', ['initial'], protocol)
            (source/'metadata/all_samples.csv').write_text('changed', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'metadata|checksum|SHA'):
                bridge.prepare(source, base/'prepared', ['initial'], protocol)

    def test_existing_output_is_never_overwritten(self):
        bridge = self.bridge()
        with local_tmp() as temporary:
            base = Path(temporary)
            source, protocol, _ = self.fixture(base)
            output = base/'prepared'
            output.mkdir()
            (output/'keep.txt').write_text('keep', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                bridge.prepare(source, output, ['initial'], protocol)
            self.assertEqual((output/'keep.txt').read_text(encoding='utf-8'), 'keep')
