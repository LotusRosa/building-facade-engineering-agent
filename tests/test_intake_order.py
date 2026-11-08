import hashlib
import unittest
from pathlib import Path
from test_logic import local_tmp
from facade_next.intake import prepare_folder


class IntakeOrderTests(unittest.TestCase):
    def test_table_order_preserves_original_checkpoint_manifest_ids(self):
        with local_tmp() as d:
            root = Path(d)
            (root/'a.jpg').write_bytes(b'a')
            (root/'b.jpg').write_bytes(b'b')
            (root/'labels.csv').write_text(
                'image_id,filename,hollow,spalling,crack,scene,group\n'
                'original-b,b.jpg,1,0,0,site-b,building-b\n'
                'original-a,a.jpg,0,1,1,site-a,building-a\n', encoding='utf-8')

            rows = prepare_folder(root)['rows']

            self.assertEqual([r['image_id'] for r in rows], ['original-b', 'original-a'])
            self.assertEqual([r['path'] for r in rows], ['b.jpg', 'a.jpg'])
            self.assertEqual([r['group'] for r in rows], ['building-b', 'building-a'])
            self.assertEqual([r['scene'] for r in rows], ['site-b', 'site-a'])

    def test_unlisted_photos_follow_table_rows_with_identity_and_unknown_labels(self):
        with local_tmp() as d:
            root = Path(d)
            images = root/'images'
            images.mkdir()
            for name in ('a', 'b', 'c', 'd'):
                (images/(name+'.jpg')).write_bytes(name.encode())
            (root/'labels.csv').write_text(
                'image_id,filename,hollow,spalling,crack\n'
                'original-d,d.jpg,1,,0\n'
                'original-b,b.jpg,0,1,1\n', encoding='utf-8')

            bundle = prepare_folder(root)
            rows = bundle['rows']

            self.assertEqual([r['path'] for r in rows], ['d.jpg', 'b.jpg', 'a.jpg', 'c.jpg'])
            self.assertEqual([r['image_id'] for r in rows[:2]], ['original-d', 'original-b'])
            self.assertEqual([rows[0][c] for c in ('hollow', 'spalling', 'crack')], [1, None, 0])
            for row, content in zip(rows[2:], (b'a', b'c')):
                fingerprint = hashlib.sha256(content).hexdigest()
                self.assertEqual(row['image_id'], 'photo_'+fingerprint)
                self.assertEqual(row['source_sha256'], fingerprint)
                self.assertEqual([row[c] for c in ('hollow', 'spalling', 'crack')], [None]*3)
                self.assertEqual((row['scene'], row['group']), ('default', ''))
            self.assertEqual((bundle['count'], bundle['complete_images'], bundle['partial_images'], bundle['unknown_images']), (4, 1, 1, 2))

    def test_table_reordering_keeps_duplicate_identity_validation(self):
        with local_tmp() as d:
            root = Path(d)
            (root/'a.jpg').write_bytes(b'a')
            (root/'b.jpg').write_bytes(b'b')
            (root/'labels.csv').write_text(
                'image_id,filename,hollow,spalling,crack\n'
                'duplicate,b.jpg,1,0,0\n'
                'duplicate,a.jpg,0,1,1\n', encoding='utf-8')

            with self.assertRaisesRegex(ValueError, 'Duplicate image_id'):
                prepare_folder(root)
