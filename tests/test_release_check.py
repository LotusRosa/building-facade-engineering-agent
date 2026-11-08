import importlib.util
import unittest
import json
from pathlib import Path
from test_logic import local_tmp

spec=importlib.util.spec_from_file_location('release_check',Path(__file__).resolve().parents[1]/'scripts/check_release.py')
release_check=importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_check)


class ReleaseCheckTests(unittest.TestCase):
    def test_package_worker_and_citation_versions_match(self):
        import facade_next
        import tomllib
        root=Path(__file__).resolve().parents[1]
        version=tomllib.loads((root/'pyproject.toml').read_text(encoding='utf-8'))['project']['version']
        self.assertEqual(facade_next.__version__,version)
        self.assertEqual(json.loads((root/'facade_next/worker_manifest.json').read_text(encoding='utf-8'))['version'],version)
        self.assertIn('version: '+version,(root/'CITATION.cff').read_text(encoding='utf-8').splitlines())

    def test_source_has_all_necessary_files_without_model_import(self):
        result=release_check.check(Path(__file__).resolve().parents[1])
        self.assertEqual(result['errors'],[])
        self.assertFalse(result['model_imported'])

    def test_missing_files_and_manifest_escape_are_rejected(self):
        with local_tmp() as d:
            p=Path(d)
            (p/'RELEASE_MANIFEST.json').write_text(json.dumps({'files':{'../outside':'0'*64}}),encoding='utf-8')
            result=release_check.check(p)
            self.assertEqual(result['status'],'failed')
            self.assertFalse(result['manifest_hashes_verified'])
            self.assertIn('Missing facade_next/worker_manifest.json',result['errors'])
            self.assertIn('Release file missing/changed: ../outside',result['errors'])
