"""Regression tests for snapshot boundaries and redacted credential reports."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from check_release import check, source_files


class ReleaseChecksTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)

    def track(self, name, content):
        (self.root / name).write_text(content)
        subprocess.run(['git', 'add', '--', name], cwd=self.root, check=True)

    def test_excludes_ignored_files_even_if_previously_tracked(self):
        self.track('checkpoint.pth', 'old tracked weights')
        self.track('.env', 'local settings')
        self.track('.codex', '')
        (self.root / '.gitignore').write_text('*.pth\n.env\n')
        (self.root / 'README.md').write_text('public documentation')
        names = {p.name for p in source_files(self.root)}
        self.assertEqual(names, {'.gitignore', 'README.md'})

    def test_reports_credential_location_without_credential_value(self):
        credential = 'sk-' + 'a' * 32
        (self.root / 'client.py').write_text('key = "' + credential + '"\n')
        errors = check(self.root)
        self.assertTrue(any('client.py:1: possible credential' in e for e in errors))
        self.assertFalse(any(credential in e for e in errors))

    def test_rejects_symlinks_in_source_snapshot(self):
        target = self.root / 'public.txt'
        target.write_text('example')
        (self.root / 'alias.txt').symlink_to(target)
        self.assertTrue(any('symlinks require explicit review' in e for e in check(self.root)))


if __name__ == '__main__':
    unittest.main()
