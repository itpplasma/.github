"""Behavioral checks using real Git indexes, histories and Gitleaks."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

GATE = Path(__file__).resolve().parents[1] / 'scripts/repo_hygiene.py'


class HygieneTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / 'repo'
        self.repo.mkdir()
        self.git('init', '-q')
        self.git('config', 'user.email', 'test@example.org')
        self.git('config', 'user.name', 'Test')
        self.git('commit', '--allow-empty', '-qm', 'base')
        self.base = self.git('rev-parse', 'HEAD').strip()

    def git(self, *args):
        return subprocess.check_output(['git', '-c', 'core.hooksPath=/dev/null', *args],
                                       cwd=self.repo, text=True)

    def add(self, name, data=b'x'):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.git('add', '--', name)

    def gate(self, *args):
        return subprocess.run([sys.executable, str(GATE), *(args or ('--staged',))],
                              cwd=self.repo, capture_output=True, text=True)

    def test_plain_source_and_deletion(self):
        self.add('source.py', b'print(1)\n')
        self.assertEqual(self.gate().returncode, 0)
        self.git('commit', '-qm', 'source')
        self.git('rm', 'source.py')
        self.assertEqual(self.gate().returncode, 0)

    def test_type_and_only_staged_allowance(self):
        self.add('test/fixtures/a with space.npz')
        self.assertEqual(self.gate().returncode, 1)
        allow = self.repo / '.repo-hygiene-allow'
        allow.write_text('test/fixtures/**\tSynthetic fixture from generator.py\n')
        self.assertEqual(self.gate().returncode, 1)
        self.git('add', '.repo-hygiene-allow')
        self.assertEqual(self.gate().returncode, 0)

    def test_size_and_aggregate(self):
        self.add('large.dat', b'x' * (1024 * 1024 + 1))
        self.assertIn('exceeds 1 MiB', self.gate().stderr)
        self.git('reset', '-q')
        for i in range(6):
            self.add(f'{i}.txt', bytes([65 + i]) * (900 * 1024))
        self.assertIn('aggregate additions', self.gate().stderr)

    def test_transient_blob_in_range(self):
        self.add('trace.npy')
        self.git('commit', '-qm', 'add output')
        self.git('rm', 'trace.npy')
        self.git('commit', '-qm', 'remove output')
        self.assertEqual(self.gate('--base', self.base).returncode, 1)

    def test_transient_blob_introduced_by_merge(self):
        self.git('checkout', '-qb', 'side')
        self.add('side.txt')
        self.git('commit', '-qm', 'side change')
        self.git('checkout', '-q', '-')
        self.add('main.txt')
        self.git('commit', '-qm', 'main change')
        self.git('merge', '--no-ff', '--no-commit', 'side')
        self.add('merge-output.npy')
        self.git('commit', '-qm', 'resolution introduces output')
        self.git('rm', 'merge-output.npy')
        self.git('commit', '-qm', 'remove resolution output')
        result = self.gate('--base', self.base)
        self.assertEqual(result.returncode, 1)
        self.assertIn('merge-output.npy', result.stderr)

    def test_secret_staged_and_range(self):
        # Construct a known rule's synthetic credential; never a live token.
        token = 'gh' + 'p_' + 'A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8'
        self.add('credentials.txt', ('token=' + token + '\n').encode())
        # Neither repo config nor ignore comments should silently suppress it.
        self.add('.gitleaks.toml', b'[allowlist]\npaths = [".*"]\n')
        self.assertEqual(self.gate().returncode, 1)
        self.git('commit', '-qm', 'synthetic secret')
        self.git('rm', 'credentials.txt')
        self.git('commit', '-qm', 'remove')
        self.assertEqual(self.gate('--base', self.base).returncode, 1)

    def local_allow(self):
        common = Path(self.git('rev-parse', '--git-common-dir').strip())
        if not common.is_absolute():
            common = self.repo / common
        info = common / 'info'
        info.mkdir(parents=True, exist_ok=True)
        return info / 'repo-hygiene-allow'

    def test_staged_local_allowance(self):
        self.add('runs/output.npz')
        result = self.gate()
        self.assertEqual(result.returncode, 1)
        self.assertIn(str(self.local_allow()), result.stderr)
        allow = self.local_allow()
        allow.write_text('runs/**\tLocal scratch artifact from run.py\n')
        self.assertEqual(self.gate().returncode, 0)  # passes with reasoned local rule
        self.assertNotIn('repo-hygiene-allow', self.git('status', '--porcelain'))
        allow.write_text('runs/**\n')  # missing reason
        self.assertEqual(self.gate().returncode, 2)  # malformed fails
        allow.unlink()
        self.assertEqual(self.gate().returncode, 1)  # missing file: blocked again

    def test_local_allowance_never_waives_secrets(self):
        token = 'gh' + 'p_' + 'A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8'
        self.add('runs/notes.txt', ('token=' + token + '\n').encode())
        self.local_allow().write_text('runs/**\tLocal scratch output\n')
        result = self.gate()
        self.assertEqual(result.returncode, 1)  # gitleaks still runs and blocks

    def test_local_allowance_rejects_identities_and_absolute_paths(self):
        self.add('runs/output.npz')
        self.local_allow().write_text('identities:\nbot@example.org\tSome bot\n')
        self.assertEqual(self.gate().returncode, 2)
        self.local_allow().write_text('/runs/**\tAbsolute pattern\n')
        self.assertEqual(self.gate().returncode, 2)

    def test_range_ignores_local_allowance(self):
        self.local_allow().write_text('trace.npy\tLocal scratch artifact\n')
        self.add('trace.npy')
        self.git('commit', '-qm', 'add output')
        result = self.gate('--base', self.base)
        self.assertEqual(result.returncode, 1)
        self.assertIn('.repo-hygiene-allow (glob<TAB>reason)', result.stderr)
        self.assertNotIn(str(self.local_allow()), result.stderr)

    def test_tracked_allowance_applies_to_range(self):
        (self.repo / '.repo-hygiene-allow').write_text(
            'trace.npy\tSynthetic fixture from generator.py\n')
        self.git('add', '.repo-hygiene-allow')
        self.git('commit', '-qm', 'allow fixture')
        self.add('trace.npy')
        self.git('commit', '-qm', 'add fixture')
        self.assertEqual(self.gate('--base', self.base).returncode, 0)

    def test_linked_worktree(self):
        worktree = self.repo.parent / 'worktree'
        self.git('worktree', 'add', '-qb', 'worktree-test', str(worktree))
        self.repo = worktree
        self.add('results/archive.csv.gz')
        self.assertEqual(self.gate().returncode, 1)
        self.local_allow().write_text('results/**\tLocal scratch output\n')
        self.assertEqual(self.gate().returncode, 0)


if __name__ == '__main__':
    unittest.main()
