"""Identity policy against real Git histories and synthetic GitHub responses."""
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('gate', Path(__file__).resolve().parents[1] / 'scripts/repo_hygiene.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class IdentityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.previous = os.getcwd()
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, self.previous)
        self.git('init', '-q')
        self.git('config', 'user.name', 'Test')
        self.git('config', 'user.email', 'test@example.org')
        self.git('commit', '--allow-empty', '-qm', 'base')
        self.base = self.git('rev-parse', 'HEAD').strip()

    def git(self, *args):
        return subprocess.check_output(['git', '-c', 'core.hooksPath=/dev/null', *args], text=True)

    def commit(self, **env):
        with patch.dict(os.environ, env):
            self.git('commit', '--allow-empty', '-qm', 'change')
        return self.git('rev-parse', 'HEAD').strip()

    def response(self, commit, author='account', committer='account'):
        raw = self.git('show', '-s', '--format=%ae%x00%ce', commit).strip().split('\0')
        return {'sha': commit, 'author': {'login': author} if author else None,
                'committer': {'login': committer} if committer else None,
                'commit': dict(zip(('author', 'committer'), ({'email': email} for email in raw)))}

    def check(self, commits, responses, allowed=()):
        def fetch(request, timeout):
            sha = request.full_url.rsplit('/', 1)[-1]
            return io.BytesIO(json.dumps(responses[sha]).encode())
        with patch.dict(os.environ, {'GITHUB_TOKEN': 'synthetic-test-token'}), patch.object(gate.urllib.request, 'urlopen', side_effect=fetch):
            return gate.check_identities(commits, set(allowed), 'test/repo')

    def test_both_roles_must_resolve_for_every_commit(self):
        first = self.commit(GIT_AUTHOR_EMAIL='invented@example.org')
        second = self.commit(GIT_COMMITTER_EMAIL='other@example.org')
        commits = self.git('rev-list', '--reverse', f'{self.base}..HEAD').splitlines()
        failures = self.check(commits, {first: self.response(first, author=None),
                                      second: self.response(second, committer=None)})
        self.assertEqual(len(failures), 2)
        self.assertIn(first + ': author email invented@example.org', failures[0])
        self.assertIn(second + ': committer email other@example.org', failures[1])
        self.assertEqual(self.check(commits, {sha: self.response(sha) for sha in commits}), [])

    def test_allowlist_is_tracked_exact_and_keeps_paths(self):
        Path('.repo-hygiene-allow').write_text('tests/fixtures/**\tSynthetic inputs\nidentities:\n  legacy@example.org\tPublished legacy contributor\npaths:\nother/**\tSynthetic\n')
        self.assertEqual(gate.allowances(None), ([], set()))
        self.git('add', '.repo-hygiene-allow')
        rules, emails = gate.allowances(None)
        self.assertEqual(rules, [('tests/fixtures/**', 'Synthetic inputs'), ('other/**', 'Synthetic')])
        self.assertEqual(emails, {'legacy@example.org'})
        sha = self.commit(GIT_AUTHOR_EMAIL='legacy@example.org')
        self.assertEqual(self.check([sha], {sha: self.response(sha, author=None)}, emails), [])
        self.assertTrue(self.check([sha], {sha: self.response(sha, committer=None)}, emails))
        Path('.repo-hygiene-allow').write_text('identities:\n*@example.org\tNo wildcard identities\n')
        self.git('add', '.repo-hygiene-allow')
        with self.assertRaisesRegex(ValueError, 'exact email'):
            gate.allowances(None)

    def test_bots_and_unavailable_api(self):
        for bot in ('github-actions', 'dependabot', 'renovate[bot]'):
            sha = self.commit(GIT_AUTHOR_NAME=bot, GIT_COMMITTER_NAME=bot)
            self.assertEqual(self.check([sha], {sha: self.response(sha, author=bot, committer=bot)}), [])
            self.assertEqual(len(self.check([sha], {sha: self.response(sha, author=None, committer=None)})), 2)
        sha = self.commit()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'GITHUB_TOKEN'):
                gate.check_identities([sha], set(), 'test/repo')
        data = self.response(sha)
        data['sha'] = self.base
        with self.assertRaisesRegex(ValueError, 'different commit'):
            self.check([sha], {sha: data})

    def test_local_identity_override_warns_without_blocking(self):
        with patch.dict(os.environ, {'GIT_AUTHOR_EMAIL': 'override@example.org',
                                     'GIT_COMMITTER_EMAIL': 'committer@example.org'}):
            result = subprocess.run(['python3', str(Path(gate.__file__)), '--staged'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('author email override@example.org differs', result.stderr)
        self.assertIn('committer email committer@example.org differs', result.stderr)
        result = subprocess.run(['python3', str(Path(gate.__file__)), '--staged'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('differs from', result.stderr)


if __name__ == '__main__':
    unittest.main()
