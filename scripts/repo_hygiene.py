#!/usr/bin/env python3
"""Staged / commit-range gate. Exceptions live in .repo-hygiene-allow."""
import argparse
import fnmatch
import json
import re
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request

MIB = 1024 * 1024
DATA = ('.npz', '.npy', '.h5', '.hdf5', '.nc', '.zip', '.tar', '.gz', '.zst',
        '.bz2', '.xz', '.tgz', '.tbz2', '.txz', '.7z', '.rar', '.pptx', '.pdf',
        '.docx', '.xlsx', '.parquet', '.arrow', '.feather', '.sqlite', '.db')


def git(*args, **kwargs):
    return subprocess.check_output(['git', *args], **kwargs)


def allowances(ref):
    name = f'{ref}:.repo-hygiene-allow' if ref else ':.repo-hygiene-allow'
    if subprocess.run(['git', 'cat-file', '-e', name], stderr=subprocess.DEVNULL).returncode:
        return [], set()
    text = git('show', name).decode('utf-8')
    rules, identities = [], set()
    section = "paths"
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        if line.strip() in ('identities:', 'paths:'):
            section = line.strip()[:-1]
            continue
        fields = line.split('\t', 1)
        if len(fields) != 2 or not all(field.strip() for field in fields):
            raise ValueError(f'.repo-hygiene-allow:{number}: expected {section} entry<TAB>reason')
        pattern, reason = (field.strip() for field in fields)
        if section == 'identities':
            if not re.fullmatch(r'[^\s<>@*?]+@[^\s<>@*?]+', pattern):
                raise ValueError(f'.repo-hygiene-allow:{number}: expected exact email<TAB>reason')
            identities.add(pattern.casefold())
            continue
        if pattern.startswith('/') or '..' in pattern.split('/'):
            raise ValueError(f'.repo-hygiene-allow:{number}: use repository-relative globs')
        rules.append((pattern, reason))
    return rules, identities


def warn_local_identity():
    """Offline and advisory: Git environment overrides can change either identity."""
    configured = subprocess.run(['git', 'config', 'user.email'], capture_output=True,
                                text=True).stdout.strip()
    for role in ('AUTHOR', 'COMMITTER'):
        effective = git('var', f'GIT_{role}_IDENT').decode().strip()
        email = effective.rsplit('<', 1)[-1].split('>', 1)[0]
        if configured and email.casefold() != configured.casefold():
            print(f'Warning: {role.lower()} email {email} differs from '
                  f'git config user.email ({configured}); CI verifies both identities.',
                  file=sys.stderr)


def check_identities(commits, allowed, repository):
    """GitHub resolves the exact commit's author and committer email to accounts."""
    token = os.environ.get('GITHUB_TOKEN')
    if not token:
        raise ValueError('GITHUB_TOKEN is required for commit identity verification')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository):
        raise ValueError('invalid GitHub repository')
    failures = []
    for commit in commits:
        raw = git('show', '-s', '--format=%an%x00%ae%x00%cn%x00%ce', commit).decode().strip()
        author_name, author_email, committer_name, committer_email = raw.split('\0')
        roles = [('author', author_name, author_email),
                 ('committer', committer_name, committer_email)]
        unresolved = [(role, name, email) for role, name, email in roles
                      if email.casefold() not in allowed]
        if not unresolved:
            continue
        request = urllib.request.Request(
            f'https://api.github.com/repos/{repository}/commits/{commit}',
            headers={'Authorization': f'Bearer {token}',
                     'Accept': 'application/vnd.github+json',
                     'X-GitHub-Api-Version': '2022-11-28'})
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.load(response)
        if data.get('sha') != commit:
            raise ValueError(f'{commit}: GitHub returned a different commit')
        for role, name, email in unresolved:
            login = (data.get(role) or {}).get('login')
            api_email = data.get('commit', {}).get(role, {}).get('email')
            if api_email != email or not login:
                failures.append(f'{commit}: {role} email {email} has no GitHub account; '
                                'use a registered identity or justify it under '
                                'identities: in .repo-hygiene-allow')
    return failures


def entries(raw):
    fields = raw.split(b'\0')
    for i in range(0, len(fields) - 1, 2):
        header = fields[i].split()
        # Raw diffs have one path per entry because rename detection is disabled.
        mode, oid = header[1], header[3].decode()
        if mode == b'160000':  # gitlink; no file blob
            continue
        yield os.fsdecode(fields[i + 1]), oid


def check(staged, base, head, repository=None):
    commits = []
    if staged:
        warn_local_identity()
        blobs = list(entries(git('diff', '--cached', '--raw', '-z', '--no-abbrev',
                                 '--no-renames', '--diff-filter=AMT')))
    else:
        blobs = []
        commits = git('rev-list', '--reverse', f'{base}..{head}').decode().splitlines()
        for commit in commits:
            blobs.extend(entries(git('diff-tree', '--root', '--no-commit-id', '-r',
                                     '--raw', '-z', '--no-abbrev', '--no-renames',
                                     '--diff-filter=AMT', commit)))
        # Includes resolutions introduced by merge commits (diff-tree skips merges).
        blobs.extend(entries(git('diff', '--raw', '-z', '--no-abbrev', '--no-renames',
                                 '--diff-filter=AMT', base, head)))
    rules, allowed_identities = allowances(None if staged else head)
    oids = sorted({oid for _, oid in blobs})
    sizes = {}
    if oids:
        output = git('cat-file', '--batch-check=%(objectname) %(objectsize)',
                     input=('\n'.join(oids) + '\n').encode())
        sizes = {oid: int(size) for oid, size in (row.split() for row in output.decode().splitlines())}
    failures = check_identities(commits, allowed_identities, repository) if repository else []
    counted = set()
    for path, oid in sorted(set(blobs)):
        if any(fnmatch.fnmatchcase(path, pattern) for pattern, _ in rules):
            continue
        counted.add(oid)
        if sizes[oid] > MIB:
            failures.append(f'{path}: {sizes[oid]} bytes exceeds 1 MiB')
        if path.lower().endswith(DATA) or '.tar.' in path.lower():
            failures.append(f'{path}: data/archive type needs a fixture/provenance allowance')
    total = sum(sizes[oid] for oid in counted)
    if total > 5 * MIB:
        failures.append(f'aggregate additions: {total} bytes exceeds 5 MiB')
    if failures:
        print('\n'.join(failures[:30]), file=sys.stderr)
        print('Archive results externally; justify synthetic/published fixtures in '
              '.repo-hygiene-allow (glob<TAB>reason). No environment override.', file=sys.stderr)
        return 1
    # Force upstream defaults: repo config, ignore files, inline comments and env
    # must not silently bypass secret detection. Path allowances never waive secrets.
    with tempfile.TemporaryDirectory(prefix='repo-hygiene-') as tmp:
        config = Path(tmp) / 'gitleaks.toml'
        config.write_text('[extend]\nuseDefault = true\n')
        args = ['gitleaks', 'protect' if staged else 'detect', '--no-banner',
                '--redact', '--log-level=error', '--config', str(config),
                '--gitleaks-ignore-path', os.devnull, '--ignore-gitleaks-allow']
        args += ['--staged'] if staged else ['--log-opts', f'{base}..{head}']
        env = {k: v for k, v in os.environ.items() if not k.startswith('GITLEAKS_')}
        return subprocess.run(args, env=env).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staged', action='store_true')
    parser.add_argument('--base')
    parser.add_argument('--head', default='HEAD')
    parser.add_argument('--github-repository', help='verify range identities via GitHub (needs GITHUB_TOKEN)')
    args = parser.parse_args()
    if args.staged == bool(args.base):
        parser.error('choose --staged or --base REF [--head REF]')
    if args.staged and args.github_repository:
        parser.error('--github-repository requires --base')
    os.chdir(git('rev-parse', '--show-toplevel').decode().strip())
    try:
        return check(args.staged, args.base, args.head, args.github_repository)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f'repo-hygiene: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
