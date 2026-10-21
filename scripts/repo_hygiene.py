#!/usr/bin/env python3
"""Offline staged / commit-range gate. Exceptions live in .repo-hygiene-allow."""
import argparse
import fnmatch
import os
from pathlib import Path
import subprocess
import sys
import tempfile

MIB = 1024 * 1024
DATA = ('.npz', '.npy', '.h5', '.hdf5', '.nc', '.zip', '.tar', '.gz', '.zst',
        '.bz2', '.xz', '.tgz', '.tbz2', '.txz', '.7z', '.rar', '.pptx', '.pdf',
        '.docx', '.xlsx', '.parquet', '.arrow', '.feather', '.sqlite', '.db')


def git(*args, **kwargs):
    return subprocess.check_output(['git', *args], **kwargs)


def allowances(ref):
    name = f'{ref}:.repo-hygiene-allow' if ref else ':.repo-hygiene-allow'
    if subprocess.run(['git', 'cat-file', '-e', name], stderr=subprocess.DEVNULL).returncode:
        return []
    text = git('show', name).decode('utf-8')
    rules = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        fields = line.split('\t', 1)
        if len(fields) != 2 or not all(field.strip() for field in fields):
            raise ValueError(f'.repo-hygiene-allow:{number}: expected glob<TAB>reason')
        pattern, reason = (field.strip() for field in fields)
        if pattern.startswith('/') or '..' in pattern.split('/'):
            raise ValueError(f'.repo-hygiene-allow:{number}: use repository-relative globs')
        rules.append((pattern, reason))
    return rules


def entries(raw):
    fields = raw.split(b'\0')
    for i in range(0, len(fields) - 1, 2):
        header = fields[i].split()
        # Raw diffs have one path per entry because rename detection is disabled.
        mode, oid = header[1], header[3].decode()
        if mode == b'160000':  # gitlink; no file blob
            continue
        yield os.fsdecode(fields[i + 1]), oid


def check(staged, base, head):
    if staged:
        blobs = list(entries(git('diff', '--cached', '--raw', '-z', '--no-abbrev',
                                 '--no-renames', '--diff-filter=AMT')))
    else:
        blobs = []
        for commit in git('rev-list', '--reverse', f'{base}..{head}').splitlines():
            blobs.extend(entries(git('diff-tree', '--root', '--no-commit-id', '-r',
                                     '--raw', '-z', '--no-abbrev', '--no-renames',
                                     '--diff-filter=AMT', commit.decode())))
        # Includes resolutions introduced by merge commits (diff-tree skips merges).
        blobs.extend(entries(git('diff', '--raw', '-z', '--no-abbrev', '--no-renames',
                                 '--diff-filter=AMT', base, head)))
    rules = allowances(None if staged else head)
    oids = sorted({oid for _, oid in blobs})
    sizes = {}
    if oids:
        output = git('cat-file', '--batch-check=%(objectname) %(objectsize)',
                     input=('\n'.join(oids) + '\n').encode())
        sizes = {oid: int(size) for oid, size in (row.split() for row in output.decode().splitlines())}
    failures, counted = [], set()
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
    args = parser.parse_args()
    if args.staged == bool(args.base):
        parser.error('choose --staged or --base REF [--head REF]')
    os.chdir(git('rev-parse', '--show-toplevel').decode().strip())
    try:
        return check(args.staged, args.base, args.head)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f'repo-hygiene: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
