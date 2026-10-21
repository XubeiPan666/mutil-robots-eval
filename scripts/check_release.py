#!/usr/bin/env python3
"""Fast source checks; never import simulator code or print credential values."""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {'.py', '.sh', '.md', '.json', '.yaml', '.yml', '.toml', '.cff', '.bib', '.txt', '.svg'}
SECRET = re.compile(r'\b(?:sk-(?:proj-|or-v1-)?[A-Za-z0-9_-]{24,}|hf_[A-Za-z0-9]{24,}|gh[pousr]_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16})\b')
PRIVATE_IP = re.compile(r'(?<![\w.])(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})(?![\w.])')
PERSONAL_PATH = re.compile(r'(?<![\w.])/(?:home|Users|media)/([A-Za-z0-9_.-]+)/')
PLACEHOLDERS = {'user', 'username', 'your-user', 'path', 'your', 'example'}


def source_files(root: Path = ROOT) -> list[Path]:
    result = subprocess.run(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
        cwd=root, check=True, stdout=subprocess.PIPE,
    )
    names = [name for name in result.stdout.split(b'\0') if name]
    ignored = subprocess.run(
        ['git', 'check-ignore', '--no-index', '-z', '--stdin'],
        cwd=root, input=b'\0'.join(names) + b'\0', stdout=subprocess.PIPE,
    ) if names else None
    if ignored is not None and ignored.returncode not in (0, 1):
        raise RuntimeError('Could not evaluate source ignore rules')
    ignored_names = set(ignored.stdout.split(b'\0')) if ignored else set()
    paths = {root / name.decode('utf-8') for name in names if name not in ignored_names}
    excluded = {'.git', '.codex', '.agents'}
    return sorted(path for path in paths if not (set(path.relative_to(root).parts) & excluded) and path.is_file())


def check(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    files = source_files(root)
    for path in files:
        rel = path.relative_to(root).as_posix()
        if path.name.startswith('.env') and path.name != '.env.example':
            errors.append(f'{rel}: local environment file must not be exported')
        if path.suffix in {'.pem', '.key'}:
            errors.append(f'{rel}: private key file must not be exported')
        if path.is_symlink():
            errors.append(f'{rel}: symlinks require explicit review before export')
            continue
        if path.suffix not in TEXT_SUFFIXES and path.name != '.env.example':
            continue
        text = path.read_text(encoding='utf-8')
        for lineno, line in enumerate(text.splitlines(), 1):
            if SECRET.search(line):
                errors.append(f'{rel}:{lineno}: possible credential (value redacted)')
            if PRIVATE_IP.search(line):
                errors.append(f'{rel}:{lineno}: machine-specific private IP address')
            match = PERSONAL_PATH.search(line)
            if match and match.group(1) not in PLACEHOLDERS:
                errors.append(f'{rel}:{lineno}: machine-specific absolute path')
        if path.suffix == '.py':
            try:
                ast.parse(text, filename=rel)
            except SyntaxError as exc:
                errors.append(f'{rel}:{exc.lineno}: invalid Python syntax')
        elif path.suffix == '.sh':
            result = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
            if result.returncode:
                errors.append(f'{rel}: invalid Bash syntax')
        elif path.suffix == '.json':
            try:
                json.loads(text)
            except json.JSONDecodeError as exc:
                errors.append(f'{rel}:{exc.lineno}: invalid JSON')
        # Check maintained local documentation targets; archival upstream docs
        # may refer to files that are outside this evaluation release.
        maintained_doc = rel in {'README.md', 'CONTRIBUTING.md', 'SECURITY.md', 'THIRD_PARTY_NOTICES.md', 'RoboFactory/script/MACG/STACKBOWL_README.md', 'RoboFactory/script/MACG/eval_scripts_rule_base/REPRODUCIBILITY.md'} or (rel.startswith('docs/') and not rel.startswith('docs/archive/'))
        if path.suffix == '.md' and maintained_doc:
            for match in re.finditer(r'\]\(<?([^\s)>]+)>?(?:\s+"[^"]*")?\)', text):
                target = unquote(match.group(1))
                parsed = urlsplit(target)
                if parsed.scheme or target.startswith('#'):
                    continue
                target_path = parsed.path
                if target_path and not (path.parent / target_path).exists():
                    errors.append(f'{rel}: missing link target {target_path}')
    return errors


def main() -> int:
    errors = check()
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        print(f'Release checks failed: {len(errors)} issue(s).', file=sys.stderr)
        return 1
    print(f'Release checks passed: {len(source_files())} source files; Python / Bash / JSON syntax, maintained documentation links, credential and local-configuration patterns.')
    print('These static checks do not validate GPU simulation, checkpoint behavior, or all possible secrets.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
