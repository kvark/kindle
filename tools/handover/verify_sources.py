"""Verify library source against the handover receipt; no GPU calls."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('kindle', 'meganeura', 'blade'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name('source-manifest.json').read_text())
    failures = []
    for repo, files in manifest['files'].items():
        for relative, expected in files.items():
            path = getattr(args, repo) / relative
            if not path.is_file():
                failures.append(str(path) + ': missing')
                continue
            with path.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != expected:
                failures.append(str(path) + ': differs')
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f"Verified {sum(map(len, manifest['files'].values()))} source files; no GPU calls")


if __name__ == '__main__':
    main()
