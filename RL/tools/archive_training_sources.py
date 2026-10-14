"""Preserve the exact Python source bytes bound by an existing run identity."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('training_directory',type=Path)
    args = parser.parse_args()
    root = Path(os.environ['SOKURL_PLATFORM_ROOT'])
    run = args.training_directory.resolve()
    identity = json.loads((run/'identity.json').read_text())
    source = run/'source_bundle'
    entries = []
    for name,expected in identity['source_hashes'].items():
        prefix,relative = name.split('/',1)
        base = root/'RL' if prefix=='RL' else root if prefix=='Platform' else None
        if base is None:
            raise RuntimeError(f'unknown source label:{name}')
        path = (base/relative).resolve()
        if not path.is_relative_to(base.resolve()):
            raise RuntimeError('source path escapes owner directory')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'current source differs from run identity:{name}')
        entries.append((name,path,expected))
    source.mkdir(exist_ok=False)
    for name,path,expected in entries:
        dest = source/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest)
        if hashlib.sha256(dest.read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'copy mismatch:{name}')
    git = lambda *args:subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
    (source/'manifest.json').write_text(json.dumps({'status':'PASS','files':identity['source_hashes'],
        'git_head':git('rev-parse','HEAD'),'git_branch':git('branch','--show-current'),
        'git_status_at_archive':git('status','--porcelain'),
        'scope':'run-bound Python bytes;game andnative binaries remain external'},indent=2),encoding='utf-8')
    print(f'PASS:{len(entries)} source files matched andarchived')

if __name__=='__main__':
    main()
