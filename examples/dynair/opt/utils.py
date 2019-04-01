import os
import subprocess
import uuid
import time
import torch
from hashlib import md5

def git_rev():
    git_dir = os.path.dirname(__file__)
    return subprocess.check_output(['git', '-C', git_dir, 'rev-parse', '--short', 'HEAD']).decode().strip()

def git_wd_is_clean():
    git_dir = os.path.dirname(__file__)
    out = subprocess.check_output(['git', '-C', git_dir, 'status', '--porcelain']).decode().strip()
    return len(out) == 0

def make_output_dir(base_path):
    name = str(uuid.uuid1())
    path = os.path.join(base_path, name)
    if os.path.exists(path):
        raise Exception('output directory already exists')
    else:
        os.makedirs(path)
        symlink_path = os.path.join(base_path, 'latest')
        if os.path.exists(symlink_path):
            os.remove(symlink_path)
        # When creating lots of jobs concurrently creating the symlink
        # can fail, since another job can create the symlink in
        # between this job deleting the existing symlink and creating
        # the new one. We just swallow the error, since the symlink
        # isn't used in this setting.
        try:
            os.symlink(name, symlink_path)
        except FileExistsError:
            pass
        with open(os.path.join(path, 'uuid.txt'), 'w') as f:
            f.write(name)
        with open(os.path.join(path, 'timestamp.txt'), 'w') as f:
            f.write(str(int(time.time())))
        return path

def describe_env():
    is_clean = git_wd_is_clean()
    if not is_clean:
        print('warning: working directory is not clean')
    return '\n'.join([
        'git rev: {}'.format(git_rev()),
        'clean working directory: {}'.format(git_wd_is_clean()),
        'torch version: {}'.format(torch.version.__version__),
    ])

def append_line(fn, s):
    with open(fn, 'a') as f:
        f.write('{}\n'.format(s))

# https://bitbucket.org/prologic/tools/src/tip/md5sum
def md5sum(fn):
    hash = md5()
    with open(fn, 'rb') as f:
        for chunk in iter(lambda: f.read(128 * hash.block_size), b""):
            hash.update(chunk)
    return hash.hexdigest()

# Crude parser for (parts of the) guide architecture CLA.
# 'foo-bar|baz', 'foo-bar' => ['foo', 'bar']
# 'foo-bar|baz', 'foo-baz' => ['foo', 'baz']
# 'foo-bar|baz', 'foo-baz-10-20' => ['foo', 'baz', 10, 20]
# 'foo-bar|baz', 'blah' => error
# 'foo-bar|baz', 'foo-blah' => error
def parse_cla(fmt, input_string):
    fmts = fmt.split('-')
    toks = input_string.split('-')
    assert len(toks) >= len(fmts), 'input is too short'
    out = []
    for f, t in zip(fmts, toks):
        vals = f.split('|')
        assert t in vals, 'input "{}" is not of the expected format "{}"'.format(input_string, fmt)
        out.append(t)

    # Trailing tokens are assumed to be ints.
    out = out + [int(i) for i in toks[len(fmts):]]
    return out