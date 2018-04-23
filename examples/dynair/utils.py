import os
import subprocess
import time
import torch
from hashlib import md5

def git_rev():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()

def git_diff():
    return subprocess.check_output(['git', 'diff']).decode().strip()

def make_output_dir(rev=None, i=0):
    assert i < 100, "Lot's of runs, something probably went wrong."
    if rev is None:
        rev = git_rev()
    path = './runs/{}-{}'.format(rev, i)
    if os.path.exists(path):
        return make_output_dir(rev, i + 1)
    else:
        os.makedirs(path)
        return path

def describe_env():
    return '\n'.join([
        time.strftime('%d %b %Y %H:%M %Z'),
        'torch version: {}'.format(torch.version.__version__),
        git_diff()
    ])

def append_line(s, fn):
    with open(fn, 'a') as f:
        f.write('{}\n'.format(s))

# https://bitbucket.org/prologic/tools/src/tip/md5sum
def md5sum(fn):
    hash = md5()
    with open(fn, 'rb') as f:
        for chunk in iter(lambda: f.read(128 * hash.block_size), b""):
            hash.update(chunk)
    return hash.hexdigest()
