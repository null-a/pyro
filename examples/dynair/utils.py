import os
import subprocess
import time
import torch
from hashlib import md5

def git_rev():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()

def git_wd_is_clean():
    out = subprocess.check_output(['git', 'status', '--porcelain']).decode().strip()
    return len(out) == 0

def make_output_dir(base_path):
    ts = int(time.time())
    path = os.path.join(base_path, str(ts))
    if os.path.exists(path):
        raise 'failed to create output directory'
    else:
        os.makedirs(path)
        return path

def describe_env():
    is_clean = git_wd_is_clean()
    if not is_clean:
        print('warning: working directory is not clean')
    return '\n'.join([
        'start: {}'.format(time.strftime('%d %b %Y %H:%M %Z')),
        'torch version: {}'.format(torch.version.__version__),
        'clean working directory: {}'.format(git_wd_is_clean())
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
