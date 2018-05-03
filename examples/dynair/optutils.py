import os
import subprocess
import time
import torch
from hashlib import md5
from functools import wraps

def git_rev():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()

def git_wd_is_clean():
    out = subprocess.check_output(['git', 'status', '--porcelain']).decode().strip()
    return len(out) == 0

def make_output_dir(base_path):
    ts = int(time.time())
    name = str(ts)
    path = os.path.join(base_path, name)
    if os.path.exists(path):
        raise 'failed to create output directory'
    else:
        os.makedirs(path)
        symlink_path = os.path.join(base_path, 'latest')
        if os.path.exists(symlink_path):
            os.remove(symlink_path)
        os.symlink(name, symlink_path)
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

class throttle(object):
    def __init__(self, period):
        self.period = period
        self.time_of_last_call = 0

    def __call__(self, f):
        @wraps(f)
        def throttled(*args, **kwargs):
            now = time.time()

            if now - self.time_of_last_call > self.period:
                self.time_of_last_call = now
                return f(*args, **kwargs)

        return throttled
