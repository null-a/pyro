from collections import defaultdict
from functools import wraps

class Cache():
    def __init__(self):
        # TODO: Figure out why this didn't work with a
        # WeakValueDictionary.
        # (Since it would be nice to not have to think about clearing
        # the cache.)
        self.store = dict()
        self.counts = defaultdict(lambda: dict(miss=0, hit=0))

    def clear(self):
        self.store.clear()

    def stats(self):
        out = dict(size=len(self.store))
        out.update(self.counts)
        return out

def cached(f):
    name = f.__name__
    @wraps(f)
    def cached_f(self, *args, **kwargs):
        assert len(kwargs) == 0, 'kwargs not supported'
        key = (name,) + args
        if key in self.cache.store:
            self.cache.counts[name]['hit'] += 1
            return self.cache.store[key]
        else:
            self.cache.counts[name]['miss'] += 1
            out = f(self, *args)
            self.cache.store[key] = out
            return out
    return cached_f
