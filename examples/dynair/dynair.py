import torch.nn as nn
import pyro.poutine as poutine
from collections import namedtuple

Config = namedtuple('Config',
                    ['seq_length', # data specified config
                     'num_chan',
                     'image_size',
                     'x_size',
                     'max_obj_count',
                     'w_size',     # global config
                     'y_size',
                     'z_size',
                     'window_size',
                    ])

def get_modules_with_cache(parent):
    out = []
    if has_cache(parent):
        out.append(parent)
    for m in parent.children():
        out.extend(get_modules_with_cache(m))
    return out

def has_cache(m):
    return hasattr(m, 'cache')

class DynAIR(nn.Module):
    def __init__(self, cfg, model, guide, use_cuda=False):
        super(DynAIR, self).__init__()

        self.cfg = cfg
        self.model = model
        self.guide = guide

        self.modules_with_cache = get_modules_with_cache(self)

        # CUDA
        if use_cuda:
            self.cuda()

    def clear_cache(self):
        for m in self.modules_with_cache:
            m.cache.clear()

    def cache_stats(self):
        return dict((m._get_name(), m.cache.stats())
                    for m in self.modules_with_cache)

    def infer(self, seqs, obj_counts, num_extra_frames=0):
        trace = poutine.trace(self.guide).get_trace((seqs, obj_counts))
        frames, _, _ = poutine.replay(self.model, trace)((seqs, obj_counts))
        wss, zss, y = trace.nodes['_RETURN']['value']

        bkg = self.model.decode_bkg(y)

        extra_wss = []
        extra_zss = []
        extra_frames = []

        ws = wss[-1]
        zs = zss[-1]

        for t in range(num_extra_frames):
            zs, ws = self.model.transition(self.cfg.seq_length + t, obj_counts, zs, ws)
            frame_mean = self.model.emission(zs, ws, bkg, obj_counts)
            extra_frames.append(frame_mean)
            extra_wss.append(ws)
            extra_zss.append(zs)

        self.clear_cache()

        return frames, wss, extra_frames, extra_wss
