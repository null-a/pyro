import torch
import torch.nn as nn
import pyro.poutine as poutine
from collections import namedtuple

Config = namedtuple('Config',
                    ['num_chan', # data specified config
                     'image_size',
                     'x_size',
                     'max_obj_count',
                     'w_size',     # global config
                     'y_size',
                     'z_size',
                     'window_size',
                     'use_depth',
                     'w_transition',
                     'z_transition',
                     'decode_obj',
                     'model_delta_w',
                     'model_delta_z',
                     'guide_w',
                     'guide_z',
                     'guide_input_embed',
                     'guide_window_embed',
                    ])

def config(module_config, data_config):
    return Config(**merge(module_config, data_config))

def merge(*dicts):
    out = {}
    for d in dicts:
        out.update(**d)
    return out

def get_modules_with_cache(parent):
    out = []
    if has_cache(parent):
        out.append(parent)
    for m in parent.children():
        out.extend(get_modules_with_cache(m))
    return out

def has_cache(m):
    return hasattr(m, 'cache')

# It's convenient to detach here for consumers of `infer`. e.g.
# visualisation code can call `numpy()` directly on these.

def frames_to_tensor(arr):
    # Turn an array of frames (of length seq_len) returned by the
    # model into a (batch, seq_len, rest...) tensor.
    return torch.cat([t.unsqueeze(0) for t in arr]).transpose(0, 1).detach()

def latents_to_tensor(xss):
    return torch.stack([torch.stack(xs) for xs in xss]).transpose(2, 0).detach()

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
        return dict((m._get_name() + '-' + str(id(m)), m.cache.stats())
                    for m in self.modules_with_cache)

    def infer(self, seqs, obj_counts, num_extra_frames=0, sample_extra=True):
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
            zs_params, ws_params = self.model.transition_params(zs, ws)
            if sample_extra:
                seq_length = seqs.size(1)
                zs, ws = self.model.sample_zs_and_ws(seq_length + t,
                                                     obj_counts,
                                                     zs_params, ws_params)
            else:
                zs = [z_mean for (z_mean, _) in zs_params]
                ws = [w_mean for (w_mean, _) in ws_params]
            frame_mean = self.model.emission(zs, ws, bkg, obj_counts)
            extra_frames.append(frame_mean)
            extra_wss.append(ws)
            extra_zss.append(zs)

        self.clear_cache()

        # TODO: Consider splitting out depth from w here to make
        # things easier for consumers.

        return (frames_to_tensor(frames),
                latents_to_tensor(wss),
                frames_to_tensor(extra_frames) if len(extra_frames) > 0 else torch.zeros(0),
                latents_to_tensor(extra_wss) if len(extra_wss) > 0 else torch.zeros(0))
