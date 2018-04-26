import torch
import torch.nn as nn
from torch.nn.functional import affine_grid, grid_sample, sigmoid, softplus
from torch.nn.utils import clip_grad_norm
from torch.autograd import Variable

import numpy as np

import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer import SVI
import pyro.poutine as poutine


import dyn3d_modules as mod
from utils import make_output_dir, append_line, describe_env

import visdom
from PIL import Image, ImageDraw

import os
import json
import argparse
from collections import defaultdict
import time
from datetime import timedelta
import enum
from functools import partial

# TODO: There's no reason the dedicated nets for t=0 for the w and z
# guides have to be used/not used in unison.

# TODO: Figure out how to conveniently specify the entire guide
# architecture. Maybe have DynAIR take sub-modules as args, allowing
# the config. to be varied by instantiating DynAIR with different
# modules? (Each of which may take its own args?) One problem with
# that is that sub-modules currently look at their parent for global
# config. info., so that may need extracting to its own structure.
# Also, some modules need to call methods on the parent, so that would
# need wiring up, perhaps in `init` for DynAIR. (By passing `self` to
# some method on the sub-module in question.)

class GuideArch(enum.Enum):
    rnn = enum.auto()
    rnnt0 = enum.auto() # dedicate nets for step t=0
    isf = enum.auto()   # image-so-far


from functools import wraps

def cached(f):
    name = f.__name__
    @wraps(f)
    def cached_f(self, *args, **kwargs):
        assert len(kwargs) == 0, 'kwargs not supported'
        key = (name,) + args
        if key in self.cache.store:
            self.cache.stats[name]['hit'] += 1
            return self.cache.store[key]
        else:
            self.cache.stats[name]['miss'] += 1
            out = f(self, *args)
            self.cache.store[key] = out
            return out
    return cached_f


class PropCache():
    def __init__(self):
        # TODO: Figure out why this didn't work with a
        # WeakValueDictionary.
        # (Since it would be nice to not have to think about clearing
        # the cache.)
        self.store = dict()
        self.stats = defaultdict(lambda: dict(miss=0, hit=0))


class DynAIR(nn.Module):
    def __init__(self, guide_arch=GuideArch.isf, use_cuda=False):

        super(DynAIR, self).__init__()

        self.cache = PropCache()

        self.guide_arch = guide_arch

        self.prototype = torch.tensor(0.).cuda() if use_cuda else torch.tensor(0.)
        self.use_cuda = use_cuda

        self.seq_length = 20

        self.max_obj_count = 3

        self.image_size = 50
        self.num_chan = 3 # not inc. the alpha channel

        self.window_size = 22

        self.y_size = 50
        self.z_size = 50
        self.w_size = 2 # (x, y)
        # This may need increasing /slightly/ to allow for the
        # rotation of the avatars.
        self.window_scale = 0.25

        self.x_size = self.num_chan * self.image_size**2
        self.x_att_size = self.num_chan * self.window_size**2 # patches cropped from the input
        self.x_obj_size = (self.num_chan+1) * self.window_size**2 # contents of the object window

        # Priors:

        self.y_prior_mean = self.prototype.new_zeros(self.y_size)
        self.y_prior_sd = self.prototype.new_ones(self.y_size)

        # TODO: Using a (reparameterized) uniform would probably be
        # better for the cubes data set. (Though this would makes less
        # sense if we allowed objects to begin off screen.)

        self.w_0_prior_mean = Variable(torch.Tensor([0, 0]))
        self.w_0_prior_sd = Variable(torch.Tensor([0.7, 0.7]),
                                     requires_grad=False)
        if use_cuda:
            self.w_0_prior_mean = self.w_0_prior_mean.cuda()
            self.w_0_prior_sd = self.w_0_prior_sd.cuda()


        self.z_0_prior_mean = self.prototype.new_zeros(self.z_size)
        self.z_0_prior_sd = self.prototype.new_ones(self.z_size)

        self.likelihood_sd = 0.3

        # Modules

        # Guide modules:

        if self.guide_arch == GuideArch.isf:
            self.guide_w_params = GuideW_ImageSoFar(self)
        else:
            self.guide_w_params = GuideW_ObjRnn(self, dedicated_t0=self.guide_arch==GuideArch.rnnt0)

        self.guide_z_params = GuideZ(self, dedicated_t0=self.guide_arch==GuideArch.rnnt0)

        self.y_param = mod.ParamY([200, 200], self.x_size, self.y_size)

        # Model modules:

        self.decode_obj = mod.DecodeObj([200, 200], self.z_size, self.num_chan, self.window_size, alpha_bias=-2.0)
        self._decode_bkg = mod.DecodeBkg([200, 200], self.y_size, self.num_chan, self.image_size)

        self.w_transition = mod.WTransition(self.z_size, self.w_size, 50)
        self.z_transition = mod.ZTransition(self.z_size, 50)
        #self.z_transition = mod.ZGatedTransition(self.z_size, 50, 50)

        # CUDA
        if use_cuda:
            self.cuda()


    # We wrap `_decode_bkg` here to enable caching without
    # interferring with the magic behaviour that comes from having an
    # `nn.Module` as a property of the model class.
    @cached
    def decode_bkg(self, *args, **kwargs):
        return self._decode_bkg(*args, **kwargs)

    # TODO: This do_likelihood business is unpleasant.
    def model(self, batch, obj_counts, do_likelihood=True):
        batch_size = batch.size(0)
        assert_size(obj_counts, (batch_size,))
        assert all(0 <= obj_counts) and all(obj_counts <= self.max_obj_count), 'Object count out of range.'

        zss = []
        wss = []
        frames = []

        with pyro.iarange('data'):

            y = self.model_sample_y(batch_size)
            bkg = self.decode_bkg(y)

            for t in range(self.seq_length):
                if t>0:
                    zs_prev, ws_prev = zss[-1], wss[-1]
                else:
                    zs_prev, ws_prev = None, None

                zs, ws = self.model_transition(t, obj_counts, zs_prev, ws_prev)
                frame_mean = self.model_emission(zs, ws, bkg, obj_counts)

                zss.append(zs)
                wss.append(ws)
                frames.append(frame_mean)

                if do_likelihood:
                    self.likelihood(t, frame_mean, batch[:, t])

        # It's possible the cache will still need clearing manually,
        # but this catches the common cases of SVI & infer.
        self.cache.store.clear()

        return frames, wss, zss

    def likelihood(self, t, frame_mean, obs):
        frame_sd = (self.likelihood_sd * self.prototype.new_ones(1)).expand_as(frame_mean)
        # TODO: Using a normal here isn't very sensible since the data
        # is in [0, 1]. Do something better.
        pyro.sample('x_{}'.format(t),
                    dist.Normal(frame_mean, frame_sd).reshape(extra_event_dims=3),
                    obs=obs)

    def model_sample_y(self, batch_size):
        return pyro.sample('y', dist.Normal(self.y_prior_mean.expand(batch_size, -1),
                                            self.y_prior_sd.expand(batch_size, -1))
                                .reshape(extra_event_dims=1))

    def model_sample_w_0(self, i, batch_size):
        return pyro.sample('w_0_{}'.format(i),
                           dist.Normal(
                               self.w_0_prior_mean.expand(batch_size, -1),
                               self.w_0_prior_sd.expand(batch_size, -1))
                           .reshape(extra_event_dims=1))

    def model_sample_w(self, t, i, w_mean, w_sd):
        return pyro.sample('w_{}_{}'.format(t, i),
                           dist.Normal(w_mean, w_sd).reshape(extra_event_dims=1))

    def model_sample_z_0(self, i, batch_size):
        return pyro.sample('z_0_{}'.format(i),
                           dist.Normal(
                               self.z_0_prior_mean.expand(batch_size, -1),
                               self.z_0_prior_sd.expand(batch_size, -1))
                           .reshape(extra_event_dims=1))

    def model_sample_z(self, t, i, z_mean, z_sd):
        return pyro.sample('z_{}_{}'.format(t, i),
                           dist.Normal(z_mean, z_sd).reshape(extra_event_dims=1))

    def model_transition_one(self, t, i, z_prev, w_prev):
        batch_size = z_prev.size(0)
        assert_size(z_prev, (batch_size, self.z_size))
        assert_size(w_prev, (batch_size, self.w_size))
        z_mean, z_sd = self.z_transition(z_prev)
        w_mean, w_sd = self.w_transition(z_prev, w_prev)
        z = self.model_sample_z(t, i, z_mean, z_sd)
        w = self.model_sample_w(t, i, w_mean, w_sd)
        return z, w

    def model_transition(self, t, obj_counts, zs_prev, ws_prev):
        batch_size = obj_counts.size(0)
        assert_size(obj_counts, (batch_size,))

        if t==0:
            assert zs_prev is None and ws_prev is None
        else:
            assert t>0
            assert len(zs_prev) == self.max_obj_count # was zs[t-1]
            assert len(ws_prev) == self.max_obj_count # was ws[t-1]

        zs = []
        ws = []

        # To begin with, we'll sample max_obj_count objects for all
        # sequences, and throw out the extra objects. We can consider
        # refining this to avoid this unnecessary sampling later.

        for i in range(self.max_obj_count):

            mask = Variable((obj_counts > i).float())

            with poutine.scale(None, mask):
                if t > 0:
                    z, w = self.model_transition_one(t, i, zs_prev[i], ws_prev[i])
                else:
                    z = self.model_sample_z_0(i, batch_size)
                    w = self.model_sample_w_0(i, batch_size)

            zs.append(z)
            ws.append(w)

        return zs, ws

    def model_emission(self, zs, ws, bkg, obj_counts):
        #batch_size = z.size(0)
        acc = bkg
        for i, (z, w) in enumerate(zip(zs, ws)):
            mask = tuple((obj_counts > i).tolist())
            acc = self.model_composite_object(z, w, mask, acc)
        return acc

    @cached
    def model_composite_object(self, z, w, mask, image_so_far):
        assert type(mask) == tuple # to facilitate caching on the mask
        assert z.size(0) == w.size(0) == image_so_far.size(0)
        mask = Variable(torch.Tensor(mask).type_as(z)) # move to gpu
        x_att = self.decode_obj(z) * mask.view(-1, 1)
        return over(self.window_to_image(w, x_att), image_so_far)


    # ==GUIDE==================================================

    def guide(self, batch, obj_counts, *args):

        # I'd rather register model/guide modules in their methods,
        # but this is easier.
        pyro.module('dynair', self)
        # for name, _ in self.named_parameters():
        #     print(name)

        batch_size = batch.size(0)
        assert_size(batch, (batch_size, self.seq_length, self.num_chan, self.image_size, self.image_size))

        assert_size(obj_counts, (batch_size,))
        assert all(0 <= obj_counts) and all(obj_counts <= self.max_obj_count), 'Object count out of range.'

        ws = mk_matrix(self.seq_length, self.max_obj_count)
        zs = mk_matrix(self.seq_length, self.max_obj_count)

        with pyro.iarange('data'):

            # NOTE: Here we're guiding y based on the contents of the
            # first frame only.
            y = self.guide_y(batch[:, 0])

            for t in range(self.seq_length):

                x = batch[:, t]
                w_guide_state_prev = None
                mask_prev = None

                # As in the model, we'll sample max_obj_count objects for
                # all sequences, and ignore the ones we don't need.

                for i in range(self.max_obj_count):

                    w_prev_i = ws[t-1][i] if t > 0 else None
                    z_prev_i = zs[t-1][i] if t > 0 else None
                    w_t_prev = ws[t][i-1] if i > 0 else None
                    z_t_prev = zs[t][i-1] if i > 0 else None

                    mask = Variable((obj_counts > i).float())

                    with poutine.scale(None, mask):
                        w, w_guide_state = self.guide_w(t, i, x, y, w_prev_i, z_prev_i, w_t_prev, z_t_prev, mask_prev, w_guide_state_prev)
                        x_att = self.image_to_window(w, x)
                        z = self.guide_z(t, i, w, x_att, z_prev_i)

                    ws[t][i] = w
                    zs[t][i] = z

                    w_guide_state_prev = w_guide_state
                    mask_prev = mask

        return ws, zs, y

    def guide_y(self, x0):
        y_mean, y_sd = self.y_param(x0)
        return pyro.sample('y', dist.Normal(y_mean, y_sd).reshape(extra_event_dims=1))

    def guide_w(self, t, i, *args, **kwargs):
        w_mean, w_sd, w_guide_state = self.guide_w_params(t, i, *args, **kwargs)
        w = pyro.sample('w_{}_{}'.format(t, i), dist.Normal(w_mean, w_sd).reshape(extra_event_dims=1))
        return w, w_guide_state

    def guide_z(self, t, i, *args, **kwargs):
        z_mean, z_sd = self.guide_z_params(t, i, *args, **kwargs)
        return pyro.sample('z_{}_{}'.format(t, i), dist.Normal(z_mean, z_sd).reshape(extra_event_dims=1))

    def image_to_window(self, w, images):
        n = w.size(0)
        assert_size(w, (n, self.w_size))
        assert_size(images, (n, self.num_chan, self.image_size, self.image_size))
        theta_inv = expand_theta(w2_to_theta_inv(w, self.window_scale))
        grid = affine_grid(theta_inv, torch.Size((n, self.num_chan, self.window_size, self.window_size)))
        return grid_sample(images, grid, padding_mode='border').view(n, -1)

    def window_to_image(self, w, windows):
        n = w.size(0)
        assert_size(w, (n, self.w_size))
        assert_size(windows, (n, self.x_obj_size))
        theta = expand_theta(w2_to_theta(w, self.window_scale))
        assert_size(theta, (n, 2, 3))
        grid = affine_grid(theta, torch.Size((n, self.num_chan+1, self.image_size, self.image_size)))
        # first arg to grid sample should be (n, c, in_w, in_h)
        return grid_sample(windows.view(n, self.num_chan+1, self.window_size, self.window_size), grid)

    def params_with_nan(self):
        return (name for (name, param) in self.named_parameters() if np.isnan(param.data.view(-1)[0]))

    def infer(self, batch, obj_counts, num_extra_frames=0):
        trace = poutine.trace(self.guide).get_trace(batch, obj_counts)
        frames, _, _ = poutine.replay(self.model, trace)(batch, obj_counts, do_likelihood=False)
        wss, zss, y = trace.nodes['_RETURN']['value']

        bkg = self.decode_bkg(y)

        extra_wss = []
        extra_zss = []
        extra_frames = []

        ws = wss[-1]
        zs = zss[-1]

        for t in range(num_extra_frames):
            zs, ws = self.model_transition(self.seq_length + t, obj_counts, zs, ws)
            frame_mean = self.model_emission(zs, ws, bkg, obj_counts)
            extra_frames.append(frame_mean)
            extra_wss.append(ws)
            extra_zss.append(zs)

        return frames, wss, extra_frames, extra_wss

# TODO: Extend with CNN variants used for guide for w. (Are there
# opportunities for code re-use, since for both w and z compute
# parameters from main + side input. The different been that there is
# no recurrent part of the guide for z.)
class GuideZ(nn.Module):
    def __init__(self, parent, dedicated_t0):
        super(GuideZ, self).__init__()

        x_att_embed_size = 200

        self.z_param = mod.ParamZ(
            [200, 200],
            parent.w_size + x_att_embed_size + parent.z_size, # input size
            parent.z_size)

        if dedicated_t0:
            self.z0_param = mod.ParamZ(
                [200, 200],
                parent.w_size + x_att_embed_size, # input size
                parent.z_size)
        else:
            self.z_init = nn.Parameter(torch.zeros(parent.z_size))

        self.x_att_embed = mod.MLP(parent.x_att_size, [x_att_embed_size], nn.ReLU, True)


    def forward(self, t, i, w, x_att, z_prev_i):
        batch_size = w.size(0)
        x_att_flat = x_att.view(x_att.size(0), -1)
        x_att_embed = self.x_att_embed(x_att_flat)
        if t == 0 and hasattr(self, 'z0_param'):
            z_mean, z_sd = self.z0_param(torch.cat((w, x_att_embed), 1))
        else:
            if t == 0:
                assert z_prev_i is None
                z_prev_i = batch_expand(self.z_init, batch_size)
            z_delta, z_sd = self.z_param(torch.cat((w, x_att_embed, z_prev_i), 1))
            z_mean = z_prev_i + z_delta
        return z_mean, z_sd


class GuideW_ObjRnn(nn.Module):
    def __init__(self, parent, dedicated_t0):
        super(GuideW_ObjRnn, self).__init__()

        x_embed_size = 800
        rnn_hid_size = 200

        # Use parent's cache for simplicity.
        self.cache = parent.cache

        self._x_embed = mod.MLP(parent.x_size, [800, x_embed_size], nn.ReLU)

        self.w_param = mod.ParamW(
            x_embed_size + parent.w_size + parent.z_size, # input size
            rnn_hid_size, [], parent.w_size, parent.z_size,
            sd_bias=-2.25)

        if dedicated_t0:
            self.w0_param = mod.ParamW(
                x_embed_size, # input size
                rnn_hid_size, [], parent.w_size, parent.z_size,
                sd_bias=0.0) # TODO: This could probably stand to be increased a little.
        else:
            self.w_init = Variable(parent.prototype.new_zeros(parent.w_size))
            # TODO: Small init.
            self.z_init = nn.Parameter(torch.zeros(parent.z_size))

    @cached
    def x_embed(self, *args, **kwargs):
        return self._x_embed(*args, **kwargs)

    def forward(self, t, i, x, y, w_prev_i, z_prev_i, w_t_prev, z_t_prev, mask_prev, rnn_hid_prev):
        batch_size = x.size(0)

        x_flat = x.view(batch_size, -1)
        x_embed = self.x_embed(x_flat)

        if t == 0 and hasattr(self, 'w0_param'):
            w_mean, w_sd, rnn_hid = self.w0_param(x_embed, w_t_prev, z_t_prev, rnn_hid_prev)
        else:
            if t == 0:
                assert w_prev_i is None and z_prev_i is None
                w_prev_i = batch_expand(self.w_init, batch_size)
                z_prev_i = batch_expand(self.z_init, batch_size)
            w_delta, w_sd, rnn_hid = self.w_param(torch.cat((x_embed, w_prev_i, z_prev_i), 1), w_t_prev, z_t_prev, rnn_hid_prev)
            w_mean = w_prev_i + w_delta

        return w_mean, w_sd, rnn_hid


class GuideW_ImageSoFar(nn.Module):
    def __init__(self, parent):
        super(GuideW_ImageSoFar, self).__init__()

        self.decode_bkg = parent.decode_bkg
        self.model_composite_object = parent.model_composite_object

        # TODO: Figure out how best to specify the desired architecture.
        self.w_param = mod.ParamW_Isf_Mlp(parent)
        #self.w_param = mod.ParamW_Isf_Cnn_Mixin(parent)
        #self.w_param = mod.ParamW_Isf_Cnn_AM(parent)

        # TODO: I don't see a problem making this a parameter in the
        # CNN+AM case, assuming we're computing absolute position. In
        # other cases it's less clear. OTOH, it's 2/3 real numbers,
        # and z_init alone can probably do the job.
        self.w_init = Variable(parent.prototype.new_zeros(parent.w_size))
        self.z_init = nn.Parameter(torch.zeros(parent.z_size))



    def forward(self, t, i, x, y, w_prev_i, z_prev_i, w_t_prev, z_t_prev, mask_prev, image_so_far_prev):
        batch_size = x.size(0)

        if i == 0:
            assert image_so_far_prev is None
            assert mask_prev is None
            image_so_far = self.decode_bkg(y)
        else:
            assert image_so_far_prev is not None
            assert w_t_prev is not None
            assert z_t_prev is not None
            # Add previous object to image so far.
            image_so_far = self.model_composite_object(z_t_prev, w_t_prev,
                                                       tuple(mask_prev.tolist()),
                                                       image_so_far_prev)

        # TODO: Should we be preventing gradients from flowing back
        # from the guide through image_so_far?

        if t == 0:
            assert w_prev_i is None and z_prev_i is None
            w_prev_i = batch_expand(self.w_init, batch_size)
            z_prev_i = batch_expand(self.z_init, batch_size)

        # For simplicity, feed `x - image_so_far` into the net, though
        # note that alternatives exist.
        diff = x - image_so_far
        w_delta, w_sd = self.w_param(diff, w_prev_i, z_prev_i)
        # TODO: Prefer absolute position here?
        w_mean = w_prev_i + w_delta
        return w_mean, w_sd, image_so_far



def batch_expand(t, b):
    return t.expand((b,) + t.size())


# Spatial transformer helpers. (Taken from the AIR example.)

expansion_indices = torch.LongTensor([1, 0, 2, 0, 1, 3])

def expand_theta(theta):
    # Take a batch of three-vectors, and massages them into a batch of
    # 2x3 matrices with elements like so:
    # [s,x,y] -> [[s,0,x],
    #             [0,s,y]]
    n = theta.size(0)
    assert_size(theta, (n, 3))
    out = torch.cat((torch.zeros([1, 1]).type_as(theta).expand(n, 1), theta), 1)
    ix = Variable(expansion_indices)
    if theta.is_cuda:
        ix = ix.cuda()
    out = torch.index_select(out, 1, ix)
    out = out.view(n, 2, 3)
    return out

def w_to_theta(w):
    # Takes w = (log(scale), pos_x, pos_y) to parameter theta of the
    # spatial transform.
    scale = torch.exp(w[:, 0:1])
    scale_inv = 1 / scale
    xy = -w[:, 1:] * scale_inv
    out = torch.cat((scale_inv, xy), 1)
    return out

def w_to_theta_inv(w):
    # Takes w to theta inverse.
    scale = torch.exp(w[:, 0:1])
    xy = w[:, 1:]
    out = torch.cat((scale, xy), 1)
    return out


def w2_to_theta(w, scale):
    # Takes w = (x, y) & scale to parameter theta of the spatial
    # transform.
    batch_size = w.size(0)
    scale_inv = torch.tensor(1.0 / scale).type_as(w).expand(batch_size, 1)
    return torch.cat((scale_inv, w), 1)

def w2_to_theta_inv(w, scale):
    batch_size = w.size(0)
    scale = torch.tensor(scale).type_as(w).expand(batch_size, 1)
    return torch.cat((scale, w), 1)


def assert_size(t, expected_size):
    actual_size = t.size()
    assert actual_size == expected_size, 'Expected size {} but got {}.'.format(expected_size, tuple(actual_size))



# This assumes that the number of channels is 3 + the alpha channel.
def over(a, b):
    # a over b
    # https://en.wikipedia.org/wiki/Alpha_compositing
    # assert a.size() == (n, 4, image_size, image_size)
    assert a.size(1) == 4
    assert b.size(1) == 3
    rgb_a = a[:, 0:3] # .size() == (n, 3, image_size, image_size)
    alpha_a = a[:, 3:4] # .size() == (n, 1, image_size, image_size)
    return rgb_a * alpha_a + b * (1 - alpha_a)


def split(t, batch_size, num_test_batches):
    n = t.size(0)
    assert batch_size > 0
    assert num_test_batches >= 0
    assert n % batch_size == 0
    num_train_batches = (n // batch_size) - num_test_batches
    assert batch_size * (num_train_batches + num_test_batches) == n
    batches = t.chunk(n // batch_size)
    train = batches[0:num_train_batches]
    test = batches[num_train_batches:(num_train_batches+num_test_batches)]
    return train, test


def run_vis(vis, dynair, X, Y, epoch, step):
    n = X.size(0)

    frames, wss, extra_frames, extra_wss = dynair.infer(X, Y, 15)

    frames = frames_to_tensor(frames)
    ws = latents_to_tensor(wss)
    extra_frames = frames_to_tensor(extra_frames)
    extra_ws = latents_to_tensor(extra_wss)

    for k in range(n):
        out = overlay_multiple_window_outlines(dynair, frames[k], ws[k], Y[k])
        vis.images(frames_to_rgb_list(X[k].cpu()), nrow=10,
                   opts=dict(title='input {} after epoch {} step {}'.format(k, epoch, step)))
        vis.images(frames_to_rgb_list(out.cpu()), nrow=10,
                   opts=dict(title='recon {} after epoch {} step {}'.format(k, epoch, step)))

        out = overlay_multiple_window_outlines(dynair, extra_frames[k], extra_ws[k], Y[k])
        vis.images(frames_to_rgb_list(out.cpu()), nrow=10,
                   opts=dict(title='extra {} after epoch {} step {}'.format(k, epoch, step)))

def vis_hook(vis, dynair, X, Y, epoch, step):
    # TODO: Add logic for perform vis. at required intervals.
    run_vis(vis, dynair, X, Y, epoch, step)

def run_svi(dynair, X_split, Y_split, num_epochs, vis_hook):
    t0 = time.time()
    output_path = make_output_dir()
    append_line(describe_env(), os.path.join(output_path, 'env.txt'))

    X_train, X_test = X_split
    Y_train, Y_test = Y_split
    batch_size = X_train[0].size(0)

    # Produce visualisations for the first train & test data points
    # where possible.
    if len(X_test) > 0:
        X_vis = torch.cat((X_train[0][0:1], X_test[0][0:1]))
        Y_vis = torch.cat((Y_train[0][0:1], Y_test[0][0:1]))
    else:
        X_vis = X_train[0][0:1]
        Y_vis = Y_train[0][0:1]

    def per_param_optim_args(module_name, param_name):
        return {'lr': 1e-4}

    svi = SVI(dynair.model, dynair.guide,
              optim.Adam(per_param_optim_args),
              loss='ELBO',
              trace_graph=False) # We don't have discrete choices.

    for i in range(num_epochs):

        for j, (X_batch, Y_batch) in enumerate(zip(X_train, Y_train)):
            loss = svi.step(X_batch, Y_batch)
            nan_params = list(dynair.params_with_nan())
            assert len(nan_params) == 0, 'The following parameters include NaN:\n  {}'.format("\n  ".join(nan_params))
            elbo = -loss / (dynair.seq_length * batch_size) # elbo per datum, per frame
            elapsed = timedelta(seconds=time.time()- t0)
            print('\33[2K\repoch={}, batch={}, elbo={:.2f}, elapsed={}'.format(i, j, elbo, elapsed), end='')
            append_line('{:.1f},{:.2f}'.format(elapsed.total_seconds(), elbo), os.path.join(output_path, 'elbo.csv'))
            if not vis_hook is None:
                vis_hook(X_vis, Y_vis, i, j)

        if (i+1) % 1000 == 0:
            torch.save(dynair.state_dict(),
                       os.path.join(output_path, 'params-{}.pytorch'.format(i+1)))


def load_data(data_path, use_cuda):
    print('loading {}'.format(data_path))
    data = np.load(data_path)
    X_np = data['X']
    # print(X_np.shape)
    X_np = X_np.astype(np.float32)
    X_np /= 255.0
    X = Variable(torch.from_numpy(X_np))
    # Drop the alpha channel.
    X = X[:,:,0:3]
    Y = torch.from_numpy(data['Y'].astype(np.uint8))
    assert X.size(0) == Y.size(0)
    if use_cuda:
        X = X.cuda()
        Y = Y.cuda()
    return X, Y

def frames_to_rgb_list(frames):
    return frames[:, 0:3].data.numpy().tolist()

def img_to_arr(img):
    assert img.mode == 'RGBA'
    channels = 4
    w, h = img.size
    arr = np.fromstring(img.tobytes(), dtype=np.uint8)
    return arr.reshape(w * h, channels).T.reshape(channels, h, w)

def draw_rect(size, color):
    img = Image.new('RGBA', (size, size))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, size - 1, size - 1], outline=color)
    return torch.from_numpy(img_to_arr(img).astype(np.float32) / 255.0)

def draw_window_outline(dynair, z_where, color):
    n = z_where.size(0)
    rect = draw_rect(dynair.window_size, color)
    if z_where.is_cuda:
        rect = rect.cuda()
    rect_batch = Variable(batch_expand(rect.contiguous().view(-1), n).contiguous())
    return dynair.window_to_image(z_where, rect_batch)

def overlay_window_outlines(dynair, frame, z_where, color):
    return over(draw_window_outline(dynair, z_where, color), frame)

def overlay_window_outlines_conditionally(dynair, frame, z_where, color, ii):
    batch_size = z_where.size(0)
    presence_mask = ii.view(-1, 1, 1, 1)
    borders = batch_expand(Variable(torch.Tensor([-0.08, 0, 0])), batch_size).type_as(ii)
    return over(draw_window_outline(dynair, borders, color) * presence_mask,
                over(draw_window_outline(dynair, z_where, color) * presence_mask,
                     frame))

def overlay_multiple_window_outlines(dynair, frame, ws, obj_count):
    acc = frame
    for i in range(obj_count):
        acc = overlay_window_outlines(dynair, acc, ws[i], ['red', 'green', 'blue'][i % 3])
    return acc

def frames_to_tensor(arr):
    # Turn an array of frames (of length seq_len) returned by the
    # model into a (batch, seq_len, rest...) tensor.
    return torch.cat([t.unsqueeze(0) for t in arr]).transpose(0, 1)

def latents_to_tensor(xss):
    return torch.stack([torch.stack(xs) for xs in xss]).transpose(2, 0)

def arr_to_img(nparr):
    assert nparr.shape[0] == 4 # required for RGBA
    shape = nparr.shape[1:]
    return Image.frombuffer('RGBA', shape, (nparr.transpose((1,2,0)) * 255).astype(np.uint8).tostring(), 'raw', 'RGBA', 0, 1)

def save_frames(frames, path_fmt_str, offset=0):
    n = frames.shape[0]
    assert_size(frames, (n, 4, frames.size(2), frames.size(3)))
    for i in range(n):
        arr_to_img(frames[i].data.numpy()).save(path_fmt_str.format(i + offset))

def mk_matrix(m, n):
    return [[None] * n for _ in range(m)]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('-b', '--batch-size', type=int, required=True, help='batch size')
    parser.add_argument('-e', '--epochs', type=int, default=10**6,
                        help='number of optimisation epochs to perform')
    parser.add_argument('--hold-out', type=int, default=0,
                        help='number of batches to hold out')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='visualise inferences during optimisation')
    parser.add_argument('--cuda', action='store_true', default=False, help='use CUDA')
    args = parser.parse_args()

    data = load_data(args.data_path, args.cuda)
    X, Y = data # (sequences, counts)
    X_split = split(X, args.batch_size, args.hold_out)
    Y_split = split(Y, args.batch_size, args.hold_out)
    print('data split: {}/{}'.format(len(X_split[0]), len(X_split[1])))

    dynair = DynAIR(use_cuda=args.cuda)
    run_svi(dynair, X_split, Y_split, args.epochs,
            partial(vis_hook, visdom.Visdom(), dynair) if args.vis else None)
