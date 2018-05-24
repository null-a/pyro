import torch
import torch.nn as nn
from torch.nn.functional import sigmoid, softplus
import pyro
import pyro.poutine as poutine
import pyro.distributions as dist

from cache import Cache, cached
from modules import MLP, NormalParams, NormalMeanWithSdParam
from utils import assert_size, delta_mean
from transform import window_to_image, over

class Model(nn.Module):
    def __init__(self, cfg, arch, delta_w=False, delta_z=False, use_cuda=False):
        super(Model, self).__init__()
        self.cache = Cache()
        self.prototype = torch.tensor(0.).cuda() if use_cuda else torch.tensor(0.)
        self.cfg = cfg

        # When enabled transitions output the delta from the previous
        # value of a variable, rather than outputting the next value
        # directly.
        self.delta_w = delta_w
        self.delta_z = delta_z

        # Priors:
        self.y_prior_mean = self.prototype.new_zeros(cfg.y_size)
        self.y_prior_sd = self.prototype.new_ones(cfg.y_size)

        self.w_0_prior_mean = torch.Tensor([3, 0, 0]).type_as(self.prototype)
        self.w_0_prior_sd = torch.Tensor([0.8, 0.7, 0.7]).type_as(self.prototype)

        self.z_0_prior_mean = self.prototype.new_zeros(cfg.z_size)
        self.z_0_prior_sd = self.prototype.new_ones(cfg.z_size)

        self.likelihood_sd = 0.3

        self.decode_obj = arch['decode_obj']
        self._decode_bkg = arch['decode_bkg']
        self.w_transition = arch['w_transition']
        self.z_transition = arch['z_transition']


    # We wrap `_decode_bkg` here to enable caching without
    # interferring with the magic behaviour that comes from having an
    # `nn.Module` as a property of the model class.
    @cached
    def decode_bkg(self, *args, **kwargs):
        return self._decode_bkg(*args, **kwargs)

    def forward(self, batch):

        pyro.module('model', self)
        # for name, _ in self.named_parameters():
        #     print(name)

        seqs, obj_counts = batch
        batch_size = obj_counts.size(0)

        if not seqs is None:
            assert_size(seqs, (batch_size,
                               self.cfg.seq_length, self.cfg.num_chan,
                               self.cfg.image_size, self.cfg.image_size))
            assert_size(obj_counts, (batch_size,))
        assert all(0 <= obj_counts) and all(obj_counts <= self.cfg.max_obj_count), 'Object count out of range.'

        zss = []
        wss = []
        frames = []

        # Supply batch size as recommended:
        # http://pyro.ai/examples/tensor_shapes.html
        with pyro.iarange('data', batch_size):

            y = self.sample_y(batch_size)
            bkg = self.decode_bkg(y)

            for t in range(self.cfg.seq_length):
                if t>0:
                    zs_prev, ws_prev = zss[-1], wss[-1]
                else:
                    zs_prev, ws_prev = None, None

                zs, ws = self.transition(t, obj_counts, zs_prev, ws_prev)
                frame_mean = self.emission(zs, ws, bkg, obj_counts)

                zss.append(zs)
                wss.append(ws)
                frames.append(frame_mean)

                obs = seqs[:,t] if not seqs is None else None
                self.likelihood(t, frame_mean, obs)

        return frames, wss, zss

    def likelihood(self, t, frame_mean, obs):
        frame_sd = (self.likelihood_sd * self.prototype.new_ones(1)).expand_as(frame_mean)
        # TODO: Using a normal here isn't very sensible since the data
        # is in [0, 1]. Do something better.
        pyro.sample('x_{}'.format(t),
                    dist.Normal(frame_mean, frame_sd).independent(3),
                    obs=obs)

    def sample_y(self, batch_size):
        return pyro.sample('y', dist.Normal(self.y_prior_mean.expand(batch_size, -1),
                                            self.y_prior_sd.expand(batch_size, -1))
                                .independent(1))

    def sample_w_0(self, i, batch_size):
        return pyro.sample('w_0_{}'.format(i),
                           dist.Normal(
                               self.w_0_prior_mean.expand(batch_size, -1),
                               self.w_0_prior_sd.expand(batch_size, -1))
                           .independent(1))

    def sample_w(self, t, i, w_prev, w_mean_or_delta, w_sd):
        w_mean = delta_mean(w_prev, w_mean_or_delta, self.delta_w)
        return pyro.sample('w_{}_{}'.format(t, i),
                           dist.Normal(w_mean, w_sd).independent(1))

    def sample_z_0(self, i, batch_size):
        return pyro.sample('z_0_{}'.format(i),
                           dist.Normal(
                               self.z_0_prior_mean.expand(batch_size, -1),
                               self.z_0_prior_sd.expand(batch_size, -1))
                           .independent(1))

    def sample_z(self, t, i, z_prev, z_mean_or_delta, z_sd):
        z_mean = delta_mean(z_prev, z_mean_or_delta, self.delta_z)
        return pyro.sample('z_{}_{}'.format(t, i),
                           dist.Normal(z_mean, z_sd).independent(1))

    def transition_one(self, t, i, z_prev, w_prev):
        batch_size = z_prev.size(0)
        assert_size(z_prev, (batch_size, self.cfg.z_size))
        assert_size(w_prev, (batch_size, self.cfg.w_size))
        z_mean_or_delta, z_sd = self.z_transition(z_prev)
        w_mean_or_delta, w_sd = self.w_transition(z_prev, w_prev)
        z = self.sample_z(t, i, z_prev, z_mean_or_delta, z_sd)
        w = self.sample_w(t, i, w_prev, w_mean_or_delta, w_sd)
        return z, w

    def transition(self, t, obj_counts, zs_prev, ws_prev):
        batch_size = obj_counts.size(0)
        assert_size(obj_counts, (batch_size,))

        if t==0:
            assert zs_prev is None and ws_prev is None
        else:
            assert t>0
            assert len(zs_prev) == self.cfg.max_obj_count # was zs[t-1]
            assert len(ws_prev) == self.cfg.max_obj_count # was ws[t-1]

        zs = []
        ws = []

        # To begin with, we'll sample max_obj_count objects for all
        # sequences, and throw out the extra objects. We can consider
        # refining this to avoid this unnecessary sampling later.

        for i in range(self.cfg.max_obj_count):

            mask = (obj_counts > i).float()

            with poutine.scale(None, mask):
                if t > 0:
                    z, w = self.transition_one(t, i, zs_prev[i], ws_prev[i])
                else:
                    z = self.sample_z_0(i, batch_size)
                    w = self.sample_w_0(i, batch_size)

            zs.append(z)
            ws.append(w)

        return zs, ws

    def emission(self, zs, ws, bkg, obj_counts):
        #batch_size = z.size(0)
        acc = bkg
        for i, (z, w) in enumerate(zip(zs, ws)):
            mask = tuple((obj_counts > i).tolist())
            acc = self.composite_object(z, w, mask, acc)
        return acc

    @cached
    def composite_object(self, z, w, mask, image_so_far):
        assert type(mask) == tuple # to facilitate caching on the mask
        assert z.size(0) == w.size(0) == image_so_far.size(0)
        mask = torch.Tensor(mask).type_as(z) # move to gpu
        x_att = self.decode_obj(z) * mask.view(-1, 1)
        return over(window_to_image(self.cfg, w, x_att), image_so_far)


class WTransition(nn.Module):
    def __init__(self, cfg, netfn, state_dependent_sd):
        super(WTransition, self).__init__()
        net = netfn(cfg.z_size + cfg.w_size)
        params_module = NormalParams if state_dependent_sd else NormalMeanWithSdParam
        params = params_module(net.output_size, cfg.w_size, sd_bias=-2.25)
        self.net = nn.Sequential(net, params)

    def forward(self, z_prev, w_prev):
        assert z_prev.size(0) == w_prev.size(0)
        return self.net(torch.cat((w_prev, z_prev), 1))


class ZTransition(nn.Module):
    def __init__(self, cfg, netfn, state_dependent_sd):
        super(ZTransition, self).__init__()
        net = netfn(cfg.z_size)
        params_module = NormalParams if state_dependent_sd else NormalMeanWithSdParam
        params = params_module(net.output_size, cfg.z_size, sd_bias=-2.25)
        self.net = nn.Sequential(net, params)

    def forward(self, z_prev):
        return self.net(z_prev)


class ZGatedTransition(nn.Module):
    def __init__(self, z_size, g_hid, h_hid):
        super(ZGatedTransition, self).__init__()
        self.g_mlp = nn.Sequential(MLP(z_size, [g_hid, z_size], nn.ReLU),
                                   nn.Sigmoid())
        self.h_mlp = MLP(z_size, [h_hid, z_size], nn.ReLU)
        self.mean_lm = nn.Linear(z_size, z_size)
        self.sd_lm = nn.Linear(z_size, z_size)
        nn.init.eye(self.mean_lm.weight)
        nn.init.constant(self.mean_lm.bias, 0)

    def forward(self, z_prev):
        g = self.g_mlp(z_prev)
        h = self.h_mlp(z_prev)
        z_mean = (1 - g) * self.mean_lm(z_prev) + g * h
        z_sd = softplus(self.sd_lm(relu(h)))
        return z_mean, z_sd


class DecodeObj(nn.Module):
    def __init__(self, cfg, netfn, alpha_bias=0.):
        super(DecodeObj, self).__init__()
        net = netfn(cfg.z_size)
        output_layer = nn.Linear(net.output_size, (cfg.num_chan+1) * cfg.window_size**2)
        # Adjust bias of the alpha channel.
        output_layer.bias.data[(cfg.num_chan * cfg.window_size ** 2):] += alpha_bias
        self.net = nn.Sequential(net, output_layer, nn.Sigmoid())

    def forward(self, z):
        return self.net(z)


class DecodeBkg(nn.Module):
    def __init__(self, cfg):
        super(DecodeBkg, self).__init__()
        self.num_chan = cfg.num_chan
        self.image_size = cfg.image_size
        self.mlp = MLP(cfg.y_size, [200, 200, cfg.num_chan * cfg.image_size**2], nn.ReLU)

    def forward(self, y):
        batch_size = y.size(0)
        out_flat = sigmoid(self.mlp(y))
        return out_flat.view(batch_size, self.num_chan, self.image_size, self.image_size)
