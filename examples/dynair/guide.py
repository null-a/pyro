import functools
import operator
import torch
import torch.nn as nn
from torch.nn.functional import softplus, relu
import pyro
import pyro.poutine as poutine
import pyro.distributions as dist
from cache import Cache, cached
from modules import MLP, ResNet, Flatten, split_at
from utils import assert_size, batch_expand, delta_mean
from transform import image_to_window

product = functools.partial(functools.reduce, operator.mul)

def mk_matrix(m, n):
    return [[None] * n for _ in range(m)]

class Guide(nn.Module):
    def __init__(self, cfg, arch, delta_w=False, delta_z=False, use_cuda=False):
        super(Guide, self).__init__()

        self.prototype = torch.tensor(0.).cuda() if use_cuda else torch.tensor(0.)
        self.cfg = cfg

        # When enabled the guide outputs the delta from the previous
        # value of the variable rather than outputting the next value
        # directly for steps t>0. This seems most natural when there
        # is a separate guide for t=0.
        self.delta_w = delta_w
        self.delta_z = delta_z

        # Modules

        # Guide modules:
        self.guide_w = arch['guide_w']
        self.guide_y = arch['guide_y']
        self.guide_z = arch['guide_z']

    def forward(self, batch):

        pyro.module('guide', self)
        # for name, _ in self.named_parameters():
        #     print(name)

        seqs, obj_counts = batch
        batch_size = seqs.size(0)

        assert_size(seqs, (batch_size,
                           self.cfg.seq_length, self.cfg.num_chan,
                           self.cfg.image_size, self.cfg.image_size))

        assert_size(obj_counts, (batch_size,))
        assert all(0 <= obj_counts) and all(obj_counts <= self.cfg.max_obj_count), 'Object count out of range.'

        ws = mk_matrix(self.cfg.seq_length, self.cfg.max_obj_count)
        zs = mk_matrix(self.cfg.seq_length, self.cfg.max_obj_count)

        with pyro.iarange('data', batch_size):

            # NOTE: Here we're guiding y based on the contents of the
            # first frame only.
            y = self.sample_y(*self.guide_y(seqs[:, 0]))

            for t in range(self.cfg.seq_length):

                x = seqs[:, t]
                w_guide_state_prev = None
                mask_prev = None

                # As in the model, we'll sample max_obj_count objects for
                # all sequences, and ignore the ones we don't need.

                for i in range(self.cfg.max_obj_count):

                    w_prev_i = ws[t-1][i] if t > 0 else None
                    z_prev_i = zs[t-1][i] if t > 0 else None
                    w_t_prev = ws[t][i-1] if i > 0 else None
                    z_t_prev = zs[t][i-1] if i > 0 else None

                    mask = (obj_counts > i).float()

                    with poutine.scale(None, mask):
                        w_params, w_guide_state = self.guide_w(t, i, x, y,
                                                               w_prev_i, z_prev_i,
                                                               w_t_prev, z_t_prev,
                                                               mask_prev, w_guide_state_prev)
                        w = self.sample_w(t, i, w_prev_i, *w_params)
                        x_att = image_to_window(self.cfg, w, x)
                        z = self.sample_z(t, i, z_prev_i, *self.guide_z(t, i, w, x_att, z_prev_i))

                    ws[t][i] = w
                    zs[t][i] = z

                    w_guide_state_prev = w_guide_state
                    mask_prev = mask

        return ws, zs, y

    def sample_y(self, y_mean, y_sd):
        return pyro.sample('y', dist.Normal(y_mean, y_sd).independent(1))

    def sample_w(self, t, i, w_prev, w_mean_or_delta, w_sd):
        #w_mean = delta_mean(w_prev, w_mean_or_delta, self.delta_w)
        # TODO: Figure out how to init. the guide such that we have
        # sensible window scales at the start of optimisation (e.g.
        # following the prior?), while also supporting optional delta
        # style.
        assert not self.delta_w
        w_mean = w_mean_or_delta + torch.tensor([3.0, 0, 0]).type_as(w_mean_or_delta)
        return pyro.sample('w_{}_{}'.format(t, i), dist.Normal(w_mean, w_sd).independent(1))

    def sample_z(self, t, i, z_prev, z_mean_or_delta, z_sd):
        z_mean = delta_mean(z_prev, z_mean_or_delta, self.delta_z)
        return pyro.sample('z_{}_{}'.format(t, i), dist.Normal(z_mean, z_sd).independent(1))


# TODO: Extend with CNN variants used for guide for w. (Are there
# opportunities for code re-use, since for both w and z compute
# parameters from main + side input. The different been that there is
# no recurrent part of the guide for z.)
class GuideZ(nn.Module):
    def __init__(self, cfg):
        super(GuideZ, self).__init__()

        x_att_size = cfg.num_chan * cfg.window_size**2 # patches cropped from the input
        x_att_embed_size = 100

        # TODO: Having only a single hidden layer between z_prev and z
        # may be insufficient?
        self.z_param = ParamZ(
            [100],
            cfg.w_size + x_att_embed_size + cfg.z_size, # input size
            cfg.z_size)

        self.z_init = nn.Parameter(torch.zeros(cfg.z_size))

        self.x_att_embed = MLP(x_att_size, [100, x_att_embed_size], nn.ReLU, True)


    def forward(self, t, i, w, x_att, z_prev_i):
        batch_size = w.size(0)
        x_att_flat = x_att.view(x_att.size(0), -1)
        x_att_embed = self.x_att_embed(x_att_flat)
        if t == 0:
            assert z_prev_i is None
            z_prev_i = batch_expand(self.z_init, batch_size)
        z_mean, z_sd = self.z_param(torch.cat((w, x_att_embed, z_prev_i), 1))
        return z_mean, z_sd


class ImgEmbedMlp(nn.Module):
    def __init__(self, in_size, hids):
        super(ImgEmbedMlp, self).__init__()
        assert len(hids) >= 1
        self.output_size = hids[-1]
        self.cache = Cache()
        self.net = MLP(in_size, hids, nn.ReLU, True)

    @cached
    def forward(self, img):
        batch_size = img.size(0)
        img_flat = img.view(batch_size, -1)
        return self.net(img_flat)


class ImgEmbedResNet(nn.Module):
    def __init__(self, in_size, hids):
        super(ImgEmbedResNet, self).__init__()
        assert len(hids) >= 1
        self.output_size = hids[-1]
        self.cache = Cache()
        self.net = ResNet(in_size, hids)

    @cached
    def forward(self, img):
        batch_size = img.size(0)
        img_flat = img.view(batch_size, -1)
        return self.net(img_flat)


class GuideW_ObjRnn(nn.Module):
    def __init__(self, cfg, rnn_hid_sizes, x_embed, rnn_cell_use_tanh):
        super(GuideW_ObjRnn, self).__init__()

        self.x_embed = x_embed

        self.w_param = ParamW(
            x_embed.output_size + 2 * cfg.w_size + 2 * cfg.z_size, # input size
            rnn_hid_sizes, [], cfg.w_size,
            rnn_cell_use_tanh=rnn_cell_use_tanh,
            sd_bias=-2.25)
        self.w_t_prev_init = nn.Parameter(torch.zeros(cfg.w_size))
        self.z_t_prev_init = nn.Parameter(torch.zeros(cfg.z_size))

        # TODO: Does it make sense that this is a parameter
        # (rather than fixed) in the case where the guide
        # computing the delta?
        self.w_init = nn.Parameter(torch.zeros(cfg.w_size))
        # TODO: Small init.
        self.z_init = nn.Parameter(torch.zeros(cfg.z_size))

    def forward(self, t, i, x, y, w_prev_i, z_prev_i, w_t_prev, z_t_prev, mask_prev, rnn_hid_prev):
        batch_size = x.size(0)

        x_embed = self.x_embed(x)

        if i == 0:
            assert w_t_prev is None and z_t_prev is None
            w_t_prev = batch_expand(self.w_t_prev_init, batch_size)
            z_t_prev = batch_expand(self.z_t_prev_init, batch_size)

        if t == 0:
            assert w_prev_i is None and z_prev_i is None
            w_prev_i = batch_expand(self.w_init, batch_size)
            z_prev_i = batch_expand(self.z_init, batch_size)

        # Could consider other ways of combining x with and the
        # ws/zs.
        rnn_input = torch.cat((x_embed, w_prev_i, z_prev_i, w_t_prev, z_t_prev), 1)
        w_mean, w_sd, rnn_hid = self.w_param(rnn_input, rnn_hid_prev)

        return (w_mean, w_sd), rnn_hid


class GuideW_ImageSoFar(nn.Module):
    def __init__(self, cfg, model):
        super(GuideW_ImageSoFar, self).__init__()

        self.w_param = ParamW_Isf_Mlp(cfg)
        #self.w_param = ParamW_Isf_Cnn_Mixin(cfg)
        #self.w_param = ParamW_Isf_Cnn_AM(cfg)

        # TODO: Does it make sense that this is a parameter (rather
        # than fixed) in the case where the guide computing the delta?
        # (Though this guide perhaps lends itself to computing
        # absolute position.)
        self.w_init = nn.Parameter(torch.zeros(cfg.w_size))
        self.z_init = nn.Parameter(torch.zeros(cfg.z_size))

        self.decode_bkg = model.decode_bkg
        self.composite_object = model.composite_object

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
            image_so_far = self.composite_object(z_t_prev, w_t_prev,
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
        w_mean, w_sd = self.w_param(diff, w_prev_i, z_prev_i)
        return (w_mean, w_sd), image_so_far


class ParamW_Isf_Mlp(nn.Module):
    def __init__(self, cfg):
        super(ParamW_Isf_Mlp, self).__init__()
        self.col_widths = [cfg.w_size, cfg.w_size]
        self.mlp = MLP(cfg.x_size + cfg.w_size + cfg.z_size,
                       [500, 200, 200, sum(self.col_widths)], nn.ReLU)

    def forward(self, img, w_prev, z_prev):
        batch_size = img.size(0)
        # TODO: Make a "Flatten" nn.Module.
        flat_img = img.view(batch_size, -1)
        out = self.mlp(torch.cat((flat_img, w_prev, z_prev), 1))
        cols = split_at(out, self.col_widths)
        w_mean = cols[0]
        w_sd = softplus(cols[1])
        return w_mean, w_sd


# TODO: Think more carefully about this architecture. Consider
# switching to inputs of a more convenient size.
class InputCnn(nn.Module):
    def __init__(self, cfg):
        super(InputCnn, self).__init__()
        assert cfg.image_size == 50
        self.output_size = 256 * 2 * 2
        self.cnn = nn.Sequential(
            nn.Conv2d(cfg.num_chan, 32, 4, stride=2, padding=0), # => 32 x 24 x 24
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # => 64 x 12 x 12
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # => 128 x 6 x 6
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=0), # => 256 x 2 x 2
            nn.ReLU(),
            Flatten()
        )

    def forward(self, img):
        return self.cnn(img)


class ParamW_Isf_Cnn_Mixin(nn.Module):
    def __init__(self, cfg):
        super(ParamW_Isf_Cnn_Mixin, self).__init__()

        self.col_widths = [cfg.w_size, cfg.w_size]

        self.cnn = InputCnn(cfg)

        self.mlp = MLP(self.cnn.output_size + cfg.w_size + cfg.z_size,
                       [200, 200, sum(self.col_widths)],
                       nn.ReLU)


    def forward(self, img, w_prev, z_prev):
        cnn_out = self.cnn(img)
        out = self.mlp(torch.cat((cnn_out, w_prev, z_prev), 1))
        cols = split_at(out, self.col_widths)
        w_mean = cols[0]
        w_sd = softplus(cols[1])
        return w_mean, w_sd


# "Activation Map" architecture.
# https://users.cs.duke.edu/~yilun/pdf/icra2017incorporating.pdf
class ParamW_Isf_Cnn_AM(nn.Module):
    def __init__(self, cfg):
        super(ParamW_Isf_Cnn_AM, self).__init__()

        self.num_chan = cfg.num_chan
        self.image_size = cfg.image_size

        self.col_widths = [cfg.w_size, cfg.w_size]

        # TODO: Could use transposed convolution here?
        # TODO: Check appropriateness of hidden sizes here.
        self.am_mlp = MLP(cfg.w_size + cfg.z_size, [200, 200, cfg.x_size], nn.ReLU)
        self.cnn = InputCnn(cfg)
        self.output_mlp = MLP(self.cnn.output_size, [sum(self.col_widths)], nn.ReLU)


    def forward(self, img, w_prev, z_prev):
        batch_size = img.size(0)
        am_flat = self.am_mlp(torch.cat((w_prev, z_prev), 1))
        am = am_flat.view(batch_size, self.num_chan, self.image_size, self.image_size)
        cnn_out = self.cnn(img * am)
        out = self.output_mlp(cnn_out)
        cols = split_at(out, self.col_widths)
        w_mean = cols[0]
        w_sd = softplus(cols[1])
        return w_mean, w_sd


class ParamW(nn.Module):
    def __init__(self, input_size, rnn_hid_sizes, hids, w_size, rnn_cell_use_tanh, sd_bias=0.0):
        super(ParamW, self).__init__()

        assert len(rnn_hid_sizes) > 0

        self.input_size = input_size

        rnn_input_sizes = [input_size] + rnn_hid_sizes[0:-1]

        nonlinearity = 'tanh' if rnn_cell_use_tanh else 'relu'
        self.rnns = nn.ModuleList(
            [nn.RNNCell(i, h, nonlinearity=nonlinearity) for i, h in zip(rnn_input_sizes, rnn_hid_sizes)])

        self.rnn_hid_inits = nn.ParameterList(
            [nn.Parameter(torch.zeros(h)) for h in rnn_hid_sizes])

        for rnn_hid_init in self.rnn_hid_inits:
            nn.init.normal_(rnn_hid_init, std=0.01)

        assert len(self.rnns) == len(self.rnn_hid_inits)

        self.col_widths = [w_size, w_size]
        self.mlp = MLP(rnn_hid_sizes[-1], hids + [sum(self.col_widths)], nn.ReLU)

        # Adjust the init. of the parameter MLP in an attempt to:

        # 1) Have the w delta output by the network be close to zero
        # at the start of optimisation. The motivation is that we want
        # minimise drift, in the hope that this helps prevent the
        # guide moving all of the windows out of frame during the
        # first few steps. (Because I assume this hurts optimisation.)

        # 2) Have the sd start out at around 0.1 for much the same
        # reason. Here we match the sd the initial sd used in the
        # model. (Is the latter sensible/helpful?)

        nn.init.normal_(self.mlp.seq[-1].weight, std=0.01)
        self.mlp.seq[-1].bias.data *= 0.0
        self.mlp.seq[-1].bias.data[w_size:] += sd_bias

    def forward(self, inp, rnn_hids_prev):
        batch_size = inp.size(0)
        assert inp.size(1) == self.input_size

        if rnn_hids_prev is None:
            hids_prev = [rnn_hid_init.expand(batch_size, -1)
                         for rnn_hid_init in self.rnn_hid_inits]
        else:
            hids_prev = rnn_hids_prev

        hids = self.apply_rnn_stack(hids_prev, inp)
        out = self.mlp(hids[-1])
        cols = split_at(out, self.col_widths)
        w_mean = cols[0]
        w_sd = softplus(cols[1])

        return w_mean, w_sd, hids

    def apply_rnn_stack(self, hids_prev, inp):
        assert len(hids_prev) == len(self.rnns)
        cur_inp = inp
        hids = []
        for rnn, hid_prev in zip(self.rnns, hids_prev):
            hid = rnn(cur_inp, hid_prev)
            hids.append(hid)
            cur_inp = hid
        return hids


class ParamZ(nn.Module):
    def __init__(self, hids, in_size, z_size):
        super(ParamZ, self).__init__()
        self.col_widths = [z_size, z_size]
        self.mlp = MLP(in_size, hids + [sum(self.col_widths)], nn.ReLU)

        nn.init.normal_(self.mlp.seq[-1].weight, std=0.01)
        self.mlp.seq[-1].bias.data *= 0.0
        self.mlp.seq[-1].bias.data[z_size:] -= 2.25

    def forward(self, inp):
        out = self.mlp(inp)
        cols = split_at(out, self.col_widths)
        z_mean = cols[0]
        z_sd = softplus(cols[1])
        return z_mean, z_sd


class ParamY(nn.Module):
    def __init__(self, cfg):
        super(ParamY, self).__init__()
        self.col_widths = [cfg.y_size, cfg.y_size]
        self.mlp = MLP(cfg.x_size, [200, 200, sum(self.col_widths)], nn.ReLU)

    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        out = self.mlp(x_flat)
        cols = split_at(out, self.col_widths)
        y_mean = cols[0]
        y_sd = softplus(cols[1])
        return y_mean, y_sd
