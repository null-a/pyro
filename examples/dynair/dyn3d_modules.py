import torch
import torch.nn as nn
from torch.nn.functional import sigmoid, softplus, tanh, relu

import functools
import operator

product = functools.partial(functools.reduce, operator.mul)

# A general purpose module to construct networks that look like:
# [Linear (256 -> 1)]
# [Linear (256 -> 256), ReLU (), Linear (256 -> 1)]
# [Linear (256 -> 256), ReLU (), Linear (256 -> 1), ReLU ()]
# etc.
class MLP(nn.Module):
    def __init__(self, in_size, out_sizes, non_linear_layer, output_non_linearity=False):
        super(MLP, self).__init__()
        assert len(out_sizes) >= 1
        layers = []
        in_sizes = [in_size] + out_sizes[0:-1]
        sizes = list(zip(in_sizes, out_sizes))
        for (i, o) in sizes[0:-1]:
            layers.append(nn.Linear(i, o))
            layers.append(non_linear_layer())
        layers.append(nn.Linear(sizes[-1][0], sizes[-1][1]))
        if output_non_linearity:
            layers.append(non_linear_layer())
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


# TODO: Make this into a nn.Module to allow more idiomatic PyTorch
# usage. With this (and Flatten) a lot of code in `forward` methods
# can be replaced with the use of `nn.Sequential`. If this doesn't
# work out, have this return a tuple for more concise usage.

# Split a matrix in a bunch of columns with specified widths.
def split_at(t, widths):
    assert t.dim() == 2
    assert t.size(1) == sum(widths)
    csum = torch.LongTensor(widths).cumsum(0).tolist()
    return [t[:,start:stop] for (start, stop) in zip([0] + csum, csum)]



# Transition

# A simple non-linear transition. With learnable (constant) sd.

class WTransition(nn.Module):
    def __init__(self, cfg, hid_size):
        super(WTransition, self).__init__()
        self.w_mean_net = MLP(cfg.z_size + cfg.w_size, [hid_size, cfg.w_size], nn.ReLU)
        # Initialize to ~0.1 (after softplus).
        self._w_sd = nn.Parameter(torch.ones(cfg.w_size) * -2.25)

    def forward(self, z_prev, w_prev):
        assert z_prev.size(0) == w_prev.size(0)
        wz_prev = torch.cat((w_prev, z_prev), 1)
        w_mean = w_prev + self.w_mean_net(wz_prev)
        w_sd = softplus(self._w_sd).expand_as(w_mean)
        return w_mean, w_sd

class ZTransition(nn.Module):
    def __init__(self, cfg, hid_size):
        super(ZTransition, self).__init__()
        self.z_mean_net = MLP(cfg.z_size, [hid_size, cfg.z_size], nn.ReLU)
        # Initialize to ~0.1 (after softplus).
        self._z_sd = nn.Parameter(torch.ones(cfg.z_size) * -2.25)

    def forward(self, z_prev):
        z_mean = z_prev + self.z_mean_net(z_prev)
        z_sd = softplus(self._z_sd).expand_as(z_mean)
        return z_mean, z_sd

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
        self.output_size = (256, 2, 2)
        self.cnn = nn.Sequential(
            nn.Conv2d(cfg.num_chan, 32, 4, stride=2, padding=0), # => 32 x 24 x 24
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # => 64 x 12 x 12
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # => 128 x 6 x 6
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=0), # => 256 x 2 x 2
            nn.ReLU(),
        )

    def forward(self, img):
        out = self.cnn(img)
        assert(out.size()[1:] == self.output_size)
        return out


class ParamW_Isf_Cnn_Mixin(nn.Module):
    def __init__(self, cfg):
        super(ParamW_Isf_Cnn_Mixin, self).__init__()

        self.col_widths = [cfg.w_size, cfg.w_size]

        self.cnn = InputCnn(cfg)

        self.mlp = MLP(product(self.cnn.output_size) + cfg.w_size + cfg.z_size,
                       [200, 200, sum(self.col_widths)],
                       nn.ReLU)


    def forward(self, img, w_prev, z_prev):
        batch_size = img.size(0)
        cnn_out_flat = self.cnn(img).view(batch_size, -1)
        out = self.mlp(torch.cat((cnn_out_flat, w_prev, z_prev), 1))
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
        self.output_mlp = MLP(product(self.cnn.output_size),
                              [sum(self.col_widths)], nn.ReLU)


    def forward(self, img, w_prev, z_prev):
        batch_size = img.size(0)
        am_flat = self.am_mlp(torch.cat((w_prev, z_prev), 1))
        am = am_flat.view(batch_size, self.num_chan, self.image_size, self.image_size)
        cnn_out_flat = self.cnn(img * am).view(batch_size, -1)
        out = self.output_mlp(cnn_out_flat)
        cols = split_at(out, self.col_widths)
        w_mean = cols[0]
        w_sd = softplus(cols[1])
        return w_mean, w_sd


class ParamW(nn.Module):
    def __init__(self, input_size, rnn_hid_sizes, hids, w_size, z_size, sd_bias=0.0):
        super(ParamW, self).__init__()

        assert len(rnn_hid_sizes) > 0

        self.input_size = input_size
        self.w_size = w_size
        self.z_size = z_size

        rnn_input_sizes = [input_size + w_size + z_size] + rnn_hid_sizes[0:-1]

        self.rnns = nn.ModuleList(
            [nn.RNNCell(i, h) for i, h in zip(rnn_input_sizes, rnn_hid_sizes)])

        self.rnn_hid_inits = nn.ParameterList(
            [nn.Parameter(torch.zeros(h)) for h in rnn_hid_sizes])

        for rnn_hid_init in self.rnn_hid_inits:
            nn.init.normal_(rnn_hid_init, std=0.01)

        assert len(self.rnns) == len(self.rnn_hid_inits)

        self.col_widths = [w_size, w_size]
        self.mlp = MLP(rnn_hid_sizes[-1], hids + [sum(self.col_widths)], nn.ReLU)

        self.w_t_prev_init = nn.Parameter(torch.zeros(w_size))
        self.z_t_prev_init = nn.Parameter(torch.zeros(z_size))

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

    def forward(self, inp, w_t_prev, z_t_prev, rnn_hids_prev):
        batch_size = inp.size(0)
        assert inp.size(1) == self.input_size

        if rnn_hids_prev is None:
            hids_prev = [rnn_hid_init.expand(batch_size, -1)
                         for rnn_hid_init in self.rnn_hid_inits]
        else:
            hids_prev = rnn_hids_prev

        if w_t_prev is None:
            w_t_prev = self.w_t_prev_init.expand(batch_size, self.w_size)
        if z_t_prev is None:
            z_t_prev = self.z_t_prev_init.expand(batch_size, self.z_size)

        rnn_input = torch.cat((inp, w_t_prev, z_t_prev), 1)
        hids = self.apply_rnn_stack(hids_prev, rnn_input)
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


class DecodeObj(nn.Module):
    def __init__(self, cfg, hids, alpha_bias=0.):
        super(DecodeObj, self).__init__()
        self.mlp = MLP(cfg.z_size, hids + [(cfg.num_chan+1) * cfg.window_size**2], nn.ReLU)
        # Adjust bias of the alpha channel.
        self.mlp.seq[-1].bias.data[(cfg.num_chan * cfg.window_size ** 2):] += alpha_bias

    def forward(self, z):
        return sigmoid(self.mlp(z))


class DecodeBkg(nn.Module):
    def __init__(self, cfg, hids):
        super(DecodeBkg, self).__init__()
        self.num_chan = cfg.num_chan
        self.image_size = cfg.image_size
        self.mlp = MLP(cfg.y_size, hids + [cfg.num_chan * cfg.image_size**2], nn.ReLU)

    def forward(self, y):
        batch_size = y.size(0)
        out_flat = sigmoid(self.mlp(y))
        return out_flat.view(batch_size, self.num_chan, self.image_size, self.image_size)
