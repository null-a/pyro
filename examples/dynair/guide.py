import functools
import operator
import torch
import torch.nn as nn
from torch.nn.functional import softplus, relu
import pyro
import pyro.poutine as poutine
import pyro.distributions as dist
from cache import Cache, cached
from modules import MLP, ResNet, Flatten, NormalParams, Cached, split_at
from utils import assert_size, batch_expand, delta_mean
from transform import image_to_window, append_channel

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
        seq_length = seqs.size(1)

        assert_size(seqs, (batch_size, seq_length, self.cfg.num_chan,
                           self.cfg.image_size, self.cfg.image_size))

        assert_size(obj_counts, (batch_size,))
        assert all(0 <= obj_counts) and all(obj_counts <= self.cfg.max_obj_count), 'Object count out of range.'

        ws = mk_matrix(seq_length, self.cfg.max_obj_count)
        zs = mk_matrix(seq_length, self.cfg.max_obj_count)

        with pyro.iarange('data', batch_size):

            # NOTE: Here we're guiding y based on the contents of the
            # first frame only.
            y = self.sample_y(*self.guide_y(seqs[:, 0])) if self.cfg.use_bkg_model else None

            for t in range(seq_length):

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
                        w_params, w_guide_state, aux = self.guide_w(t, i, x, y,
                                                                    w_prev_i, z_prev_i,
                                                                    w_t_prev, z_t_prev,
                                                                    mask_prev, w_guide_state_prev)
                        w = self.sample_w(t, i, w_prev_i, *w_params)
                        x_att = image_to_window(self.cfg, w[:,0:3], x)
                        z = self.sample_z(t, i, z_prev_i, *self.guide_z(t, i, w, x_att, z_prev_i, aux))

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


class GuideZ(nn.Module):
    def __init__(self, cfg, combine_module, aux_size, aux_method):
        super(GuideZ, self).__init__()

        self.aux_method = aux_method
        assert type(aux_size) == tuple
        assert aux_method in ['ignore', 'main', 'side']

        if aux_method == 'main':
            # We allow the aux input to have extra channels. See
            # `precombine` for how this is handled.
            assert (aux_size[0] >= cfg.num_chan and
                    aux_size[1:] == (cfg.window_size, cfg.window_size)), 'aux/main input size mismatch'

        main_input_size = (aux_size[0] if aux_method == 'main' else cfg.num_chan,
                           cfg.window_size,
                           cfg.window_size)
        side_input_size = (cfg.w_size + cfg.z_size +
                           (product(aux_size) if aux_method == 'side' else 0))


        self.combine = combine_module(main_input_size, side_input_size)

        self.params = NormalParams(self.combine.output_size, cfg.z_size, sd_bias=-2.25)

        self.z_init = nn.Parameter(torch.zeros(cfg.z_size))

    def precombine(self, x_att, w, z_prev_i, aux):
        if self.aux_method == 'ignore':
            return x_att, torch.cat((w, z_prev_i), 1)
        elif self.aux_method == 'main':
            aux_input = aux(w)
            x_att_num_chan = x_att.size(1)
            aux_num_chan = aux_input.size(1)
            # When the aux input has more channels than the main input
            # (e.g. when it contains the depth channel), the main
            # input will (in effect) be padding with additional
            # channels set to zero everywhere before the diff is
            # computed.
            if aux_num_chan > x_att_num_chan:
                diff = x_att - aux_input[:,0:x_att_num_chan]
                extra_aux_chans = aux_input[:,x_att_num_chan:]
                main_input = torch.cat((diff, extra_aux_chans), 1)
            else:
                main_input = x_att - aux_input
            return main_input, torch.cat((w, z_prev_i), 1)
        elif self.aux_method == 'side':
            batch_size = x_att.size(0)
            aux_flat = aux(w).view(batch_size, -1)
            return x_att, torch.cat((w, z_prev_i, aux_flat), 1)
        else:
            raise Exception('impossible')

    def forward(self, t, i, w, x_att, z_prev_i, aux):
        batch_size = w.size(0)
        if t == 0:
            assert z_prev_i is None
            z_prev_i = batch_expand(self.z_init, batch_size)

        main, side = self.precombine(x_att, w, z_prev_i, aux)
        return self.params(self.combine(main, side))


class ImgEmbedMlp(nn.Module):
    def __init__(self, in_size, hids):
        super(ImgEmbedMlp, self).__init__()
        assert len(in_size) == 3
        mlp = MLP(product(in_size), hids)
        self.net = nn.Sequential(Flatten(), mlp)
        self.output_size = mlp.output_size

    def forward(self, img):
        return self.net(img)


class ImgEmbedResNet(nn.Module):
    def __init__(self, in_size, hids):
        super(ImgEmbedResNet, self).__init__()
        assert len(in_size) == 3
        resnet = ResNet(product(in_size), hids)
        self.net = nn.Sequential(Flatten(), resnet)
        self.output_size = resnet.output_size

    def forward(self, img):
        return self.net(img)


# Identity image embed net.
class ImgEmbedId(nn.Module):
    def __init__(self, in_size):
        super(ImgEmbedId, self).__init__()
        assert len(in_size) == 3
        self.output_size = product(in_size)
        self.flatten = Flatten()

    def forward(self, x):
        return self.flatten(x)


class GuideW_ObjRnn(nn.Module):
    def __init__(self, cfg, rnn_hid_sizes, x_embed_module, rnn_cell_use_tanh):
        super(GuideW_ObjRnn, self).__init__()

        assert len(rnn_hid_sizes) >= 1

        # Cache the output of the embed net to avoid computing the
        # same embedding at each object step.
        self.x_embed = Cached(x_embed_module((cfg.num_chan, cfg.image_size, cfg.image_size)))

        self.rnn_stack = RnnStack(
            self.x_embed.output_size + 2 * cfg.w_size + 2 * cfg.z_size, # input size
            rnn_hid_sizes,
            use_tanh=rnn_cell_use_tanh)

        self.params = NormalParams(rnn_hid_sizes[-1], cfg.w_size, sd_bias=-2.25)

        self.aux_size = (rnn_hid_sizes[-1],)

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

        # Could use CombineMixin here (with Identity as output), and
        # then parameterize combine network to allow other ways of
        # combining the input with w and z.
        rnn_input = torch.cat((x_embed, w_prev_i, z_prev_i, w_t_prev, z_t_prev), 1)
        rnn_hid = self.rnn_stack(rnn_input, rnn_hid_prev)

        return self.params(rnn_hid[-1]), rnn_hid, lambda _: rnn_hid[-1]


class GuideW_ImageSoFar(nn.Module):
    def __init__(self, cfg, model, combine_module, block_grad, include_bkg):
        super(GuideW_ImageSoFar, self).__init__()

        self.block_grad = block_grad
        self.include_bkg = include_bkg

        self.combine = combine_module((cfg.num_chan, cfg.image_size, cfg.image_size), # image so far diff
                                      cfg.w_size + cfg.z_size)                        # side input

        self.params = NormalParams(self.combine.output_size, cfg.w_size, sd_bias=-2.25)

        # Size of the window-so-far.
        self.aux_size = (cfg.num_chan + (1 if cfg.use_depth else 0),
                         cfg.window_size, cfg.window_size)

        # TODO: Does it make sense that this is a parameter (rather
        # than fixed) in the case where the guide computing the delta?
        # (Though this guide perhaps lends itself to computing
        # absolute position.)
        self.w_init = nn.Parameter(torch.zeros(cfg.w_size))
        self.z_init = nn.Parameter(torch.zeros(cfg.z_size))

        self.cfg = cfg
        self.decode_bkg = model.decode_bkg
        self.composite_object = model.composite_object

    def forward(self, t, i, x, y, w_prev_i, z_prev_i, w_t_prev, z_t_prev, mask_prev, image_so_far_prev):
        batch_size = x.size(0)

        if i == 0:
            assert image_so_far_prev is None
            assert mask_prev is None
            image_so_far = self.decode_bkg(y) if (self.cfg.use_bkg_model and self.include_bkg) else torch.zeros_like(x)
            if self.cfg.use_depth:
                # Add depth channel.
                image_so_far = append_channel(image_so_far)

        else:
            assert image_so_far_prev is not None
            assert w_t_prev is not None
            assert z_t_prev is not None
            # Add previous object to image so far.
            image_so_far = self.composite_object(z_t_prev, w_t_prev,
                                                 tuple(mask_prev.tolist()),
                                                 image_so_far_prev)

        # TODO: Should we be preventing gradients from flowing back
        # from the guide to the model, through image_so_far?
        if self.block_grad:
            image_so_far = image_so_far.detach()

        if t == 0:
            assert w_prev_i is None and z_prev_i is None
            w_prev_i = batch_expand(self.w_init, batch_size)
            z_prev_i = batch_expand(self.z_init, batch_size)

        # For simplicity, feed `x - image_so_far` into the net, though
        # note that alternatives exist.
        diff = x - image_so_far[:,0:3]

        params = self.params(self.combine(diff, torch.cat((w_prev_i, z_prev_i), 1)))

        # Return (a function that computes) the window-so-far for use
        # as auxiliary input to z guide.
        aux = lambda w: image_to_window(self.cfg, w, image_so_far)

        return params, image_so_far, aux


# TODO: Think more carefully about these CNN architectures. Consider
# switching to inputs of a more convenient size.

class InputCnn(nn.Module):
    def __init__(self, in_size):
        super(InputCnn, self).__init__()
        num_chan, w, h = in_size
        assert w == 50 and h == 50, 'input cnn requires an input size of 50'
        self.output_size = 256 * 2 * 2
        self.cnn = nn.Sequential(
            nn.Conv2d(num_chan, 32, 4, stride=2, padding=0), # => 32 x 24 x 24
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


class WindowCnn(nn.Module):
    def __init__(self, in_size):
        super(WindowCnn, self).__init__()
        num_chan, w, h = in_size
        assert w == 24 and h == 24, 'window cnn requires a window size of 24'
        self.output_size = 128 * 3 * 3
        self.cnn = nn.Sequential(
            nn.Conv2d(num_chan, 32, 4, stride=2, padding=1), # => 32 x 12 x 12
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # => 64 x 6 x 6
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # => 128 x 3 x 3
            nn.ReLU(),
            Flatten()
        )

    def forward(self, img):
        return self.cnn(img)


class RnnStack(nn.Module):
    def __init__(self, input_size, hid_sizes, use_tanh):
        super(RnnStack, self).__init__()

        assert len(hid_sizes) > 0
        self.input_size = input_size
        input_sizes = [input_size] + hid_sizes[0:-1]

        nonlinearity = 'tanh' if use_tanh else 'relu'
        self.rnns = nn.ModuleList(
            [nn.RNNCell(i, h, nonlinearity=nonlinearity) for i, h in zip(input_sizes, hid_sizes)])

        self.hid_inits = nn.ParameterList(
            [nn.Parameter(torch.zeros(h)) for h in hid_sizes])

        for hid_init in self.hid_inits:
            nn.init.normal_(hid_init, std=0.01)

        assert len(self.rnns) == len(self.hid_inits)

    def forward(self, inp, hids_prev):
        batch_size = inp.size(0)
        assert inp.size(1) == self.input_size

        if hids_prev is None:
            hids_prev = [hid_init.expand(batch_size, -1)
                         for hid_init in self.hid_inits]
        assert len(hids_prev) == len(self.rnns)

        cur_inp = inp
        hids = []
        for rnn, hid_prev in zip(self.rnns, hids_prev):
            hid = rnn(cur_inp, hid_prev)
            hids.append(hid)
            cur_inp = hid
        return hids


class CombineMixin(nn.Module):
    def __init__(self, embed_module, output_module,
                 main_size, side_size):
        super(CombineMixin, self).__init__()
        self.embed_net = embed_module(main_size)
        self.output_net = output_module(self.embed_net.output_size + side_size)
        self.output_size = self.output_net.output_size

    def forward(self, main, side):
        main_embedding = self.embed_net(main)
        return self.output_net(torch.cat((main_embedding, side), 1))

# TODO: Add combine net for the "Activation Map" architecture.
# https://users.cs.duke.edu/~yilun/pdf/icra2017incorporating.pdf


class GuideY(nn.Module):
    def __init__(self, cfg):
        super(GuideY, self).__init__()
        self.col_widths = [cfg.y_size, cfg.y_size]
        self.mlp = MLP(cfg.x_size, [200, 200, sum(self.col_widths)], output_non_linearity=False)

    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        out = self.mlp(x_flat)
        cols = split_at(out, self.col_widths)
        y_mean = cols[0]
        y_sd = softplus(cols[1])
        return y_mean, y_sd


# TODO: Reinstate this simpler implementation. (Also drop split_at import.)
# Requires reoptimizing bkg model params or modifying the existing
# params so that the names align.

# class GuideY(nn.Module):
#     def __init__(self, cfg):
#         super(GuideY, self).__init__()
#         mlp = MLP(cfg.x_size, [200, 200])
#         params = NormalParams(mlp.output_size, cfg.y_size)
#         self.net = nn.Sequential(Flatten(), mlp, params)

#     def forward(self, x):
#         return self.net(x)
