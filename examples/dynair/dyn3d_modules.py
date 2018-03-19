import torch
import torch.nn as nn
from torch.nn.functional import sigmoid, softplus, tanh, relu


from torch.autograd import Variable

from pyro.util import ng_zeros, zeros

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

# Split a matrix in a bunch of columns with specified widths.
def split_at(t, widths):
    assert t.dim() == 2
    assert t.size(1) == sum(widths)
    csum = torch.LongTensor(widths).cumsum(0).tolist()
    return [t[:,start:stop] for (start, stop) in zip([0] + csum, csum)]



# Transition

# A simple non-linear transition. With learnable (constant) sd.

class WTransition(nn.Module):
    def __init__(self, z_size, w_size, hid_size):
        super(WTransition, self).__init__()
        self.w_mean_net = MLP(z_size + w_size, [hid_size, w_size], nn.ReLU)
        self._w_sd = nn.Parameter(torch.zeros(w_size))

    def forward(self, z_prev, w_prev):
        assert z_prev.size(0) == w_prev.size(0)
        wz_prev = torch.cat((w_prev, z_prev), 1)
        w_mean = w_prev + self.w_mean_net(wz_prev)
        w_sd = softplus(self._w_sd).expand_as(w_mean)
        return w_mean, w_sd

class ZTransition(nn.Module):
    def __init__(self, z_size, hid_size):
        super(ZTransition, self).__init__()
        self.z_mean_net = MLP(z_size, [hid_size, z_size], nn.ReLU)
        # Initialize to ~0.1 (after softplus).
        self._z_sd = nn.Parameter(torch.ones(z_size) * -2.25)

    def forward(self, z_prev):
        z_mean = self.z_mean_net(z_prev)
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


class ITransition(nn.Module):
    def __init__(self, i_size, w_size, z_size, hid_size):
        super(ITransition, self).__init__()
        in_size = w_size + z_size
        self.mlp = MLP(in_size, [hid_size, i_size], nn.ReLU)

    def forward(self, w_prev, z_prev):
        out = self.mlp(torch.cat((w_prev, z_prev), 1))
        return sigmoid(out)

# TODO: What would make a good network arch. for this task. (i.e.
# Locating an object given a hint about where to look and what to look
# for.)

# Perhaps we could reduce the resolution of x before putting it
# through the network. Or if that looses useful precision, maybe we
# could stack windows, having a first stage that uses a low-res input
# to position the first window, from which we determine the position
# more accurately.

class ParamIW(nn.Module):
    def __init__(self, embed_hids, hids, x_size, i_size, w_size, z_size):
        super(ParamIW, self).__init__()
        self.embed = MLP(x_size, embed_hids, nn.ReLU, True)
        self.col_widths = [i_size, w_size, w_size]

        # Prior MLP
        prior_in_size = embed_hids[-1]
        self.mlp_prior = MLP(prior_in_size, hids + [sum(self.col_widths)], nn.ReLU)

        # Object continuation MLP
        # cont_in_size = embed_hids[-1] + w_size + z_size
        # self.mlp_cont = MLP(cont_in_size, hids + [sum(self.col_widths)], nn.ReLU)

        # Dummy parameters. These won't ever move from zero.
        # self.w_init = nn.Parameter(torch.zeros(w_size))
        # self.z_init = nn.Parameter(torch.zeros(z_size))

    def forward(self, x, i_prev, w_prev, z_prev):
        batch_size = x.size(0)
        x_flat = x.contiguous().view(batch_size, -1)
        x_embed = self.embed(x_flat)

        # TODO: Here we apply the prior and continuation nets to all
        # data points in the batch. Try partitioning the batch and
        # applying conditionally.
        out_prior = self.mlp_prior(x_embed)
        # w_prev_arg = self.w_init.expand(batch_size, -1)
        # z_prev_arg = self.z_init.expand(batch_size, -1))
        #out_cont = self.mlp_cont(x_embed, 1)
        #out = _if(i_prev, out_cont, out_prior)

        cols = split_at(out_prior, self.col_widths)
        i_ps = sigmoid(cols[0])
        w_mean = cols[1]
        w_sd = softplus(cols[2])
        return i_ps, w_mean, w_sd

def _if(cond, cons, alt):
    return cond * cons + (1 - cond) * alt



# TODO: Similarly, what should this look like. Re-visit DMM for
# inspiration?

class ParamZ(nn.Module):
    def __init__(self, x_att_hids, hids, w_size, x_att_size, z_size):
        super(ParamZ, self).__init__()
        self.col_widths = [z_size, z_size]
        self.x_att_mlp = MLP(x_att_size, x_att_hids, nn.ReLU, True)
        in_size = w_size + x_att_hids[-1] + z_size
        self.mlp = MLP(in_size, hids + [sum(self.col_widths)], nn.ReLU)

    def forward(self, w, x_att, z_prev):
        x_att_flat = x_att.view(x_att.size(0), -1)
        x_att_h = self.x_att_mlp(x_att_flat)
        out = self.mlp(torch.cat((w, x_att_h, z_prev), 1))
        cols = split_at(out, self.col_widths)
        z_mean = cols[0]
        z_sd = softplus(cols[1])
        return z_mean, z_sd

class ParamY(nn.Module):
    def __init__(self, hids, x_size, y_size):
        super(ParamY, self).__init__()
        self.col_widths = [y_size, y_size]
        self.mlp = MLP(x_size, hids + [sum(self.col_widths)], nn.ReLU)

    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.contiguous().view(batch_size, -1)
        out = self.mlp(x_flat)
        cols = split_at(out, self.col_widths)
        y_mean = cols[0]
        y_sd = softplus(cols[1])
        return y_mean, y_sd



class Baseline(nn.Module):
    def __init__(self, seq_length):
        super(Baseline, self).__init__()
        self.seq_length = seq_length
        self.mlp = nn.Linear(seq_length, 1)

    def forward(self, cur_step):
        one_hot = Variable(torch.zeros(self.seq_length), requires_grad=False)
        one_hot[cur_step] = 1
        one_hot = one_hot.type_as(self.mlp.weight)
        return self.mlp(one_hot)

class DecodeObj(nn.Module):
    def __init__(self, hids, z_size, num_chan, window_size, alpha_bias=0.):
        super(DecodeObj, self).__init__()
        self.mlp = MLP(z_size, hids + [num_chan * window_size**2], nn.ReLU)
        self.mlp.seq[4].bias.data[(num_chan - 1) * window_size**2:] += alpha_bias

    def forward(self, z):
        return sigmoid(self.mlp(z))


class DecodeBkg(nn.Module):
    def __init__(self, hids, y_size, num_chan, image_size):
        super(DecodeBkg, self).__init__()
        self.num_chan = num_chan
        self.image_size = image_size
        self.mlp = MLP(y_size, hids + [(num_chan-1) * image_size**2], nn.ReLU)

    def forward(self, y):
        batch_size = y.size(0)
        out_flat = sigmoid(self.mlp(y))
        return out_flat.view(batch_size, self.num_chan-1, self.image_size, self.image_size)
