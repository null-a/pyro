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

class Transition(nn.Module):
    def __init__(self, z_size, w_size, z_hid_size, w_hid_size):
        super(Transition, self).__init__()

        # Means
        self.z_mean_net = MLP(z_size, [z_hid_size, z_size], nn.ReLU)
        self.w_mean_net = MLP(z_size + w_size, [w_hid_size, w_size], nn.ReLU)

        # SDs
        # Initialize to ~0.1 (after softplus).
        self._z_sd = nn.Parameter(torch.ones(z_size) * -2.25)
        self._w_sd = nn.Parameter(torch.ones(w_size) * -2.25)

    def forward(self, z_prev, w_prev):
        assert z_prev.size(0) == w_prev.size(0)
        batch_size = z_prev.size(0)

        wz_prev = torch.cat((w_prev, z_prev), 1)

        z_mean = self.z_mean_net(z_prev)
        w_mean = self.w_mean_net(wz_prev)

        z_sd = softplus(self._z_sd).expand_as(z_mean)
        w_sd = softplus(self._w_sd).expand_as(w_mean)

        return z_mean, z_sd, w_mean, w_sd


# TODO: What would make a good network arch. for this task. (i.e.
# Locating an object given a hint about where to look and what to look
# for.)
class ParamW(nn.Module):
    def __init__(self, hids, x_size, w_size, z_size):
        super(ParamW, self).__init__()
        in_size = x_size + w_size + z_size
        self.col_widths = [w_size, w_size]
        self.mlp = MLP(in_size, hids + [sum(self.col_widths)], nn.ReLU)

    def forward(self, x, w_prev, z_prev):
        x_flat = x.view(x.size(0), -1)
        out = self.mlp(torch.cat((x_flat, w_prev, z_prev), 1))
        cols = split_at(out, self.col_widths)
        w_mean = cols[0]
        w_sd = softplus(cols[1])
        return w_mean, w_sd

# TODO: Similarly, what should this look like. Re-visit DMM for
# inspiration?

# One thought is that we might compute a representation the window
# contents before trying to combine this with the previous state.

class ParamZ(nn.Module):
    def __init__(self, hids, w_size, x_att_size, z_size):
        super(ParamZ, self).__init__()
        in_size = w_size + x_att_size + z_size
        self.col_widths = [z_size, z_size]
        self.mlp = MLP(in_size, hids + [sum(self.col_widths)], nn.ReLU)

    def forward(self, w, x_att, z_prev):
        x_att_flat = x_att.view(x_att.size(0), -1)
        out = self.mlp(torch.cat((w, x_att_flat, z_prev), 1))
        cols = split_at(out, self.col_widths)
        z_mean = cols[0]
        z_sd = softplus(cols[1])
        return z_mean, z_sd
