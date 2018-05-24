import torch
import torch.nn as nn
from torch.nn.functional import relu, softplus

from cache import Cache, cached

# A general purpose module to construct networks that look like:
# [Linear (256 -> 1)]
# [Linear (256 -> 256), ReLU (), Linear (256 -> 1)]
# [Linear (256 -> 256), ReLU (), Linear (256 -> 1), ReLU ()]
# etc.
class MLP(nn.Module):
    def __init__(self, in_size, hids, non_linear_layer, output_non_linearity=False):
        super(MLP, self).__init__()
        assert len(hids) > 0
        layers = []
        in_sizes = [in_size] + hids[0:-1]
        sizes = list(zip(in_sizes, hids))
        for (i, o) in sizes[0:-1]:
            layers.append(nn.Linear(i, o))
            layers.append(non_linear_layer())
        layers.append(nn.Linear(sizes[-1][0], sizes[-1][1]))
        if output_non_linearity:
            layers.append(non_linear_layer())
        self.seq = nn.Sequential(*layers)
        self.output_size = hids[-1]

    def forward(self, x):
        return self.seq(x)


# This implements the original formulation of ResNets.
class ResidualBlockFC(nn.Module):
    def __init__(self, size):
        super(ResidualBlockFC, self).__init__()
        in_size, hid_size, out_size = size
        self.l1 = nn.Linear(in_size, hid_size)
        self.l2 = nn.Linear(hid_size, out_size)
        if in_size != out_size:
            self.w = nn.Linear(in_size, out_size, bias=False)
        else:
            self.w = lambda x: x

    def forward(self, x):
        h = relu(self.l1(x))
        return relu(self.l2(h) + self.w(x))



class ResNet(nn.Module):
    def __init__(self, in_size, hids, block=ResidualBlockFC):
        super(ResNet, self).__init__()
        assert len(hids) > 0
        assert len(hids) % 2 == 0, 'number of hidden layers must be even'
        num_blocks = len(hids) // 2
        all_sizes = [in_size] + hids
        block_sizes = [tuple(all_sizes[(i*2):(i*2)+3]) for i in range(num_blocks)]
        blocks = [block(size) for size in block_sizes]
        self.seq = nn.Sequential(*blocks)
        self.output_size = hids[-1]

    def forward(self, x):
        return self.seq(x)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1)

# Takes a batch of vectors to a tuple containing a batch of mean and a
# batch of sd parameters.
class NormalParams(nn.Module):
    def __init__(self, in_size, param_size, sd_bias=0.0):
        super(NormalParams, self).__init__()
        self.param_size = param_size
        self.output_layer = nn.Linear(in_size, 2*param_size)

        # These notes below described the motivation for this init
        # when originally added to the guide for w. The same thing was
        # also subsequently used in the guide for z. I don't know
        # whether this is necessary with bkg model pre-training.

        # 1) Have the w delta output by the network be close to zero
        # at the start of optimisation. The motivation is that we want
        # minimise drift, in the hope that this helps prevent the
        # guide moving all of the windows out of frame during the
        # first few steps. (Because I assume this hurts optimisation.)

        # 2) Have the sd start out at around 0.1 for much the same
        # reason. Here we match the sd the initial sd used in the
        # model. (Is the latter sensible/helpful?)

        nn.init.normal_(self.output_layer.weight, std=0.01)
        self.output_layer.bias.data *= 0.0
        self.output_layer.bias.data[param_size:] += sd_bias

    def forward(self, x):
        out = self.output_layer(x)
        mean, pre_sd = split_at(out, (self.param_size, self.param_size))
        return mean, softplus(pre_sd)


class NormalMeanWithSdParam(nn.Module):
    def __init__(self, in_size, param_size, sd_bias=0.0):
        super(NormalMeanWithSdParam, self).__init__()
        self.output_layer = nn.Linear(in_size, param_size)
        self.pre_sd = nn.Parameter(torch.ones(param_size) * sd_bias)

    def forward(self, x):
        mean = self.output_layer(x)
        sd = softplus(self.pre_sd).expand_as(mean)
        return mean, sd


# Split a matrix in a bunch of columns with specified widths.
def split_at(t, widths):
    assert t.dim() == 2
    assert t.size(1) == sum(widths)
    csum = torch.LongTensor(widths).cumsum(0).tolist()
    return [t[:,start:stop] for (start, stop) in zip([0] + csum, csum)]


class Cached(nn.Module):
    def __init__(self, net):
        super(Cached, self).__init__()
        self.net = net
        if hasattr(self.net, 'output_size'):
            self.output_size = self.net.output_size
        self.cache = Cache()

    @cached
    def forward(self, x):
        return self.net(x)
