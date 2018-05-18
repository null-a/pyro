import torch
import torch.nn as nn
from torch.nn.functional import relu

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
        assert len(hids) % 2 == 0, 'number of hidden layers must be even'
        num_blocks = len(hids) // 2
        all_sizes = [in_size] + hids
        block_sizes = [tuple(all_sizes[(i*2):(i*2)+3]) for i in range(num_blocks)]
        blocks = [block(size) for size in block_sizes]
        self.seq = nn.Sequential(*blocks)

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
