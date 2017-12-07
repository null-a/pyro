import torch
import torch.nn as nn
from torch.nn.functional import sigmoid, softplus


from torch.autograd import Variable

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

class SquishStateParams(nn.Module):
    def __init__(self, z_size):
        super(SquishStateParams, self).__init__()
        self.col_widths = 2*[z_size]

    def forward(self, x):
        cols = split_at(x, self.col_widths)
        z_mean = cols[0]
        z_sd = softplus(cols[1])
        return z_mean, z_sd

class DummyTransition(nn.Module):
    def __init__(self):
        super(DummyTransition, self).__init__()
        self.w = Variable(torch.Tensor([[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [1, 0, 1, 0],
                                        [0, 1, 0, 1]]))

    def forward(self, z_prev):
        z_mean = torch.mm(z_prev, self.w)
        z_sd = Variable(torch.ones([z_prev.size(0), 4]) * 0.01)
        return z_mean, z_sd

# Not used by dynair2.
class SquishObjParams(nn.Module):
    def __init__(self, z_pres_size, z_where_size, z_what_size):
        super(SquishObjParams, self).__init__()
        self.col_widths = [z_pres_size] + 2*[z_where_size] + 2*[z_what_size]

    def forward(self, x):
        cols = split_at(x, self.col_widths)
        z_pres_p = sigmoid(cols[0])
        z_where_mean = cols[1]
        z_where_sd = softplus(cols[2])
        z_what_mean = cols[3]
        z_what_sd = softplus(cols[4])
        return z_pres_p, z_where_mean, z_where_sd, z_what_mean, z_what_sd

# Split a matrix in a bunch of columns with specified widths.
def split_at(t, widths):
    assert t.dim() == 2
    assert t.size(1) == sum(widths)
    csum = torch.LongTensor(widths).cumsum(0).tolist()
    return [t[:,start:stop] for (start, stop) in zip([0] + csum, csum)]
