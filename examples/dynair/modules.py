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

class SquishStateParams(nn.Module):
    def __init__(self, z_size):
        super(SquishStateParams, self).__init__()
        self.col_widths = 2*[z_size]

    def forward(self, x):
        cols = split_at(x, self.col_widths)
        z_mean = cols[0]
        z_sd = softplus(cols[1])
        return z_mean, z_sd

class FixedTransition(nn.Module):
    def __init__(self):
        super(FixedTransition, self).__init__()
        self.w = Variable(torch.Tensor([[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [1, 0, 1, 0],
                                        [0, 1, 0, 1]]))

    def forward(self, z_prev):
        z_mean = torch.mm(z_prev, self.w)
        z_sd = Variable(torch.ones([z_prev.size(0), 4]) * 0.01)
        return z_mean, z_sd

class LinearTransition(nn.Module):
    def __init__(self, z_size):
        super(LinearTransition, self).__init__()
        self.lin = nn.Linear(z_size, z_size, bias=False)
        nn.init.normal(self.lin.weight, std=0.01)
        self.lin.weight.data += torch.eye(z_size)
        # nn.init.eye(self.lin.weight)
        # self.lin.weight.data = torch.Tensor([[1, 0, 1, 0],
        #                                      [0, 1, 0, 1],
        #                                      [0, 0, 1, 0],
        #                                      [0, 0, 0, 1]])


    def forward(self, z):
        return self.lin(z)

# Linear transition + fixed Gaussian noise.
class BasicTransition(nn.Module):
    def __init__(self, z_size, z_sd):
        super(BasicTransition, self).__init__()
        self.lin = nn.Linear(z_size, z_size, bias=False)
        nn.init.normal(self.lin.weight, std=0.1)
        self.lin.weight.data += torch.eye(z_size)
        self.z_sd = z_sd

    def forward(self, z):
        z_mean = self.lin(z)
        return z_mean, self.z_sd.expand_as(z_mean)

# Either a linear or a simple non-linear transition. With (optionally
# learnable) constant sd.
class BasicTransition2(nn.Module):
    def __init__(self, z_size, linear, opt_sd, use_cuda):
        super(BasicTransition2, self).__init__()
        if linear:
            # linear transition
            self.tr = nn.Linear(z_size, z_size, bias=False)
            nn.init.normal(self.tr.weight, std=0.1)
            self.tr.weight.data += torch.eye(z_size)
        else:
            self.tr = MLP(z_size, [50, z_size], nn.ReLU)
        # sd
        # Init. to about 0.1 (after softplus)
        sd0 = torch.ones(z_size) * -2.25
        if opt_sd:
            self._sd = nn.Parameter(sd0)
        else:
            self._sd = Variable(sd0, requires_grad=False)

        if use_cuda:
            self._sd = self._sd.cuda()

    def sd(self):
        return softplus(self._sd)

    def forward(self, z_prev):
        z_mean = self.tr(z_prev)
        z_sd = self.sd().expand_as(z_mean)
        return z_mean, z_sd


# The DMM gated transition function.
class TransitionDMM(nn.Module):
    def __init__(self, z_size):
        super(TransitionDMM, self).__init__()
        # Could vary MLP hidden sizes if desired.
        self.mlp_g = MLP(z_size, [z_size, z_size], nn.ReLU)
        self.mlp_h = MLP(z_size, [z_size, z_size], nn.ReLU)
        self.lin_mean = nn.Linear(z_size, z_size)
        self.lin_sd = nn.Linear(z_size, z_size)
        nn.init.eye(self.lin_mean.weight)
        nn.init.constant(self.lin_mean.bias, 0)

    def forward(self, z_prev):
        g = sigmoid(self.mlp_g(z_prev))
        h = self.mlp_h(z_prev)
        z_mean = (1 - g) * self.lin_mean(z_prev) + g * h
        z_sd = softplus(self.lin_sd(relu(h)))
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

# dynair3

# Simple combine function, mostly for testing.
class Combine(nn.Module):
    def __init__(self, input_rnn_hid_size, z_size):
        super(Combine, self).__init__()
        self.mlp = MLP(input_rnn_hid_size + z_size, [z_size * 2], nn.ReLU)
        self.col_sizes = [z_size, z_size]

    def forward(self, input_rnn_hid, z_prev):
        x = self.mlp(torch.cat((input_rnn_hid, z_prev), 1))
        cols = split_at(x, self.col_sizes)
        z_mean = cols[0]
        z_sd = softplus(cols[1])
        return z_mean, z_sd

# The DMM combiner function.
class CombineDMM(nn.Module):
    def __init__(self, input_rnn_hid_size, z_size):
        super(CombineDMM, self).__init__()
        self.l_h = nn.Linear(z_size, input_rnn_hid_size)
        self.l_mean = nn.Linear(input_rnn_hid_size, z_size)
        self.l_sd = nn.Linear(input_rnn_hid_size, z_size)

    def forward(self, input_rnn_hid, z_prev):
        h = 0.5 * (tanh(self.l_h(z_prev)) + input_rnn_hid)
        z_mean = self.l_mean(h)
        z_sd = softplus(self.l_sd(h))
        return z_mean, z_sd

class InputRNN(nn.Module):
    def __init__(self, image_embed_size, hid_size):
        super(InputRNN, self).__init__()
        self.rnn = nn.LSTM(image_embed_size, hid_size, batch_first=True)
        self.h0 = zeros(hid_size)
        self.c0 = zeros(hid_size)
        nn.init.normal(self.h0)
        nn.init.normal(self.c0)
        self.hid_size = hid_size

    def forward(self, seq):
        batch_size = seq.size(0)
        # CUDNN complains if h0, c0 are not contiguous.
        h0 = self.h0.expand(1, batch_size, self.hid_size).contiguous()
        c0 = self.c0.expand(1, batch_size, self.hid_size).contiguous()
        hid_seq, _, = self.rnn(seq, (h0, c0))
        return hid_seq

# I don't have a good sense about what net arch. is sensible here.
# Could experiment with reversing the seq., extra RNN layers (since
# the non-recurrent encoder in AIR has hidden layers), or something
# bi-directional even?
class EncodeRNN(nn.Module):
    def __init__(self, window_size, num_chan, hid_size, z_what_size):
        super(EncodeRNN, self).__init__()
        self.hid_size = hid_size
        self.rnn = nn.GRU(num_chan * window_size**2, hid_size)
        self.h0 = zeros(hid_size)
        nn.init.normal(self.h0)
        self.mlp = MLP(hid_size, [z_what_size * 2], nn.ReLU)
        self.col_widths = [z_what_size, z_what_size]

    def forward(self, seq):
        # seq.size(0) is sequence length for this RNN.
        batch_size = seq.size(1)
        # CUDNN complains if h0 isn't contiguous.
        h0 = self.h0.expand(1, batch_size, self.hid_size).contiguous()
        _, hid = self.rnn(seq, h0)
        assert hid.size() == (1, batch_size, self.hid_size)
        x = self.mlp(hid.view(batch_size, self.hid_size))
        cols = split_at(x, self.col_widths)
        z_what_mean = cols[0]
        z_what_sd = softplus(cols[1])
        return z_what_mean, z_what_sd

class Decoder(nn.Module):
    def __init__(self, z_what_size, hidden_layer_sizes, num_chan, window_size, bias):
        super(Decoder, self).__init__()
        out_size = num_chan * window_size**2
        self.bias = bias
        self.mlp = MLP(z_what_size, hidden_layer_sizes + [out_size], nn.ReLU)

    def forward(self, z_what):
        return sigmoid(self.mlp(z_what) + self.bias)

# dynair4


class Combine4(nn.Module):
    def __init__(self, input_rnn_hid_size, hids, z_size, zero_mean=False):
        super(Combine4, self).__init__()
        self.zero_mean = zero_mean
        self.mlp = MLP(input_rnn_hid_size + z_size, hids + [2 * z_size], nn.ReLU)
        self.col_widths = [z_size, z_size]

    def forward(self, input_rnn_hid, z_prev):
        x = self.mlp(torch.cat((input_rnn_hid, z_prev), 1))
        cols = split_at(x, self.col_widths)
        mean = cols[0]
        sd = softplus(cols[1])
        if self.zero_mean:
            mean = mean * 0
        return mean, sd

class InitialState(nn.Module):
    def __init__(self, input_rnn_hid_size, hids, z_size):
        super(InitialState, self).__init__()
        self.mlp = MLP(input_rnn_hid_size, hids + [z_size * 2], nn.ReLU)
        self.col_widths = [z_size, z_size]

    def forward(self, input_rnn_hid):
        x = self.mlp(input_rnn_hid)
        cols = split_at(x, self.col_widths)
        mean = cols[0]
        sd = softplus(cols[1])
        return mean, sd

# dynair5

class Combine5(nn.Module):
    def __init__(self, input_rnn_hid_size, hids, z_size, use_skip):
        super(Combine5, self).__init__()
        self.mlp = MLP(input_rnn_hid_size + z_size, hids + [2 * z_size], nn.ReLU)
        self.col_widths = [z_size, z_size]
        self.use_skip = use_skip

    def forward(self, input_rnn_hid, z_pre):
        x = self.mlp(torch.cat((input_rnn_hid, z_pre), 1))
        cols = split_at(x, self.col_widths)
        mean = cols[0]
        if self.use_skip:
            mean = mean + z_pre
        sd = softplus(cols[1])
        return mean, sd
