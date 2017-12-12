import torch
import torch.nn as nn
from torch.nn.functional import sigmoid, softplus


from torch.autograd import Variable

from pyro.util import ng_zeros

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

# dynair3

# TODO: This is network is unlikely to be rich enough, this is just
# something to get this to run. Replace with something more like the
# DMM combiner perhaps?
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

class InputRNN(nn.Module):
    def __init__(self, image_size, num_chan, hid_size):
        super(InputRNN, self).__init__()
        self.rnn = nn.GRU(num_chan * image_size**2, hid_size, batch_first=True)
        self.h0 = ng_zeros(hid_size) # TODO: Make optimizable.
        self.hid_size = hid_size

    def forward(self, seq):
        batch_size = seq.size(0)
        h0 = self.h0.expand(1, batch_size, self.hid_size)
        hid_seq, _ = self.rnn(seq, h0)
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
        self.h0 = ng_zeros(hid_size) # TODO: Make optimizable.
        self.mlp = MLP(hid_size, [z_what_size * 2], nn.ReLU)
        self.col_widths = [z_what_size, z_what_size]

    def forward(self, seq):
        # seq.size(0) is sequence length for this RNN.
        batch_size = seq.size(1)
        h0 = self.h0.expand(1, batch_size, self.hid_size)
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
