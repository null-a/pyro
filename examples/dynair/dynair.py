import torch
import torch.nn as nn
from torch.nn.functional import affine_grid, grid_sample
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
from pyro.util import zeros, ng_ones#ng_zeros

from modules import MLP, SquishStateParams, SquishObjParams

class DynAIR(nn.Module):
    def __init__(self):

        super(DynAIR, self).__init__()

        self.seq_length = 5
        self.max_num_obj = 3

        self.image_size = 32
        self.num_chan = 4

        self.window_size = 16

        self.z_size = 10
        self.z_pres_size = 1
        self.z_where_size = 2
        self.z_what_size = 20

        self.emission_rnn_size = 10

        # Parameters.
        # TODO: Make these optimizable.
        self.z_0 = zeros(self.z_size)
        self.emission_rnn_h_0 = zeros(self.emission_rnn_size)
        # I think we'll want fixed alpha=1 for the background. (i.e. background is opaque)
        self.background = torch.cat((zeros(self.num_chan - 1, self.image_size, self.image_size),
                                     ng_ones(1, self.image_size, self.image_size)))

        # Modules.
        self.decode = nn.Sequential(
            MLP(self.z_what_size, [100, self.num_chan * self.window_size**2], nn.ReLU),
            nn.Sigmoid())

        # Use something more like the DMM transition here.
        self.transition = nn.Sequential(
            nn.Linear(self.z_size, 2*self.z_size),
            SquishStateParams(self.z_size))

        # These could be something fancier.
        self.emission_rnn = nn.RNNCell(self.z_size, self.emission_rnn_size)
        self.emission_predict_params = nn.Sequential(
            nn.Linear(self.emission_rnn_size, self.z_pres_size + 2*self.z_where_size + 2*self.z_what_size),
            SquishObjParams(self.z_pres_size, self.z_where_size, self.z_what_size))

    def model(self, b):
        # TODO: iarange here (or somewhere)
        z = batch_expand(self.z_0, b) # Hidden state.
        for t in range(self.seq_length):
            z = self.model_transition(t, z)
            frame_mean = self.model_emission(t, z)
            print(frame_mean)
            # TODO: Add likelihood/observations.

    def model_transition(self, t, z_prev):
        b = z_prev.size(0)
        z_mean, z_sd = self.transition(z_prev)
        z = pyro.sample('z_{}'.format(t),
                        dist.normal,
                        z_mean,
                        z_sd)
        return z

    # Currently uses an RNN to parameterize the dependency of the
    # per-frame choices on the model's hidden state. (This isn't the
    # only choice one could make.)

    # 1. Since FF net, since we compute all params anyway.
    # 2. Use z as rnn state, no extra input.

    def model_emission(self, t, z):
        b = z.size(0)
        rnn_h = batch_expand(self.emission_rnn_h_0, b)
        z_pres = ng_ones(b)
        img_so_far = batch_expand(self.background, b)
        for i in range(self.max_num_obj):
            rnn_h, z_pres, img_so_far = self.model_object(t, i, z, rnn_h, z_pres, img_so_far)
        return img_so_far

    def model_object(self, t, i, z, rnn_h_prev, z_pres_prev, img_so_far_prev):
        assert z.size(0) == rnn_h_prev.size(0) == z_pres_prev.size(0) == img_so_far_prev.size(0)
        b = z.size(0)

        # RNN step.
        rnn_h = self.emission_rnn(z, rnn_h_prev)

        # One difference from AIR is that now everything is
        # conditioned on the model's hidden state. (Through the
        # emission RNN.)
        z_pres_p, z_where_mean, z_where_sd, z_what_mean, z_what_sd = self.emission_predict_params(rnn_h)

        z_pres = self.model_sample_z_pres(t, i, z_pres_p, z_pres_prev)
        z_where = self.model_sample_z_where(t, i, z_where_mean, z_where_sd)
        z_what = self.model_sample_z_what(t, i, z_what_mean, z_what_sd)

        # assert z_what.size() == (n, z_what_size)
        y_att = self.decode(z_what)
        y = self.window_to_image(z_where, y_att)

        img_so_far = composite(img_so_far_prev, y, z_pres)

        return rnn_h, z_pres, img_so_far

    # TODO: How do we do the eqv. of setting z_pres_prior_p=0.01 here?
    def model_sample_z_pres(self, t, i, z_pres_p, z_pres_prev):
        z_pres = pyro.sample('z_pres_{}_{}'.format(t, i),
                             dist.bernoulli,
                             z_pres_p * z_pres_prev)
        return z_pres

    def model_sample_z_where(self, t, i, z_where_mean, z_where_sd):
        return pyro.sample('z_where_{}_{}'.format(t, i),
                           dist.normal,
                           z_where_mean,
                           z_where_sd)

    def model_sample_z_what(self, t, i, z_what_mean, z_what_sd):
        return pyro.sample('z_what_{}_{}'.format(t, i),
                           dist.normal,
                           z_what_mean,
                           z_what_sd)


    def window_to_image(self, z_where, windows):
        assert z_where.size(0) == windows.size(0)
        # assert z_where.size(1) == 2
        assert windows.size(1) == self.num_chan * self.window_size**2, 'Size mismatch.'
        n = windows.size(0)
        theta = expand_z_where(z_where)
        assert theta.size() == (n, 2, 3)
        grid = affine_grid(theta, torch.Size((n, self.num_chan, self.image_size, self.image_size)))
        # first arg to grid sample should be (n, c, in_w, in_h)
        out = grid_sample(windows.view(n, self.num_chan, self.window_size, self.window_size), grid)
        # TODO: Maybe don't need to view if that only exists to drop singleton dim?
        # Also better off not flat for easier compositing?
        return out#.view(n, num_chan, self.image_size, self.image_size)


def batch_expand(t, b):
    return t.expand((b,) + t.size())

def expand_z_where(z_where):
    b = z_where.size(0)
    assert z_where.size(1) == 2
    # assert z_where.size() == (n, 2)
    # Translation only to begin, full affine later.
    # [x,y] -> [[1,0,x],
    #           [0,1,y]]
    I = batch_expand(Variable(torch.eye(2)), b)
    return torch.cat((I, z_where.view(b, 2, 1)), 2)

# Note, all of these need to operate on mini-batches.
def composite(img_so_far, y, z_pres):
    # assert y.size() == (n, 4, image_size, image_size)
    # assert z_pres.size() == (n,)

    # TODO: I'm multiplying all channels by z_pres for simplicity.
    # This might not be the right thing to do if z_pres can take on
    # values other than 0 and 1. (Since z_pres will be incorporated
    # once here, then again when the rgb is multiplied by alpha.)

    return over(y * z_pres.view(-1, 1, 1, 1), img_so_far)

# TODO: I suspect this can be simplified if the background is always
# opaque and we always composite an object on to the image so far.
# image_so_far will always have opacity=1 I think, so we can probably
# avoid representing that explicity and simplify the over computation.

# This assumes that the number of channels is 4.
def over(a, b):
    # a over b
    # https://en.wikipedia.org/wiki/Alpha_compositing
    # assert a.size() == (n, 4, image_size, image_size)


    rgb_a = a[:, 0:3] # .size() == (n, 3, image_size, image_size)
    rgb_b = b[:, 0:3]
    alpha_a = a[:, 3:4] # .size() == (n, 1, image_size, image_size)
    alpha_b = b[:, 3:4]


    c = alpha_b * (1 - alpha_a)
    alpha = alpha_a + c
    rgb = (rgb_a * alpha_a + rgb_b * c) / alpha

    return torch.cat((rgb, alpha), 1)


if __name__ == '__main__':
    dynair = DynAIR()
    dynair.model(1)
