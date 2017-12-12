import torch
import torch.nn as nn
from torch.nn.functional import affine_grid, grid_sample
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
from pyro.util import zeros, ng_zeros, ng_ones

from modules import MLP, SquishStateParams, DummyTransition

from matplotlib import pyplot as plt

class DynAIR(nn.Module):
    def __init__(self):

        super(DynAIR, self).__init__()

        self.seq_length = 10

        self.image_size = 32
        self.num_chan = 4

        self.window_size = 16

        self.z_size = 4
        self.z_what_size = 10
        self.z_where_size = 2

        assert self.z_where_size <= self.z_size # z_where is a slice of z

        self.z_0_prior_mean = ng_zeros(self.z_size)
        #                                         [z_where , rest... ]
        self.z_0_prior_sd = Variable(torch.Tensor([1.3, 1.3, 0.4, 0.4]))

        self.z_what_prior_mean = ng_zeros(self.z_what_size)
        self.z_what_prior_sd = ng_ones(self.z_what_size)

        # Parameters.
        # TODO: Make these optimizable.

        # I think we'll want fixed alpha=1 for the background. (i.e. background is opaque)
        # self.background = torch.cat((zeros(self.num_chan - 1, self.image_size, self.image_size),
        #                              ng_ones(1, self.image_size, self.image_size)))

        # TODO: Replace with opt. param.
        # Sample dummy background:
        param = zeros(self.num_chan - 1, self.image_size, self.image_size)
        onpixels = pyro.sample('bkg1', dist.bernoulli, param + 0.1)
        intensities = pyro.sample('bkg2', dist.uniform, param, param + 1)
        self.background = torch.cat((intensities * onpixels,
                                     ng_ones(1, self.image_size, self.image_size)))

        # Modules.
        self.decode = nn.Sequential(
            MLP(self.z_what_size, [100, self.num_chan * self.window_size**2], nn.ReLU),
            nn.Sigmoid())

        # Use something more like the DMM transition here.
        self.transition = nn.Sequential(
            nn.Linear(self.z_size, 2*self.z_size),
            SquishStateParams(self.z_size))

        # self.transition = DummyTransition()

    def model(self, batch_size):
        z_what = self.model_sample_z_what(batch_size)
        y_att = self.decode(z_what)

        z = self.model_sample_z_0(batch_size)
        frame_mean = self.model_emission(z, y_att)
        frames = [frame_mean]

        # TODO: iarange here (or somewhere)
        for t in range(1, self.seq_length):
            z = self.model_transition(t, z)
            frame_mean = self.model_emission(z, y_att)
            #print(frame_mean)
            frames.append(frame_mean)
            # TODO: Add likelihood/observations.
        return frames


    def model_sample_z_what(self, batch_size):
        return pyro.sample('z_what',
                           dist.normal,
                           self.z_what_prior_mean,
                           self.z_what_prior_sd,
                           batch_size=batch_size)

    # TODO: Having a uniform dist. over the (x,y) of z_where would
    # probably be a little better than the normal, though it makes
    # sense to use normal for other parts of z.
    def model_sample_z_0(self, batch_size):
        return pyro.sample('z_0',
                           dist.normal,
                           self.z_0_prior_mean,
                           self.z_0_prior_sd,
                           batch_size=batch_size)

    # The z variable combines z_where (the window transform
    # parameters) and any other latent dynamic object state.
    def model_sample_z(self, t, z_mean, z_sd):
        return pyro.sample('z_{}'.format(t), dist.normal, z_mean, z_sd)

    def model_transition(self, t, z):
        z_mean, z_sd = self.transition(z)
        return self.model_sample_z(t, z_mean, z_sd)

    def model_emission(self, z, y_att):
        assert z.size(0) == y_att.size(0)
        batch_size = z.size(0)
        z_where = z[:, 0:self.z_where_size]
        y = self.window_to_image(z_where, y_att)
        return over(y, batch_expand(self.background, batch_size))

    def window_to_image(self, z_where, windows):
        assert z_where.size(0) == windows.size(0)
        # assert z_where.size(1) == 2
        assert windows.size(1) == self.num_chan * self.window_size**2, 'Size mismatch.'
        n = windows.size(0)
        theta = expand_z_where(z_where)
        assert theta.size() == (n, 2, 3)
        grid = affine_grid(theta, torch.Size((n, self.num_chan, self.image_size, self.image_size)))
        # first arg to grid sample should be (n, c, in_w, in_h)
        return grid_sample(windows.view(n, self.num_chan, self.window_size, self.window_size), grid)


def batch_expand(t, b):
    return t.expand((b,) + t.size())

def expand_z_where(z_where):
    b = z_where.size(0)
    assert z_where.size(1) == 2
    # assert z_where.size() == (n, 2)
    # Translation only to begin, full affine later.
    # [x,y] -> [[1,0,x],
    #           [0,1,y]]
    I = batch_expand(Variable(torch.eye(2) * 3), b)
    return torch.cat((I, z_where.view(b, 2, 1)), 2)



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
    frames = dynair.model(1)
    #print(frames)
    for frame in frames:
        #print(frame[0].data.size())
        img = frame[0].data.numpy().transpose(1,2,0)
        plt.imshow(img)
        plt.show()
    input('press a key...')
