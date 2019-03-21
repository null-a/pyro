import torch
import torch.nn as nn

import pyro
import pyro.poutine as poutine
import pyro.distributions as dist

from modules import MLP, NormalParams
from utils import assert_size, batch_expand


class Model(nn.Module):
    def __init__(self, z_size, num_chan, image_size, use_cuda=False):
        super(Model, self).__init__()

        self.prototype = torch.tensor(0.).cuda() if use_cuda else torch.tensor(0.)

        self.likelihood_sd = 0.3

        self.z_size = z_size
        self.num_chan = num_chan
        self.image_size = image_size
        x_size = num_chan * image_size**2

        self.transition = MLP(z_size, [z_size, z_size], nn.Tanh)
        self.emission_net = MLP(z_size, [500, x_size], nn.ELU, output_non_linearity=False)

    def emission(self, z):
        return torch.sigmoid(self.emission_net(z) + 2.0).reshape(-1, self.num_chan, self.image_size, self.image_size)

    def forward(self, batch):
        pyro.module('model', self)

        seqs, obj_counts = batch
        batch_size = seqs.size(0)
        seq_length = seqs.size(1)

        assert_size(seqs, (batch_size, seq_length, self.num_chan,
                           self.image_size, self.image_size))
        assert_size(obj_counts, (batch_size,))
        assert all(1 == obj_counts)


        frames = []
        zs = []

        with pyro.iarange('data', batch_size):
            z_prev = None
            for t in range(seq_length):

                w_mean = batch_expand(self.prototype.new_zeros(self.z_size), batch_size)
                # Tighten after first frame? (so that guide is
                # encouraged to use w only to fix up learned
                # transitions, and not to output latent state directly
                # from the observation. (e.g. fix up predicted
                # position rather than output position directly.)
                w_sd = batch_expand(self.prototype.new_ones(self.z_size), batch_size)
                if t == 0:
                    w_sd = w_sd * 0.1
                w = self.sample_w(t, w_mean, w_sd)

                if t == 0:
                    z = w # TODO: make this z = z0 + w, where z0 is learnable?
                else:
                    assert z_prev is not None
                    # *additive* noise for now. the paper has (in one example) an extra mat. mul. in here.
                    z = z_prev + self.transition(z_prev) + w

                frame_mean = self.emission(z)

                obs = seqs[:,t]
                self.likelihood(t, frame_mean, obs)

                frames.append(frame_mean)
                zs.append(z)
                z_prev = z

        return frames, zs

    def sample_w(self, t, w_mean, w_sd):
        return pyro.sample('w_{}'.format(t),
                           dist.Normal(w_mean, w_sd).independent(1))

    def likelihood(self, t, frame_mean, obs):
        frame_sd = (self.likelihood_sd * self.prototype.new_ones(1)).expand_as(frame_mean)
        pyro.sample('x_{}'.format(t),
                    dist.Normal(frame_mean, frame_sd).independent(3),
                    obs=obs)

class Guide(nn.Module):
    def __init__(self, model, z_size, num_chan, image_size, use_cuda=False):
        super(Guide, self).__init__()

        self.model = model
        self.prototype = torch.tensor(0.).cuda() if use_cuda else torch.tensor(0.)

        self.z_size = z_size
        self.num_chan = num_chan
        self.image_size = image_size

        x_size = num_chan * image_size**2
        self.x_size = x_size

        self.z_prev_init = nn.Parameter(self.prototype.new_zeros(z_size))
        self.predict = nn.Sequential(MLP(x_size + z_size, [500, 250], nn.ELU), NormalParams(250, z_size))

    def forward(self, batch):
        pyro.module('guide', self)


        seqs, obj_counts = batch
        batch_size = seqs.size(0)
        seq_length = seqs.size(1)

        assert_size(seqs, (batch_size, seq_length, self.num_chan,
                           self.image_size, self.image_size))
        assert_size(obj_counts, (batch_size,))
        assert all(1 == obj_counts)

        with pyro.iarange('data', batch_size):

            z_prev = batch_expand(self.z_prev_init, batch_size)

            for t in range(seq_length):

                x = seqs[:,t].reshape(-1, self.x_size)

                if t == 0:
                    w_mean, w_sd = self.predict(torch.cat((x, z_prev), 1))
                    w = pyro.sample('w_{}'.format(t),
                                    dist.Normal(w_mean, w_sd).independent(1))
                    z = w
                else:
                    # Compute proposed transition. (This is z without the additive w.)
                    z_prop = z_prev + self.model.transition(z_prev)
                    # predict w (i.e. error in z_prop) from x_t and z_prop
                    w_mean, w_sd = self.predict(torch.cat((x, z_prop), 1))
                    w = pyro.sample('w_{}'.format(t),
                                    dist.Normal(w_mean, w_sd).independent(1))
                    z = z_prop + w

                z_prev = z

def frames_to_tensor(arr):
    # Turn an array of frames (of length seq_len) returned by the
    # model into a (batch, seq_len, rest...) tensor.
    return torch.cat([t.unsqueeze(0) for t in arr]).transpose(0, 1).detach()

class DVBF(nn.Module):
    def __init__(self, z_size=10, num_chan=1, image_size=50, use_cuda=False):
        super(DVBF, self).__init__()

        self.num_chan = num_chan
        self.image_size = image_size

        self.model = Model(z_size, num_chan, image_size, use_cuda)
        self.guide = Guide(self.model, z_size, num_chan, image_size, use_cuda)

        if use_cuda:
            self.cuda()

    def infer(self, seqs, obj_counts):
        trace = poutine.trace(self.guide).get_trace((seqs, obj_counts))
        frames, zs = poutine.replay(self.model, trace)((seqs, obj_counts))

        z_prev = zs[-1]

        extra_frames = []

        for t in range(5):
            z = z_prev + self.model.transition(z_prev) # deterministic
            frame_mean = self.model.emission(z)
            extra_frames.append(frame_mean)
            z_prev = z

        return frames_to_tensor(frames), frames_to_tensor(extra_frames)

    def clear_cache(self, *args, **kwargs):
        pass
