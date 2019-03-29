import torch
import torch.nn as nn
from torch.nn.functional import softplus

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

        self.emission_net = MLP(z_size, [200, 200, 200, x_size], nn.ELU, output_non_linearity=False)

        # Locally linear transitions.
        # This follows notation in paper.
        self.M = 16 # number of transition matrices
        self.A = nn.Parameter(self.prototype.new_zeros((self.M, z_size, z_size)))
        torch.nn.init.normal_(self.A, 0., 1e-3)

        # TODO: I don't see the arch. specified in the paper.
        self.alpha_net = nn.Sequential(nn.Linear(z_size, self.M), nn.Softmax(dim=1))


    def emission(self, z):
        return torch.sigmoid(self.emission_net(z) + 2.0).reshape(-1, self.num_chan, self.image_size, self.image_size)

    def transition(self, t, z_prev, w, deterministic=False):
        assert t >= 0
        if t == 0:
            assert z_prev is None
            # TODO: make this z = z0 + w, where z0 is learnable?
            # (check paper, i think they do something different.)
            return w
        else:

            # TODO: Use more recent PyTorch to avoid having to clone
            # inputs to einsum.
            # https://github.com/pytorch/pytorch/issues/7763

            batch_size = z_prev.size(0)
            alpha = self.alpha_net(z_prev)
            # Per-data point transition matrices:
            A = torch.einsum('ij,jkl->ikl', (alpha.clone(), self.A.clone()))
            assert A.shape == (batch_size, self.z_size, self.z_size)

            # Batched matrix-vector multiple (between per data point
            # transition matrices and batch of z_prev)
            Az_prev = torch.einsum('ijk,ik->ij', (A.clone(), z_prev.clone()))
            assert Az_prev.shape == (batch_size, self.z_size)

            if not deterministic:
                # *additive* noise for now. the paper has (in one example)
                # an extra mat. mul. in here.
                return Az_prev + w
            else:
                assert w is None
                return Az_prev

    def forward(self, batch, annealing_factor=1.0):
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

        w_sd0 = batch_expand(self.prototype.new_ones(self.z_size), batch_size)

        # Prior over scale of w for t > 0.

        # TODO: Remove this hack if this sticks around.
        # (Scale is a hack to account for data sub-sampling without
        # changing too much code. Hard-coded assuming data set/batch
        # size I'm currently working with.)
        with poutine.scale(None, 1. / 200):
            w_sd = pyro.sample('w_sd',
                               dist.Gamma(self.prototype.new_ones(self.z_size) * 1.,
                                          self.prototype.new_ones(self.z_size) * 3.).independent(1))

        with pyro.iarange('data', batch_size):
            z_prev = None
            for t in range(seq_length):

                w_mean = batch_expand(self.prototype.new_zeros(self.z_size), batch_size)
                # TODO: Tighten after first frame? (so that guide is
                # encouraged to use w only to fix up learned
                # transitions, and not to output latent state directly
                # from the observation. (e.g. fix up predicted
                # position rather than output position directly.)

                # if t == 0:
                #     w_sd = w_sd * 0.1
                w = self.sample_w(t, w_mean, w_sd if t>0 else w_sd0)

                z = self.transition(t, z_prev, w)

                frame_mean = self.emission(z)

                obs = seqs[:,t]
                self.likelihood(t, frame_mean, obs, annealing_factor)

                frames.append(frame_mean)
                zs.append(z)
                z_prev = z

        return frames, zs

    def sample_w(self, t, w_mean, w_sd):
        return pyro.sample('w_{}'.format(t),
                           dist.Normal(w_mean, w_sd).independent(1))

    def likelihood(self, t, frame_mean, obs, annealing_factor):
        frame_sd = (self.likelihood_sd * self.prototype.new_ones(1)).expand_as(frame_mean)
        with poutine.scale(None, annealing_factor):
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

        self.rnn_hid_size = 200
        self.predict0_rnn = nn.RNN(x_size, self.rnn_hid_size, nonlinearity='relu', bidirectional=True)
        self.predict0_net = nn.Sequential(MLP(self.rnn_hid_size * 2, [200], nn.ELU), NormalParams(200, z_size))

        self.predict_net = nn.Sequential(MLP(x_size + z_size, [200, 200], nn.ELU), NormalParams(200, z_size))

        self.w_sd_param = nn.Parameter(torch.zeros(self.z_size) -1.5) # init. to ~0.2 (after softplus)

    def forward(self, batch, annealing_factor=None):
        pyro.module('guide', self)

        seqs, obj_counts = batch
        batch_size = seqs.size(0)
        seq_length = seqs.size(1)

        assert_size(seqs, (batch_size, seq_length, self.num_chan,
                           self.image_size, self.image_size))
        assert_size(obj_counts, (batch_size,))
        assert all(1 == obj_counts)

        pyro.sample('w_sd', dist.Delta(softplus(self.w_sd_param)).independent(1))

        with pyro.iarange('data', batch_size):

            z_prev = None

            for t in range(seq_length):

                # TODO: try (for a second time) predicting w from z
                # rather than z_prev. (this amounts having the guide
                # predict the error in the transition.)
                if t == 0:
                    # predict w0 from entire sequence.

                    # TODO: It looks like two RNNs (fwd/back) don't
                    # interact. So here I'm taking the final hidden
                    # state of the backward net, but I'm not making
                    # any use of the forward net. One thing to do
                    # would be to take the final hidden state of the
                    # forward net and concat it to the thing I'm
                    # already using. That hid. state is:
                    # rnn_outputs[-1,:,0:200]
                    rnn_outputs, _ = self.predict0_rnn(seqs.reshape(batch_size, seq_length, -1).transpose(0, 1))
                    predict_hid = torch.cat((rnn_outputs[0, :, self.rnn_hid_size:],
                                             rnn_outputs[-1, :, 0:self.rnn_hid_size]), 1)
                    w_mean, w_sd = self.predict0_net(predict_hid)
                else:
                    # predict w from x and z_prev
                    x = seqs[:,t].reshape(-1, self.x_size)
                    w_mean, w_sd = self.predict_net(torch.cat((x, z_prev), 1))

                w = pyro.sample('w_{}'.format(t),
                                dist.Normal(w_mean, w_sd).independent(1))
                z = self.model.transition(t, z_prev, w)
                z_prev = z


def frames_to_tensor(arr):
    # Turn an array of frames (of length seq_len) returned by the
    # model into a (batch, seq_len, rest...) tensor.
    return torch.cat([t.unsqueeze(0) for t in arr]).transpose(0, 1).detach()

class DVBF(nn.Module):
    def __init__(self, z_size=4, num_chan=1, image_size=20, use_cuda=False):
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
        seq_length = seqs.size(1)

        for t in range(10):
            z = self.model.transition(seq_length+t, z_prev, None, deterministic=True)
            frame_mean = self.model.emission(z)
            extra_frames.append(frame_mean)
            z_prev = z

        return frames_to_tensor(frames), frames_to_tensor(extra_frames)

    def clear_cache(self, *args, **kwargs):
        pass
