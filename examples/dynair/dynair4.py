import torch
import torch.nn as nn
from torch.nn.functional import affine_grid, grid_sample, sigmoid
from torch.autograd import Variable

import numpy as np

import pyro
import pyro.distributions as dist
from pyro.util import zeros, ng_zeros, ng_ones
import pyro.optim as optim
from pyro.infer import SVI
import pyro.poutine as poutine

from modules import MLP, SquishStateParams, FixedTransition, TransitionDMM, InputRNN, EncodeRNN, Decoder, Combine, CombineDMM, LinearTransition, CombineSD

from matplotlib import pyplot as plt

import visdom
from PIL import Image, ImageDraw

class DynAIR(nn.Module):
    def __init__(self):

        super(DynAIR, self).__init__()

        self.seq_length = 14

        self.image_size = 32
        self.num_chan = 4

        self.window_size = 16
        self.window_scale = self.image_size / self.window_size

        self.z_size = 4
        self.z_what_size = 20
        self.z_where_size = 2

        assert self.z_where_size <= self.z_size # z_where is a slice of z

        # TODO: Are these sensible?
        #                                         [z_where , rest... ]
        #self.w_0_prior_sd = Variable(torch.Tensor([1.3, 1.3, 0.4, 0.4]), requires_grad=False)
        self.w_0_prior_sd = Variable(torch.Tensor([2, 2, 0.5, 0.5]), requires_grad=False)
        self.w_prior_sd = ng_ones(self.z_size) * 0.1

        self.z_what_prior_mean = ng_zeros(self.z_what_size)
        self.z_what_prior_sd = ng_ones(self.z_what_size)

        self.input_rnn_hid_size = 100
        self.encode_rnn_hid_size = 100

        self.decoder_hidden_layers = [100]

        self.likelihood_sd = 0.3

        # Parameters.

        # Optimizable bkg:
        # TODO: Squish to [0,1].
        #self.bkg_rgb = nn.Parameter(torch.zeros(self.num_chan - 1, self.image_size, self.image_size) + 0.5)

        # Fixed bkg:
        self.bkg_rgb = ng_zeros(self.num_chan - 1, self.image_size, self.image_size)

        # I think we'll want fixed alpha=1 for the background. (i.e. background is opaque)
        self.bkg_alpha = ng_ones(1, self.image_size, self.image_size)

        # Sample dummy background when testing sampling from model.
        # param = zeros(self.num_chan - 1, self.image_size, self.image_size)
        # onpixels = pyro.sample('bkg1', dist.bernoulli, param + 0.1)
        # intensities = pyro.sample('bkg2', dist.uniform, param, param + 1)
        # self.background = torch.cat((intensities * onpixels,
        #                              ng_ones(1, self.image_size, self.image_size)))

        # Guide modules:

        self.initial_state = MLP(self.input_rnn_hid_size, [self.input_rnn_hid_size, self.z_size], nn.ReLU)

        self.encode_rnn = EncodeRNN(self.window_size, self.num_chan,
                                    self.encode_rnn_hid_size, self.z_what_size)

        # self.decode = nn.Sequential(
        #     MLP(self.z_what_size, [100, self.num_chan * self.window_size**2], nn.ReLU),
        #     nn.Sigmoid())

        # Use fixed bias so that early in optimization objects are
        # black. Without this inference will position all objects off
        # screen to avoid the penalty of placing a mid-gray square in
        # the output. (AIR does something similar, though there
        # inference can choose to "turn off" the object.)
        self.decode = Decoder(self.z_what_size, self.decoder_hidden_layers, self.num_chan, self.window_size, -2)

        self.input_rnn = InputRNN(self.image_size, self.num_chan, self.input_rnn_hid_size)

        self.combine = CombineDMM(self.input_rnn_hid_size, self.z_size)
        #self.combine = Combine(self.input_rnn_hid_size, self.z_size)


        # Model modules:
        # self.transition = nn.Sequential(
        #     nn.Linear(self.z_size, 2*self.z_size),
        #     SquishStateParams(self.z_size))

        #self.transition = TransitionDMM(self.z_size)

        # self.transition = FixedTransition()

        # TODO: This will now be used in both the model and the guide.
        # I think this will "just work", but if not compare with
        # adding to the model and guiding with a Delta, which is the
        # correct thing.
        self.transition = LinearTransition(self.z_size)


    def background(self):
        return torch.cat((self.bkg_rgb, self.bkg_alpha))

    # TODO: This do_likelihood business is unpleasant.
    def model(self, batch, do_likelihood=True):

        batch_size = batch.size(0)
        z_what = self.model_sample_z_what(batch_size)
        y_att = self.decode(z_what)

        w = self.model_sample_w_0(batch_size)
        z = w # + zero
        frame_mean = self.model_emission(z, y_att)

        zs = [z]
        frames = [frame_mean]

        # Recall, that the data are in reverse time order.
        if do_likelihood:
            self.likelihood(0, frame_mean, batch[:, -1])

        # TODO: iarange here (or somewhere)
        for t in range(1, self.seq_length):
            z = self.model_transition(t, z)
            frame_mean = self.model_emission(z, y_att)
            #print(frame_mean)
            zs.append(z)
            frames.append(frame_mean)
            if do_likelihood:
                self.likelihood(t, frame_mean, batch[:, -(t + 1)])

        return frames, zs

    def likelihood(self, t, frame_mean, obs):
        frame_sd = (self.likelihood_sd * ng_ones(1)).expand_as(frame_mean)
        # TODO: Using a normal here isn't very sensible since the data
        # is in [0, 1]. Do something better.
        pyro.sample('x_{}'.format(t),
                    dist.normal,
                    frame_mean,
                    frame_sd,
                    obs=obs)


    def model_sample_z_what(self, batch_size):
        return pyro.sample('z_what',
                           dist.normal,
                           self.z_what_prior_mean,
                           self.z_what_prior_sd,
                           batch_size=batch_size)

    def model_sample_w_0(self, batch_size):
        return pyro.sample('w_0',
                           dist.normal,
                           ng_zeros(self.z_size),
                           self.w_0_prior_sd,
                           batch_size=batch_size)

    def model_sample_w(self, t, batch_size):
        return pyro.sample('w_{}'.format(t),
                           dist.normal,
                           ng_zeros(self.z_size),
                           self.w_prior_sd,
                           batch_size=batch_size)

    def model_transition(self, t, z):
        batch_size = z.size(0)
        w = self.model_sample_w(t, batch_size)
        z = self.transition(z) + w
        assert_size(w, (batch_size, self.z_size))
        assert_size(z, (batch_size, self.z_size))
        return z

    def model_emission(self, z, y_att):
        assert z.size(0) == y_att.size(0)
        batch_size = z.size(0)
        z_where = z[:, 0:self.z_where_size]
        y = self.window_to_image(z_where, y_att)
        return over(y, batch_expand(self.background(), batch_size))

    # ==GUIDE==================================================

    def guide(self, batch, *args):

        # I'd rather register model/guide modules in their methods,
        # but this is easier.
        pyro.module('dynair', self)
        # for name, _ in self.named_parameters():
        #     print(name)

        # TODO: iarange (here, or elsewhere) (I'm assuming batch will
        # be the first dim in order to support this.)

        # Assume input videos already have frames in reverse time
        # order.
        batch_size = batch.size(0)
        assert_size(batch, (batch_size, self.seq_length, self.num_chan, self.image_size, self.image_size))

        zs = self.guide_zs(batch)
        z_what = self.guide_z_what(batch, zs)

        return zs, z_what

    def guide_zs(self, batch):
        # Run RNN over input (rev)
        batch_size = batch.size(0)
        assert_size(batch, (batch_size, self.seq_length, self.num_chan, self.image_size, self.image_size))

        flat_batch = batch.view(batch_size, self.seq_length, -1)
        assert_size(flat_batch, (batch_size, self.seq_length, self.num_chan * self.image_size ** 2))

        input_rnn_h = self.input_rnn(flat_batch)

        # Initialise z_{-1} from RNN state
        z = self.initial_state(input_rnn_h[:, -1])

        zs = []

        for t in range (self.seq_length):
            # print(t)
            # print(z.size())
            # Reminder: input_rnn_h is in reverse time order.
            w = self.guide_w(t, input_rnn_h[:, self.seq_length - (t + 1)], z)
            z = self.transition(z) + w
            zs.append(z)

        return zs

    def guide_w(self, t, rnn_hid, z_prev):
        batch_size = z_prev.size(0)
        # TODO: If the guide outputs the mean it has the option of
        # ignoring the transition and using only w to predict the
        # state for the current frame. This can lead to good
        # reconstructions without learning the dynamics. However, this
        # extra flexibility might help get inference off the ground
        # (by having a way to position windows before the transition
        # is learned) so it could be useful. Will the prior on w
        # ensure that optimization eventually finds the solution that
        # uses the transition rather than w?
        w_mean, w_sd = self.combine(rnn_hid, z_prev)
        #w_mean = ng_zeros(batch_size, self.z_size)
        w = pyro.sample('w_{}'.format(t), dist.normal, w_mean, w_sd)
        return w

    def guide_z_what(self, batch, zs):
        batch_size = batch.size(0)
        assert_size(batch, (batch_size, self.seq_length, self.num_chan, self.image_size, self.image_size))
        assert len(zs) == self.seq_length
        assert all(z.size() == (batch_size, self.z_size) for z in zs)

        z_wheres = [z[:, 0:self.z_where_size] for z in zs]
        batch_trans = batch.transpose(0, 1)
        x_att = torch.stack([self.image_to_window(z_where, frame)
                             for (z_where, frame) in zip(z_wheres, batch_trans)])

        assert_size(x_att, (self.seq_length, batch_size, self.num_chan * self.window_size ** 2))
        z_what_mean, z_what_sd = self.encode_rnn(x_att)
        z_what = pyro.sample('z_what', dist.normal, z_what_mean, z_what_sd)
        # print('z_what')
        # print(z_what)
        return z_what

    def image_to_window(self, z_where, images):
        assert z_where.size(0) == images.size(0), 'Batch size mismatch'
        n = images.size(0)
        assert_size(images, (n, self.num_chan, self.image_size, self.image_size))
        theta_inv = expand_z_where(*z_where_inv(z_where, self.window_scale))
        grid = affine_grid(theta_inv, torch.Size((n, self.num_chan, self.window_size, self.window_size)))
        return grid_sample(images, grid).view(n, -1)

    def window_to_image(self, z_where, windows):
        assert z_where.size(0) == windows.size(0)
        # assert z_where.size(1) == 2
        assert windows.size(1) == self.num_chan * self.window_size**2, 'Size mismatch.'
        n = windows.size(0)
        # TODO: The need for `contiguous` arises here because z_where
        # is a slice out of z. See if it can be avoided. This only
        # comes up in the model, since the guide passes z_where to
        # z_where_inv which creates a new (contiguous) tensor.
        theta = expand_z_where(z_where.contiguous(), self.window_scale)
        assert theta.size() == (n, 2, 3)
        grid = affine_grid(theta, torch.Size((n, self.num_chan, self.image_size, self.image_size)))
        # first arg to grid sample should be (n, c, in_w, in_h)
        return grid_sample(windows.view(n, self.num_chan, self.window_size, self.window_size), grid)


def batch_expand(t, b):
    return t.expand((b,) + t.size())

def expand_z_where(z_where, scale):
    b = z_where.size(0)
    assert z_where.size(1) == 2
    # assert z_where.size() == (n, 2)
    # Translation only to begin, full affine later.
    # (scale, [x, y]) -> [[scale, 0,     x],
    #                     [0,     scale, y]]
    I = batch_expand(Variable(torch.eye(2) * scale), b)
    return torch.cat((I, z_where.view(b, 2, 1)), 2)

def assert_size(t, expected_size):
    actual_size = t.size()
    assert actual_size == expected_size, 'Expected size {} but got {}.'.format(expected_size, tuple(actual_size))

def z_where_inv(z_where, scale):
    return -z_where / scale, 1.0 / scale



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


class Plot():
    def __init__(self, vis):
        self.win = None
        self.vis = vis

    def add(self, x, y):
        if self.win is None:
            self.win = self.vis.line(X=np.array([x]), Y=np.array([y]))
        else:
            self.vis.line(X=np.array([x]), Y=np.array([y]), win=self.win, update='append')

def run_svi(X):
    vis = visdom.Visdom()
    progress_plot = Plot(visdom.Visdom(env='progress'))
    dynair = DynAIR()

    batches = X.chunk(40)

    svi = SVI(dynair.model, dynair.guide,
              optim.Adam(dict(lr=1e-4)),
              loss='ELBO')
              # trace_graph=True) # No discrete things, yet.

    for i in range(1000):

        for j, batch in enumerate(batches):
            loss = svi.step(batch)
            elbo = -loss / (15 * 25) # elbo per datum, per frame
            print('epoch={}, batch={}, elbo={:.2f}'.format(i, j, elbo))
            progress_plot.add(i*len(batches) + j, elbo)

        ix = 7
        # TODO: Make reconstruct method.
        trace = poutine.trace(dynair.guide).get_trace(X[ix:ix+3])
        frames, zs = poutine.replay(dynair.model, trace)(X[ix:ix+3], do_likelihood=False)

        frames = latent_seq_to_tensor(frames)
        zs = latent_seq_to_tensor(zs)

        out0 = overlay_window_outlines(dynair, frames[0], zs[0, :, 0:2])
        vis.images(list(reversed(frames_to_rgb_list(X[ix]))), nrow=7)
        vis.images(frames_to_rgb_list(out0), nrow=7)

        out1 = overlay_window_outlines(dynair, frames[1], zs[1, :, 0:2])
        vis.images(list(reversed(frames_to_rgb_list(X[ix+1]))), nrow=7)
        vis.images(frames_to_rgb_list(out1), nrow=7)

        out2 = overlay_window_outlines(dynair, frames[2], zs[2, :, 0:2])
        vis.images(list(reversed(frames_to_rgb_list(X[ix+2]))), nrow=7)
        vis.images(frames_to_rgb_list(out2), nrow=7)


        # Test extrapolation.
        # TODO: Remove seq_length hack.
        # TODO: Clean-up.
        # dynair.seq_length = 7
        # zs, z_what = dynair.guide(X[ix:ix+1, 0:7])
        # y_att = dynair.decode(z_what)
        # z = zs[-1]
        # frames = []
        # extrap_zs = []
        # for t in range(7):
        #     z = dynair.model_transition(t, z)
        #     frame_mean = dynair.model_emission(z, y_att)
        #     frames.append(frame_mean)
        #     extrap_zs.append(z)
        # extrap_frames = latent_seq_to_tensor(frames)
        # extrap_zs = latent_seq_to_tensor(extrap_zs)
        # out = overlay_window_outlines(dynair, extrap_frames[0], extrap_zs[0, :, 0:2])
        # #print(extrap_frames.size())
        # # TODO: Show ground truth and extrapolation.
        # vis.images(list(reversed(frames_to_rgb_list(X[ix])))[7:] + frames_to_rgb_list(out), nrow=7)
        # dynair.seq_length = 14

        print(dynair.transition.lin.weight.data)




def load_data():
    X_np = np.load('single_object_no_bkg.npz')['X']
    #print(X_np.shape)
    X_np = X_np.astype(np.float32)
    X_np /= 255.0
    X = Variable(torch.from_numpy(X_np))
    return X

def frames_to_rgb_list(frames):
    return frames[:, 0:3].data.numpy().tolist()

def img_to_arr(img):
    assert img.mode == 'RGBA'
    channels = 4
    w, h = img.size
    arr = np.fromstring(img.tobytes(), dtype=np.uint8)
    return arr.reshape(w * h, channels).T.reshape(channels, h, w)

def draw_rect(size):
    img = Image.new('RGBA', (size, size))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, size - 1, size - 1], outline='white')
    return torch.from_numpy(img_to_arr(img).astype(np.float32) / 255.0)

def draw_window_outline(dynair, z_where):
    n = z_where.size(0)
    rect = draw_rect(dynair.window_size)
    rect_batch = Variable(batch_expand(rect.contiguous().view(-1), n).contiguous())
    return dynair.window_to_image(z_where, rect_batch)

def overlay_window_outlines(dynair, frame, z_where):
    return over(draw_window_outline(dynair, z_where), frame)

def latent_seq_to_tensor(arr):
    # Turn an array of latents (of length seq_len) returned by the
    # model into a (batch, seq_len, rest...) tensor.
    return torch.cat([t.unsqueeze(0) for t in arr]).transpose(0, 1)



if __name__ == '__main__':

    # Test model by sampling:
    # dynair = DynAIR()
    # dummy_data = Variable(torch.ones(1, 14, 4, 32, 32))
    # frames, zs = dynair.model(dummy_data, do_likelihood=False)
    # #print(frames)
    # print(zs)
    # for frame in frames:
    #     print(frame[0].data.size())
    #     img = frame[0].data.numpy().transpose(1,2,0)
    #     plt.imshow(img)
    #     plt.show()
    #input('press a key...')

    # Test guide:
    #(batch, seq, channel, w, h)
    # dynair = DynAIR()
    # data = Variable(torch.ones(1, 14, 4, 32, 32))
    # dynair.guide(data)

    run_svi(load_data())
