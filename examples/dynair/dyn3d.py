import torch
import torch.nn as nn
from torch.nn.functional import affine_grid, grid_sample, sigmoid
from torch.autograd import Variable

import numpy as np

import pyro
import pyro.distributions as dist
from pyro.util import ng_zeros, ng_ones
import pyro.optim as optim
from pyro.infer import SVI
import pyro.poutine as poutine


import dyn3d_modules as mod


import visdom
from PIL import Image, ImageDraw

import argparse

class DynAIR(nn.Module):
    def __init__(self, use_cuda=False):

        super(DynAIR, self).__init__()

        # Set here so that self.ng_ones/self.ng_zeros does the right
        # thing.
        self.use_cuda = use_cuda

        self.seq_length = 14

        self.image_size = 32
        self.num_chan = 4

        self.window_size = 16

        self.z_size = 4
        self.w_size = 3 # x,y position and scale

        # Network config:
        # Model

        # Guide
        #self.extractor_arch = [200, 200]
        #self.encode_arch = [200,200]
        #self.combiner_arch = []


        bkg_rgb = self.ng_zeros(self.num_chan - 1, self.image_size, self.image_size)
        bkg_alpha = self.ng_ones(1, self.image_size, self.image_size)
        self.bkg = torch.cat((bkg_rgb, bkg_alpha))



        # Priors:
        # TODO: The prior should ensure that scale>0.

        # TODO: Previously a prior over position + a transition
        # initialised to the identity made for a seemingly sensible
        # init. Can we do something similar here? (It seems kinda
        # pointless trying to put information into the prior for the
        # first step if the transition is free to do anything.)

        self.w_0_prior_mean = self.ng_zeros(self.w_size)
        self.w_0_prior_sd = Variable(torch.Tensor([2, 2, 2]),
                                     requires_grad=False)
        if use_cuda:
            self.z_0_prior_sd = self.z_0_prior_sd.cuda()


        self.z_0_prior_mean = self.ng_zeros(self.z_size)
        self.z_0_prior_sd = self.ng_ones(self.z_size)

        self.likelihood_sd = 0.3

        #self.encoder_arch = []
        #self.decoder_arch = []

        # Parameters.
        # ...

        self.guide_z_init = nn.Parameter(torch.zeros(self.z_size)) # TODO: rand. init?
        self.guide_w_init = nn.Parameter(torch.zeros(self.w_size))

        # Modules

        # Guide modules:

        self.z_param = mod.ParamZ([50, 50], self.w_size, self.num_chan * self.window_size**2, self.z_size)
        self.w_param = mod.ParamW([50, 50], self.num_chan * self.image_size**2, self.w_size, self.z_size)
        #self.extractor = None
        #self.encode = None # Maybe use multi-layer RNN instead/as well.
        #self.combiner = None

        # Model modules:

        self.transition = mod.Transition(self.z_size, self.w_size, 50, 50)
        self.decode = nn.Sequential(
            mod.MLP(self.z_size,
                    [100, 100, self.num_chan * self.window_size**2],
                    nn.ReLU),
            nn.Sigmoid())


        # CUDA
        if use_cuda:
            self.cuda()


    # TODO: This do_likelihood business is unpleasant.
    def model(self, batch, do_likelihood=True):

        batch_size = batch.size(0)

        z = self.model_sample_z_0(batch_size)
        w = self.model_sample_w_0(batch_size)
        frame_mean = self.model_emission(z, w)

        zs = [z]
        ws = [w]
        frames = [frame_mean]

        # Recall, that the data are in reverse time order.
        if do_likelihood:
            self.likelihood(0, frame_mean, batch[:, -1])

        # TODO: iarange here (or somewhere)
        for t in range(1, self.seq_length):
            z, w = self.model_transition(t, z, w)
            frame_mean = self.model_emission(z, w)
            zs.append(z)
            ws.append(w)
            frames.append(frame_mean)
            if do_likelihood:
                self.likelihood(t, frame_mean, batch[:, -(t + 1)])

        return frames, ws, zs

    def likelihood(self, t, frame_mean, obs):
        frame_sd = (self.likelihood_sd * self.ng_ones(1)).expand_as(frame_mean)
        # TODO: Using a normal here isn't very sensible since the data
        # is in [0, 1]. Do something better.
        pyro.sample('x_{}'.format(t),
                    dist.normal,
                    frame_mean,
                    frame_sd,
                    obs=obs)

    def model_sample_w_0(self, batch_size):
        return pyro.sample('w_0',
                           dist.normal,
                           self.w_0_prior_mean,
                           self.w_0_prior_sd,
                           batch_size=batch_size)

    def model_sample_w(self, t, w_mean, w_sd):
        return pyro.sample('w_{}'.format(t),
                           dist.normal,
                           w_mean,
                           w_sd)

    def model_sample_z_0(self, batch_size):
        return pyro.sample('z_0',
                           dist.normal,
                           self.z_0_prior_mean,
                           self.z_0_prior_sd,
                           batch_size=batch_size)

    def model_sample_z(self, t, z_mean, z_sd):
        return pyro.sample('z_{}'.format(t),
                           dist.normal,
                           z_mean,
                           z_sd)

    def model_transition(self, t, z, w):
        batch_size = z.size(0)
        assert_size(z, (batch_size, self.z_size))
        assert_size(w, (batch_size, self.w_size))
        z_mean, z_sd, w_mean, w_sd = self.transition(z, w)
        z = self.model_sample_z(t, z_mean, z_sd)
        w = self.model_sample_w(t, w_mean, w_sd)
        return z, w

    def model_emission(self, z, w):
        batch_size = z.size(0)
        assert z.size(0) == w.size(0)
        x_att = self.decode(z)
        bkg = batch_expand(self.bkg, batch_size)
        return over(self.window_to_image(w, x_att), bkg)



    # ==GUIDE==================================================

    def guide(self, batch, *args):

        # I'd rather register model/guide modules in their methods,
        # but this is easier.
        pyro.module('dynair', self)
        # for name, _ in self.named_parameters():
        #     print(name)

        # TODO: iarange (here, or elsewhere) (I'm assuming batch will
        # be the first dim in order to support this.)

        batch_size = batch.size(0)
        assert_size(batch, (batch_size, self.seq_length, self.num_chan, self.image_size, self.image_size))

        zs = []
        ws = []

        z = batch_expand(self.guide_z_init, batch_size)
        w = batch_expand(self.guide_w_init, batch_size)

        for t in range(self.seq_length):
            x = batch[:, t]
            w = self.guide_w(t, x, w, z)
            x_att = self.image_to_window(w, x)
            z = self.guide_z(t, w, x_att, z)

            ws.append(w)
            zs.append(z)

        return ws, zs

    def guide_w(self, t, batch, w_prev, z_prev):
        w_mean, w_sd = self.w_param(batch, w_prev, z_prev)
        return pyro.sample('w_{}'.format(t), dist.normal, w_mean, w_sd)

    def guide_z(self, t, w, x_att, z_prev):
        z_mean, z_sd = self.z_param(w, x_att, z_prev)
        return pyro.sample('z_{}'.format(t), dist.normal, z_mean, z_sd)


    def image_to_window(self, w, images):
        n = w.size(0)
        assert_size(w, (n, self.w_size))
        assert_size(images, (n, self.num_chan, self.image_size, self.image_size))

        theta_inv = expand_z_where(z_where_inv(w))
        grid = affine_grid(theta_inv, torch.Size((n, self.num_chan, self.window_size, self.window_size)))
        return grid_sample(images, grid).view(n, -1)

    def window_to_image(self, w, windows):
        n = w.size(0)
        assert_size(w, (n, self.w_size))
        assert_size(windows, (n, self.num_chan * self.window_size**2))

        theta = expand_z_where(w)
        assert_size(theta, (n, 2, 3))
        grid = affine_grid(theta, torch.Size((n, self.num_chan, self.image_size, self.image_size)))
        # first arg to grid sample should be (n, c, in_w, in_h)
        return grid_sample(windows.view(n, self.num_chan, self.window_size, self.window_size), grid)

    # Helpers to create zeros/ones on cpu/gpu as appropriate.
    def ng_zeros(self, *args, **kwargs):
        t = ng_zeros(*args, **kwargs)
        if self.use_cuda:
            t = t.cuda()
        return t

    def ng_ones(self, *args, **kwargs):
        t = ng_ones(*args, **kwargs)
        if self.use_cuda:
            t = t.cuda()
        return t


def batch_expand(t, b):
    return t.expand((b,) + t.size())


# Spatial transformer helpers. (Taken from the AIR example.)

expansion_indices = torch.LongTensor([1, 0, 2, 0, 1, 3])

def expand_z_where(z_where):
    # Take a batch of three-vectors, and massages them into a batch of
    # 2x3 matrices with elements like so:
    # [s,x,y] -> [[s,0,x],
    #             [0,s,y]]
    n = z_where.size(0)
    assert_size(z_where, (n, 3))
    out = torch.cat((ng_zeros([1, 1]).type_as(z_where).expand(n, 1), z_where), 1)
    ix = Variable(expansion_indices)
    if z_where.is_cuda:
        ix = ix.cuda()
    out = torch.index_select(out, 1, ix)
    out = out.view(n, 2, 3)
    return out

# An alternative to this would be to add the "missing" bottom row to
# theta, and then use `torch.inverse`.
def z_where_inv(z_where):
    # Take a batch of z_where vectors, and compute their "inverse".
    # That is, for each row compute:
    # [s,x,y] -> [1/s,-x/s,-y/s]
    # These are the parameters required to perform the inverse of the
    # spatial transform performed in the generative model.
    n = z_where.size(0)
    out = torch.cat((ng_ones([1, 1]).type_as(z_where).expand(n, 1), -z_where[:, 1:]), 1)
    # Divide all entries by the scale.
    out = out / z_where[:, 0:1]
    return out




def assert_size(t, expected_size):
    actual_size = t.size()
    assert actual_size == expected_size, 'Expected size {} but got {}.'.format(expected_size, tuple(actual_size))



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

def run_svi(X, args):
    vis = visdom.Visdom()
    progress_plot = Plot(visdom.Visdom(env='progress'))
    dynair = DynAIR(use_combiner_skip_conns=not args.no_skip,
                    use_transition_in_guide=not args.no_trans_in_guide,
                    use_linear_transition=args.use_linear_transition,
                    optimize_transition_sd=args.optimize_transition_sd,
                    use_cuda=args.cuda)

    batches = X.chunk(40)

    def per_param_optim_args(module_name, param_name, tags):
        return {'lr': 1e-1 if param_name == 'bkg_rgb' else 1e-4}

    svi = SVI(dynair.model, dynair.guide,
              #optim.Adam(dict(lr=1e-4)),
              optim.Adam(per_param_optim_args),
              loss='ELBO')
              # trace_graph=True) # No discrete things, yet.

    for i in range(1000):

        for j, batch in enumerate(batches):
            loss = svi.step(batch)
            elbo = -loss / (dynair.seq_length * batch.size(0)) # elbo per datum, per frame
            print('epoch={}, batch={}, elbo={:.2f}'.format(i, j, elbo))
            progress_plot.add(i*len(batches) + j, elbo)

        ix = 15
        n = 1

        if (i+1) % 1 == 0:

            # Show transition s.d.
            #print(dynair.transition.sd())

            # TODO: Make reconstruct method.
            trace = poutine.trace(dynair.guide).get_trace(X[ix:ix+n])
            frames, zs = poutine.replay(dynair.model, trace)(X[ix:ix+n], do_likelihood=False)

            frames = latent_seq_to_tensor(frames)
            zs = latent_seq_to_tensor(zs)

            for k in range(n):
                out = overlay_window_outlines(dynair, frames[k], zs[k, :, 0:2])
                vis.images(list(reversed(frames_to_rgb_list(X[ix+k].cpu()))), nrow=7)
                vis.images(frames_to_rgb_list(out.cpu()), nrow=7)


            # Test extrapolation.
            # TODO: Clean-up.
            ex = X[54:54+1]
            zs, y, w = dynair.guide(ex)
            bkg = dynair.model_generate_bkg(w)

            z = zs[-1]
            frames = []
            extrap_zs = []
            for t in range(14):
                #z = dynair.model_transition(14 + t, z)
                z, _ = dynair.transition(z)
                frame_mean = dynair.model_emission(w, y, z, bkg)
                frames.append(frame_mean)
                extrap_zs.append(z)
            extrap_frames = latent_seq_to_tensor(frames)
            extrap_zs = latent_seq_to_tensor(extrap_zs)
            out = overlay_window_outlines(dynair, extrap_frames[0], extrap_zs[0, :, 0:2])
            vis.images(frames_to_rgb_list(out.cpu()), nrow=7)

        # print(dynair.transition.lin.weight.data)

        if (i+1) % 50 == 0:
            print('Saving parameters...')
            torch.save(dynair.state_dict(), 'dynair6.pytorch')




def load_data():
    X_np = np.load('single_object_with_shade_and_bkg.npz')['X']
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
    if z_where.is_cuda:
        rect = rect.cuda()
    rect_batch = Variable(batch_expand(rect.contiguous().view(-1), n).contiguous())
    return dynair.window_to_image(z_where, rect_batch)

def overlay_window_outlines(dynair, frame, z_where):
    return over(draw_window_outline(dynair, z_where), frame)

def latent_seq_to_tensor(arr):
    # Turn an array of latents (of length seq_len) returned by the
    # model into a (batch, seq_len, rest...) tensor.
    return torch.cat([t.unsqueeze(0) for t in arr]).transpose(0, 1)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=False, help='Use CUDA')
    parser.add_argument('--no-skip', action='store_true', default=False, help='No skip connections in combiner net.')
    parser.add_argument('--no-trans-in-guide', action='store_true', default=False, help='Do not use model transition in guide.')
    parser.add_argument('--use-linear-transition', action='store_true', default=False, help='Use linear transition.')
    parser.add_argument('--optimize-transition-sd', action='store_true', default=False, help='Optimize transition sd.')
    args = parser.parse_args()
    print(args)


    # Test model by sampling:
    # dynair = DynAIR()
    # dummy_data = Variable(torch.ones(1, 14, 4, 32, 32))
    # frames, ws, zs = dynair.model(dummy_data, do_likelihood=False)
    # print(len(frames))
    # print(frames[0].size())

    # print(len(ws))
    # print(ws[0].size())

    # print(len(zs))
    # print(zs[0].size())
# print(zs)
    # from matplotlib import pyplot as plt
    # for frame in frames:
    #     print(frame[0].data.size())
    #     img = frame[0].data.numpy().transpose(1,2,0)
    #     plt.imshow(img)
    #     plt.show()
    # input('press a key...')

    # Test guide:
    #(batch, seq, channel, w, h)
    dynair = DynAIR()
    data = Variable(torch.ones(1, 14, 4, 32, 32))
    ws, zs = dynair.guide(data)

    print(torch.stack(ws))
    print(torch.stack(zs))

    # X = load_data()
    # if args.cuda:
    #     X = X.cuda()
    # run_svi(X, args)
