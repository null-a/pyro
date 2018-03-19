import torch
import torch.nn as nn
from torch.nn.functional import affine_grid, grid_sample, sigmoid, softplus
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

        self.seq_length = 20

        self.image_size = 50
        self.num_chan = 4

        self.window_size = 22

        self.i_size = 1
        self.y_size = 50
        self.z_size = 50
        self.w_size = 3 # (scale, x, y) = (softplus(w[0]), w[1], w[2])

        self.x_size = self.num_chan * self.image_size**2
        self.x_att_size = self.num_chan * self.window_size**2

        self.ctx_size = 50

        # bkg_rgb = self.ng_zeros(self.num_chan - 1, self.image_size, self.image_size)
        # bkg_alpha = self.ng_ones(1, self.image_size, self.image_size)
        # self.bkg = torch.cat((bkg_rgb, bkg_alpha))


        # Priors:
        self.create_prior_p = 0.5

        self.y_prior_mean = self.ng_zeros(self.y_size)
        self.y_prior_sd = self.ng_ones(self.y_size)

        # TODO: Using a (reparameterized) uniform would probably be
        # better for the cubes data set.
        self.w_prior_mean = Variable(torch.Tensor([np.log(0.3), 0, 0]))
        self.w_prior_sd = Variable(torch.Tensor([0.02, 0.3, 0.3]),
                                   requires_grad=False)
        if use_cuda:
            self.w_prior_mean = self.w_prior_mean.cuda()
            self.w_prior_sd = self.w_prior_sd.cuda()


        self.z_prior_mean = self.ng_zeros(self.z_size)
        self.z_prior_sd = self.ng_ones(self.z_size)

        self.likelihood_sd = 0.3


        # Parameters.
        self.ctx_init = nn.Parameter(torch.zeros(self.ctx_size))
        nn.init.normal(self.ctx_init, std=0.01)

        # self.guide_z_init = nn.Parameter(torch.zeros(self.z_size))

        self.bkg_alpha = self.ng_ones(1, self.image_size, self.image_size)

        # Modules

        # TODO: Review all arch. of all modules. (Currently just using
        # MLP so that I have something to test.)

        # Guide modules:
        self.y_param = mod.ParamY([200, 200], self.x_size, self.y_size)
        self.z_param = mod.ParamZ([100, 100], [100], self.w_size, self.x_att_size, self.ctx_size, self.z_size)
        self.iw_param = mod.ParamIW([500, 200], [200], self.x_size, self.ctx_size, self.i_size, self.w_size)

        self.update_ctx = mod.UpdateCtx([100, 100], self.ctx_size, self.i_size, self.w_size, self.z_size)

        self.baseline = mod.Baseline(self.seq_length)

        # Model modules:
        # TODO: Consider using init. that outputs black/transparent images.
        self.decode_obj = mod.DecodeObj([100, 100], self.z_size, self.num_chan, self.window_size, -3.0)
        self.decode_bkg_rgb = mod.DecodeBkg([200, 200], self.y_size, self.num_chan, self.image_size)

        self.i_transition = mod.ITransition(self.i_size, self.w_size, self.z_size, 50)
        self.w_transition = mod.WTransition(self.z_size, self.w_size, 50)
        self.z_transition = mod.ZTransition(self.z_size, 50)
        #self.z_transition = mod.ZGatedTransition(self.z_size, 50, 50)

        # CUDA
        if use_cuda:
            self.cuda()



    # TODO: This do_likelihood business is unpleasant.
    def model(self, batch, do_likelihood=True, **kwargs):

        batch_size = batch.size(0)

        with pyro.iarange('data'):

            y = self.model_sample_y(batch_size)
            bkg = self.decode_bkg(y)

            zs = []
            ws = []
            frames = []

            # Dummy values for z,w to which the transition can be applied.
            # (The result of which will be discarded.)
            z_prev = self.ng_zeros(batch_size, self.z_size)
            w_prev = self.ng_zeros(batch_size, self.w_size)

            i_prev = self.ng_zeros(batch_size, self.i_size)

            for t in range(0, self.seq_length):
                i = self.model_sample_i(t, i_prev, w_prev, z_prev)

                with poutine.scale(None, i.squeeze(-1)):
                    z, w = self.model_transition(t, i, i_prev, z_prev, w_prev)

                frame_mean = self.model_emission(i, z, w, bkg)

                zs.append(z)
                ws.append(w)
                frames.append(frame_mean)

                if do_likelihood:
                    self.likelihood(t, frame_mean, batch[:, t])

                z_prev, w_prev, i_prev = z, w, i

        return frames, ws, zs

    def likelihood(self, t, frame_mean, obs):
        frame_sd = (self.likelihood_sd * self.ng_ones(1)).expand_as(frame_mean)
        # TODO: Using a normal here isn't very sensible since the data
        # is in [0, 1]. Do something better.
        pyro.sample('x_{}'.format(t),
                    dist.Normal(frame_mean, frame_sd, extra_event_dims=3),
                    obs=obs)

    def model_sample_i(self, t, i_prev, w_prev, z_prev):
        if self.is_i_step(t):
            persist_p = self.i_transition(w_prev, z_prev)
            ps = _if(i_prev, persist_p, self.create_prior_p)
            return pyro.sample('i_{}'.format(t), dist.Bernoulli(ps, extra_event_dims=1))
        else:
            return i_prev

    def model_sample_y(self, batch_size):
        return pyro.sample('y', dist.Normal(self.y_prior_mean.expand(batch_size, -1),
                                            self.y_prior_sd.expand(batch_size, -1),
                                            extra_event_dims=1))

    def model_sample_w(self, t, w_mean, w_sd):
        return pyro.sample('w_{}'.format(t),
                           dist.Normal(w_mean, w_sd, extra_event_dims=1))

    def model_sample_z(self, t, z_mean, z_sd):
        return pyro.sample('z_{}'.format(t),
                           dist.Normal(z_mean, z_sd, extra_event_dims=1))

    def model_transition(self, t, i, i_prev, z_prev, w_prev):
        batch_size = i.size(0)
        assert_size(i, (batch_size, self.i_size))
        assert_size(z_prev, (batch_size, self.z_size))
        assert_size(w_prev, (batch_size, self.w_size))

        # TODO: Possible optimization -- avoid applying the transition
        # when no objects are present. (e.g. At the first time step.)
        # Better yet, only apply the transition to present objects.
        # (Would slicing tensors negate any savings?) Something
        # similar applies to model_emission/spatial transformer stuff.
        z_transition_mean, z_transition_sd = self.z_transition(z_prev)
        w_transition_mean, w_transition_sd = self.w_transition(z_prev, w_prev)

        z_mean = _if(i_prev, z_transition_mean, self.z_prior_mean)
        z_sd = _if(i_prev, z_transition_sd, self.z_prior_sd)
        w_mean = _if(i_prev, w_transition_mean, self.w_prior_mean)
        w_sd = _if(i_prev, w_transition_sd, self.w_prior_sd)

        z = self.model_sample_z(t, z_mean, z_sd)
        w = self.model_sample_w(t, w_mean, w_sd)

        return z, w

    def model_emission(self, i, z, w, bkg):
        batch_size = z.size(0)
        assert z.size(0) == w.size(0)
        # Note that neither of these currently depend on w, but doing
        # so may be useful in future.

        # Zero out the contents of windows when the object is not present.
        x_att = self.decode_obj(z) * i
        return over(self.window_to_image(w, x_att), bkg)

    def decode_bkg(self, y):
        batch_size = y.size(0)
        rgb = self.decode_bkg_rgb(y)
        alpha = batch_expand(self.bkg_alpha, batch_size)
        return torch.cat((rgb, alpha), 1)


    # ==GUIDE==================================================

    def guide(self, batch, **kwargs):

        # I'd rather register model/guide modules in their methods,
        # but this is easier.
        pyro.module('dynair', self)
        # for name, _ in self.named_parameters():
        #     print(name)

        batch_size = batch.size(0)
        assert_size(batch, (batch_size, self.seq_length, self.num_chan, self.image_size, self.image_size))

        with pyro.iarange('data'):

            # NOTE: Here we're guiding y based on the contents of the
            # first frame only.
            # TODO: Implement a better guide for y.
            y = self.guide_y(batch[:, 0])

            ii = []
            zs = []
            ws = []

            i_prev = self.ng_zeros(batch_size, self.i_size)

            ctx = batch_expand(self.ctx_init, batch_size)

            for t in range(self.seq_length):

                # This is a bit odd -- we always compute i_ps, and
                # then ignore it when not sampling i for the current
                # step.
                x = batch[:, t]
                i_ps, w_mean, w_sd = self.iw_param(x, ctx)

                i = self.guide_i(t, i_ps, i_prev)

                with poutine.scale(None, i.squeeze(-1)):
                    w = self.guide_w(t, w_mean, w_sd)
                    x_att = self.image_to_window(w, x)
                    z = self.guide_z(t, i, w, x_att, ctx)

                # TODO: Are we throwing useful information away by
                # only incorporating sampled values (and not the
                # computed parameters) into the context?
                ctx = self.update_ctx(ctx, i, w, z)

                ii.append(i)
                ws.append(w)
                zs.append(z)

                i_prev = i

        return ws, zs, y, ii

    def guide_y(self, x0):
        y_mean, y_sd = self.y_param(x0)
        return pyro.sample('y', dist.Normal(y_mean, y_sd, extra_event_dims=1))

    def guide_i(self, t, ps, i_prev):
        batch_size = ps.size(0)

        if self.is_i_step(t):
            baseline = self.baseline(t)
            return pyro.sample('i_{}'.format(t),
                               dist.Bernoulli(ps, extra_event_dims=1),
                               baseline=dict(baseline_value=batch_expand(baseline, batch_size).squeeze(-1)))
        else:
            return i_prev

    def guide_w(self, t, w_mean, w_sd):
        return pyro.sample('w_{}'.format(t), dist.Normal(w_mean, w_sd, extra_event_dims=1))

    def guide_z(self, t, i, w, x_att, ctx):
        batch_size = i.size(0)
        z_mean, z_sd = self.z_param(w, x_att, ctx)
        return pyro.sample('z_{}'.format(t), dist.Normal(z_mean, z_sd, extra_event_dims=1))


    def image_to_window(self, w, images):
        n = w.size(0)
        assert_size(w, (n, self.w_size))
        assert_size(images, (n, self.num_chan, self.image_size, self.image_size))
        theta_inv = expand_theta(w_to_theta_inv(w))
        grid = affine_grid(theta_inv, torch.Size((n, self.num_chan, self.window_size, self.window_size)))
        return grid_sample(images, grid).view(n, -1)

    def window_to_image(self, w, windows):
        n = w.size(0)
        assert_size(w, (n, self.w_size))
        assert_size(windows, (n, self.x_att_size))
        theta = expand_theta(w_to_theta(w))
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

    def params_with_nan(self):
        return (name for (name, param) in self.named_parameters() if np.isnan(param.data.view(-1)[0]))

    def infer(self, batch, num_extra_frames=0):
        trace = poutine.trace(self.guide).get_trace(batch)
        frames, _, _ = poutine.replay(self.model, trace)(batch, do_likelihood=False)
        ws, zs, y, ii = trace.nodes['_RETURN']['value']
        bkg = self.decode_bkg(y)

        extra_ws = []
        extra_zs = []
        extra_ii = []
        extra_frames = []

        w_prev = ws[-1]
        z_prev = zs[-1]
        i_prev = ii[-1]

        for t in range(num_extra_frames):
            i = self.model_sample_i(num_extra_frames + t, i_prev, w_prev, z_prev)
            z, w = self.model_transition(num_extra_frames + t, i, i_prev, z_prev, w_prev)
            frame_mean = self.model_emission(i, z, w, bkg)
            extra_frames.append(frame_mean)
            extra_ii.append(i)
            extra_ws.append(w)
            extra_zs.append(z)
            w_prev, z_prev, i_prev = w, z, i

        return frames, ws, ii, extra_frames, extra_ws, extra_ii

    def is_i_step(self, t):
        # Controls when i is sampled.
        #return (t % 4 == 0) or (t >= (self.seq_length-2))
        return True


def _if(cond, cons, alt):
    return cond * cons + (1 - cond) * alt

def batch_expand(t, b):
    return t.expand((b,) + t.size())


# Spatial transformer helpers. (Taken from the AIR example.)

expansion_indices = torch.LongTensor([1, 0, 2, 0, 1, 3])

def expand_theta(theta):
    # Take a batch of three-vectors, and massages them into a batch of
    # 2x3 matrices with elements like so:
    # [s,x,y] -> [[s,0,x],
    #             [0,s,y]]
    n = theta.size(0)
    assert_size(theta, (n, 3))
    out = torch.cat((ng_zeros([1, 1]).type_as(theta).expand(n, 1), theta), 1)
    ix = Variable(expansion_indices)
    if theta.is_cuda:
        ix = ix.cuda()
    out = torch.index_select(out, 1, ix)
    out = out.view(n, 2, 3)
    return out

def w_to_theta(w):
    # Takes w = (log(scale), pos_x, pos_y) to parameter theta of the
    # spatial transform.
    scale = torch.exp(w[:, 0:1])
    scale_inv = 1 / scale
    xy = -w[:, 1:] * scale_inv
    out = torch.cat((scale_inv, xy), 1)
    return out

def w_to_theta_inv(w):
    # Takes w to theta inverse.
    scale = torch.exp(w[:, 0:1])
    xy = w[:, 1:]
    out = torch.cat((scale, xy), 1)
    return out



def assert_size(t, expected_size):
    actual_size = t.size()
    assert actual_size == expected_size, 'Expected size {} but got {}.'.format(expected_size, tuple(actual_size))


# TODO: I suspect this can be simplified if the background is always
# opaque and we always composite an object on to the image so far.
# image_so_far will always have opacity=1 I think, so we can probably
# avoid representing that explicity and simplify the over computation.
# We might also be able to drop the alpha channel from everywhere
# except the window contents, adopting the convention that alpha=1
# where omitted.

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



def run_svi(X, args):
    vis = visdom.Visdom()

    dynair = DynAIR(use_cuda=args.cuda)

    # Don't train on the last batch.
    num_batches = 39
    all_batches = X.chunk(num_batches + 1)
    batches = all_batches[0:-1]

    def per_param_optim_args(module_name, param_name, tags):
        lr = 1e2 if param_name.startswith('baseline') else 1e-4
        return {'lr': lr}

    svi = SVI(dynair.model, dynair.guide,
              optim.Adam(per_param_optim_args),
              loss='ELBO',
              trace_graph=True)

    for i in range(5000):

        for j, batch in enumerate(batches):
            loss = svi.step(batch)
            nan_params = list(dynair.params_with_nan())
            assert len(nan_params) == 0, 'The following parameters include NaN:\n  {}'.format("\n  ".join(nan_params))
            elbo = -loss / (dynair.seq_length * batch.size(0)) # elbo per datum, per frame
            print('\33[2K\repoch={}, batch={}, elbo={:.2f}'.format(i, j, elbo), end='')

        if (i+1) % 1 == 0:
            ix = 40
            # Produce visualization for train & test data points.
            test_batch = torch.cat((X[ix:ix+1], all_batches[-1][0:1]))
            n = test_batch.size(0)

            frames, ws, ii, extra_frames, extra_ws, extra_ii = [latent_seq_to_tensor(x) for x in dynair.infer(test_batch, 15)]

            for k in range(n):
                out = overlay_window_outlines_conditionally(dynair, frames[k], ws[k], ii[k])
                vis.images(frames_to_rgb_list(test_batch[k].cpu()), nrow=10)
                vis.images(frames_to_rgb_list(out.cpu()), nrow=10)

                out = overlay_window_outlines_conditionally(dynair, extra_frames[k], extra_ws[k], extra_ii[k])
                vis.images(frames_to_rgb_list(out.cpu()), nrow=10)

        if (i+1) % 50 == 0:
            #print('Saving parameters...')
            torch.save(dynair.state_dict(), 'dyn3d.pytorch')

    print('\nDone')


def load_data():
    X_np = np.load('cube2.npz')['X']
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

def overlay_window_outlines_conditionally(dynair, frame, z_where, ii):
    batch_size = z_where.size(0)
    presence_mask = ii.view(-1, 1, 1, 1)
    borders = batch_expand(Variable(torch.Tensor([-0.08, 0, 0])), batch_size).type_as(ii)
    return over(draw_window_outline(dynair, borders) * presence_mask,
                over(draw_window_outline(dynair, z_where) * presence_mask,
                     frame))


def latent_seq_to_tensor(arr):
    # Turn an array of latents (of length seq_len) returned by the
    # model into a (batch, seq_len, rest...) tensor.
    return torch.cat([t.unsqueeze(0) for t in arr]).transpose(0, 1)

def arr_to_img(nparr):
    assert nparr.shape[0] == 4 # required for RGBA
    shape = nparr.shape[1:]
    return Image.frombuffer('RGBA', shape, (nparr.transpose((1,2,0)) * 255).astype(np.uint8).tostring(), 'raw', 'RGBA', 0, 1)

def save_frames(frames, path_fmt_str, offset=0):
    n = frames.shape[0]
    assert_size(frames, (n, 4, frames.size(2), frames.size(3)))
    for i in range(n):
        arr_to_img(frames[i].data.numpy()).save(path_fmt_str.format(i + offset))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=False, help='Use CUDA')
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
    # # print(zs)
    # from matplotlib import pyplot as plt
    # for frame in frames:
    #     print(frame[0].data.size())
    #     img = frame[0].data.numpy().transpose(1,2,0)
    #     plt.imshow(img)
    #     plt.show()
    # input('press a key...')

    # Test guide:
    #(batch, seq, channel, w, h)
    # dynair = DynAIR()
    # data = Variable(torch.ones(1, 14, 4, 32, 32))
    # ws, zs, y = dynair.guide(data)

    # # print(torch.stack(ws))
    # # print(torch.stack(zs))
    # print(y)

    X = load_data()
    if args.cuda:
        X = X.cuda()
    run_svi(X, args)

    # ====================
    # Code to visualise latent variables and extrapolated frames.
    #
    # These can be turned into movies with something like this:
    # ffmpeg -framerate 10 -i frame_%2d.png -r 25 out.mpg
    #
    # dynair = DynAIR(use_cuda=args.cuda)
    # dynair.load_state_dict(torch.load('dyn3d.pytorch', map_location=lambda storage, loc: storage))

    # vis = visdom.Visdom()
    # ix = 40
    # test_batch = X[ix:ix+1]

    # frames, ws, extra_frames, extra_ws = [latent_seq_to_tensor(x) for x in dynair.infer(test_batch, 15)]

    # #out = frames[0]
    # out = overlay_window_outlines(dynair, frames[0], ws[0])
    # vis.images(frames_to_rgb_list(test_batch[0].cpu()), nrow=10)
    # vis.images(frames_to_rgb_list(out.cpu()), nrow=10)
    # #save_frames(out, 'path/to/frame_{:02d}.png')

    # out = overlay_window_outlines(dynair, extra_frames[0], extra_ws[0])
    # vis.images(frames_to_rgb_list(out.cpu()), nrow=10)
    # #save_frames(out, 'path/to/frame_{:02d}.png', offset=20)
