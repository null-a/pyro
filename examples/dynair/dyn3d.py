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

        self.y_size = 50
        self.z_size = 50
        self.w_size = 3 # (scale, x, y) = (softplus(w[0]), w[1], w[2])

        self.x_size = self.num_chan * self.image_size**2
        self.x_att_size = self.num_chan * self.window_size**2


        # bkg_rgb = self.ng_zeros(self.num_chan - 1, self.image_size, self.image_size)
        # bkg_alpha = self.ng_ones(1, self.image_size, self.image_size)
        # self.bkg = torch.cat((bkg_rgb, bkg_alpha))


        # Priors:

        self.y_prior_mean = self.ng_zeros(self.y_size)
        self.y_prior_sd = self.ng_ones(self.y_size)

        # TODO: Using a (reparameterized) uniform would probably be
        # better for the cubes data set.
        self.w_0_prior_mean = Variable(torch.Tensor([3, 0, 0]))
        self.w_0_prior_sd = Variable(torch.Tensor([0.8, 0.7, 0.7]),
                                     requires_grad=False)
        if use_cuda:
            self.w_0_prior_mean = self.w_0_prior_mean.cuda()
            self.w_0_prior_sd = self.w_0_prior_sd.cuda()


        self.z_0_prior_mean = self.ng_zeros(self.z_size)
        self.z_0_prior_sd = self.ng_ones(self.z_size)

        self.likelihood_sd = 0.3


        # Parameters.
        self.guide_z_init = nn.Parameter(torch.zeros(self.z_size))
        self.guide_w_init = nn.Parameter(torch.zeros(self.w_size))


        self.bkg_alpha = self.ng_ones(1, self.image_size, self.image_size)

        # Modules

        # TODO: Review all arch. of all modules. (Currently just using
        # MLP so that I have something to test.)

        # Guide modules:
        self.y_param = mod.ParamY([200, 200], self.x_size, self.y_size)
        self.z_param = mod.ParamZ([100, 100], [100], self.w_size, self.x_att_size, self.z_size)
        self.w_param = mod.ParamW([500, 200], [200], self.x_size, self.w_size, self.z_size)

        # Model modules:
        # TODO: Consider using init. that outputs black/transparent images.
        self.decode_obj = mod.DecodeObj([100, 100], self.z_size, self.num_chan, self.window_size)
        self.decode_bkg_rgb = mod.DecodeBkg([200, 200], self.y_size, self.num_chan, self.image_size)

        #self.w_transition = mod.WTransition(self.z_size, self.w_size, 50)
        self.w_emit = mod.EmitW(self.z_size, self.w_size, [50])

        self.z_transition = mod.ZTransition(self.z_size, 50)
        #self.z_transition = mod.ZGatedTransition(self.z_size, 50, 50)

        # CUDA
        if use_cuda:
            self.cuda()



    # TODO: This do_likelihood business is unpleasant.
    def model(self, batch, do_likelihood=True):

        batch_size = batch.size(0)

        y = self.model_sample_y(batch_size)
        bkg = self.decode_bkg(y)

        z = self.model_sample_z_0(batch_size)
        w, frame_mean = self.model_emission(0, z, bkg)

        zs = [z]
        ws = [w]
        frames = [frame_mean]

        if do_likelihood:
            self.likelihood(0, frame_mean, batch[:, 0])

        # TODO: iarange here (or somewhere)
        for t in range(1, self.seq_length):
            z = self.model_transition(t, z)
            w, frame_mean = self.model_emission(t, z, bkg)
            zs.append(z)
            ws.append(w)
            frames.append(frame_mean)
            if do_likelihood:
                self.likelihood(t, frame_mean, batch[:, t])

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

    def model_sample_y(self, batch_size):
        return pyro.sample('y',
                           dist.normal,
                           self.y_prior_mean.expand(batch_size, -1),
                           self.y_prior_sd.expand(batch_size, -1))

    def model_sample_w(self, t, w_mean, w_sd):
        return pyro.sample('w_{}'.format(t),
                           dist.normal,
                           w_mean,
                           w_sd)

    def model_sample_z_0(self, batch_size):
        return pyro.sample('z_0',
                           dist.normal,
                           self.z_0_prior_mean.expand(batch_size, -1),
                           self.z_0_prior_sd.expand(batch_size, -1))

    def model_sample_z(self, t, z_mean, z_sd):
        return pyro.sample('z_{}'.format(t),
                           dist.normal,
                           z_mean,
                           z_sd)

    def model_transition(self, t, z_prev):
        batch_size = z_prev.size(0)
        assert_size(z_prev, (batch_size, self.z_size))
        z_mean, z_sd = self.z_transition(z_prev)
        z = self.model_sample_z(t, z_mean, z_sd)
        return z

    def model_emission(self, t, z, bkg):
        batch_size = z.size(0)
        # Note that neither of these currently depend on w, but doing
        # so may be useful in future.
        w_mean, w_sd = self.w_emit(z)
        w = self.model_sample_w(t, w_mean, w_sd)
        x_att = self.decode_obj(z)
        x_mean = over(self.window_to_image(w, x_att), bkg)
        return w, x_mean

    def decode_bkg(self, y):
        batch_size = y.size(0)
        rgb = self.decode_bkg_rgb(y)
        alpha = batch_expand(self.bkg_alpha, batch_size)
        return torch.cat((rgb, alpha), 1)


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

        # NOTE: Here we're guiding y based on the contents of the
        # first frame only.
        # TODO: Implement a better guide for y.
        y = self.guide_y(batch[:, 0])

        zs = []
        ws = []

        z = batch_expand(self.guide_z_init, batch_size)
        #w = batch_expand(self.guide_w_init, batch_size)

        for t in range(self.seq_length):
            x = batch[:, t]
            w = self.guide_w(t, x, z)
            x_att = self.image_to_window(w, x)
            z = self.guide_z(t, w, x, x_att, z)

            ws.append(w)
            zs.append(z)

        return ws, zs, y

    def guide_y(self, x0):
        y_mean, y_sd = self.y_param(x0)
        return pyro.sample('y', dist.normal, y_mean, y_sd)

    def guide_w(self, t, batch, z_prev):
        w_mean, w_sd = self.w_param(batch, z_prev)
        return pyro.sample('w_{}'.format(t), dist.normal, w_mean, w_sd)

    def guide_z(self, t, w, x, x_att, z_prev):
        z_mean, z_sd = self.z_param(w, x_att, z_prev)
        return pyro.sample('z_{}'.format(t), dist.normal, z_mean, z_sd)


    def image_to_window(self, w, images):
        n = w.size(0)
        assert_size(w, (n, self.w_size))
        assert_size(images, (n, self.num_chan, self.image_size, self.image_size))
        z_where = w_to_z_where(w)
        theta_inv = expand_z_where(z_where_inv(z_where))
        grid = affine_grid(theta_inv, torch.Size((n, self.num_chan, self.window_size, self.window_size)))
        return grid_sample(images, grid).view(n, -1)

    def window_to_image(self, w, windows):
        n = w.size(0)
        assert_size(w, (n, self.w_size))
        assert_size(windows, (n, self.x_att_size))
        z_where = w_to_z_where(w)
        theta = expand_z_where(z_where)
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
        ws, zs, y = trace.nodes['_RETURN']['value']
        bkg = self.decode_bkg(y)

        extra_ws = []
        extra_zs = []
        extra_frames = []

        w = ws[-1]
        z = zs[-1]

        for t in range(num_extra_frames):
            z = self.model_transition(num_extra_frames + t, z)
            w, frame_mean = self.model_emission(num_extra_frames + t, z, bkg)
            extra_frames.append(frame_mean)
            extra_ws.append(w)
            extra_zs.append(z)

        return frames, ws, extra_frames, extra_ws


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

def w_to_z_where(w):
    # Unsquish the `scale` component of w.
    scale = softplus(w[:, 0:1])
    xy = w[:, 1:] * scale
    out = torch.cat((scale, xy), 1)
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

    batches = X.chunk(20)

    svi = SVI(dynair.model, dynair.guide,
              optim.Adam(dict(lr=1e-4)),
              loss='ELBO')
              # trace_graph=True) # No discrete things, yet.

    for i in range(5000):

        for j, batch in enumerate(batches):
            loss = svi.step(batch)
            nan_params = list(dynair.params_with_nan())
            assert len(nan_params) == 0, 'The following parameters include NaN:\n  {}'.format("\n  ".join(nan_params))
            elbo = -loss / (dynair.seq_length * batch.size(0)) # elbo per datum, per frame
            print('epoch={}, batch={}, elbo={:.2f}'.format(i, j, elbo))

        if (i+1) % 1 == 0:
            ix = 40
            n = 1
            test_batch = X[ix:ix+n]

            frames, ws, extra_frames, extra_ws = [latent_seq_to_tensor(x) for x in dynair.infer(test_batch, 15)]

            for k in range(n):
                out = overlay_window_outlines(dynair, frames[k], ws[k])
                vis.images(frames_to_rgb_list(test_batch[k].cpu()), nrow=10)
                vis.images(frames_to_rgb_list(out.cpu()), nrow=10)

                out = overlay_window_outlines(dynair, extra_frames[k], extra_ws[k])
                vis.images(frames_to_rgb_list(out.cpu()), nrow=10)

        if (i+1) % 50 == 0:
            print('Saving parameters...')
            torch.save(dynair.state_dict(), 'dyn3d.pytorch')




def load_data():
    X_np = np.load('cube.npz')['X']
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
