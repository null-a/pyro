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

import json
import argparse
from collections import defaultdict
import time
from datetime import timedelta

class DynAIR(nn.Module):
    def __init__(self, use_cuda=False):

        super(DynAIR, self).__init__()

        # Set here so that self.ng_ones/self.ng_zeros does the right
        # thing.
        self.use_cuda = use_cuda

        self.seq_length = 20

        self.max_obj_count = 3

        self.image_size = 50
        self.num_chan = 4

        self.window_size = 22

        self.y_size = 50
        self.z_size = 50
        self.w_size = 3 # (scale, x, y) = (softplus(w[0]), w[1], w[2])

        self.x_size = self.num_chan * self.image_size**2
        self.x_att_size = self.num_chan * self.window_size**2

        self.x_embed_size = 800

        # bkg_rgb = self.ng_zeros(self.num_chan - 1, self.image_size, self.image_size)
        # bkg_alpha = self.ng_ones(1, self.image_size, self.image_size)
        # self.bkg = torch.cat((bkg_rgb, bkg_alpha))


        # Priors:

        self.y_prior_mean = self.ng_zeros(self.y_size)
        self.y_prior_sd = self.ng_ones(self.y_size)

        # TODO: Using a (reparameterized) uniform would probably be
        # better for the cubes data set. (Though this would makes less
        # sense if we allowed objects to begin off screen.)

        # TODO: This is probably too extreme. Adjusted prior scale
        # from {mean=log(0.3), sd=0.7} to this while attempting to
        # make optimising for multiple objects work.
        self.w_0_prior_mean = Variable(torch.Tensor([np.log(0.1), 0, 0]))
        self.w_0_prior_sd = Variable(torch.Tensor([0.05, 0.7, 0.7]),
                                     requires_grad=False)
        if use_cuda:
            self.w_0_prior_mean = self.w_0_prior_mean.cuda()
            self.w_0_prior_sd = self.w_0_prior_sd.cuda()


        self.z_0_prior_mean = self.ng_zeros(self.z_size)
        self.z_0_prior_sd = self.ng_ones(self.z_size)

        self.likelihood_sd = 0.3


        # Parameters.
        self.guide_w_t_init = nn.Parameter(torch.zeros(self.w_size))
        self.guide_w_init = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.w_size)) for _ in range(self.max_obj_count)])
        self.guide_z_init = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.z_size)) for _ in range(self.max_obj_count)])


        self.bkg_alpha = self.ng_ones(1, self.image_size, self.image_size)

        # Modules

        # TODO: Review all arch. of all modules. (Currently just using
        # MLP so that I have something to test.)

        # Guide modules:
        self.y_param = mod.ParamY([200, 200], self.x_size, self.y_size)
        self.z_param = mod.ParamZ([100, 100], [100], self.w_size, self.x_att_size, self.z_size)
        self.w_param = mod.ParamW(200, [], self.x_embed_size, self.w_size, self.z_size)
        self.x_embed = mod.EmbedX([800], self.x_embed_size, self.x_size)


        # Model modules:
        # TODO: Consider using init. that outputs black/transparent images.
        self.decode_obj = mod.DecodeObj([100, 100], self.z_size, self.num_chan, self.window_size, alpha_bias=-2.0)
        self.decode_bkg_rgb = mod.DecodeBkg([200, 200], self.y_size, self.num_chan, self.image_size)

        self.w_transition = mod.WTransition(self.z_size, self.w_size, 50)
        self.z_transition = mod.ZTransition(self.z_size, 50)
        #self.z_transition = mod.ZGatedTransition(self.z_size, 50, 50)

        # CUDA
        if use_cuda:
            self.cuda()



    # TODO: This do_likelihood business is unpleasant.
    def model(self, batch, obj_counts, do_likelihood=True):
        batch_size = batch.size(0)
        assert_size(obj_counts, (batch_size,))
        assert all(0 <= obj_counts) and all(obj_counts <= self.max_obj_count), 'Object count out of range.'

        frames = []
        zs = mk_matrix(self.seq_length, self.max_obj_count)
        ws = mk_matrix(self.seq_length, self.max_obj_count)

        with pyro.iarange('data'):

            y = self.model_sample_y(batch_size)
            bkg = self.decode_bkg(y)

            for t in range(self.seq_length):

                # To begin with, we'll sample max_obj_count objects for
                # all sequences, and throw out the extra objects. We can
                # consider refining this to avoid this unnecessary
                # sampling later.

                for i in range(self.max_obj_count):

                    mask = Variable((obj_counts > i).float())

                    with poutine.scale(None, mask):
                        if t > 0:
                            z, w = self.model_transition(t, i, zs[t-1][i], ws[t-1][i])
                        else:
                            z = self.model_sample_z_0(i, batch_size)
                            w = self.model_sample_w_0(i, batch_size)

                    # Zero out unused samples here. This isn't necessary
                    # for correctness, but might help spot any mistakes
                    # elsewhere.
                    zs[t][i] = z * mask.view(-1, 1)
                    ws[t][i] = w * mask.view(-1, 1)

                frame_mean = self.model_emission(zs[t], ws[t], bkg, obj_counts)
                frames.append(frame_mean)

                if do_likelihood:
                    self.likelihood(t, frame_mean, batch[:, t])

        return frames, ws, zs

    def likelihood(self, t, frame_mean, obs):
        frame_sd = (self.likelihood_sd * self.ng_ones(1)).expand_as(frame_mean)
        # TODO: Using a normal here isn't very sensible since the data
        # is in [0, 1]. Do something better.
        pyro.sample('x_{}'.format(t),
                    dist.Normal(frame_mean, frame_sd).reshape(extra_event_dims=3),
                    obs=obs)

    def model_sample_y(self, batch_size):
        return pyro.sample('y', dist.Normal(self.y_prior_mean.expand(batch_size, -1),
                                            self.y_prior_sd.expand(batch_size, -1))
                                .reshape(extra_event_dims=1))

    def model_sample_w_0(self, i, batch_size):
        return pyro.sample('w_0_{}'.format(i),
                           dist.Normal(
                               self.w_0_prior_mean.expand(batch_size, -1),
                               self.w_0_prior_sd.expand(batch_size, -1))
                           .reshape(extra_event_dims=1))

    def model_sample_w(self, t, i, w_mean, w_sd):
        return pyro.sample('w_{}_{}'.format(t, i),
                           dist.Normal(w_mean, w_sd).reshape(extra_event_dims=1))

    def model_sample_z_0(self, i, batch_size):
        return pyro.sample('z_0_{}'.format(i),
                           dist.Normal(
                               self.z_0_prior_mean.expand(batch_size, -1),
                               self.z_0_prior_sd.expand(batch_size, -1))
                           .reshape(extra_event_dims=1))

    def model_sample_z(self, t, i, z_mean, z_sd):
        return pyro.sample('z_{}_{}'.format(t, i),
                           dist.Normal(z_mean, z_sd).reshape(extra_event_dims=1))

    def model_transition(self, t, i, z_prev, w_prev):
        batch_size = z_prev.size(0)
        assert_size(z_prev, (batch_size, self.z_size))
        assert_size(w_prev, (batch_size, self.w_size))
        z_mean, z_sd = self.z_transition(z_prev)
        w_mean, w_sd = self.w_transition(z_prev, w_prev)
        z = self.model_sample_z(t, i, z_mean, z_sd)
        w = self.model_sample_w(t, i, w_mean, w_sd)
        return z, w

    def model_emission(self, zs, ws, bkg, obj_counts):
        #batch_size = z.size(0)
        #assert z.size(0) == w.size(0)
        acc = bkg
        for i, (z, w) in enumerate(zip(zs, ws)):
            assert z.size(0) == w.size(0)
            mask = Variable((obj_counts > i).float())
            x_att = self.decode_obj(z) * mask.view(-1, 1)
            acc = over(self.window_to_image(w, x_att), acc)
        return acc

    def decode_bkg(self, y):
        batch_size = y.size(0)
        rgb = self.decode_bkg_rgb(y)
        alpha = batch_expand(self.bkg_alpha, batch_size)
        return torch.cat((rgb, alpha), 1)


    # ==GUIDE==================================================

    def guide(self, batch, obj_counts, *args):

        # I'd rather register model/guide modules in their methods,
        # but this is easier.
        pyro.module('dynair', self)
        # for name, _ in self.named_parameters():
        #     print(name)

        batch_size = batch.size(0)
        assert_size(batch, (batch_size, self.seq_length, self.num_chan, self.image_size, self.image_size))

        assert_size(obj_counts, (batch_size,))
        assert all(0 <= obj_counts) and all(obj_counts <= self.max_obj_count), 'Object count out of range.'

        ws = mk_matrix(self.seq_length, self.max_obj_count)
        zs = mk_matrix(self.seq_length, self.max_obj_count)

        w_init = [batch_expand(w_init_i, batch_size) for w_init_i in self.guide_w_init]
        z_init = [batch_expand(z_init_i, batch_size) for z_init_i in self.guide_z_init]
        w_t_init = batch_expand(self.guide_w_t_init, batch_size)

        with pyro.iarange('data'):

            # NOTE: Here we're guiding y based on the contents of the
            # first frame only.
            # TODO: Implement a better guide for y.
            y = self.guide_y(batch[:, 0])

            for t in range(self.seq_length):

                x = batch[:, t]
                x_embed = self.x_embed(x)
                rnn_hid_prev = None

                # As in the model, we'll sample max_obj_count objects for
                # all sequences, and ignore the ones we don't need.

                for i in range(self.max_obj_count):

                    w_prev_i = ws[t-1][i] if t > 0 else w_init[i]
                    z_prev_i = zs[t-1][i] if t > 0 else z_init[i]
                    w_t_prev = ws[t][i-1] if i > 0 else w_t_init

                    mask = Variable((obj_counts > i).float())

                    with poutine.scale(None, mask):
                        w, rnn_hid = self.guide_w(t, i, x_embed, w_prev_i, z_prev_i, w_t_prev, rnn_hid_prev)
                        x_att = self.image_to_window(w, x)
                        z = self.guide_z(t, i, w, x_att, z_prev_i)

                    # Zero out unused samples here. This isn't necessary
                    # for correctness, but might help spot any mistakes
                    # elsewhere.
                    ws[t][i] = w * mask.view(-1, 1)
                    zs[t][i] = z * mask.view(-1, 1)

                    rnn_hid_prev = rnn_hid

        return ws, zs, y

    def guide_y(self, x0):
        y_mean, y_sd = self.y_param(x0)
        return pyro.sample('y', dist.Normal(y_mean, y_sd).reshape(extra_event_dims=1))

    def guide_w(self, t, i, x_embed, w_prev_i, z_prev_i, w_t_prev, rnn_hid_prev):
        w_mean, w_sd, rnn_hid = self.w_param(x_embed, w_prev_i, z_prev_i, w_t_prev, rnn_hid_prev)
        w = pyro.sample('w_{}_{}'.format(t, i), dist.Normal(w_mean, w_sd).reshape(extra_event_dims=1))
        return w, rnn_hid

    def guide_z(self, t, i, w, x_att, z_prev_i):
        z_mean, z_sd = self.z_param(w, x_att, z_prev_i)
        return pyro.sample('z_{}_{}'.format(t, i), dist.Normal(z_mean, z_sd).reshape(extra_event_dims=1))

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

    def infer(self, batch, obj_counts, num_extra_frames=0):
        trace = poutine.trace(self.guide).get_trace(batch, obj_counts)
        frames, _, _ = poutine.replay(self.model, trace)(batch, obj_counts, do_likelihood=False)
        ws, zs, y = trace.nodes['_RETURN']['value']

        # TODO: Reinstate extrapolation code. (Re-use model code.)

        #bkg = self.decode_bkg(y)

        # extra_ws = []
        # extra_zs = []
        # extra_frames = []

        # w = ws[-1]
        # z = zs[-1]

        # for t in range(num_extra_frames):
        #     z, w = self.model_transition(num_extra_frames + t, z, w)
        #     frame_mean = self.model_emission(z, w, bkg)
        #     extra_frames.append(frame_mean)
        #     extra_ws.append(w)
        #     extra_zs.append(z)

        extra_frames = None
        extra_ws = None

        return frames, ws, extra_frames, extra_ws


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



def split(t, batch_size, num_train_batches, num_test_batches):
    n = t.size(0)
    assert batch_size > 0
    assert num_train_batches >= 0
    assert num_test_batches >= 0
    assert n % batch_size == 0
    assert batch_size * (num_train_batches + num_test_batches) == n
    batches = t.chunk(n // batch_size)
    train = batches[0:num_train_batches]
    test = batches[num_train_batches:(num_train_batches+num_test_batches)]
    return train, test


# Borrowed from:
# https://github.com/uber/pyro/blob/5b67518dc1ded8aac59b6dfc51d0892223e8faad/tutorial/source/gmm.ipynb
def add_grad_hooks(module):
    norms = defaultdict(list)
    def hook(name, grad):
        norm = grad.norm().item()
        # print(name)
        # print(norm)
        norms[name].append(norm)
    for name, value in module.named_parameters():
        value.register_hook(lambda grad, name=name: hook(name, grad))
    return norms


def run_svi(data, args):
    t0 = time.time()
    vis = visdom.Visdom()
    dynair = DynAIR(use_cuda=args.cuda)
    norms = add_grad_hooks(dynair)

    X, Y = data # (sequences, counts)
    batch_size = 25
    X_train, X_test = split(X, batch_size, 39, 1)
    Y_train, Y_test = split(Y, batch_size, 39, 1)

    def per_param_optim_args(module_name, param_name, tags):
        return {'lr': 1e-4}

    svi = SVI(dynair.model, dynair.guide,
              #optim.Adam(per_param_optim_args),
              optim.ClippedAdam({'lr': 1e-4, 'clip_norm': 100}), # This is applied pointwise.
              loss='ELBO',
              trace_graph=False) # We don't have discrete choices.

    for i in range(10**6):

        for j, (X_batch, Y_batch) in enumerate(zip(X_train, Y_train)):
            loss = svi.step(X_batch, Y_batch)
            nan_params = list(dynair.params_with_nan())
            assert len(nan_params) == 0, 'The following parameters include NaN:\n  {}'.format("\n  ".join(nan_params))
            elbo = -loss / (dynair.seq_length * batch_size) # elbo per datum, per frame
            elapsed = str(timedelta(seconds=time.time()- t0))
            print('\33[2K\repoch={}, batch={}, elbo={:.2f}, elapsed={}'.format(i, j, elbo, elapsed), end='')

        if i < 50 or (i+1) % 50 == 0:
            ix = 40
            # Produce visualization for train & test data points.
            X_vis = torch.cat((X[ix:ix+1], X_test[0][0:1]))
            Y_vis = torch.cat((Y[ix:ix+1], Y_test[0][0:1]))
            n = X_vis.size(0)

            frames, ws, extra_frames, extra_ws = dynair.infer(X_vis, Y_vis, 15)
            frames = frames_to_tensor(frames)
            ws = latents_to_tensor(ws)

            for k in range(n):
                out = overlay_multiple_window_outlines(dynair, frames[k], ws[k], Y_vis[k])
                vis.images(frames_to_rgb_list(X_vis[k].cpu()), nrow=10)
                vis.images(frames_to_rgb_list(out.cpu()), nrow=10)

                # out = overlay_window_outlines(dynair, extra_frames[k], extra_ws[k])
                # vis.images(frames_to_rgb_list(out.cpu()), nrow=10)

        if (i+1) % 50 == 0:
            #print('Saving parameters...')
            torch.save(dynair.state_dict(), 'dyn3d.pytorch')

        # Write grad norms to disk.
        with open('grad_norms.json', 'w') as f:
            json.dump(norms, f)





def load_data(use_cuda):
    data = np.load('./data/multi_obj.npz')
    X_np = data['X']
    # print(X_np.shape)
    X_np = X_np.astype(np.float32)
    X_np /= 255.0
    X = Variable(torch.from_numpy(X_np))
    Y = torch.from_numpy(data['Y'].astype(np.uint8))
    if use_cuda:
        X = X.cuda()
        Y = Y.cuda()
    return X, Y

def frames_to_rgb_list(frames):
    return frames[:, 0:3].data.numpy().tolist()

def img_to_arr(img):
    assert img.mode == 'RGBA'
    channels = 4
    w, h = img.size
    arr = np.fromstring(img.tobytes(), dtype=np.uint8)
    return arr.reshape(w * h, channels).T.reshape(channels, h, w)

def draw_rect(size, color):
    img = Image.new('RGBA', (size, size))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, size - 1, size - 1], outline=color)
    return torch.from_numpy(img_to_arr(img).astype(np.float32) / 255.0)

def draw_window_outline(dynair, z_where, color):
    n = z_where.size(0)
    rect = draw_rect(dynair.window_size, color)
    if z_where.is_cuda:
        rect = rect.cuda()
    rect_batch = Variable(batch_expand(rect.contiguous().view(-1), n).contiguous())
    return dynair.window_to_image(z_where, rect_batch)

def overlay_window_outlines(dynair, frame, z_where, color):
    return over(draw_window_outline(dynair, z_where, color), frame)

def overlay_window_outlines_conditionally(dynair, frame, z_where, color, ii):
    batch_size = z_where.size(0)
    presence_mask = ii.view(-1, 1, 1, 1)
    borders = batch_expand(Variable(torch.Tensor([-0.08, 0, 0])), batch_size).type_as(ii)
    return over(draw_window_outline(dynair, borders, color) * presence_mask,
                over(draw_window_outline(dynair, z_where, color) * presence_mask,
                     frame))

def overlay_multiple_window_outlines(dynair, frame, ws, obj_count):
    acc = frame
    for i in range(obj_count):
        acc = overlay_window_outlines(dynair, acc, ws[i], ['red', 'green', 'blue'][i % 3])
    return acc

def frames_to_tensor(arr):
    # Turn an array of frames (of length seq_len) returned by the
    # model into a (batch, seq_len, rest...) tensor.
    return torch.cat([t.unsqueeze(0) for t in arr]).transpose(0, 1)

def latents_to_tensor(xss):
    return torch.stack([torch.stack(xs) for xs in xss]).transpose(2, 0)

def arr_to_img(nparr):
    assert nparr.shape[0] == 4 # required for RGBA
    shape = nparr.shape[1:]
    return Image.frombuffer('RGBA', shape, (nparr.transpose((1,2,0)) * 255).astype(np.uint8).tostring(), 'raw', 'RGBA', 0, 1)

def save_frames(frames, path_fmt_str, offset=0):
    n = frames.shape[0]
    assert_size(frames, (n, 4, frames.size(2), frames.size(3)))
    for i in range(n):
        arr_to_img(frames[i].data.numpy()).save(path_fmt_str.format(i + offset))

def mk_matrix(m, n):
    return [[None] * n for _ in range(m)]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=False, help='Use CUDA')
    args = parser.parse_args()
    print(args)


    # Test model by sampling:
    # dynair = DynAIR()
    # dummy_data = Variable(torch.ones(3, 20, 4, 50, 50))
    # obj_counts = torch.LongTensor([1, 2, 3])
    # frames, ws, zs = dynair.model(dummy_data, obj_counts, do_likelihood=False)

    # Debugging log_pdf masking:
    # trace = poutine.trace(dynair.model).get_trace(dummy_data, obj_counts)
    # trace.compute_batch_log_pdf()
    # for name, site in trace.nodes.items():
    #     print(name)
    #     if site['type'] == 'sample':
    #         print(site['batch_log_pdf'])



    # print(len(frames))
    # print(frames[0].size())
    # print(len(ws))
    # print(ws[0].size())
    # print(len(zs))
    # print(zs[0].size())
    # # print(zs)

    # Visualize prior samples with matplotlib:
    # from matplotlib import pyplot as plt
    # for frame in frames:
    #     print(frame[0].data.size())
    #     img = frame[0].data.numpy().transpose(1,2,0)
    #     plt.imshow(img)
    #     plt.show()
    # input('press a key...')

    # Visualise prior samples using visdom:
    # vis = visdom.Visdom()
    # frames = frames_to_tensor(frames)
    # ws = latents_to_tensor(ws)

    # ix = 2
    # out = overlay_multiple_window_outlines(dynair, frames[ix], ws[ix], obj_counts[ix])
    # vis.images(frames_to_rgb_list(out), nrow=10)



    # Test guide:
    #(batch, seq, channel, w, h)
    # dynair = DynAIR()
    # data = Variable(torch.ones(1, 14, 4, 32, 32))
    # ws, zs, y = dynair.guide(data)

    # # print(torch.stack(ws))
    # # print(torch.stack(zs))
    # print(y)

    data = load_data(args.cuda)
    # X = Variable(torch.zeros(1000, 20, 4, 50, 50))
    # nn.init.normal(X)
    run_svi(data, args)

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
