import torch
from torch.nn.functional import affine_grid, grid_sample, sigmoid, softplus

from utils import assert_size

def image_to_window(cfg, w, images):
    n = w.size(0)
    assert_size(w, (n, cfg.w_size))
    assert_size(images, (n, cfg.num_chan, cfg.image_size, cfg.image_size))
    theta_inv = expand_theta(theta_inverse(w_to_theta(w)))
    grid = affine_grid(theta_inv, torch.Size((n, cfg.num_chan, cfg.window_size, cfg.window_size)))
    # TODO: Consider using padding_mode='border' with grid_sample,
    # seems pretty sensible, though may not make much difference.
    return grid_sample(images, grid)

def window_to_image(cfg, w, windows):
    n = w.size(0)
    assert_size(w, (n, cfg.w_size))
    x_obj_size = (cfg.num_chan+1) * cfg.window_size**2 # contents of the object window
    assert_size(windows, (n, x_obj_size))
    theta = expand_theta(w_to_theta(w))
    assert_size(theta, (n, 2, 3))
    grid = affine_grid(theta, torch.Size((n, cfg.num_chan+1, cfg.image_size, cfg.image_size)))
    # first arg to grid sample should be (n, c, in_w, in_h)
    return grid_sample(windows.view(n, cfg.num_chan+1, cfg.window_size, cfg.window_size), grid)

expansion_indices = torch.LongTensor([1, 0, 2, 0, 1, 3])

def expand_theta(theta):
    # Take a batch of three-vectors, and massages them into a batch of
    # 2x3 matrices with elements like so:
    # [s,x,y] -> [[s,0,x],
    #             [0,s,y]]
    n = theta.size(0)
    assert_size(theta, (n, 3))
    out = torch.cat((torch.zeros([1, 1]).type_as(theta).expand(n, 1), theta), 1)
    ix = expansion_indices
    if theta.is_cuda:
        ix = ix.cuda()
    out = torch.index_select(out, 1, ix)
    out = out.view(n, 2, 3)
    return out

def w_to_theta(w):
    # Unsquish the `scale` component of w.
    scale = softplus(w[:, 0:1])
    xy = w[:, 1:] * scale
    out = torch.cat((scale, xy), 1)
    return out

# An alternative to this would be to add the "missing" bottom row to
# theta, and then use `torch.inverse`.
def theta_inverse(theta):
    # Take a batch of z_where vectors, and compute their "inverse".
    # That is, for each row compute:
    # [s,x,y] -> [1/s,-x/s,-y/s]
    # These are the parameters required to perform the inverse of the
    # spatial transform performed in the generative model.
    n = theta.size(0)
    out = torch.cat((torch.ones([1, 1]).type_as(theta).expand(n, 1), -theta[:, 1:]), 1)
    # Divide all entries by the scale.
    out = out / theta[:, 0:1]
    return out

# This assumes that the number of channels is 3 + the alpha channel.
def over(a, b):
    # a over b
    # https://en.wikipedia.org/wiki/Alpha_compositing
    # assert a.size() == (n, 4, image_size, image_size)
    assert a.size(1) == 4
    assert b.size(1) == 3
    rgb_a = a[:, 0:3] # .size() == (n, 3, image_size, image_size)
    alpha_a = a[:, 3:4] # .size() == (n, 1, image_size, image_size)
    return rgb_a * alpha_a + b * (1 - alpha_a)