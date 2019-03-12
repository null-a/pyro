import random
import glob
import math
import os.path
import argparse
from functools import partial
import json

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
from PIL.ImageTransform import AffineTransform

import torch
from torchvision.utils import make_grid, save_image

from transform import over

SIZE = 50

def interp1(n, a, b, t):
    return a + (b - a) * (t / float(n - 1))

# Interpolate between (x1,y1) and (x2,y2).
def interp(n, xy1, xy2, t):
    x1, y1 = xy1
    x2, y2 = xy2
    return (interp1(n, x1, x2, t), interp1(n, y1, y2, t))

# I'm saving in TIFF format since using GIF causes frames to be
# dropped. Will convert to GIF (or some other format) using
# ImageMagick. e.g. `convert out.tiff out.gif`
def save_seq(frames):
    first = frames[0]
    rest = frames[1:]
    first.save('out.tiff', save_all=True, append_images=rest)

# https://www.emojione.com/
def load_avatars(path):
    fns = glob.glob(os.path.join(path, '*.png'))
    print('found {} avatar images'.format(len(fns)))
    return [Image.open(fn).convert('RGBA').resize((SIZE, SIZE), resample=Image.BILINEAR) for fn in fns]

def rot(img, angle):
    return img.rotate(angle)

def scale(img, s):
    img = img.transform((SIZE, SIZE),
                        AffineTransform((s, 0, 0,
                                         0, s, 0)),
                        resample=Image.BILINEAR)
    # Translate so that we are effectively scaling around the centre.
    # This makes working with result easier to think about.
    return trans(img, SIZE/2*(1-1/s), SIZE/2*(1-1/s))

def trans(img, x, y):
    return img.transform((SIZE, SIZE),
                         AffineTransform((1, 0, -x,
                                          0, 1, -y)),
                         resample=Image.BILINEAR)

# Each object is located in the centre of an image that has the same
# size as the output. I want to think about placing that in a frame
# that has 0,0 at the top-left. This help does that. i.e. It takes an
# object, and positions it relative to the top-left.

# e.g. position(s1, 0, 0) places the centre of the object as the top
# left of the output.

def position(img, x, y):
    return trans(img, x - SIZE/2, y - SIZE/2)

def sample_end_points():

    # Avoid placing objects in a border of the following width. This
    # ensures objects are not partially out of shot.
    b = 7
    assert SIZE == 50, 'SIZE != 50 -- border width needs tuning for this new image size.'

    x0 = np.random.uniform(b, SIZE-b)
    x1 = np.random.uniform(b, SIZE-b)
    y0 = np.random.uniform(b, SIZE-b)
    y1 = np.random.uniform(b, SIZE-b)

    # Compute path length.
    path_length = math.sqrt((x0-x1)**2 + (y0-y1)**2)
    # print(path_length)

    MIN_PATH_LENGTH = SIZE / 2

    if path_length < MIN_PATH_LENGTH:
        return sample_end_points()
    else:
        return (x0, y0), (x1, y1)

def sample_position():
    b = 7
    assert SIZE == 50, 'SIZE != 50 -- border width needs tuning for this new image size.'
    x = np.random.uniform(b, SIZE-b)
    y = np.random.uniform(b, SIZE-b)
    return x, y

# Note, unused.
def sample_natural_scene_bkg(path, size):
    # I'm using some images from here:
    # http://cvcl.mit.edu/database.htm
    # So this assumes images are 256x256 JPGs.
    fns = glob.glob(os.path.join(path, '*.jpg'))
    n = len(fns)
    ix = np.random.randint(n)
    im = Image.open(fns[ix])
    assert im.size == (256, 256)

    # Crop a (256 - t) sized image out of the full image to help avoid
    # duplicating images.
    t = 64
    x = np.random.randint(t)
    y = np.random.randint(t)

    cropped = im.crop((x,y,x+256-t,y+256-t)).resize((size, size), resample=Image.BILINEAR).convert('RGBA')
    #arr = img_to_arr(cropped)
    return cropped, (n, ix, x, y)

def sample_scene(seq_len, min_num_objs, max_num_objs, rotate, translate, avatars, output_grayscale, bkg):

    assert max_num_objs <= len(avatars)
    assert bkg.size == (SIZE, SIZE)

    # Sample objects (without replacement)
    num_objs = np.random.randint(max_num_objs - min_num_objs + 1) + min_num_objs
    avatars_ix = list(range(len(avatars)))
    random.shuffle(avatars_ix)

    objs = []
    for i in range(num_objs):
        if translate:
            xy1, xy2 = sample_end_points()
        else:
            # no translation
            xy1 = sample_position()
            xy2 = None
        objs.append(dict(
            xy1=xy1,
            xy2=xy2,
            shape=avatars[avatars_ix[i]],
            rot_init=np.random.uniform(360),
            rot_vel=np.random.uniform(0, 15)
        ))

    frames = []
    tracks = []

    obj_downscale_factor = 3.5

    for t in range(seq_len):
        acc = bkg
        annotations = []
        for ix, obj in enumerate(objs):
            if obj['xy2'] is None:
                x, y = obj['xy1']
            else:
                x, y = interp(seq_len, obj['xy1'], obj['xy2'], t)

            s = obj['shape']
            if rotate:
                s = rot(s, obj['rot_init'] + obj['rot_vel'] * t)
            s = scale(s, obj_downscale_factor) # add variable scale, possibly dynamic
            s = position(s, x, y)
            acc = Image.alpha_composite(acc, s)

            # x,y refer to the corner nearest the origin, which is top-left here.
            obj_pixel_size = SIZE/float(obj_downscale_factor)
            annotations.append([x-(obj_pixel_size/2),
                                y-(obj_pixel_size/2),
                                obj_pixel_size,  # width
                                obj_pixel_size]) # height

        frames.append(acc.convert('L' if output_grayscale else 'RGB')) # convert to L/RGB since all pixels now opaque
        tracks.append(annotations)

    # tracks for seq with zero objects need special handling to ensure
    # they have shape (seq_len, 0, 4)
    if num_objs == 0:
        np_tracks = np.zeros((seq_len, 0, 4))
    else:
        np_tracks = np.array(tracks)

    return frames, num_objs, np_tracks, avatars_ix

# We pad tracks to that we can pack tracks with differing numbers of
# objects into a single regular array for convenience.
def pad_tracks(tracks, max_num_objs):
    # tracks. np array (seq_len, num_objs, len(x,y,w,h))
    seq_len = tracks.shape[0]
    num_objs = tracks.shape[1]
    assert tracks.shape[2] == 4
    assert num_objs <= max_num_objs
    return np.hstack((tracks, np.zeros((seq_len, max_num_objs - num_objs, 4))))

# Arrays of pixel intensities that range over 0..255
def img_to_arr(img):
    #assert img.mode == 'RGBA'
    #channels = 4 if img.mode == 'RGBA' else 3
    if img.mode == 'RGBA':
        channels = 4
    elif img.mode == 'RGB':
        channels = 3
    elif img.mode == 'LA':
        channels = 2
    elif img.mode == 'L':
        channels = 1
    else:
        raise Exception('unsupport image mode')
    w, h = img.size
    arr = np.fromstring(img.tobytes(), dtype=np.uint8)
    return arr.reshape(w * h, channels).T.reshape(channels, h, w)

def arr_to_img(nparr):
    assert nparr.shape[0] == 4 # required for RGBA
    shape = nparr.shape[1:]
    return Image.frombuffer('RGBA', shape, (nparr.transpose((1,2,0))).astype(np.uint8).tostring(), 'raw', 'RGBA', 0, 1)

# This is here primarily to sanity check the ground-truth object
# tracks.

# Eventually the model/guide will also need to output window info
# x,y,w,h in order to compute metrics, at which point it might be
# worth switching to this implementation of box drawing.

# a) This doesn't rely on window_to_image, so boxes render more
# crisply.

# b) Since we only create an overlay, it could be useful for showing
# both model and guide window positions in a variant in which can
# differ.

# Before doing so, I might want to first switch this over to using
# torch rather than numpy.

def draw_bounding_box(tracks, output_grayscale):
    # tracks. np tensor (seq_len, num_obj, len([x,y,w,h]))
    # returns np tensor of size (seq_len, num_chan=3, w, h), with pixel intensities in 0..255
    out = []
    for obj_positions in tracks:
        img = Image.new('LA' if output_grayscale else 'RGBA', (SIZE,SIZE), 0)
        draw = ImageDraw.Draw(img)
        for ix, obj_pos in enumerate(obj_positions):
            x, y, w, h = obj_pos
            draw.rectangle([x, y, x+w, y+h], outline=['red','green','blue'][ix])
        out.append(img_to_arr(img))
    return np.array(out)

def main_one(sample_one, args, avatar_set_size, num_bkg, get_bkg):
    frames, num_objs, tracks, _ = sample_one(get_bkg(np.random.randint(num_bkg)))

    if args.f == 'png':
        frames_arr = np.stack(img_to_arr(frame) for frame in frames)
        frames_arr_t = torch.from_numpy(frames_arr / 255.0)

        # test round tripping through padding.
        tracks = pad_tracks(tracks, args.max)[:,0:num_objs]

        # This will be ints in 0..255
        boxes_overlay = draw_bounding_box(tracks, args.g)
        boxes_overlay_t = torch.from_numpy(boxes_overlay / 255.)

        save_image(make_grid(over(boxes_overlay_t, frames_arr_t), nrow=10), 'out.png')
    elif args.f == 'tiff':
        save_seq(frames)
    else:
        raise Exception('impossible')

def main_dataset(sample_one, args, avatar_set_size, num_bkg, get_bkg):
    n = args.n
    seq_len = args.l

    if n > num_bkg:
        print('WARNING: too few background images, backgrounds will not be unique')

    perm = list(range(min(n, num_bkg)))
    # For a given n, the same set of background images will always be
    # used, though they will, in general, appear in a different order
    # in the final dataset.
    random.shuffle(perm)

    seqs, counts, trackss, obj_ids = tuple(zip(*[sample_one(get_bkg(perm[i % num_bkg])) for i in range(n)]))

    seqs_np = np.stack(np.stack(img_to_arr(frame) for frame in seq) for seq in seqs)
    counts_np = np.array(counts, dtype=np.int8)
    trackss_np = np.stack([pad_tracks(tracks, args.max) for tracks in trackss])
    obj_ids_np = np.stack(obj_ids)

    # print(trackss_np[0,:,0])
    # print(trackss_np.dtype)
    # print(seqs_np.shape)
    # print(counts_np.shape)
    # print(trackss_np.shape)
    # print(obj_ids_np.shape)
    # print(obj_ids_np.dtype)

    assert(seqs_np.shape == (n, seq_len, 1 if args.g else 3, SIZE, SIZE))
    assert(counts_np.shape == (n,))
    assert(trackss_np.shape == (n, seq_len, args.max, 4))
    assert(obj_ids_np.shape == (n, 3))

    np.savez_compressed('out.npz',
                        X=seqs_np,
                        Y=counts_np,
                        T=trackss_np,
                        O=obj_ids_np,
                        avatar_set_size=np.array([avatar_set_size]))


def bkgs_from_dir(bkg_path):
    fns = sorted(glob.glob(os.path.join(bkg_path, '*.jpg')))
    print('found {} background images'.format(len(fns)))
    def get_bkg(i):
        return Image.open(fns[i]).convert('RGBA')
    return len(fns), get_bkg

def empty_bkg():
    print('using empty background')
    def get_bkg(i):
        assert i == 0
        return Image.new('RGBA', (SIZE,SIZE), 'black')
    return 1, get_bkg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('avatar_path')
    parser.add_argument('--bkg-path')
    parser.add_argument('--min', type=int, default=0)
    parser.add_argument('--max', type=int, required=True)
    parser.add_argument('-l', type=int, required=True, help='sequence length')
    parser.add_argument('-r', action='store_true', default=False, help='rotate objects over time')
    parser.add_argument('-t', action='store_true', default=False, help='translate objects over time')
    parser.add_argument('-g', action='store_true', default=False, help='generate grayscale frames')
    subparsers = parser.add_subparsers(dest='target')
    one_parser = subparsers.add_parser('one')
    one_parser.add_argument('-f', choices=['png', 'tiff'], default='png')
    dataset_parser = subparsers.add_parser('dataset')
    dataset_parser.add_argument('n', type=int, help='number of datapoints to generate')
    one_parser.set_defaults(main=main_one)
    dataset_parser.set_defaults(main=main_dataset)

    args = parser.parse_args()
    assert args.min >= 0
    assert args.max >= 0
    assert args.max >= args.min

    # bkgs is a pair of (num_bkgs, get_bkg_fn)
    bkgs = bkgs_from_dir(args.bkg_path) if args.bkg_path else empty_bkg()

    avatars = load_avatars(args.avatar_path)
    sample_one = partial(sample_scene,
                         args.l, args.min, args.max, args.r, args.t,
                         avatars, args.g)
    args.main(sample_one, args, len(avatars), *bkgs)
