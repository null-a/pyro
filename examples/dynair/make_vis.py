import os
import os.path
import argparse
import json
import subprocess
import torch
from torchvision.utils import make_grid, save_image
import numpy as np
from matplotlib import pyplot as plt
import pyro
from PIL import Image, ImageDraw

from dynair import config
from data import load_data, data_params, trunc_seqs
from opt.all import build_module
from vis import overlay_multiple_window_outlines

def frame_to_img(frame):
    num_chan = frame.shape[0]
    assert num_chan == 3 or num_chan == 1
    shape = frame.shape[1:]
    byte_str = (frame.transpose(1,2,0) * 255).astype(np.uint8).tostring()
    mode = 'RGB' if num_chan == 3 else 'L'
    img = Image.frombuffer(mode, shape, byte_str, 'raw', mode, 0, 1)
    if num_chan == 1:
        img = img.convert('RGB')
    return img

def movie_main(dynair, X, Y, args):
    tmp_dir = './tmp'
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    for ix in args.indices:
        make_movie(dynair, X[ix], Y[ix], tmp_dir, 'movie_{}.{}'.format(ix, args.f), not args.d, args.n)

def make_movie(dynair, x, y, tmp_dir, out_fn, sample_extra, num_extra_frames):
    # x is an input sequence
    # y is the object count
    size = dynair.cfg.image_size

    # Compute recon/extra.
    # Here we unsqueeze x and y into a "batch" of size 1, as expected by the model/guide.
    frames, ws, _, _, extra_frames, extra_ws = dynair.infer(x.unsqueeze(0), y.unsqueeze(0), num_extra_frames, sample_extra)

    input_seq = x
    recon_seq = overlay_multiple_window_outlines(dynair.cfg, frames[0], ws[0,:,:,0:3], y)
    extra_seq = overlay_multiple_window_outlines(dynair.cfg, extra_frames[0], extra_ws[0,:,:,0:3], y)

    for i, (input_frame, recon_frame) in enumerate(zip(input_seq, recon_seq)):
        input_img = frame_to_img(input_frame.numpy())
        infer_img = frame_to_img(recon_frame.numpy())
        img = Image.new('RGB', (100,50))
        img.paste(input_img, (0,0))
        img.paste(infer_img, (size,0))
        img.save('{}/frame_{:02d}.png'.format(tmp_dir, i))

    for i, extra_frame in enumerate(extra_seq):
        extra_img = frame_to_img(extra_frame.numpy())
        img = Image.new('RGB', (2*size,size))
        img.paste(extra_img, box=(size,0))
        img.save('{}/frame_{:02d}.png'.format(tmp_dir, i + input_seq.shape[0]))

    # Create palette to fix artifacts seen in gray scale gifs.
    # https://engineering.giphy.com/how-to-make-gifs-with-ffmpeg/
    subprocess.call(['ffmpeg', '-i', '{}/frame_%2d.png'.format(tmp_dir), '-filter_complex', '[0:v] palettegen', '-y', 'palette.png'])
    # Adding '-s', '400x200' (after -y) is still problematic though, so avoiding.
    subprocess.call(['ffmpeg', '-framerate', '8', '-i', '{}/frame_%2d.png'.format(tmp_dir), '-i', 'palette.png', '-filter_complex', '[0:v][1:v] paletteuse', '-y', out_fn])

def frames_main(dynair, X, Y, args):
    for ix in args.indices:
        x = X[ix]
        y = Y[ix]
        frames, ws, zs, bkg, extra_frames, extra_ws = dynair.infer(x.unsqueeze(0), y.unsqueeze(0), args.n, sample_extra=not args.d)
        frames_with_windows = overlay_multiple_window_outlines(dynair.cfg, frames[0], ws[0,:,:,0:3], y)
        extra_frames_with_windows = overlay_multiple_window_outlines(dynair.cfg, extra_frames[0], extra_ws[0,:,:,0:3], y)

        if args.s:
            save_individual_frames(x, './frames_{}_input'.format(ix))
            save_individual_frames(frames_with_windows, './frames_{}_recon'.format(ix))
            save_individual_frames(extra_frames_with_windows, './frames_{}_extra'.format(ix))
            save_bkg_and_objs(dynair, ix, bkg, zs)
        else:
            save_image(make_grid(x, nrow=10), 'frames_{}_input.png'.format(ix))
            save_image(make_grid(frames_with_windows, nrow=10), 'frames_{}_recon.png'.format(ix))
            save_image(make_grid(extra_frames_with_windows, nrow=10), 'frames_{}_extra.png'.format(ix))

def save_individual_frames(t, path):
    if not os.path.exists(path):
        os.mkdir(path)
    for i, img in enumerate(t):
        save_image(img, os.path.join(path, 'frame_{:02d}.png').format(i))


def save_bkg_and_objs(dynair, ix, bkg, zs):
    save_image(bkg[0], './background_{}.png'.format(ix))

    objs_path = './objects_{}'.format(ix)
    if not os.path.exists(objs_path):
        os.mkdir(objs_path)

    for i, obj in enumerate(zs[0]):
        for j, z in enumerate(obj):
            obj_img = dynair.model.decode_obj(z).view(dynair.cfg.num_chan + 1, dynair.cfg.window_size, dynair.cfg.window_size)
            save_image(obj_img, objs_path + '/obj_{}_frame_{:02d}.png'.format(i, j))

    if dynair.cfg.use_depth:
        depths = []
        for i, obj in enumerate(zs[0]):
            for j, z in enumerate(obj):
                depth = dynair.model.decode_obj_depth(z)
                depths.append([i,j,depth.item()])
        with open('./depths_{}.txt'.format(ix), 'w') as f:
            f.writelines(', '.join(str(v) for v in row) + '\n' for row in depths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('module_config_path')
    parser.add_argument('params_path')
    parser.add_argument('indices', type=int, nargs='+',
                        help='indices of data points for which to create visualisations')

    parser.add_argument('-l', type=int, help='sequence length')
    parser.add_argument('-n', type=int, default=5, help='number of frames to extrapolate')
    parser.add_argument('-d', action='store_true', default=False,
                        help='do not sample latent variables when generating extra frames')

    subparsers = parser.add_subparsers(dest='target')
    movie_parser = subparsers.add_parser('movie')
    frames_parser = subparsers.add_parser('frames')
    movie_parser.set_defaults(main=movie_main)
    frames_parser.set_defaults(main=frames_main)

    # e.g. gif, mp4, etc. (relies on the fact that ffmpeg will
    # determine the format based on the file extension.)
    movie_parser.add_argument('-f', default='gif', help='output format')

    frames_parser.add_argument('-s', default=False, action='store_true',
                               help='Save each frame as a separate file.')

    args = parser.parse_args()

    data = trunc_seqs(load_data(args.data_path), args.l)
    X, Y, _ = data

    with open(args.module_config_path) as f:
        module_config = json.load(f)
    cfg = config(module_config, data_params(data))

    dynair = build_module(cfg, use_cuda=False)
    dynair.load_state_dict(torch.load(args.params_path, map_location=lambda storage, loc: storage))

    args.main(dynair, X, Y, args)

if __name__ == '__main__':
    main()
