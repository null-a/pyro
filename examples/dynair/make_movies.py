import os
import argparse
import json
import subprocess
import torch
import numpy as np
from matplotlib import pyplot as plt
import pyro
from PIL import Image, ImageDraw

from dynair import config
from data import load_data, data_params
from opt.all import build_module
from vis import overlay_multiple_window_outlines

def frame_to_img(frame, mark=False):
    assert frame.shape[0] == 3
    shape = frame.shape[1:]
    byte_str = (frame.transpose(1,2,0) * 255).astype(np.uint8).tostring()
    img = Image.frombuffer('RGB', shape, byte_str, 'raw', 'RGB', 0, 1)
    if mark:
        draw = ImageDraw.Draw(img, mode='RGBA')
        draw.rectangle([(0,0), (5,5)], fill=(255,255,255,127))
    return img

def make_video(dynair, x, y, tmp_dir, out_fn):
    # x is an input sequence
    # y is the object count

    assert X.shape[3] == X.shape[4] # I've assumed square inputs throughout.
    size = X.shape[3]

    # Compute recon/extra.
    # Here we unsqueeze x and y into a "batch" of size 1, as expected by the model/guide.
    frames, ws, extra_frames, extra_ws = dynair.infer(x.unsqueeze(0), y.unsqueeze(0), 20)

    input_seq = x
    recon_seq = overlay_multiple_window_outlines(dynair.cfg, frames[0], ws[0], y)
    extra_seq = overlay_multiple_window_outlines(dynair.cfg, extra_frames[0], extra_ws[0], y)

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

    subprocess.call(['ffmpeg', '-framerate', '8', '-i', '{}/frame_%2d.png'.format(tmp_dir), '-y', '-s', '400x200', out_fn])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('module_config_path')
    parser.add_argument('params_path')
    parser.add_argument('indices', type=int, nargs='+',
                        help='indices of data points for which to make movies')
    # e.g. gif, mp4, etc. (relies on the fact that ffmpeg will
    # determine the format based on the file extension.)
    parser.add_argument('-f', default='gif')
    args = parser.parse_args()

    data = load_data(args.data_path)
    X, Y = data

    with open(args.module_config_path) as f:
        module_config = json.load(f)
    cfg = config(module_config, data_params(data))

    dynair = build_module(cfg, use_cuda=False)
    dynair.load_state_dict(torch.load(args.params_path, map_location=lambda storage, loc: storage))

    tmp_dir = './tmp'
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    for ix in args.indices:
        make_video(dynair, X[ix], Y[ix], tmp_dir, 'movie_{}.{}'.format(ix, args.f))
