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
from vis import frames_to_tensor, latents_to_tensor, frames_to_rgb_list, overlay_multiple_window_outlines

def show_seq(seq, layout_shape):
    cols, rows = layout_shape
    plt.figure(figsize=(8, 2))
    for r in range(rows):
        for c in range(cols):
            plt.subplot(rows, cols, r * cols + c + 1)
            plt.axis('off')
            plt.imshow(seq[r * cols + c].transpose((1,2,0)))

def frame_to_img(frame, mark=False):
    assert frame.shape[0] == 3
    shape = frame.shape[1:]
    byte_str = (frame.transpose(1,2,0) * 255).astype(np.uint8).tostring()
    img = Image.frombuffer('RGB', shape, byte_str, 'raw', 'RGB', 0, 1)
    if mark:
        draw = ImageDraw.Draw(img, mode='RGBA')
        draw.rectangle([(0,0), (5,5)], fill=(255,255,255,127))
    return img

def make_video(dynair, x, y, out_fn):
    # x is an input sequence
    # y is the object count

    assert X.shape[3] == X.shape[4] # I've assumed square inputs throughout.
    size = X.shape[3]

    # Compute recon/extra.
    # Here we unsqueeze x and y into a "batch" of size 1, as expected by the model/guide.
    frames, wss, extra_frames, extra_wss = dynair.infer(x.unsqueeze(0), y.unsqueeze(0), 20)

    frames = frames_to_tensor(frames)
    ws = latents_to_tensor(wss)
    extra_frames = frames_to_tensor(extra_frames)
    extra_ws = latents_to_tensor(extra_wss)

    # TODO: Converting this to a list and back is silly. I guess this
    # is happening in the model vis too. I think visdom now support
    # numpy and/or torch directly, so fix?
    input_seq = np.array(frames_to_rgb_list(x))

    out = overlay_multiple_window_outlines(dynair.cfg, frames[0], ws[0], y)
    recon_seq = np.array(frames_to_rgb_list(out))

    out2 = overlay_multiple_window_outlines(dynair.cfg, extra_frames[0], extra_ws[0], y)
    extra_seq = np.array(frames_to_rgb_list(out2))

    # show_seq(input_seq, (10,2))
    # show_seq(recon_seq, (10,2))
    # show_seq(extra_seq, (10,2))
    # plt.show()

    for i, (input_frame, recon_frame) in enumerate(zip(input_seq, recon_seq)):
        input_img = frame_to_img(input_frame)
        infer_img = frame_to_img(recon_frame)
        img = Image.new('RGB', (100,50))
        img.paste(input_img, (0,0))
        img.paste(infer_img, (size,0))
        img.save('{}/frame_{:02d}.png'.format(TMP_DIR, i))

    for i, extra_frame in enumerate(extra_seq):
        extra_img = frame_to_img(extra_frame)
        img = Image.new('RGB', (2*size,size))
        img.paste(extra_img, box=(size,0))
        img.save('{}/frame_{:02d}.png'.format(TMP_DIR, i + input_seq.shape[0]))

    subprocess.call(['ffmpeg', '-framerate', '8', '-i', '{}/frame_%2d.png'.format(TMP_DIR), '-y', '-s', '400x200', out_fn])

def get_ix_of_first_ex_of_each_count(Y, max_obj_count):
    return [(count,int((Y==count).nonzero()[0])) for count in range(1, max_obj_count+1)]

if __name__ == '__main__':

    #pyro.set_rng_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('module_config_path')
    parser.add_argument('params_path')
    parser.add_argument('indices', type=int, nargs='+',
                        help='indices of data points for which to make movies')
    args = parser.parse_args()

    TMP_DIR = './tmp'
    FORMAT = 'gif' # e.g. gif, mp4, etc. (ffmpeg will determine the format based on the file extension.)

    data = load_data(args.data_path)
    X, Y = data

    with open(args.module_config_path) as f:
        module_config = json.load(f)
    cfg = config(module_config, data_params(data))

    dynair = build_module(cfg, use_cuda=False)
    dynair.load_state_dict(torch.load(args.params_path, map_location=lambda storage, loc: storage))

    for ix in args.indices:
        make_video(dynair, X[ix], Y[ix], 'movie_{}.{}'.format(ix, FORMAT))
