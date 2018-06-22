import argparse
import json
import torch
from torch.nn.functional import softplus
from torchvision.utils import make_grid, save_image
from pyro.infer import Trace_ELBO

import numpy as np

from PIL import Image, ImageDraw, ImageFilter
from PIL.ImageTransform import AffineTransform

import motmetrics as mm

from dynair import config
from data import load_data, data_params, split, trunc_seqs
from opt.all import build_module
from opt.run_svi import elbo_from_batches
from transform import over
from vis import overlay_multiple_window_outlines

def elbo_main(dynair, X, Y, T, args):
    X, Y, T = trunc_seqs((X, Y, T), args.l)
    print(X.size())
    X_batches, _ = split(X, args.batch_size, 0)
    Y_batches, _ = split(Y, args.batch_size, 0)
    elbo = elbo_from_batches(dynair, list(zip(X_batches, Y_batches)), args.n)
    seq_length = X.size(1)
    print(elbo / float(seq_length * args.batch_size))

# TODO: Add ability to use euclidean distance.

# TODO: Consider using mean rather than sampling when generated
# extrapolated frames.

def tracking_main(dynair, X, Y, T, args):
    # Determine infer/extra split.
    seq_len = X.size(1)
    infer_len = args.l if args.l is not None else seq_len
    extra_len = seq_len - infer_len
    assert 0 < infer_len <= seq_len

    X_batches, _ = split(X, args.batch_size, 0)
    Y_batches, _ = split(Y, args.batch_size, 0)
    T_batches, _ = split(T, args.batch_size, 0)

    # For each data point we build an 'accumulator' (defined by in
    # motmetrics library) which stores inferences and ground truth
    # positions. These are then used to compute the metrics.
    accs = []

    for i, (X_batch, Y_batch, T_batch) in enumerate(zip(X_batches, Y_batches, T_batches)):

        _, infer_wss, _, extra_wss = dynair.infer(X_batch[:,0:infer_len], Y_batch, extra_len)
        wss = torch.cat((infer_wss, extra_wss), 2)
        #print(wss.size()) # (batch_size, max_num_objs, seq_len, |w|)

        # Iterate over each data point in batch.
        for j, (ws, x, obj_count, ground_truth) in enumerate(zip(wss, X_batch, Y_batch, T_batch)):
            obj_count = obj_count.item()

            # Drop the padding added for non-existent objects.
            ws = ws[0:obj_count].contiguous()
            ground_truth = ground_truth[:,0:obj_count]

            # Here we:
            # 1) flatten ws
            # 2) map from ws to (x,y,w,h)
            # 3) undo flatten
            # 4) transpose so that layout matches ground_truth
            bbs = ws_to_bounding_box(ws.view(-1, 3), dynair.cfg.image_size).view(obj_count, -1, 4).transpose(0,1)
            #print(bbs.size()) # (seq_len, obj_count, |(x,y,w,h)|)

            # Sanity check (x,y,w,h) is computed correctly by visualising bounding boxes.
            #overlay = torch.from_numpy(draw_bounding_box(ground_truth, dynair.cfg.image_size) / 255.).float()
            #overlay = torch.from_numpy(draw_bounding_box(bbs, dynair.cfg.image_size) / 255.).float()
            #save_image(make_grid(over(overlay, x), nrow=10), 'bb_{}_{}.png'.format(i,j))

            # Here we also remove the padding from ground_truth.
            acc = mot_populate_acc(bbs, ground_truth)
            accs.append(acc)

    summary = compute_metrics([acc.events.loc[0:infer_len-1] for acc in accs])
    print('inferred:')
    print(summary)
    summary = compute_metrics([acc.events.loc[infer_len:] for acc in accs])
    print('extrapolated:')
    print(summary)

def compute_metrics(accs):
    mh = mm.metrics.create()
    return mh.compute_many(accs,
                           metrics=['num_frames', 'mota', 'motp'],
                           names=list(range(len(accs))),
                           generate_overall=True)

def mot_populate_acc(preds, truth):
    # preds (seq_len, obj_count, 4)
    # truth (seq_len, obj_count, 4)
    preds = preds.numpy()
    truth = truth.numpy()
    assert preds.shape == truth.shape
    assert preds.shape[0] == 20
    assert truth.shape[2] == 4

    obj_count = preds.shape[1]
    assert obj_count <= 10

    acc = mm.MOTAccumulator(auto_id=True)

    # Iterate over frames.
    for preds_i, truth_i in zip(preds, truth):
        distances = mm.distances.iou_matrix(
            preds_i,
            truth_i,
            max_iou=0.5)
        #print(distances)
        acc.update(
            list("abcdefghij")[0:obj_count], # ground truth object
            list("0123456789")[0:obj_count], # inferred objects
            distances)

    return acc

# Map (a batch of) w (as used by the spatial transformer) to (x,y,w,h)
def ws_to_bounding_box(ws, x_size):
    ws_scale = softplus(ws[:,0])
    ws_x = ws[:,1]
    ws_y = ws[:,2]
    w = x_size / ws_scale
    h = x_size / ws_scale
    xtrans = -ws_x * x_size / 2.
    ytrans = -ws_y * x_size / 2.
    x = (x_size - w) / 2 + xtrans  # origin is top left
    y = (x_size - h) / 2 + ytrans
    return torch.cat((x.unsqueeze(1), y.unsqueeze(1), w.unsqueeze(1), h.unsqueeze(1)), 1)

# TODO: This is duplicated in make_data.py. Consolidate.
def draw_bounding_box(tracks, image_size):
    # tracks. np tensor (seq_len, num_obj, len([x,y,w,h]))
    # returns np tensor of size (seq_len, num_chan=3, w, h), with pixel intensities in 0..255
    out = []
    for obj_positions in tracks:
        img = Image.new('RGBA', (image_size, image_size), 0)
        draw = ImageDraw.Draw(img)
        for ix, obj_pos in enumerate(obj_positions):
            x, y, w, h = obj_pos
            draw.rectangle([x, y, x+w, y+h], outline=['red','green','blue'][ix])
        out.append(img_to_arr(img))
    return np.array(out)

def img_to_arr(img):
    #assert img.mode == 'RGBA'
    channels = 4 if img.mode == 'RGBA' else 3
    w, h = img.size
    arr = np.fromstring(img.tobytes(), dtype=np.uint8)
    return arr.reshape(w * h, channels).T.reshape(channels, h, w)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('module_config_path')
    parser.add_argument('params_path')

    parser.add_argument('start', type=int, help='start index of test set')
    parser.add_argument('end', type=int, help='end index of test set')

    parser.add_argument('-b', '--batch-size', type=int, required=True, help='batch size')
    parser.add_argument('-l', type=int, help='sequence length')

    subparsers = parser.add_subparsers(dest='target')
    elbo_parser = subparsers.add_parser('elbo')
    elbo_parser.set_defaults(main=elbo_main)

    tracking_parser = subparsers.add_parser('tracking')
    tracking_parser.set_defaults(main=tracking_main)

    elbo_parser.add_argument('-n', type=int, default=1, help='number of particles')

    args = parser.parse_args()

    # We don't truncate seqs here as we typically do elsewhere, as
    # computing metrics always requires full sequences in order to
    # compute metrics for extrapolated frames.
    data = load_data(args.data_path)
    X, Y, T = data

    with open(args.module_config_path) as f:
        module_config = json.load(f)
    cfg = config(module_config, data_params(data))

    dynair = build_module(cfg, use_cuda=False)
    dynair.load_state_dict(torch.load(args.params_path, map_location=lambda storage, loc: storage))

    args.main(dynair,
              X[args.start:args.end],
              Y[args.start:args.end],
              T[args.start:args.end] if T is not None else None,
              args)

if __name__ == '__main__':
    main()
