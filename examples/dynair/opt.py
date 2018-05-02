import os
import argparse
import time
from datetime import timedelta
from pprint import pprint as pp
from functools import partial
import visdom
from pyro.infer import SVI, Trace_ELBO
import pyro.optim as optim

import torch
import torch.nn as nn

from dyn3d import Config, DynAIR
from model import Model
from guide import Guide, GuideW_ObjRnn, GuideW_ImageSoFar, GuideZ
from data import split, load_data, data_params
from optutils import make_output_dir, append_line, describe_env, md5sum
from vis import frames_to_tensor, latents_to_tensor, overlay_multiple_window_outlines, frames_to_rgb_list
import dyn3d_modules as mod

def run_vis(vis, dynair, X, Y, epoch, batch):
    n = X.size(0)

    frames, wss, extra_frames, extra_wss = dynair.infer(X, Y, 15)

    frames = frames_to_tensor(frames)
    ws = latents_to_tensor(wss)
    extra_frames = frames_to_tensor(extra_frames)
    extra_ws = latents_to_tensor(extra_wss)

    for k in range(n):
        out = overlay_multiple_window_outlines(dynair.cfg, frames[k], ws[k], Y[k])
        vis.images(frames_to_rgb_list(X[k].cpu()), nrow=10,
                   opts=dict(title='input {} after epoch {} batch {}'.format(k, epoch, batch)))
        vis.images(frames_to_rgb_list(out.cpu()), nrow=10,
                   opts=dict(title='recon {} after epoch {} batch {}'.format(k, epoch, batch)))

        out = overlay_multiple_window_outlines(dynair.cfg, extra_frames[k], extra_ws[k], Y[k])
        vis.images(frames_to_rgb_list(out.cpu()), nrow=10,
                   opts=dict(title='extra {} after epoch {} batch {}'.format(k, epoch, batch)))

def vis_hook(period, vis, dynair, X, Y, epoch, batch, step):
    if period > 0 and (step+1) % period == 0:
        run_vis(vis, dynair, X, Y, epoch, batch)

def run_svi(dynair, X_split, Y_split, num_epochs, vis_hook, output_path):
    t0 = time.time()

    X_train, X_test = X_split
    Y_train, Y_test = Y_split
    num_batches = len(X_train)
    batch_size = X_train[0].size(0)

    # Produce visualisations for the first train & test data points
    # where possible.
    if len(X_test) > 0:
        X_vis = torch.cat((X_train[0][0:1], X_test[0][0:1]))
        Y_vis = torch.cat((Y_train[0][0:1], Y_test[0][0:1]))
    else:
        X_vis = X_train[0][0:1]
        Y_vis = Y_train[0][0:1]

    def per_param_optim_args(module_name, param_name):
        return {'lr': 1e-4}

    svi = SVI(dynair.model, dynair.guide,
              optim.Adam(per_param_optim_args),
              loss=Trace_ELBO())

    for i in range(num_epochs):

        for j, (X_batch, Y_batch) in enumerate(zip(X_train, Y_train)):
            loss = svi.step(batch_size, X_batch, Y_batch)
            elbo = -loss / (dynair.cfg.seq_length * batch_size) # elbo per datum, per frame
            elapsed = timedelta(seconds=time.time()- t0)
            print('\33[2K\repoch={}, batch={}, elbo={:.2f}, elapsed={}'.format(i, j, elbo, elapsed), end='')
            append_line(os.path.join(output_path, 'elbo.csv'), '{:.1f},{:.2f}'.format(elapsed.total_seconds(), elbo))
            if not vis_hook is None:
                vis_hook(X_vis, Y_vis, i, j, num_batches*i+j)
            dynair.clear_cache()
            print()
            pp(dynair.cache_stats())


        if (i+1) % 1000 == 0:
            torch.save(dynair.state_dict(),
                       os.path.join(output_path, 'params-{}.pytorch'.format(i+1)))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('-b', '--batch-size', type=int, required=True, help='batch size')
    parser.add_argument('-e', '--epochs', type=int, default=10**6,
                        help='number of optimisation epochs to perform')
    parser.add_argument('--hold-out', type=int, default=0,
                        help='number of batches to hold out')
    parser.add_argument('--vis', type=int, default=0,
                        help='visualise inferences during optimisation (zero disables, otherwise specifies period)')
    parser.add_argument('-o', default='./runs', help='base output path')
    parser.add_argument('--y-size', type=int, default=50, help='size of y')
    parser.add_argument('--z-size', type=int, default=50, help='size of z')
    parser.add_argument('--window-size', type=int, default=22,
                        help='size of the object window')
    parser.add_argument('--cuda', action='store_true', default=False, help='use CUDA')
    args = parser.parse_args()

    data = load_data(args.data_path, args.cuda)
    X, Y = data # (sequences, counts)
    X_split = split(X, args.batch_size, args.hold_out)
    Y_split = split(Y, args.batch_size, args.hold_out)

    cfg = Config(w_size=3,
                 y_size=args.y_size,
                 z_size=args.z_size,
                 window_size=args.window_size,
                 **data_params(data))

    output_path = make_output_dir(args.o)
    print('output path: {}'.format(output_path))
    log_to_cond = partial(append_line, os.path.join(output_path, 'condition.txt'))
    log_to_cond(describe_env())
    log_to_cond('data path: {}'.format(args.data_path))
    log_to_cond('data md5: {}'.format(md5sum(args.data_path)))
    log_to_cond('data split: {}/{}'.format(len(X_split[0]), len(X_split[1])))
    log_to_cond(cfg)

    model = Model(cfg, use_cuda=args.cuda)
    guide = Guide(cfg,
                  dict(guide_w=GuideW_ObjRnn(cfg, dedicated_t0=False),
                       #guide_w=GuideW_ImageSoFar(cfg, model),
                       guide_y=mod.ParamY(cfg),
                       guide_z=GuideZ(cfg, dedicated_t0=False)),
                  use_cuda=args.cuda)

    dynair = DynAIR(cfg, model, guide, use_cuda=args.cuda)

    run_svi(dynair, X_split, Y_split, args.epochs,
            partial(vis_hook, args.vis, visdom.Visdom(), dynair),
            output_path)
