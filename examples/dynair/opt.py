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

from dynair import Config, DynAIR
from vae import VAE
from model import Model, DecodeBkg
from guide import Guide, GuideW_ObjRnn, GuideW_ImageSoFar, GuideZ, ParamY
from data import split, load_data, data_params
from opt.utils import make_output_dir, append_line, describe_env, md5sum, throttle
from vis import frames_to_tensor, latents_to_tensor, overlay_multiple_window_outlines, frames_to_rgb_list

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

def run_svi(mod, batches, num_epochs, hook, output_path, save_period, elbo_scale=1.0):
    t0 = time.time()
    num_batches = len(batches)

    def per_param_optim_args(module_name, param_name):
        return {'lr': 1e-4}

    svi = SVI(mod.model, mod.guide,
              optim.Adam(per_param_optim_args),
              loss=Trace_ELBO())

    for i in range(num_epochs):
        for j, batch in enumerate(batches):
            step = num_batches*i+j
            loss = svi.step(batch)
            elbo = -loss * elbo_scale
            report_progress(i, j, step, elbo, t0, output_path)
            if not hook is None:
                hook(i, j, step)

        if save_period > 0 and (i+1) % save_period == 0:
            torch.save(mod.state_dict(),
                       os.path.join(output_path, 'params-{}.pytorch'.format(i+1)))

@throttle(1)
def report_progress(i, j, step, elbo, t0, output_path):
    elapsed = timedelta(seconds=time.time() - t0)
    print('\33[2K\repoch={}, batch={}, elbo={:.2f}, elapsed={}'.format(i, j, elbo, elapsed), end='')
    append_line(os.path.join(output_path, 'elbo.csv'),
                '{:.2f},{:.1f},{}'.format(elbo, elapsed.total_seconds(), step))

def opt_all(data, X_split, Y_split, cfg, args, output_path, log_to_cond):
    vis = visdom.Visdom()

    X_train, X_test = X_split
    Y_train, Y_test = Y_split

    if args.cuda:
        X_train = X_train.cuda()
        Y_train = Y_train.cuda()
        X_tes = X_test.cuda()
        Y_test = Y_test.cuda()

    batch_size = X_train[0].size(0)

    # Produce visualisations for the first train & test data points
    # where possible.
    if len(X_test) > 0:
        X_vis = torch.cat((X_train[0][0:1], X_test[0][0:1]))
        Y_vis = torch.cat((Y_train[0][0:1], Y_test[0][0:1]))
    else:
        X_vis = X_train[0][0:1]
        Y_vis = Y_train[0][0:1]

    model = Model(cfg,
                  delta_w=True, # Previous experiment use delta style here only.
                  use_cuda=args.cuda)
    guide = Guide(cfg,
                  dict(guide_w=GuideW_ObjRnn(cfg, dedicated_t0=False),
                       #guide_w=GuideW_ImageSoFar(cfg, model),
                       guide_y=ParamY(cfg),
                       guide_z=GuideZ(cfg, dedicated_t0=False)),
                  use_cuda=args.cuda)

    dynair = DynAIR(cfg, model, guide, use_cuda=args.cuda)

    def hook(epoch, batch, step):
        vis_hook(args.vis, vis, dynair, X_vis, Y_vis, epoch, batch, step)
        dynair.clear_cache()
        print()
        pp(dynair.cache_stats())

    run_svi(dynair, list(zip(X_train, Y_train)), args.epochs, hook, output_path, args.s,
            elbo_scale=1.0/(cfg.seq_length*batch_size))


def opt_bkg(data, X_split, Y_split, cfg, args, output_path, log_to_cond):
    vis = visdom.Visdom()
    vae = VAE(ParamY(cfg), DecodeBkg(cfg), cfg.y_size, use_cuda=args.cuda)

    X_train, _ = X_split
    # Extract backgrounds from the input sequences.
    batches = X_train.mode(2)[0].view(X_train.size()[0:2] + (-1,))
    if args.cuda:
        batches = batches.cuda()

    vis_batch = batches[0, 0:10]
    batch_size = batches.size(1)

    if args.vis > 0:
        vis.images(vis_batch.cpu().view(-1, cfg.num_chan, cfg.image_size, cfg.image_size), nrow=10)

    def hook(epoch, batch, step):
        if args.vis > 0 and (step + 1) % args.vis == 0:
            x_mean = vae.recon(vis_batch).cpu().view(-1, cfg.num_chan, cfg.image_size, cfg.image_size)
            vis.images(frames_to_rgb_list(x_mean), nrow=10)

    run_svi(vae, batches, args.epochs, hook, output_path, args.s, elbo_scale=1.0/batch_size)


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
    parser.add_argument('-s', type=int, default=0,
                        help='save parameters (zero disables, otherwise specifies period in epochs')
    parser.add_argument('--cuda', action='store_true', default=False, help='use CUDA')

    parser.add_argument('--y-size', type=int, default=50, help='size of y')
    parser.add_argument('--z-size', type=int, default=50, help='size of z')
    parser.add_argument('--window-size', type=int, default=22,
                        help='size of the object window')

    subparsers = parser.add_subparsers(dest='target')
    all_parser = subparsers.add_parser('all')
    bkg_parser = subparsers.add_parser('bkg')
    all_parser.set_defaults(main=opt_all)
    bkg_parser.set_defaults(main=opt_bkg)


    args = parser.parse_args()

    data = load_data(args.data_path)
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
    log_to_cond('target: {}'.format(args.target))
    log_to_cond('data path: {}'.format(args.data_path))
    log_to_cond('data md5: {}'.format(md5sum(args.data_path)))
    log_to_cond('data split: {}/{}'.format(len(X_split[0]), len(X_split[1])))
    log_to_cond(cfg)

    args.main(data, X_split, Y_split, cfg, args, output_path, log_to_cond)
