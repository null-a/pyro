from functools import partial
from pprint import pprint as pp

import visdom
import torch

from dynair import DynAIR
from model import Model, DecodeBkg
from guide import Guide, GuideW_ObjRnn, GuideW_ImageSoFar, GuideZ, ParamY
from opt.run_svi import run_svi
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

def hook(vis_period, vis, dynair, X, Y, epoch, batch, step):
    if vis_period > 0 and (step+1) % vis_period == 0:
        run_vis(vis, dynair, X, Y, epoch, batch)
    dynair.clear_cache()
    print()
    pp(dynair.cache_stats())

def load_bkg_params(dynair, path):
    print('loading {}'.format(path))
    state_dict = dict((bkg_to_dynair_name_map(k),v) for k, v in
                      torch.load(path, map_location=lambda storage, loc: storage).items())
    dynair.load_state_dict(state_dict, strict=False)

def bkg_to_dynair_name_map(name):
    if name.startswith('encode'):
        return name.replace('encode', 'guide.guide_y', 1)
    elif name.startswith('decode'):
        return name.replace('decode', 'model._decode_bkg', 1)
    else:
        raise 'unexpected parameter name encountered'

def is_bkg_param(module_name, param_name):
    return ((module_name == 'guide' and param_name.startswith('guide_y')) or
            (module_name == 'model' and param_name.startswith('_decode_bkg')))

def opt_all(X_split, Y_split, cfg, args, output_path):

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

    if args.bkg_params is not None:
        load_bkg_params(dynair, args.bkg_params)

    def optim_args(module_name, param_name):
        if args.fix_bkg_params and is_bkg_param(module_name, param_name):
            return {'lr': 0.0}
        else:
            return {'lr': 1e-4}

    run_svi(dynair, list(zip(X_train, Y_train)), args.epochs, optim_args,
            partial(hook, args.vis, visdom.Visdom(), dynair, X_vis, Y_vis),
            output_path, args.s,
            elbo_scale=1.0/(cfg.seq_length*batch_size))
