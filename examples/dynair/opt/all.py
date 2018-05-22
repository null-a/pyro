from functools import partial
from pprint import pprint as pp

import visdom
import torch
import torch.nn as nn

from dynair import DynAIR
from model import Model, DecodeObj, DecodeBkg, WTransition, ZTransition
from guide import (Guide, GuideW_ObjRnn, GuideW_ImageSoFar, GuideZ, GuideY, CombineMixin,
                   ImgEmbedMlp, ImgEmbedResNet, ImgEmbedId, InputCnn, WindowCnn)
from modules import MLP, ResNet
from opt.run_svi import run_svi
from opt.utils import md5sum, parse_cla
from vis import overlay_multiple_window_outlines

def run_vis(vis, dynair, X, Y, epoch, batch):
    n = X.size(0)
    frames, ws, extra_frames, extra_ws = dynair.infer(X, Y, 15)

    for k in range(n):
        out = overlay_multiple_window_outlines(dynair.cfg, frames[k], ws[k], Y[k])
        vis.images(X[k].cpu(), nrow=10,
                   opts=dict(title='input {} after epoch {} batch {}'.format(k, epoch, batch)))
        vis.images(out.cpu(), nrow=10,
                   opts=dict(title='recon {} after epoch {} batch {}'.format(k, epoch, batch)))

        out = overlay_multiple_window_outlines(dynair.cfg, extra_frames[k], extra_ws[k], Y[k])
        vis.images(out.cpu(), nrow=10,
                   opts=dict(title='extra {} after epoch {} batch {}'.format(k, epoch, batch)))

def hook(vis_period, vis, dynair, X, Y, epoch, batch, step):
    if vis_period > 0 and (step+1) % vis_period == 0:
        run_vis(vis, dynair, X, Y, epoch, batch)
    dynair.clear_cache()
    # print()
    # pp(dynair.cache_stats())

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

def build_module(cfg, use_cuda):
    decode_bkg, guide_y = bkg_modules(cfg)
    model = Model(cfg,
                  dict(decode_obj=DecodeObj(cfg, [100, 100]),
                       decode_bkg=decode_bkg,
                       w_transition=WTransition(cfg, 50),
                       z_transition=ZTransition(cfg, 50)),
                  delta_w=cfg.model_delta_w,
                  delta_z=cfg.model_delta_z,
                  use_cuda=use_cuda)

    if cfg.guide_input_embed.startswith('mlp'):
        _, *hids = parse_cla('mlp', cfg.guide_input_embed)
        x_embed = partial(ImgEmbedMlp, hids=hids)
    elif cfg.guide_input_embed.startswith('resnet'):
        _, *hids = parse_cla('resnet', cfg.guide_input_embed)
        x_embed = partial(ImgEmbedResNet, hids=hids)
    elif cfg.guide_input_embed == 'cnn':
        x_embed = InputCnn
    elif cfg.guide_input_embed == 'id':
        x_embed = ImgEmbedId
    else:
        raise Exception('unknown guide_input_embed: {}'.format(cfg.guide_input_embed))

    if cfg.guide_w.startswith('rnn'):
        _, nl, *hids = parse_cla('rnn-tanh|relu', cfg.guide_w)
        use_tanh = nl == 'tanh'
        guide_w = GuideW_ObjRnn(cfg, hids, x_embed, rnn_cell_use_tanh=use_tanh)
    elif cfg.guide_w.startswith('isf'):
        _, arch, *hids = parse_cla('isf-mlp|resnet', cfg.guide_w)
        if arch == 'mlp':
            output_net = partial(MLP,
                                 hids=hids,
                                 non_linear_layer=nn.ReLU,
                                 output_non_linearity=True)
        elif arch == 'resnet':
            output_net = partial(ResNet, hids=hids)
        else:
            raise Exception('impossible')

        guide_w = GuideW_ImageSoFar(cfg, model,
                                    partial(CombineMixin, x_embed, output_net))
    else:
        raise Exception('unknown guide_w: {}'.format(cfg.guide_w))

    if cfg.guide_window_embed.startswith('mlp'):
        _, *hids = parse_cla('mlp', cfg.guide_window_embed)
        x_att_embed = partial(ImgEmbedMlp, hids=hids)
    elif cfg.guide_window_embed.startswith('resnet'):
        _, *hids = parse_cla('resnet', cfg.guide_window_embed)
        x_att_embed = partial(ImgEmbedResNet, hids=hids)
    elif cfg.guide_window_embed == 'cnn':
        x_att_embed = WindowCnn
    elif cfg.guide_window_embed == 'id':
        x_att_embed = ImgEmbedId
    else:
        raise Exception('unknown guide_window_embed: {}'.format(cfg.guide_window_embed))

    if cfg.guide_z.startswith('mlp') or cfg.guide_z.startswith('resnet'):
        arch, *hids = parse_cla('mlp|resnet', cfg.guide_z)
        if arch == 'mlp':
            output_net = partial(MLP,
                                 hids=hids,
                                 non_linear_layer=nn.ReLU,
                                 output_non_linearity=True)
        elif arch == 'resnet':
            output_net = partial(ResNet, hids=hids)
        else:
            raise Exception('impossible')

        guide_z = GuideZ(cfg, partial(CombineMixin, x_att_embed, output_net))
    else:
        raise Exception('unknown guide_z: {}'.format(cfg.guide_z))

    guide = Guide(cfg,
                  dict(guide_w=guide_w,
                       guide_y=guide_y,
                       guide_z=guide_z
                  ),
                  use_cuda=use_cuda)

    return DynAIR(cfg, model, guide, use_cuda=use_cuda)

def bkg_modules(cfg):
    return DecodeBkg(cfg), GuideY(cfg)

def opt_all(X_split, Y_split, cfg, args, output_path, log_to_cond):

    X_train, X_test = X_split
    Y_train, Y_test = Y_split

    if args.cuda:
        X_train = X_train.cuda()
        Y_train = Y_train.cuda()
        X_test = X_test.cuda()
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

    dynair = build_module(cfg, args.cuda)

    if args.show:
        print(dynair)

    if args.bkg_params is not None:
        load_bkg_params(dynair, args.bkg_params)
        log_to_cond('bkg params path: {}'.format(args.bkg_params))
        log_to_cond('bkg params md5: {}'.format(md5sum(args.bkg_params)))

    log_to_cond('fixed bkg params: {}'.format(args.fix_bkg_params))

    def optim_args(module_name, param_name):
        if args.fix_bkg_params and is_bkg_param(module_name, param_name):
            return {'lr': 0.0}
        else:
            return {'lr': 1e-4}

    run_svi(dynair, list(zip(X_train, Y_train)), args.epochs, optim_args,
            partial(hook, args.v, visdom.Visdom(), dynair, X_vis, Y_vis),
            output_path, args.s, args.t, args.g, args.c,
            elbo_scale=1.0/(cfg.seq_length*batch_size))
