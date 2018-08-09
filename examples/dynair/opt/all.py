from functools import partial
from pprint import pprint as pp

import visdom
import torch
import torch.nn as nn

from dynair import DynAIR
from model import Model, DecodeObj, DecodeObjDepth, DecodeBkg, WTransition, ZTransition
from guide import (Guide, GuideW_ObjRnn, GuideW_ImageSoFar, GuideZ, GuideY, CombineMixin,
                   ImgEmbedMlp, ImgEmbedResNet, ImgEmbedId, InputCnn, WindowCnn)
from modules import MLP, ResNet
from opt.run_svi import run_svi
from opt.utils import md5sum, parse_cla
from vis import overlay_multiple_window_outlines

def run_vis(vis, dynair, X, Y, epoch, batch):
    n = X.size(0)
    frames, ws, _, _, extra_frames, extra_ws = dynair.infer(X, Y, 15)

    for k in range(n):
        out = overlay_multiple_window_outlines(dynair.cfg, frames[k], ws[k,:,:,0:3], Y[k])
        vis.images(X[k].cpu(), nrow=10,
                   opts=dict(title='input {} after epoch {} batch {}'.format(k, epoch, batch)))
        vis.images(out.cpu(), nrow=10,
                   opts=dict(title='recon {} after epoch {} batch {}'.format(k, epoch, batch)))

        out = overlay_multiple_window_outlines(dynair.cfg, extra_frames[k], extra_ws[k,:,:,0:3], Y[k])
        vis.images(out.cpu(), nrow=10,
                   opts=dict(title='extra {} after epoch {} batch {}'.format(k, epoch, batch)))

def hook(vis_period, vis, dynair, X_train, Y_train, X_test, Y_test, epoch, batch, step):
    if vis_period > 0 and (step+1) % vis_period == 0:
        # Produce visualisations for the first train & test data points
        # where possible.
        if len(X_test) > 0:
            X_vis = torch.cat((X_train[0][0:1], X_test[0][0:1]))
            Y_vis = torch.cat((Y_train[0][0:1], Y_test[0][0:1]))
        else:
            X_vis = X_train[0][0:1]
            Y_vis = Y_train[0][0:1]
        run_vis(vis, dynair, X_vis, Y_vis, epoch, batch)
    dynair.clear_cache()
    # print()
    # pp(dynair.cache_stats())

def load_bkg_params(dynair, path):
    print('loading {}'.format(path))
    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    dynair.guide.guide_y.load_state_dict(get_params_by_prefix(state_dict, 'encode'))
    dynair.model._decode_bkg.load_state_dict(get_params_by_prefix(state_dict, 'decode'))

def get_params_by_prefix(state_dict, prefix):
    return dict((k.replace(prefix + '.', '', 1), v)
                for k, v in state_dict.items()
                if k.startswith(prefix))

def is_bkg_param(module_name, param_name):
    return ((module_name == 'guide' and param_name.startswith('guide_y')) or
            (module_name == 'model' and param_name.startswith('_decode_bkg')))

arch_lookup = dict(mlp=MLP, resnet=ResNet)

def build_module(cfg, use_cuda):
    decode_bkg, guide_y = bkg_modules(cfg) if cfg.use_bkg_model else (None, None)

    arch, width, *hids = parse_cla('mlp|resnet-half|full', cfg.decode_obj)
    decode_obj = DecodeObj(cfg,
                           partial(arch_lookup[arch], hids=hids),
                           use_half_z=dict(half=True, full=False)[width],
                           alpha_bias=(0.0 if cfg.use_bkg_model else -0.8))

    arch, *hids = parse_cla('mlp', cfg.decode_obj_depth)
    decode_obj_depth = DecodeObjDepth(cfg, hids) if cfg.use_depth else None

    sd_opt, arch, *hids = parse_cla('sdparam|sdstate-mlp|resnet', cfg.w_transition)
    w_transition = WTransition(cfg,
                               partial(arch_lookup[arch], hids=hids),
                               state_dependent_sd=dict(sdstate=True, sdparam=False)[sd_opt])

    sd_opt, arch, *hids = parse_cla('sdparam|sdstate-mlp|resnet', cfg.z_transition)
    z_transition = ZTransition(cfg,
                               partial(arch_lookup[arch], hids=hids),
                               state_dependent_sd=dict(sdstate=True, sdparam=False)[sd_opt])

    model = Model(cfg,
                  dict(decode_obj=decode_obj,
                       decode_obj_depth=decode_obj_depth,
                       decode_bkg=decode_bkg,
                       w_transition=w_transition,
                       z_transition=z_transition),
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
        guide_w = GuideW_ObjRnn(cfg, hids, x_embed, rnn_cell_use_tanh=dict(tanh=True, relu=False)[nl])
    elif cfg.guide_w.startswith('isf'):
        _, block, bkg, arch, *hids = parse_cla('isf-block|noblock-bkg|nobkg-mlp|resnet', cfg.guide_w)
        output_net = partial(arch_lookup[arch], hids=hids)
        guide_w = GuideW_ImageSoFar(cfg, model,
                                    partial(CombineMixin, x_embed, output_net),
                                    dict(block=True, noblock=False)[block],
                                    include_bkg=dict(bkg=True, nobkg=False)[bkg])
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

    # TODO: Drop this if?
    if True:
        aux, arch, *hids = parse_cla('auxignore|auxmain|auxside-mlp|resnet', cfg.guide_z)
        output_net = partial(arch_lookup[arch], hids=hids)
        guide_z = GuideZ(cfg,
                         partial(CombineMixin, x_att_embed, output_net),
                         aux_size=guide_w.aux_size,
                         aux_method=aux[3:])
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

def weight_init(m):
    def mlp_init(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight, torch.nn.init.calculate_gain('relu'))
    def norm_init(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)

    if type(m) == MLP:
        m.apply(mlp_init)
    elif m._get_name().startswith('Normal'):
        # This overwrites the custom init. implemented in the module.
        #It causes the magnitude of gradients to increase by a factor
        #of ~2, and optimisation to fail. (elbo decrease after first
        #few epochs.) This suggests that the init of these modules may
        #be important?

        #m.apply(norm_init)
        pass
    elif type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight)
    elif type(m) == nn.RNNCell:
        # Assuming ReLU, although tanh can be used.
        torch.nn.init.xavier_normal_(m.weight_ih, torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_normal_(m.weight_hh, torch.nn.init.calculate_gain('relu'))
    else:
        for c in m.children():
            weight_init(c)

def opt_all(X_split, Y_split, cfg, args, use_cuda, output_path, log_to_cond):

    X_train, X_test = X_split
    Y_train, Y_test = Y_split

    if use_cuda:
        X_train = X_train.cuda()
        Y_train = Y_train.cuda()
        X_test = X_test.cuda()
        Y_test = Y_test.cuda()

    batch_size = X_train[0].size(0)
    seq_length = X_train[0].size(1)
    elbo_scale = 1.0/(seq_length*batch_size)

    dynair = build_module(cfg, use_cuda)

    if args.show:
        print(dynair)

    if args.xinit:
        weight_init(dynair)
    log_to_cond('xavier init: {}'.format(args.xinit))

    if args.bkg_params is not None:
        load_bkg_params(dynair, args.bkg_params)
        log_to_cond('bkg params md5: {}'.format(md5sum(args.bkg_params)))

    log_to_cond('fixed bkg params: {}'.format(args.fix_bkg_params))

    def optim_args(module_name, param_name):
        if args.fix_bkg_params and is_bkg_param(module_name, param_name):
            lr = 0.0
        else:
            lr = 1e-4
        ret = {'lr': lr}
        if not args.w is None:
            ret['weight_decay'] = args.w
        return ret

    run_svi(dynair, list(zip(X_train, Y_train)), args.epochs, optim_args,
            partial(hook, args.v, visdom.Visdom(), dynair,
                    X_train, Y_train, X_test, Y_test),
            output_path,
            args.s, args.t, args.log_elbo, args.g, args.n, args.c,
            test_batches=list(zip(X_test, Y_test)), elbo_scale=elbo_scale)
    print()
