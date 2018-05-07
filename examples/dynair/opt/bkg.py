import visdom
import torch

from vae import VAE
from model import DecodeBkg
from guide import ParamY
from opt.run_svi import run_svi
from vis import frames_to_rgb_list

def opt_bkg(X_split, Y_split, cfg, args, output_path, log_to_cond):
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

    optim_args = {'lr': 1e-4}

    run_svi(vae, batches, args.epochs, optim_args, hook,
            output_path, args.s, args.g, elbo_scale=1.0/batch_size)
