import visdom
import torch

from vae import VAE
from opt.all import bkg_modules
from opt.run_svi import run_svi

def opt_bkg(X_split, Y_split, cfg, args, use_cuda, output_path, log_to_cond):
    vis = visdom.Visdom()
    decode_bkg, guide_y = bkg_modules(cfg)
    vae = VAE(guide_y, decode_bkg, cfg.y_size, use_cuda)

    if args.show:
        print(vae)

    X_train, X_test = X_split
    # Extract backgrounds from the input sequences.
    batches = X_train.mode(2)[0].view(X_train.size()[0:2] + (-1,))
    test_batches = X_test.mode(2)[0].view(X_test.size()[0:2] + (-1,)) if len(X_test) > 0 else []
    if use_cuda:
        batches = batches.cuda()
        test_batches = test_batches.cuda()

    vis_batch = batches[0, 0:10]
    batch_size = batches.size(1)

    if args.v > 0:
        vis.images(vis_batch.cpu().view(-1, cfg.num_chan, cfg.image_size, cfg.image_size), nrow=10)

    def hook(epoch, batch, step):
        if args.v > 0 and (step + 1) % args.v == 0:
            x_mean = vae.recon(vis_batch).cpu().view(-1, cfg.num_chan, cfg.image_size, cfg.image_size)
            vis.images(x_mean, nrow=10)

    optim_args = {'lr': 1e-4}
    if not args.w is None:
        optim_args['weight_decay'] = args.w

    run_svi(vae, batches, args.epochs, optim_args, hook,
            output_path, args.s, args.t, args.log_elbo, args.g, args.n, args.c,
            test_batches=test_batches, elbo_scale=1.0/batch_size)
    print()