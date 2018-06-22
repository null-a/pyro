import os
import sys
import argparse
from functools import partial
import json
import torch.cuda

from dynair import config
from data import split, load_data, data_params
from opt.utils import make_output_dir, append_line, describe_env, md5sum
from opt.all import opt_all
from opt.bkg import opt_bkg

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('-b', '--batch-size', type=int, required=True, help='batch size')
    parser.add_argument('-l', type=int, help='sequence length')
    parser.add_argument('-e', '--epochs', type=int, default=10**6,
                        help='number of optimisation epochs to perform')
    parser.add_argument('--hold-out', type=int, default=0,
                        help='number of batches to hold out')
    parser.add_argument('-v', type=int, default=0,
                        help='visualise inferences during optimisation (zero disables, otherwise specifies period in steps)')
    parser.add_argument('-o', default='./runs', help='base output path')
    parser.add_argument('-s', type=int, default=0,
                        help='save parameters (zero disables, otherwise specifies period in epochs')
    parser.add_argument('-g', action='store_true', default=False,
                        help='perform inf/nan checks on grad before taking a step')
    parser.add_argument('-n', action='store_true', default=False,
                        help='record norm of gradient during optimisation')
    parser.add_argument('-t', type=float, default=0.0,
                        help='throttle progress updates')
    parser.add_argument('-c', type=float, default=float('inf'),
                        help='threshold at which to clip the l2 norm of the gradient')
    parser.add_argument('--cpu', action='store_true', default=False, help='always use cpu')

    parser.add_argument('--y-size', type=int, default=50, help='size of y')
    parser.add_argument('--z-size', type=int, default=50, help='size of z')
    parser.add_argument('--window-size', type=int, default=22,
                        help='size of the object window')

    parser.add_argument('--w-transition', default='sdparam-mlp-50',
                        help='architecture of w transition')
    parser.add_argument('--z-transition', default='sdparam-mlp-50',
                        help='architecture of z transition')

    parser.add_argument('--decode-obj', default='mlp-100-100',
                        help='architecture of decode object network')

    parser.add_argument('--model-delta-w', action='store_true', default=False,
                        help='use w transition output as delta from previous value to next mean')
    parser.add_argument('--model-delta-z', action='store_true', default=False,
                        help='use z transition output as delta from previous value to next mean')

    parser.add_argument('--use-depth', action='store_true', default=False,
                        help='render scene using explicit object depths')

    parser.add_argument('--guide-w', default='rnn-tanh-200-200',
                        help='architecture of guide for w')
    parser.add_argument('--guide-z', default='auxignore-mlp-100',
                        help='architecture of guide for z')
    parser.add_argument('--guide-input-embed', default='mlp-500-200',
                        help='architecture of input embedding network')
    parser.add_argument('--guide-window-embed', default='mlp-100-100',
                        help='architecture of window embedding network')

    parser.add_argument('--show', action='store_true', default=False,
                        help='show module information')

    parser.add_argument('--log-elbo', type=int, default=0,
                        help='record the elbo of the full training set (and test set when present) during optimisation ' +
                             '(zero disables, otherwise specifies period in epochs)')

    parser.add_argument('--desc', help='argument value is written to desc.txt in output directory to aid identification')

    subparsers = parser.add_subparsers(dest='target')
    all_parser = subparsers.add_parser('all')
    bkg_parser = subparsers.add_parser('bkg')
    all_parser.set_defaults(main=opt_all)
    bkg_parser.set_defaults(main=opt_bkg)

    all_parser.add_argument('--bkg-params',
                            help='path to pre-trained background model/guide parameters')
    all_parser.add_argument('--fix-bkg-params', action='store_true', default=False,
                            help='do not optimise the background model/guide parameters')

    args = parser.parse_args()

    data = load_data(args.data_path, args.l)
    X, Y, _ = data # (sequences, counts)
    X_split = split(X, args.batch_size, args.hold_out)
    Y_split = split(Y, args.batch_size, args.hold_out)

    use_cuda = torch.cuda.is_available() and not args.cpu
    print('using cuda: {}'.format(use_cuda))

    module_config = dict(w_size=3,
                         y_size=args.y_size,
                         z_size=args.z_size,
                         window_size=args.window_size,
                         use_depth=args.use_depth,
                         w_transition=args.w_transition,
                         z_transition=args.z_transition,
                         decode_obj=args.decode_obj,
                         model_delta_w=args.model_delta_w,
                         model_delta_z=args.model_delta_z,
                         guide_w=args.guide_w,
                         guide_z=args.guide_z,
                         guide_input_embed=args.guide_input_embed,
                         guide_window_embed=args.guide_window_embed)
    cfg = config(module_config, data_params(data))

    output_path = make_output_dir(args.o)
    print('output path: {}'.format(output_path))
    log_to_cond = partial(append_line, os.path.join(output_path, 'condition.txt'))
    log_to_cond(describe_env())
    log_to_cond('target: {}'.format(args.target))
    log_to_cond('data md5: {}'.format(md5sum(args.data_path)))
    log_to_cond('data split: {}/{}'.format(len(X_split[0]), len(X_split[1])))
    log_to_cond('data seq length: {}'.format(X.size(1)))
    log_to_cond('gradient clipping threshold: {:e}'.format(args.c))
    log_to_cond(cfg)

    append_line(os.path.join(output_path, 'argv.txt'), ' '.join(sys.argv))
    if not args.desc is None:
        append_line(os.path.join(output_path, 'desc.txt'), args.desc)
    with open(os.path.join(output_path, 'module_config.json'), 'w') as f:
        json.dump(module_config, f)

    args.main(X_split, Y_split, cfg, args, use_cuda, output_path, log_to_cond)
