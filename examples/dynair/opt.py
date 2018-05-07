import os
import argparse
from functools import partial

from dynair import Config
from data import split, load_data, data_params
from opt.utils import make_output_dir, append_line, describe_env, md5sum
from opt.all import opt_all
from opt.bkg import opt_bkg

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
    parser.add_argument('-g', action='store_true', default=False,
                        help='record norm of gradient during optimisation')
    parser.add_argument('-t', type=float, default=0.0,
                        help='throttle progress updates')
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

    all_parser.add_argument('--bkg-params',
                            help='path to pre-trained background model/guide parameters')
    all_parser.add_argument('--fix-bkg-params', action='store_true', default=False,
                            help='do not optimise the background model/guide parameters')

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

    args.main(X_split, Y_split, cfg, args, output_path, log_to_cond)
