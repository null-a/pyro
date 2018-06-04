import argparse
import json
import torch
from pyro.infer import Trace_ELBO

from dynair import config
from data import load_data, data_params, split
from opt.all import build_module

def elbo_main(dynair, X, Y, args):
    X_batches, _ = split(X[args.start:args.end], args.batch_size, 0)
    Y_batches, _ = split(Y[args.start:args.end], args.batch_size, 0)
    elbo = Trace_ELBO(num_particles=args.n)
    loss = 0.0
    with torch.no_grad():
        for batch in zip(X_batches, Y_batches):
            loss += elbo.loss(dynair.model, dynair.guide, batch)
    print(loss / float(dynair.cfg.seq_length * (args.end - args.start)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('module_config_path')
    parser.add_argument('params_path')

    parser.add_argument('start', type=int, help='start index of test set')
    parser.add_argument('end', type=int, help='end index of test set')

    parser.add_argument('-b', '--batch-size', type=int, required=True, help='batch size')

    subparsers = parser.add_subparsers(dest='target')
    elbo_parser = subparsers.add_parser('elbo')
    elbo_parser.set_defaults(main=elbo_main)

    elbo_parser.add_argument('-n', type=int, default=1, help='number of particles')

    args = parser.parse_args()

    data = load_data(args.data_path)
    X, Y = data

    with open(args.module_config_path) as f:
        module_config = json.load(f)
    cfg = config(module_config, data_params(data))

    dynair = build_module(cfg, use_cuda=False)
    dynair.load_state_dict(torch.load(args.params_path, map_location=lambda storage, loc: storage))

    args.main(dynair, X, Y, args)

if __name__ == '__main__':
    main()
