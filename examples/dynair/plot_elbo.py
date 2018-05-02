import os
import glob
import argparse
from os.path import dirname, abspath
from matplotlib import pyplot as plt

plt.style.use('seaborn-ticks')

def read(path):
    with open(path) as f:
        data = [[float(c.strip()) for c in line.split(',')] for line in f]
    return tuple(zip(*data))

def main(path, num_batches):

    runs = [(fn,) + read(fn) for fn in glob.glob(path)]

    for (fn, wall, elbo) in runs:

        label = dirname(abspath(fn)).split(os.sep)[-1]

        xs = range(len(elbo))
        if not num_batches is None:
            xs = [(float(x) / num_batches) for x in xs]

        plt.plot(xs, elbo, label=label)
        plt.xlabel('steps' if num_batches is None else 'epochs')
        plt.ylabel('elbo')

    plt.grid()
    plt.legend()

    # fig, ax1 = plt.subplots()

    # ax1.plot(xs, elbo, label=label)
    # ax1.set_xlabel('steps' if num_batches is None else 'epochs')
    # ax1.set_ylabel('elbo')
    # ax1.grid()
    # ax1.legend()

    # TODO: Re-enable to also show wall clock time.
    # ax2 = ax1.twiny()
    # ax2.plot([w/3600 for w in wall], [0] * len(wall), visible=False)

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('-n', '--num-batches', type=int,
                        help='number of batches per epoch, shows epochs on x axis')
    args = parser.parse_args()
    main(args.path, args.num_batches)
