import os
import glob
import argparse
from os.path import dirname, abspath
from matplotlib import pyplot as plt

plt.style.use('seaborn-ticks')

def read(path):
    with open(path) as f:
        data = [parse_line(line) for line in f]
    return tuple(zip(*data))

def parse_line(line):
    strs = [col.strip() for col in line.split(',')]
    return (float(strs[0]), # elbo
            float(strs[1]), # wall
            int(strs[2]))   # step

def main(path, num_batches):

    runs = [(fn,) + read(fn) for fn in glob.glob(path)]

    # print(runs)
    # assert False

    for (fn, elbo, wall, step) in runs:

        label = dirname(abspath(fn)).split(os.sep)[-1]

        plt.plot(step, elbo, label=label)
        plt.xlabel('step')
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
