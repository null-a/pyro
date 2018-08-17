import subprocess
import os
import argparse
import glob
import random
import numpy as np

# This generates a bunch of movies and writes them to disk as
# individual frames.

def main(n, out_path):

    # Get all available backgrounds.
    bkg_fns = glob.glob('/Users/paul/Downloads/bkg/*.jpg')
    random.shuffle(bkg_fns)
    num_bkg = len(bkg_fns)
    assert num_bkg > 0, 'no backgrounds found'

    for i in range(n):

        # Make a sub-directory.
        movie_path ='./{}/movie{}'.format(out_path, i)
        os.mkdir(movie_path)

        # Make frames.
        subprocess.check_call(['python3', 'cube.py', '-o', movie_path])

        # Add background.
        bkg_fn = bkg_fns[i] # [np.random.randint(num_bkg)]
        subprocess.call(['python3', 'add_bkg.py', '-m', movie_path, '-b', bkg_fn])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int)
    parser.add_argument('-o', type=str)
    args = parser.parse_args()
    assert args.n is not None, 'argument error'
    assert args.o is not None, 'argument error'
    main(args.n, args.o)
