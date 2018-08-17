import sys
import glob
import argparse
from PIL import Image
import numpy as np

#from matplotlib import pyplot as plt

def get_background(filename, crop_size, out_size):
    assert 0 < crop_size < 256

    im = Image.open(filename)
    assert im.size == (256, 256)

    x = np.random.randint(256 - crop_size)
    y = np.random.randint(256 - crop_size)

    cropped = im.crop((x,y,x+crop_size,y+crop_size))
    resized = cropped.resize((out_size, out_size), resample=Image.BILINEAR)
    out = resized.convert('RGBA')

    return out

# This is destructive, is overwrites the input frame.
def add_bkg_to_frame(frame_path, bkg):
    frame = Image.open(frame_path).convert('RGBA')
    print(frame)
    print(bkg)
    im = Image.alpha_composite(bkg, frame)
    im.save(frame_path)


# I'm using some images from here:
# http://cvcl.mit.edu/database.htm
# So this assumes images are 256x256 JPGs.

def add_bkg_to_movie(movie_path, bkg):
    frame_paths = glob.glob('{}/frame_*.png'.format(movie_path))
    for frame_path in frame_paths:
        add_bkg_to_frame(frame_path, bkg)


def main(bkg_filepath, movie_path):
    bkg = get_background(bkg_filepath, 180, 50)
    #plt.imshow(bkg)
    #plt.show()
    add_bkg_to_movie(movie_path, bkg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=str)
    parser.add_argument('-m', type=str)
    args = parser.parse_args()
    assert args.b is not None and args.m is not None, 'argument error'
    main(args.b, args.m)
