import random
import glob
import math
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
from PIL.ImageTransform import AffineTransform

# Turn multiple frames (gif,tiff,etc.) into a single strip of frames
# with:
# montage out.tiff -tile x1 -geometry +2 out.png
# http://www.imagemagick.org/Usage/montage/

SIZE = 50

def interp1(n, a, b, t):
    return a + (b - a) * (t / float(n - 1))

# Interpolate between (x1,y1) and (x2,y2).
def interp(n, xy1, xy2, t):
    x1, y1 = xy1
    x2, y2 = xy2
    return (interp1(n, x1, x2, t), interp1(n, y1, y2, t))

# I'm saving in TIFF format since using GIF causes frames to be
# dropped. Will convert to GIF (or some other format) using
# ImageMagick. e.g. `convert out.tiff out.gif`
def save_seq(frames):
    first = frames[0]
    rest = frames[1:]
    first.save('out.tiff', save_all=True, append_images=rest)

# triangle
def mk_s1(color):
    img = Image.new('RGBA', (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.polygon([SIZE/2, 0, (1 - 0.866) * SIZE/2, SIZE*0.75, 1.866 * SIZE/2, SIZE*0.75], fill=color)
    return img

# square
def mk_s2(color):
    img = Image.new('RGBA', (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.polygon([SIZE/2, 0, SIZE, SIZE/2, SIZE/2, SIZE, 0, SIZE/2], fill=color)
    return img

# egg timer
def mk_s3(color):
    img = Image.new('RGBA', (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.polygon([SIZE/2, 0,  SIZE/2, SIZE, 0, SIZE/2,  SIZE, SIZE/2], fill=color)
    return img

# https://www.emojione.com/
def load_sprites():
    fns = glob.glob('/Users/paul/Downloads/emoji/*.png')
    return [Image.open(fn).convert('RGBA').resize((SIZE, SIZE), resample=Image.BILINEAR) for fn in fns]

SHAPES = load_sprites()

#SHAPES = [mk_s1, mk_s2, mk_s3]
#COLORS = [['red', 'green'], ['green', 'blue'], ['blue', 'red']]

def checker_board(n):
    img = Image.new('RGBA', (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for i in range(n):
        for j in range(n):
            col= 'silver' if j%2 == i%2 else 'black'
            draw.rectangle([(i*SIZE/n, j*SIZE/n), ((i+1)*SIZE/n, (j+1)*SIZE/n)], fill=col)
    return img


def rot(img, angle):
    return img.rotate(angle)

def scale(img, s):
    img = img.transform((SIZE, SIZE),
                        AffineTransform((s, 0, 0,
                                         0, s, 0)),
                        resample=Image.BILINEAR)
    # Translate so that we are effectively scaling around the centre.
    # This makes working with result easier to think about.
    return trans(img, SIZE/2*(1-1/s), SIZE/2*(1-1/s))

def trans(img, x, y):
    return img.transform((SIZE, SIZE),
                         AffineTransform((1, 0, -x,
                                          0, 1, -y)),
                         resample=Image.BILINEAR)

# Each object is located in the centre of an image that has the same
# size as the output. I want to think about placing that in a frame
# that has 0,0 at the top-left. This help does that. i.e. It takes an
# object, and positions it relative to the top-left.

# e.g. position(s1, 0, 0) places the centre of the object as the top
# left of the output.

def position(img, x, y):
    return trans(img, x - SIZE/2, y - SIZE/2)


def in_frame(xy, width):
    m = 5 # margin.
    x, y = xy
    return (m <= x < (width-m)) and (m <= y < (width-m))

# A crude way to check whether some path will cause the object to ever
# appear in shot.
def intersects_frame(xy1, xy2, n, width):
    return any(in_frame(interp(n, xy1, xy2, t), width) for t in range(n))

def sample_end_points(n):
    s = 2
    x1, y1, x2, y2 = tuple([i-SIZE/s for i in np.random.randint(s * SIZE, size=4)])
    xy1 = (x1, y1)
    xy2 = (x2, y2)
    if intersects_frame(xy1, xy2, n, SIZE):
        return xy1, xy2
    else:
        #print('reject')
        return sample_end_points(n)

# Object start somewhere in frame and move towards to center of the frame.

def sample_end_points():

    # Avoid placing objects in a border of the following width. This
    # ensures objects are not partially out of shot.
    b = 7
    assert SIZE == 50, 'SIZE != 50 -- border width needs tuning for this new image size.'

    x0 = np.random.uniform(b, SIZE-b)
    x1 = np.random.uniform(b, SIZE-b)
    y0 = np.random.uniform(b, SIZE-b)
    y1 = np.random.uniform(b, SIZE-b)

    # Compute path length.
    path_length = math.sqrt((x0-x1)**2 + (y0-y1)**2)
    # print(path_length)

    MIN_PATH_LENGTH = SIZE / 2

    if path_length < MIN_PATH_LENGTH:
        return sample_end_points()
    else:
        return (x0, y0), (x1, y1)

def sample_shade():
    img = Image.new('RGBA', (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    gray_level = (30,) * 3 + (128,)

    hyp = SIZE * (np.sqrt(2) / 2.0)
    theta = np.random.uniform() * 2 * np.pi

    x = np.cos(theta) * hyp
    y = np.sin(theta) * hyp

    draw.line([(x+SIZE/2,y+SIZE/2), (-x+SIZE/2,-y+SIZE/2)], fill=gray_level)

    # Pick a fill seed that is not on the boundary.
    h = SIZE/4
    xf = np.cos(theta + np.pi/2) * h
    yf = np.sin(theta + np.pi/2) * h

    ImageDraw.floodfill(img, (xf + SIZE/2, yf + SIZE/2), gray_level)

    img = img.filter(ImageFilter.GaussianBlur(1))

    return img

def sample_natural_scene_bkg(size):
    # I'm using some images from here:
    # http://cvcl.mit.edu/database.htm
    # So this assumes images are 256x256 JPGs.
    fns = glob.glob('/Users/paul/Downloads/bkg/*.jpg')
    n = len(fns)
    im = Image.open(fns[np.random.randint(n)])
    assert im.size == (256, 256)

    # Crop a (256 - t) sized image out of the full image to help avoid
    # duplicating images.
    t = 64
    x = np.random.randint(t)
    y = np.random.randint(t)

    cropped = im.crop((x,y,x+256-t,y+256-t)).resize((size, size), resample=Image.BILINEAR).convert('RGBA')
    #arr = img_to_arr(cropped)
    return cropped

def sample_scene():

    # number of frames
    n = 20

    bkg = sample_natural_scene_bkg(SIZE)
    #bkg = checker_board(4)
    #bkg = Image.new('RGBA', (SIZE, SIZE), (0, 0, 0, 255))

    #shade = sample_shade()

    # Sample objects
    MAX_NUM_OBJS = 3
    num_objs = np.random.randint(MAX_NUM_OBJS) + 1

    # I choose not to use a particular sprite more than once.
    assert len(SHAPES) == MAX_NUM_OBJS
    sprites_ix = list(range(MAX_NUM_OBJS))
    random.shuffle(sprites_ix)

    objs = []
    for i in range(num_objs):
        xy1, xy2 = sample_end_points()
        objs.append(dict(
            xy1=xy1,
            xy2=xy2,
            shape=SHAPES[sprites_ix[i]],
            rot_init=np.random.uniform(360),
            rot_vel=np.random.uniform(0, 15)
        ))
    #print(objs)

    frames = []
    for t in range(n):
        acc = bkg
        for obj in objs:
            x, y = interp(n, obj['xy1'], obj['xy2'], t)
            s = obj['shape']
            s = rot(s, obj['rot_init'] + obj['rot_vel'] * t)
            s = scale(s, 4) # add variable scale, possibly dynamic
            s = position(s, x, y)
            acc = Image.alpha_composite(acc, s)

        #acc = Image.alpha_composite(acc, shade)

        frames.append(acc)

    return frames, num_objs


def sample_dataset(n):
    seqs, counts = tuple(zip(*[sample_scene() for _ in range(n)]))

    seqs_arr = np.stack(np.stack(img_to_arr(frame) for frame in seq) for seq in seqs)
    counts_arr = np.array(counts, dtype=np.int8)
    return seqs_arr, counts_arr

def img_to_arr(img):
    assert img.mode == 'RGBA'
    channels = 4
    w, h = img.size
    arr = np.fromstring(img.tobytes(), dtype=np.uint8)
    return arr.reshape(w * h, channels).T.reshape(channels, h, w)



# Save a tiff
frames, _ = sample_scene()
save_seq(frames)

# Convert a frame to correct array format for the model.
# arr = img_to_arr(frames[0])
# print(arr.shape)

# Using with `imshow` requires an additional `transpose(1, 2, 0)`.
# plt.imshow(arr.transpose(1, 2, 0))
# plt.show()

# Make a dataset
# seqs_arr, counts_arr = sample_dataset(1000)
# print(seqs_arr.shape)
# print(counts_arr)
# np.savez_compressed('multi_obj.npz', X=seqs_arr, Y=counts_arr)
