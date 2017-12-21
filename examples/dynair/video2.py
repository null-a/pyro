import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from PIL.ImageTransform import AffineTransform

# Turn multiple frames (gif,tiff,etc.) into a single strip of frames
# with:
# montage out.tiff -tile x1 -geometry +2 out.png
# http://www.imagemagick.org/Usage/montage/

SIZE = 32

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

SHAPES = [mk_s1, mk_s2, mk_s3]
COLORS = [['red', 'green'], ['green', 'blue'], ['blue', 'red']]

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

def sample_scene():

    # number of frames
    n = 14

    bkg = checker_board(4)
    #bkg = Image.new('RGBA', (SIZE, SIZE), (0, 0, 0, 255))

    # Sample objects
    num_objs = 1#np.random.randint(3) + 1
    objs = []
    for i in range(num_objs):
        xy1, xy2 = sample_end_points(n)
        objs.append(dict(
            xy1=xy1,
            xy2=xy2,
            shape=np.random.randint(3),
            color=np.random.randint(2),
            rotation=np.random.randint(4) * 90#np.random.uniform(360)
        ))
    #print(objs)

    frames = []
    for t in range(n):
        acc = bkg
        for obj in objs:
            x, y = interp(n, obj['xy1'], obj['xy2'], t)
            s = SHAPES[obj['shape']](COLORS[obj['shape']][obj['color']])
            s = rot(s, obj['rotation']) # add time dep. rotation
            s = scale(s, 2) # add variable scale, possibly dynamic
            s = position(s, x, y)
            acc = Image.alpha_composite(acc, s)
        frames.append(acc)

    return frames


def sample_dataset(n):
    return np.stack(np.stack(img_to_arr(frame) for frame in sample_scene()) for _ in range(n))

def img_to_arr(img):
    assert img.mode == 'RGBA'
    channels = 4
    w, h = img.size
    arr = np.fromstring(img.tobytes(), dtype=np.uint8)
    return arr.reshape(w * h, channels).T.reshape(channels, h, w)



# Save a tiff
frames = sample_scene()
save_seq(frames)

# Convert a frame to correct array format for the model.
#arr = img_to_arr(frames[0])
#print(arr.shape)

# Using with `imshow` requires an additional `transpose(1, 2, 0)`.
#plt.imshow(arr.transpose(1, 2, 0))
#plt.show()

# Make a dataset
# out = sample_dataset(1000)
# print(out.shape)
# np.savez_compressed('single_object_with_bkg.npz', X=out)
