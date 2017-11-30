import numpy as np

from matplotlib import pyplot as plt

from PIL import Image, ImageDraw

from PIL.ImageTransform import AffineTransform


# https://pillow.readthedocs.io/en/4.3.x/handbook/concepts.html

# The most obvious format choice seems to be:
# RGBA (4x8-bit pixels, true color with transparency mask)

# (0,0) is upper left in PIL


img1 = Image.new('RGBA', (2, 3), (0,0,0,0))

img1.putpixel((0,0), (1,0,0,0))
img1.putpixel((1,0), (0,1,0,0))
img1.putpixel((0,1), (0,0,1,0))
img1.putpixel((1,1), (1,0,0,1))
img1.putpixel((0,2), (0,1,0,1))
img1.putpixel((1,2), (0,0,1,1))

# r g  alpha = 0 0
# b r          0 1
# g b          1 1

# A list of pixels (each is a 4-tuple) starting at top row, going left
# to right.
print(list(img1.getdata()))
# [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]

print(img1.tobytes())
# b'\x01\x00\x00\x00\x00\x01\x00\x00\x00\x00\x01\x00\x01\x00\x00\x01\x00\x01\x00\x01\x00\x00\x01\x01'

arr = np.fromstring(img1.tobytes(), dtype=np.int8)

# An array of pixels. Pretty much the list above.
x = arr.reshape(6,4)
print(x)
# [[1 0 0 0]
#  [0 1 0 0]
#  [0 0 1 0]
#  [1 0 0 1]
#  [0 1 0 1]
#  [0 0 1 1]]


# I think this is some thing like the format we'd need for e.g. STN
# type stuff. i.e. Each channel as a 2d array.
print(arr.reshape(6,4).T.reshape(4, 3, 2))
# [[[1 0]    red
#   [0 1]
#   [0 0]]

#  [[0 1]    green
#   [0 0]
#   [1 0]]

#  [[0 0]    blue
#   [1 0]
#   [0 1]]

#  [[0 0]    alpha
#   [0 1]
#   [1 1]]]
