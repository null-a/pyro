import numpy as np

from matplotlib import pyplot as plt

from PIL import Image, ImageDraw

from PIL.ImageTransform import AffineTransform


# https://pillow.readthedocs.io/en/4.3.x/handbook/concepts.html

# The most obvious format choice seems to be:
# RGBA (4x8-bit pixels, true color with transparency mask)

# (0,0) is upper left in PIL


img1 = Image.new('RGBA', (100, 100), (0,0,0,0))
img2 = Image.new('RGBA', (100, 100), (0,0,0,0))

draw1 = ImageDraw.Draw(img1)
draw2 = ImageDraw.Draw(img2)

draw1.ellipse((10, 10, 90, 90), fill=(255,0,0,255))

#draw1.polygon([50, 10, 10, 90, 90, 90], fill=(0,255,0,255))
draw2.polygon([10,10,90,10,90,90,10,90], fill=(0,0,255,255))


img1 = img1.transform((100,100),
                      AffineTransform((1,0,20,0,1,20)),
                      resample=Image.BILINEAR)

# This is "over"? Perhaps:
# https://github.com/python-pillow/Pillow/blob/e11b47858c5b3b4c243667a2ef4fda0741d90466/libImaging/AlphaComposite.c#L27
img = Image.alpha_composite(img1, img2)

#img.show()
plt.axis('off')
plt.imshow(img)
plt.show()



# Try combining a few of these to make a short video.

# https://pillow.readthedocs.io/en/4.3.x/handbook/tutorial.html#image-sequences

# Include a background, overlapping objects to test alpha.
