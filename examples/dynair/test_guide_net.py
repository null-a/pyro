from functools import partial
from matplotlib import pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from data import load_data
from guide import InputCnn, CombineMixin
from modules import MLP

# Prepare training data.
def load(path):
    X, Y, T, O = load_data('./out.npz')
    imgs = []
    obj_ids = []
    positions = []
    for x, y, t, o in zip(X, Y, T, O):
        # Use only the first frame of each sequence to begin with.
        for i in range(y):
            imgs.append(x[0])
            one_hot = torch.zeros(3)
            one_hot[o[i]] = 1
            obj_ids.append(one_hot)
            position = torch.tensor(t[0, i, 0:2]).float()
            positions.append((position - 25.0) / 25.0) # center and scale (x,y) position
    return torch.stack(imgs), torch.stack(obj_ids), torch.stack(positions)

img, obj_ids, pos = load('./out.npz')


# print(img.shape)
# assert False

bs = 50 # batch_size
img_train = torch.stack(torch.split(img[0:3000], bs))
obj_ids_train = torch.stack(torch.split(obj_ids[0:3000], bs))
pos_train = torch.stack(torch.split(pos[0:3000], bs))

img_test = torch.stack(torch.split(img[3000:], bs))
obj_ids_test = torch.stack(torch.split(obj_ids[3000:], bs))
pos_test = torch.stack(torch.split(pos[3000:], bs))


net = CombineMixin(InputCnn,
                   partial(MLP, hids=[800, 800, 2], output_non_linearity=False),
                   (3,50,50), # main input size
                   3)         # side input size

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

def do_epoch(img_b, obj_ids_b, pos_b, optimize):
    epoch_loss = 0.0
    for img, obj_ids, pos in zip(img_b, obj_ids_b, pos_b):
        out = net(img, obj_ids)
        loss = torch.pow(out - pos, 2).sum()
        epoch_loss += loss.item()
        if optimize:
            loss.backward()
            optimizer.step()
            net.zero_grad()
    return epoch_loss / (img_b.size(0) * img_b.size(1))


for i in range(1000):
    train_loss = do_epoch(img_train, obj_ids_train, pos_train, True)
    test_loss  = do_epoch(img_test, obj_ids_test, pos_test, False)
    print('%6d | train : %.4f | test : %.4f' % (i, train_loss, test_loss))
