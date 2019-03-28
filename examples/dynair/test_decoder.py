from functools import partial
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from data import load_data
from guide import InputCnn, ImgEmbedMlp, ImgEmbedResNet, CombineMixin
from modules import MLP
from data import load_data

all_seqs, _, all_tracks, _ = load_data('./ball-gray-20k-len-30-20x20.npz')

all_tracks = all_tracks.float()

x_size = 20 * 20
all_img = all_seqs.reshape(-1, x_size) # flatten seqs / images
all_pos = all_tracks.reshape(-1, 4)[:, 0:2] # flattent seqs, drop singleton dim


bs = 50 # batch_size
train_img = torch.stack(torch.split(all_img[0:10000], bs))
train_pos = torch.stack(torch.split(all_pos[0:10000], bs))
test_img = torch.stack(torch.split(all_img[10000:12000], bs))
test_pos = torch.stack(torch.split(all_pos[10000:12000], bs))

print(train_img.shape)
print(train_pos.shape)
print(test_img.shape)
print(test_pos.shape)
# assert False

net = nn.Sequential(MLP(2, [200, 200, 200, x_size], nn.ELU, output_non_linearity=False), nn.Sigmoid())

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

for i in range(1000):
    # train
    train_epoch_loss = 0.0
    for img, pos in zip(train_img, train_pos):
        out = net(pos)
        loss = torch.pow(out - img, 2).sum()
        train_epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        net.zero_grad()
    train_epoch_loss /= (train_img.size(0) * train_img.size(1))

    # test
    test_epoch_loss = 0.0
    outputs = []
    losses = []
    for img, pos in zip(test_img, test_pos):
        out = net(pos).detach()
        loss = torch.pow(out - img, 2).sum(1)
        outputs.append(out)
        losses.append(loss)
        test_epoch_loss += loss.sum().item()
    test_epoch_loss /= (test_img.size(0) * test_img.size(1))

    outputs = torch.cat(outputs)
    losses = torch.cat(losses)

    # print('\n\n==================================================\n\n')
    print('%6d | train : %.4f | test : %.4f' % (i, train_epoch_loss, test_epoch_loss))

    top_n_ix = losses.argsort(descending=True)[0:3].tolist()
    for ix in top_n_ix:
        print(ix)
        print(all_pos[10000+ix])
        print(losses[ix].item())
        #print(outputs[ix])
        #print(all_tracks[18000+ix, 0])
        print('-------------------------------------')
        if (i+1) % 20 == 0:
            plt.imshow(all_img[10000+ix].reshape((20,20)), cmap='gray')
            plt.show()
            plt.imshow(outputs[ix].reshape((20,20)), cmap='gray')
            plt.show()



    # if (i+1) % 50 == 0:
    #     plt.hist(losses, bins=50)
    #     plt.show()
