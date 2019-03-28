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

_, _, all_tracks, _ = load_data('./ball-gray-20k-len-30-20x20.npz')

all_tracks = all_tracks.float()
all_tracks = all_tracks.reshape(20000, 30, 4)



tracks_prev = all_tracks[:, 0:29, :].reshape(-1, 4)
tracks_curr = all_tracks[:, 1:30, :].reshape(-1, 4)


# TODO: Shuffle these ^^ , otherwize optimizing on lots of similar examples.
ix = np.arange(tracks_prev.shape[0])
np.random.shuffle(ix)
tracks_prev = tracks_prev[ix]
tracks_curr = tracks_curr[ix]

# print(tracks_prev.shape) # 580K x 4
# print(tracks_curr.shape)
# print(tracks_prev[0:5])
# print(tracks_curr[0:5])




bs = 50 # batch_size
train_prev = torch.stack(torch.split(tracks_prev[0:10000], bs))
train_curr = torch.stack(torch.split(tracks_curr[0:10000], bs))

test_prev = torch.stack(torch.split(tracks_prev[10000:15000], bs))
test_curr = torch.stack(torch.split(tracks_curr[10000:15000], bs))


M = 16 # number of transition matrices
A = nn.Parameter(torch.zeros((M, 4, 4)))
#torch.nn.init.normal_(self.A, 0., 1e-3)
for A_i in A:
    torch.nn.init.eye_(A_i)

net = nn.Sequential(nn.Linear(4, 100), nn.ELU(), nn.Linear(100, M), nn.Softmax(dim=1))


params = []
params.extend(net.parameters())
params.append(A)
optimizer = torch.optim.Adam(params, lr=0.001)



def transition(prev):
    batch_size = prev.size(0)
    alpha = net(prev)
    # Per-data point transition matrices:
    A2 = torch.einsum('ij,jkl->ikl', (alpha.clone(), A.clone()))
    assert A2.shape == (batch_size, 4, 4)

    # Batched matrix-vector multiple (between per data point
    # transition matrices and batch of z_prev)
    Az_prev = torch.einsum('ijk,ik->ij', (A2.clone(), prev.clone()))
    assert Az_prev.shape == (batch_size, 4)
    return Az_prev


for i in range(1000):
    # train
    train_epoch_loss = 0.0
    for prev, curr in zip(train_prev, train_curr):
        out = transition(prev)
        loss = torch.pow(out - curr, 2).sum()
        train_epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        net.zero_grad()
        A.grad *= 0.0
    train_epoch_loss /= (train_prev.size(0) * train_prev.size(1))

    # test
    test_epoch_loss = 0.0
    outputs = []
    losses = []
    for prev, curr in zip(test_prev, test_curr):
        out = transition(prev)
        loss = torch.pow(out - curr, 2).sum(1)
        outputs.append(out)
        losses.append(loss)
        test_epoch_loss += loss.sum().item()
    test_epoch_loss /= (test_prev.size(0) * test_prev.size(1))

    outputs = torch.cat(outputs)
    losses = torch.cat(losses)

    # print('\n\n==================================================\n\n')
    print('%6d | train : %.4f | test : %.4f' % (i, train_epoch_loss, test_epoch_loss))

    top_n_ix = losses.argsort(descending=True)[0:3].tolist()
    for ix in top_n_ix:
        print(ix)
        print(losses[ix].item())
        print(tracks_prev[10000+ix])
        print(tracks_curr[10000+ix])
        print(outputs[ix])
        #print(all_tracks[18000+ix, 0])
        print('-------------------------------------')


    # if (i+1) % 50 == 0:
    #     plt.hist(losses, bins=50)
    #     plt.show()
