from functools import partial
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from data import load_data
from guide import InputCnn, ImgEmbedMlp, ImgEmbedResNet, CombineMixin
from modules import MLP, Flatten
from data import load_data





all_imgs, _, all_tracks, _ = load_data('./ball-gray-20k-len-30-20x20.npz')

# frames matching tracks_curr (pos/vel etc)
imgs_curr = all_imgs[:, 1:30, :].reshape(-1, 400)

all_tracks = all_tracks.float()
all_tracks = all_tracks.reshape(20000, 30, 4)



tracks_prev = all_tracks[:, 0:29, :].reshape(-1, 4)
tracks_curr = all_tracks[:, 1:30, :].reshape(-1, 4)



# TODO: Shuffle these ^^ , otherwize optimizing on lots of similar examples.
ix = np.arange(tracks_prev.shape[0])
np.random.shuffle(ix)
tracks_prev = tracks_prev[ix]
tracks_curr = tracks_curr[ix]
imgs_curr = imgs_curr[ix]

# print(tracks_prev.shape) # 580K x 4
# print(tracks_curr.shape)
# print(tracks_prev[0:5])
# print(tracks_curr[0:5])




bs = 50 # batch_size
train_prev = torch.stack(torch.split(tracks_prev[0:20000], bs))
train_curr = torch.stack(torch.split(tracks_curr[0:20000], bs))
train_img = torch.stack(torch.split(imgs_curr[0:20000], bs))

test_prev = torch.stack(torch.split(tracks_prev[20000:25000], bs))
test_curr = torch.stack(torch.split(tracks_curr[20000:25000], bs))
test_img = torch.stack(torch.split(imgs_curr[20000:25000], bs))

# transition net
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


# predict net
z_size = 4
x_size = 400
# started with this:
predict_net = MLP(x_size + z_size * 2, [200, 200, 4], nn.ELU, output_non_linearity=False)

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        num_chan = 1
        self.cnn = nn.Sequential(
            nn.Conv2d(num_chan, 16, 4, stride=2, padding=0), # => 16 x 9 x 9
            nn.ELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=0), # => 32 x 4 x 4
            nn.ReLU(),
            Flatten()
        )

    def forward(self, img):
        return self.cnn(img)

cnn = Cnn()
combine_net = MLP(512 + 2 * z_size, [200, 200, 4], nn.ELU, output_non_linearity=False)

def predict(prev, t, img):

    # img_unflat = img.reshape(bs, 1, 20, 20)
    # cnn_out = cnn(img_unflat) # batch size x 512
    # out = combine_net(torch.cat((prev, t, cnn_out), 1))

    out = predict_net(torch.cat((prev, t, img), 1))
    #out = predict_net(torch.cat((prev, img), 1))
    #out = predict_net(torch.cat((t, img), 1))
    return out


for i in range(200):
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

    # top_n_ix = losses.argsort(descending=True)[0:3].tolist()
    # for ix in top_n_ix:
    #     print(ix)
    #     print(losses[ix].item())
    #     print(tracks_prev[10000+ix])
    #     print(tracks_curr[10000+ix])
    #     print(outputs[ix])
    #     #print(all_tracks[18000+ix, 0])
    #     print('-------------------------------------')


    # if (i+1) % 50 == 0:
    #     plt.hist(losses, bins=50)
    #     plt.show()

print('\n\n========================================\n\n')

params = []
params.extend(predict_net.parameters())
#params.extend(cnn.parameters())
#params.extend(combine_net.parameters())
optimizer = torch.optim.Adam(params, lr=0.001)


for i in range(500):
    # train
    train_epoch_loss = 0.0
    for prev, curr, img in zip(train_prev, train_curr, train_img):
        t = transition(prev)
        out = predict(prev, t, img)
        loss = torch.pow(out - (curr - t), 2).sum()
        train_epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        predict_net.zero_grad()
        #cnn.zero_grad()
        #combine_net.zero_grad()
    train_epoch_loss /= (train_prev.size(0) * train_prev.size(1))

    # test
    test_epoch_loss = 0.0
    outputs = []
    losses = []
    for prev, curr, img in zip(test_prev, test_curr, test_img):
        t = transition(prev)
        out = predict(prev, t, img)
        loss = torch.pow(out - (curr - t), 2).sum(1)
        outputs.append(out)
        losses.append(loss)
        test_epoch_loss += loss.sum().item()
    test_epoch_loss /= (test_prev.size(0) * test_prev.size(1))

    outputs = torch.cat(outputs)
    losses = torch.cat(losses)

    print('%6d | train : %.4f | test : %.4f' % (i, train_epoch_loss, test_epoch_loss))

    if (i+1) % 50 == 0:
        top_n_ix = losses.argsort(descending=True)[0:3].tolist()
        for ix in top_n_ix:
            print(ix)
            print(losses[ix].item())
            print(tracks_prev[20000+ix] + outputs[ix])
            print(tracks_curr[20000+ix])
            #print(all_tracks[18000+ix, 0])
            print('-------------------------------------')
