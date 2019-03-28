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
all_seqs = all_seqs.reshape(20000, 30, x_size) # flatten images
all_tracks = all_tracks.reshape(20000, 30, 4) # drop singleton dim


bs = 50 # batch_size
train_seqs = torch.stack(torch.split(all_seqs[0:18000], bs))
train_tracks = torch.stack(torch.split(all_tracks[0:18000], bs))
test_seqs = torch.stack(torch.split(all_seqs[18000:], bs))
test_tracks = torch.stack(torch.split(all_tracks[18000:], bs))

rnn = nn.RNN(x_size, 200, nonlinearity='relu', batch_first=True, bidirectional=True)
#rnn = nn.GRU(x_size, 200, batch_first=True, bidirectional=True)
mlp = MLP(400, [200, 4], nn.ELU, output_non_linearity=False)

# net = CombineMixin(InputCnn,
#                    #partial(ImgEmbedResNet, hids=[1000, 1000]),
#                    partial(MLP, hids=[800, 800, 2], output_non_linearity=False),
#                    (3,50,50), # main input size
#                    3)         # side input size

params = []
params.extend(list(rnn.parameters()))
params.extend(list(mlp.parameters()))

optimizer = torch.optim.Adam(params, lr=0.001)


def img_to_arr(img):
    #assert img.mode == 'RGBA'
    channels = 4 if img.mode == 'RGBA' else 3
    w, h = img.size
    arr = np.fromstring(img.tobytes(), dtype=np.uint8)
    return arr.reshape(w * h, channels).T.reshape(channels, h, w)

# Expects pixel intensities in [0,1]
def arr_to_img(nparr):
    assert type(nparr) == np.ndarray
    assert nparr.shape[0] == 3 # required for RGB
    shape = nparr.shape[1:]
    return Image.frombuffer('RGB', shape, (nparr.transpose((1,2,0)) * 255.).astype(np.uint8).tostring(), 'raw', 'RGB', 0, 1)

# origin is top-left
def draw_bounding_box(nparr, x, y, w, h, color='red'):
    img = arr_to_img(nparr)
    draw = ImageDraw.Draw(img)
    draw.rectangle([x, y, x+w, y+h], outline=color)
    return img_to_arr(img)

for i in range(1000):
    # train
    train_epoch_loss = 0.0
    for seqs, tracks in zip(train_seqs, train_tracks):

        rnn_outputs, _ = rnn(seqs)
        mlp_input1 = rnn_outputs[:, 0, 200:]
        mlp_input2 = rnn_outputs[:, -1, 0:200]
        out = mlp(torch.cat((mlp_input1, mlp_input2), 1))

        # We're trying to output the pos/vel at the first first.
        loss = torch.pow(out - tracks[:, 0], 2).sum()
        train_epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        rnn.zero_grad()
        mlp.zero_grad()
    train_epoch_loss /= (train_seqs.size(0) * train_seqs.size(1))

    # test
    test_epoch_loss = 0.0
    outputs = []
    losses = []
    for seqs, tracks in zip(test_seqs, test_tracks):
        rnn_outputs, _ = rnn(seqs)
        mlp_input1 = rnn_outputs[:, 0, 200:]
        mlp_input2 = rnn_outputs[:, -1, 0:200]
        out = mlp(torch.cat((mlp_input1, mlp_input2), 1)).detach()

        loss = torch.pow(out - tracks[:, 0], 2).sum(1)
        outputs.append(out)
        losses.append(loss)
        test_epoch_loss += loss.sum().item()
    test_epoch_loss /= (test_seqs.size(0) * test_seqs.size(1))

    outputs = torch.cat(outputs)
    losses = torch.cat(losses)


    print('\n\n==================================================\n\n')
    print('%6d | train : %.4f | test : %.4f' % (i, train_epoch_loss, test_epoch_loss))

    top_n_ix = losses.argsort(descending=True)[0:10].tolist()
    for ix in top_n_ix:
        print(ix)
        print(losses[ix].item())
        print(outputs[ix])
        print(all_tracks[18000+ix, 0])
        print('-------------------------------------')

    if (i+1) % 50 == 0:
        plt.hist(losses, bins=50)
        plt.show()
