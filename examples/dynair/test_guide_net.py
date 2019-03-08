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

all_img, all_obj_ids, all_pos = load('./out.npz')

# print(all_img.shape)
# assert False

bs = 50 # batch_size
img_train = torch.stack(torch.split(all_img[0:3000], bs))
obj_ids_train = torch.stack(torch.split(all_obj_ids[0:3000], bs))
pos_train = torch.stack(torch.split(all_pos[0:3000], bs))

img_test = torch.stack(torch.split(all_img[3000:], bs))
obj_ids_test = torch.stack(torch.split(all_obj_ids[3000:], bs))
pos_test = torch.stack(torch.split(all_pos[3000:], bs))


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
    for img, obj_ids, pos in zip(img_train, obj_ids_train, pos_train):
        out = net(img, obj_ids)
        loss = torch.pow(out - pos, 2).sum()
        train_epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        net.zero_grad()
    train_epoch_loss /= (img_train.size(0) * img_train.size(1))

    # test
    test_epoch_loss = 0.0
    outputs = []
    losses = []
    for img, obj_ids, pos in zip(img_test, obj_ids_test, pos_test):
        out = net(img, obj_ids).detach()
        loss = torch.pow(out - pos, 2).sum(1)
        outputs.append(out)
        losses.append(loss)
        test_epoch_loss += loss.sum().item()
    test_epoch_loss /= (img_test.size(0) * img_test.size(1))
    outputs = torch.cat(outputs)
    losses = torch.cat(losses)

    print('%6d | train : %.4f | test : %.4f' % (i, train_epoch_loss, test_epoch_loss))

    if (i+1) % 10 == 0:

        plt.hist(losses)
        plt.show()

        top_n_ix = losses.argsort(descending=True)[0:5].tolist()
        for ix in top_n_ix:
            obj_id = all_obj_ids[3000+ix].argmax().item()
            color = ["pink", "gray", "yellow"][obj_id]
            p = outputs[ix]
            x = (p[0] * 25) + 25
            y = (p[1] * 25) + 25
            img = draw_bounding_box(all_img[3000+ix].numpy(), x, y, 14.3, 14.3, color)
            plt.imshow(img.transpose((1,2,0)))
            plt.show()
