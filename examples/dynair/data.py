import numpy as np
import torch

def split(t, batch_size, num_test_batches):
    n = t.size(0)
    assert batch_size > 0
    assert num_test_batches >= 0
    assert n % batch_size == 0
    num_train_batches = (n // batch_size) - num_test_batches
    assert batch_size * (num_train_batches + num_test_batches) == n
    batches = t.chunk(n // batch_size)
    train = batches[0:num_train_batches]
    test = batches[num_train_batches:(num_train_batches+num_test_batches)]
    return train, test

def load_data(data_path, use_cuda):
    print('loading {}'.format(data_path))
    data = np.load(data_path)
    X_np = data['X']
    # print(X_np.shape)
    X_np = X_np.astype(np.float32)
    X_np /= 255.0
    X = torch.from_numpy(X_np)
    # Drop the alpha channel.
    X = X[:,:,0:3]
    Y = torch.from_numpy(data['Y'].astype(np.uint8))
    assert X.size(0) == Y.size(0)
    if use_cuda:
        X = X.cuda()
        Y = Y.cuda()
    return X, Y

def data_params(data):
    X, Y = data
    seq_length, num_chan, image_size = X.size()[1:4]
    x_size = num_chan * image_size**2
    max_obj_count = Y.max().item()
    return dict(seq_length=seq_length,
                num_chan=num_chan,
                image_size=image_size,
                x_size=x_size,
                max_obj_count=max_obj_count)
