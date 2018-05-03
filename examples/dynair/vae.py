import torch
import torch.nn as nn
import pyro
import pyro.poutine as poutine
import pyro.distributions as dist

class VAE(nn.Module):
    def __init__(self, encode, decode, y_size, use_cuda=False):
        super(VAE, self).__init__()

        self.prototype = torch.tensor(0.).cuda() if use_cuda else torch.tensor(0.)

        self.encode = encode
        self.decode = decode

        self.y_prior_mean = self.prototype.new_zeros(y_size)
        self.y_prior_sd = self.prototype.new_ones(y_size)

        self.x_sd = 0.3

        if use_cuda:
            self.cuda()

    def model(self, batch):
        batch_size = batch.size(0)
        pyro.module('decode', self.decode)
        with pyro.iarange('data', batch_size):
            y = pyro.sample('y', dist.Normal(self.y_prior_mean.expand(batch_size, -1),
                                             self.y_prior_sd.expand(batch_size, -1))
                            .independent(1))
            x_mean = self.decode(y).view(batch_size, -1)
            x_sd = (self.x_sd * self.prototype.new_ones(1)).expand_as(x_mean)
            x = pyro.sample('x',
                            dist.Normal(x_mean, x_sd).independent(1),
                            obs=batch)
            return y, x, x_mean

    def guide(self, batch):
        batch_size = batch.size(0)
        pyro.module('encode', self.encode)
        with pyro.iarange('data', batch_size):
            y_mean, y_sd = self.encode(batch)
            y = pyro.sample('y', dist.Normal(y_mean, y_sd).independent(1))
            return y

    def recon(self, batch):
        trace = poutine.trace(self.guide).get_trace(batch)
        _, _, x_mean = poutine.replay(self.model, trace)(batch)
        return x_mean
