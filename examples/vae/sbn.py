import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import visdom

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO, TraceGraph_ELBO, CSIS
from pyro.optim import Adam
from utils.mnist_cached import MNISTCached as MNIST
from utils.mnist_cached import setup_data_loaders
from utils.vae_plots import mnist_test_tsne, plot_llk, plot_vae_samples


# define the PyTorch module that parameterizes the
# diagonal gaussian distribution q(z|x)
class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(Encoder, self).__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        #self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.reshape(-1, 784)
        # then compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_prob = torch.sigmoid(self.fc21(hidden))
        return z_prob


# define the PyTorch module that parameterizes the
# observation likelihood p(x|z)
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(Decoder, self).__init__()
        # setup the two linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, 784)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.softplus(self.fc1(z))
        # return the parameter for the output Bernoulli
        # each is of size batch_size x 784
        loc_img = torch.sigmoid(self.fc21(hidden))
        return loc_img

class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        hidden_dim = 400
        self.fc1 = nn.Linear(28**2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        hidden = self.softplus(self.fc1(x))
        baseline = self.fc2(hidden).reshape([-1])
        return baseline

class SBN(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, z_dim=50, hidden_dim=400, use_baseline=None, use_cuda=False):
        super(SBN, self).__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)
        if use_baseline == 'net':
            self.baseline = Baseline()

        self.proto = torch.zeros([1])
        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
            self.proto = self.proto.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.use_baseline = use_baseline

    # define the model p(x|z)p(z)
    def model(self, batch_size, observations=dict(x=0.)):
        #print('.')
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", batch_size):
            # setup hyperparameters for prior p(z)
            z_prob = 0.5 * self.proto.new_ones(torch.Size((batch_size, self.z_dim)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Bernoulli(z_prob).to_event(1))
            # decode the latent code z
            loc_img = self.decoder.forward(z)
            # score against actual images
            pyro.sample("x", dist.Bernoulli(loc_img).to_event(1), obs=observations['x']) # x.reshape(-1, 784))
            # return the loc so we can visualize it later
            return loc_img

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, batch_size, observations=dict(x=None)):
        #print(observations)
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", batch_size):
            # use the encoder to get the parameters used to define q(z|x)
            z_prob = self.encoder.forward(observations['x'])
            # sample the latent code z
            if self.use_baseline == 'avg':
                infer = dict(baseline={'use_decaying_avg_baseline': True,
                                       'baseline_beta': 0.9})
            elif self.use_baseline == 'net':
                pyro.module("baseline", self.baseline)
                infer = dict(baseline={'nn_baseline': self.baseline,
                                       'nn_baseline_input': observations['x']})
            else:
                infer = dict()
            pyro.sample("latent",
                        dist.Bernoulli(z_prob).to_event(1),
                        infer=infer)

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_prob = self.encoder(x)
        # sample in latent space
        z = dist.Bernoulli(z_prob).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img

def batch_log_prob(trace):
    trace.compute_log_prob()
    log_prob = 0.0
    for name, site in trace.nodes.items():
        if site["type"] == "sample" and name != "data":
            # print(name)
            # print(site)
            # print(site['log_prob'].shape)
            log_prob += site['unscaled_log_prob']
    return log_prob

def wake(model, guide, *args, **kwargs):
    n = 1
    surrogate = 0.0
    log_ws = []
    log_ps = []
    log_qs = []
    # TODO: vectorize
    for i in range(n):
        guide_trace = pyro.poutine.trace(guide).get_trace(*args, **kwargs)
        model_trace = pyro.poutine.trace(
            pyro.poutine.replay(model, trace=guide_trace)).get_trace(*args, **kwargs)
        log_p = batch_log_prob(model_trace)
        log_q = batch_log_prob(guide_trace)
        log_ps.append(log_p)
        log_qs.append(log_q)

    # sum over samples/batch
    surrogate = -torch.sum(sum(log_ps)) / float(n)

    loss = -torch.sum(sum(log_p.detach() - log_q.detach() for (log_p, log_q) in zip(log_ps, log_qs))) / float(n)
    # print(loss)
    # print(surrogate)

    return loss + surrogate - surrogate.detach()

def rws(model, guide, *args, **kwargs):
    n = 5
    surrogate = 0.0
    log_ws = []
    log_ps = []
    log_qs = []
    # TODO: vectorize
    for i in range(n):
        guide_trace = pyro.poutine.trace(guide).get_trace(*args, **kwargs)
        model_trace = pyro.poutine.trace(
            pyro.poutine.replay(model, trace=guide_trace)).get_trace(*args, **kwargs)
        log_p = batch_log_prob(model_trace)
        log_q = batch_log_prob(guide_trace)
        log_w = (log_p - log_q).detach()
        log_ws.append(log_w)
        log_ps.append(log_p)
        log_qs.append(log_q)

    all_log_ws = torch.stack(log_ws)
    assert all_log_ws.shape[0] == n
    #assert all_log_ws.shape[1] == batch_size
    log_ws_sum = torch.logsumexp(all_log_ws, 0)
    assert log_ws_sum.shape == log_ws[0].shape

    # normalised weights (log space)
    log_ws_norm = [log_w - log_ws_sum for log_w in log_ws]

    # estimate of -log p(x)
    loss = torch.sum(-log_ws_sum + torch.log(torch.tensor(float(n))))

    # model_surrogate = -torch.sum(sum(torch.exp(log_w_norm) * log_p for (log_w_norm, log_p) in zip(log_ws_norm, log_ps)))
    # guide_surrogate = -torch.sum(sum(torch.exp(log_w_norm) * log_q for (log_w_norm, log_q) in zip(log_ws_norm, log_qs)))

    surrogate = -torch.sum(sum(torch.exp(log_w_norm) * (log_p + log_q) for (log_w_norm, log_p, log_q) in zip(log_ws_norm, log_ps, log_qs)))

    return loss + surrogate - surrogate.detach()


def main(args):
    # clear param store
    pyro.clear_param_store()

    # setup MNIST data loaders
    # train_loader, test_loader
    train_loader, test_loader = setup_data_loaders(MNIST, use_cuda=args.cuda, batch_size=250)

    # setup the VAE
    vae = SBN(use_baseline=args.baseline, use_cuda=args.cuda)

    # setup the optimizer
    def adam_args(module_name, param_name):
        if 'baseline' in param_name or 'baseline' in module_name:
            return {"lr": args.learning_rate * 10.0}
        else:
            return {"lr": args.learning_rate}
    optimizer = Adam(adam_args)

    # setup the inference algorithm

    if args.loss == 'elbo':
        print('loss is elbo...')
        print('baseline=%s' % args.baseline)
        #elbo = JitTrace_ELBO() if args.jit else Trace_ELBO(num_particles=1)
        elbo = TraceGraph_ELBO(num_particles=1)
        svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)
    elif args.loss == 'rws':
        print('loss is rws...')
        svi = SVI(vae.model, vae.guide, optimizer, loss=rws)
    else:
        raise 'unknown loss'

    # Don't forget to tweak torch_distribution.py to enable/disable
    # reparameterisation as appropriate.
    # RWS:
    # [epoch 000]  average training loss: 200.4840
    # [epoch 001]  average training loss: 156.6850
    # [epoch 002]  average training loss: 146.0818
    # [epoch 003]  average training loss: 139.1793
    # [epoch 004]  average training loss: 134.4797
    # [epoch 005]  average training loss: 130.6190
    # [epoch 006]  average training loss: 127.4402
    # [epoch 007]  average training loss: 125.1244
    # [epoch 008]  average training loss: 123.1649
    # [epoch 009]  average training loss: 121.5196
    # [epoch 010]  average training loss: 120.1540
    # [epoch 011]  average training loss: 118.9896
    # [epoch 012]  average training loss: 117.9802
    # [epoch 013]  average training loss: 117.0995
    # [epoch 014]  average training loss: 116.3533
    # [epoch 015]  average training loss: 115.6856
    # [epoch 016]  average training loss: 115.0475
    # [epoch 017]  average training loss: 114.4574
    # [epoch 018]  average training loss: 114.0630
    # [epoch 019]  average training loss: 113.5262
    # [epoch 020]  average training loss: 113.1756
    # [epoch 021]  average training loss: 112.7183
    # [epoch 022]  average training loss: 112.4059
    # [epoch 023]  average training loss: 112.0213
    # [epoch 024]  average training loss: 111.7626
    # [epoch 025]  average training loss: 111.4242
    # [epoch 026]  average training loss: 111.1843
    # [epoch 027]  average training loss: 110.9030

    # TraceELBO(num_particles=5)
    # [epoch 027]  average training loss: 139.7160


    # wake-sleep
    # svi_model = SVI(vae.model, vae.guide, optimizer, loss=wake)
    # svi_guide = CSIS(vae.model, vae.guide, optimizer, training_batch_size=1)

    elapsed = 0.0

    # setup visdom for visualization
    if args.visdom_flag:
        vis = visdom.Visdom()

    train_loss = []
    test_likelihood = []
    # training loop
    for epoch in range(args.num_epochs):
        # initialize loss accumulator
        epoch_loss = 0.
        guide_loss = 0.
        # do a training epoch over each mini-batch x returned
        # by the data loader
        t0 = time.time()
        for x, _ in train_loader:
            # if on GPU put mini-batch into CUDA memory
            if args.cuda:
                x = x.cuda()
            # do ELBO gradient and accumulate loss
            epoch_loss += svi.step(x.shape[0], dict(x=x.reshape(-1, 784)))
            #epoch_loss += svi_model.step(x)
            #guide_loss += svi_guide.step(x)
            #epoch_loss += svi_model.step(x.shape[0], dict(x=x.reshape(-1, 784)))
            #guide_loss += svi_guide.step(x.shape[0])
            # print([name for (name, val) in pyro.get_param_store().named_parameters()])
            # assert False
        elapsed += time.time() - t0

        # report training diagnostics
        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = epoch_loss / normalizer_train
        train_loss.append((total_epoch_loss_train, elapsed))
        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))
        #print(guide_loss / normalizer_train)

        if (epoch+1) % args.test_frequency == 0:
            # initialize loss accumulator
            test_loss = 0.
            # compute the loss over the entire test set
            for i, (x, _) in enumerate(test_loader):
                # if on GPU put mini-batch into CUDA memory
                if args.cuda:
                    x = x.cuda()
                # compute marginal likelihood estimate
                test_loss += rws(vae.model, vae.guide, x.shape[0], dict(x=x.reshape(-1, 784))).item()


        #         # pick three random test images from the first mini-batch and
        #         # visualize how well we're reconstructing them
        #         if i == 0:
        #             if args.visdom_flag:
        #                 plot_vae_samples(vae, vis)
        #                 reco_indices = np.random.randint(0, x.shape[0], 3)
        #                 for index in reco_indices:
        #                     test_img = x[index, :]
        #                     reco_img = vae.reconstruct_img(test_img)
        #                     vis.image(test_img.reshape(28, 28).detach().cpu().numpy(),
        #                               opts={'caption': 'test image'})
        #                     vis.image(reco_img.reshape(28, 28).detach().cpu().numpy(),
        #                               opts={'caption': 'reconstructed image'})

            # report test diagnostics
            normalizer_test = len(test_loader.dataset)
            total_epoch_loss_test = test_loss / normalizer_test
            test_likelihood.append((epoch, total_epoch_loss_test))
            print("[epoch %03d]  average test loss: %.4f" % (epoch, total_epoch_loss_test))

        # if epoch == args.tsne_iter:
        #     mnist_test_tsne(vae=vae, test_loader=test_loader)
        #     plot_llk(np.array(train_elbo), np.array(test_elbo))

    if args.save:
        with open('history.json', 'w') as f:
            json.dump(dict(train=train_loss, test=test_likelihood), f)

    return vae


if __name__ == '__main__':
    assert pyro.__version__.startswith('0.3.1')
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=101, type=int, help='number of training epochs')
    parser.add_argument('-tf', '--test-frequency', default=5, type=int, help='how often we evaluate the test set')
    parser.add_argument('-lr', '--learning-rate', default=1.0e-3, type=float, help='learning rate')
    parser.add_argument('--cuda', action='store_true', default=False, help='whether to use cuda')
    parser.add_argument('--jit', action='store_true', default=False, help='whether to use PyTorch jit')
    parser.add_argument('-visdom', '--visdom_flag', action="store_true", help='Whether plotting in visdom is desired')
    parser.add_argument('-i-tsne', '--tsne_iter', default=100, type=int, help='epoch when tsne visualization runs')
    parser.add_argument('--save', default=False, action='store_true', help='write training history to fs')
    parser.add_argument('--baseline', choices='none avg net'.split(), default='none')
    parser.add_argument('--loss', choices='elbo rws'.split(), default='elbo')
    args = parser.parse_args()

    model = main(args)
