import argparse

import numpy as np
import torch
import torch.nn as nn
import visdom

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
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
        self.fc22 = nn.Linear(hidden_dim, z_dim)
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
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale


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


# define a PyTorch module for the VAE
class VAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, z_dim=50, hidden_dim=400, use_cuda=False):
        super(VAE, self).__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

    # define the model p(x|z)p(z)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            loc_img = self.decoder.forward(z)
            # score against actual images
            pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, 784))
            # return the loc so we can visualize it later
            return loc_img

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
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


def rws_guide(model, guide, *args, **kwargs):
    n = 5
    surrogate = 0.0
    log_ws = []
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
        log_qs.append(log_q)

    all_log_ws = torch.stack(log_ws)
    assert all_log_ws.shape[0] == n
    #assert all_log_ws.shape[1] == batch_size
    log_ws_sum = torch.logsumexp(all_log_ws, 0)
    assert log_ws_sum.shape == log_ws[0].shape

    log_ws_norm = [log_w - log_ws_sum for log_w in log_ws]

    # sum over samples/batch
    surrogate = -torch.sum(sum(torch.exp(log_w_norm) * log_q for (log_w_norm, log_q) in zip(log_ws_norm, log_qs)))

    loss = torch.sum(sum(torch.exp(log_w_norm) * log_w for (log_w_norm, log_w) in zip(log_ws_norm, log_ws)) - log_ws_sum + torch.log(torch.tensor(float(n))))
    # print(loss)
    # print(surrogate)

    return loss + surrogate - surrogate.detach()


def rws_model(model, guide, *args, **kwargs):
    n = 5
    surrogate = 0.0
    log_ws = []
    log_ps = []
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

    all_log_ws = torch.stack(log_ws)
    assert all_log_ws.shape[0] == n
    #assert all_log_ws.shape[1] == batch_size
    log_ws_sum = torch.logsumexp(all_log_ws, 0)
    assert log_ws_sum.shape == log_ws[0].shape

    log_ws_norm = [log_w - log_ws_sum for log_w in log_ws]

    # sum over samples/batch
    surrogate = -torch.sum(sum(torch.exp(log_w_norm) * log_p for (log_w_norm, log_p) in zip(log_ws_norm, log_ps)))

    loss = torch.sum(-log_ws_sum + torch.log(torch.tensor(float(n))))
    # print(loss)
    # print(surrogate)

    return loss + surrogate - surrogate.detach()



def main(args):
    # clear param store
    pyro.clear_param_store()

    # setup MNIST data loaders
    # train_loader, test_loader
    train_loader, test_loader = setup_data_loaders(MNIST, use_cuda=args.cuda, batch_size=256)

    # setup the VAE
    vae = VAE(use_cuda=args.cuda)

    # setup the optimizer
    adam_args = {"lr": args.learning_rate}
    optimizer = Adam(adam_args)

    # setup the inference algorithm
    # elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
    # svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)

    svi_model = SVI(vae.model, vae.guide, optimizer, loss=rws_model)
    svi_guide = SVI(vae.model, vae.guide, optimizer, loss=rws_guide)


    # setup visdom for visualization
    if args.visdom_flag:
        vis = visdom.Visdom()

    train_elbo = []
    test_elbo = []
    # training loop
    for epoch in range(args.num_epochs):
        # initialize loss accumulator
        epoch_loss = 0.
        guide_loss = 0.
        # do a training epoch over each mini-batch x returned
        # by the data loader
        for x, _ in train_loader:
            # if on GPU put mini-batch into CUDA memory
            if args.cuda:
                x = x.cuda()
            # do ELBO gradient and accumulate loss
            # epoch_loss += svi.step(x)
            epoch_loss += svi_model.step(x)
            guide_loss += svi_guide.step(x)

        # report training diagnostics
        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = epoch_loss / normalizer_train
        train_elbo.append(total_epoch_loss_train)
        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

        if epoch % args.test_frequency == 0:
            # initialize loss accumulator
            #test_loss = 0.
            # compute the loss over the entire test set
            for i, (x, _) in enumerate(test_loader):
                # if on GPU put mini-batch into CUDA memory
                if args.cuda:
                    x = x.cuda()
                # compute ELBO estimate and accumulate loss
                #test_loss += svi.evaluate_loss(x)

                # pick three random test images from the first mini-batch and
                # visualize how well we're reconstructing them
                if i == 0:
                    if args.visdom_flag:
                        plot_vae_samples(vae, vis)
                        reco_indices = np.random.randint(0, x.shape[0], 3)
                        for index in reco_indices:
                            test_img = x[index, :]
                            reco_img = vae.reconstruct_img(test_img)
                            vis.image(test_img.reshape(28, 28).detach().cpu().numpy(),
                                      opts={'caption': 'test image'})
                            vis.image(reco_img.reshape(28, 28).detach().cpu().numpy(),
                                      opts={'caption': 'reconstructed image'})

            # report test diagnostics
            # normalizer_test = len(test_loader.dataset)
            # total_epoch_loss_test = test_loss / normalizer_test
            # test_elbo.append(total_epoch_loss_test)
            # print("[epoch %03d]  average test loss: %.4f" % (epoch, total_epoch_loss_test))

        # if epoch == args.tsne_iter:
        #     mnist_test_tsne(vae=vae, test_loader=test_loader)
        #     plot_llk(np.array(train_elbo), np.array(test_elbo))

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
    args = parser.parse_args()

    model = main(args)
