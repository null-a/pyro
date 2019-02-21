import torch
import torch.nn as nn
from torch.nn.functional import softplus

import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim
from pyro.contrib.autoguide import AutoDiagonalNormal, AutoMultivariateNormal

from torch.distributions import constraints

# def model():
#     z = pyro.sample('z', dist.Bernoulli(torch.tensor(0.9)))
#     x = pyro.sample('x', dist.Bernoulli(torch.tensor(0.6 if z else 0.3)), obs=torch.tensor(0.0))


# is_posterior = pyro.infer.Importance(model, num_samples=1000).run()
# is_marginal = pyro.infer.EmpiricalMarginal(is_posterior, "z")

# print(torch.exp(is_marginal.log_prob(torch.tensor(1.0))))
# print(torch.exp(is_marginal.log_prob(torch.tensor(0.0))))


def model(obs):
    for i in pyro.plate('data', len(obs)):
        z = pyro.sample('z%d' % i, dist.Bernoulli(torch.tensor(0.9)))
        x = pyro.sample('x%d' % i, dist.Bernoulli(torch.tensor(0.6 if z else 0.3)), obs=obs[i])

def zProb(obs):
    p1 = pyro.param('p1', torch.tensor(0.0))
    p2 = pyro.param('p2', torch.tensor(0.0))
    return torch.sigmoid(p1 * obs + p2)

def guide(obs):
    for i in pyro.plate('data', len(obs)):
        z = pyro.sample('z%d' % i, dist.Bernoulli(zProb(obs[i])))

# optimizer = pyro.optim.Adam({"lr": 0.01})
# svi = pyro.infer.SVI(model, guide, optimizer, loss=pyro.infer.Trace_ELBO())

# data = [torch.tensor(0.0), torch.tensor(1.0)]

# for _ in range(2000):
#     svi.step(data)

# print(zProb(data[1]).detach())
# print(zProb(data[0]).detach())



# This doesn't work when using reparameterised samplers. (This is
# currently patched in torch_distributions.py.) The problem is that
# under reparameterisation we'd need to evaluate grad log q at the
# sampled value (once transformed). what the code below ends up doing
# is backprop through *log q(g(eps))*. (where g is the reparam
# transform that also depends on the parameters.) what we'd need
# instead (under reparam) is to backprop through `log q(
# g(eps).detach() )`. (i checked that adding such a detach to the
# computation of the log prob in (trace_struct) also makes things
# work.

# can this be made to work with reparam somehow? and even if it can,
# is there any benefit?


# TODO: Have this return both the loss and the grads. (surrogate isn't
# the loss, and having the loss is useful for monitoring/comparing
# progress etc.)

def wake_guide_update(model, guide, *args, **kwargs):
    surrogate = 0.0
    ws = []
    for i in range(5):
        guide_trace = pyro.poutine.trace(guide).get_trace(*args, **kwargs)
        model_trace = pyro.poutine.trace(
            pyro.poutine.replay(model, trace=guide_trace)).get_trace(*args, **kwargs)
        logp = model_trace.log_prob_sum()
        logq = guide_trace.log_prob_sum()
        w = torch.exp(logp - logq).detach()
        surrogate += w * (-logq)
        ws.append(w)
    # For the toy continuous model, dividing by the number of samples
    # rather than the sum of the weight produces better results,
    # particularly when the number of samples (per-gradient step) is
    # small.
    return surrogate / 5. # sum(ws)



# svi = pyro.infer.SVI(model, guide, optimizer, loss=wake_guide_update)

# for _ in range(2000):
#     svi.step(data)

# print(zProb(data[1]).detach()) # 0.9473684210526315
# print(zProb(data[0]).detach()) # 0.8372093023255814



# Attempting to observe the mode covering behaviour of guides
# optimised with this KL. This kinda works, but it appears to require
# (a) lots of samples per gradient step for stability, an not too big
# learning rate. (1e-3 seems ok, 1e-2 doesn't seem to do the right
# things.)
def model2():
    z = pyro.sample('x', dist.MultivariateNormal(torch.tensor([0., 0]), torch.tensor([[1.0, 0.9], [0.9, 1.0]])))

# This is a mean field guide. i.e. The two components of z are
# independent, and this can't capture the dependency of the zs in the
# model.
def guide2():
    cov = torch.diag(torch.exp(pyro.param('sd', torch.log(torch.tensor([1., 1.])))))
    z = pyro.sample('x', dist.MultivariateNormal(torch.tensor([0., 0]), cov))


#optimizer = pyro.optim.Adam({"lr": 0.01})

#svi = pyro.infer.SVI(model2, guide2, optimizer, loss=pyro.infer.Trace_ELBO())
# svi = pyro.infer.SVI(model2, guide2, optimizer, loss=wake_guide_update)

# for i in range(100000):
#     loss = svi.step()
#     if (i+1) % 50 == 0:
#         #print(loss)
#         print(torch.exp(pyro.param('sd')).tolist())


# --------------------------------------------------
# toy long range dep model i've played with before

# var model = function() {
#   var zs = repeat(4, function() {
#     sample(Gaussian({mu: 0, sigma: 1}))
#   })
#   observe(Gaussian({mu: zs[0] - zs[3], sigma: 0.1}), 0)
#   return [zs[0], zs[3]]
# }
# Infer({model, method: 'MCMC', kernel: 'HMC', samples: 1000})

def model3(l=2):
    assert l >= 2
    zs = [pyro.sample('z%d' % i, dist.MultivariateNormal(torch.tensor([0.]), torch.tensor([[1.]])))
          for i in range(l)]
    pyro.sample('x',
                dist.Normal(zs[0] - zs[-1], torch.tensor(0.1)),
                obs=torch.tensor(0.0))

# I need to figure out what a sensible guide for this is. i.e.
# something that's actually capable of implementing the hoped for
# computation.


pos_size = 10
hid_size = 10
in_size = 1 + pos_size # the value sampled (embedded) plus the current position param
rnn = nn.LSTMCell(in_size, hid_size)
predict_loc = nn.Sequential(nn.Linear(hid_size, 1))
predict_scale = nn.Sequential(nn.Linear(hid_size, 1), nn.Softplus())

def guide3(l=2):
    assert l >= 2

    c = pyro.param('c0', torch.zeros(1, hid_size))
    h = pyro.param('h0', torch.zeros(1, hid_size))

    pyro.module('rnn', rnn)
    pyro.module('predict_scale', predict_scale)
    pyro.module('predict_loc', predict_loc)

    zs = []
    for i in range(l):
        loc = predict_loc(h).reshape(1)
        scale = predict_scale(h).reshape(1)

        # When used with MF, Normal seems to work better than MVNormal?!?
        z = pyro.sample('z%d' % i, dist.Normal(loc, scale))
        #z = pyro.sample('z%d' % i, dist.MultivariateNormal(loc, scale))

        pos = pyro.param('pos%d'%i, torch.zeros(pos_size))
        rnn_input = torch.cat((z.reshape(1), pos)).reshape(1, -1)
        h, c = rnn(rnn_input, (h, c))

        zs.append(z)

    return (zs[0], zs[-1])


optimizer = pyro.optim.Adam({"lr": 0.001})

#guide3 = AutoDiagonalNormal(model3)
#guide3 = AutoMultivariateNormal(model3)


# NOTE: **need to noodle with torch_distributions.py to re-enable reparam**
svi = pyro.infer.SVI(model3, guide3, optimizer, loss=pyro.infer.Trace_ELBO(num_particles=5))
#svi = pyro.infer.SVI(model3, guide3, optimizer, loss=wake_guide_update)

l = 2 # length/size of latent zs
for i in range(2000):
    loss = svi.step(l)
    if (i+1) % 50 == 0:
        print('%d -- %f' % (i+1, loss))

zss = []
for _ in range(1000):
    zs = guide3(l)
    zss.append(zs)
from matplotlib import pyplot as plt
xs, ys = zip(*zss)
plt.plot(xs, ys, 'b.')
plt.axis('equal')
plt.show()

# ==================================================

# TODO: Try trad. sleep phase guide update.
