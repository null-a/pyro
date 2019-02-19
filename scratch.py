import torch
import torch.nn as nn
from torch.nn.functional import softplus

import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim


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



def wake_guide_update(model, guide, *args, **kwargs):
    surrogate = 0.0
    ws = []
    for i in range(100):
        guide_trace = pyro.poutine.trace(guide).get_trace(*args, **kwargs)
        model_trace = pyro.poutine.trace(
            pyro.poutine.replay(model, trace=guide_trace)).get_trace(*args, **kwargs)
        logp = model_trace.log_prob_sum()
        logq = guide_trace.log_prob_sum()
        w = torch.exp(logp - logq).detach()
        surrogate += w * (-logq)
        ws.append(w)
    # print(ws)
    # print(sum(ws))
    return surrogate / sum(ws)



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
    z = pyro.sample('x', dist.MultivariateNormal(torch.tensor([0., 0]), torch.tensor([[1, 0.9], [0.9, 1]])))

# This is a mean field guide. i.e. The two components of z are
# independent, and this can't capture the dependency of the zs in the
# model.
def guide2():
    cov = torch.diag(torch.exp(pyro.param('sd', torch.log(torch.tensor([1., 1.])))))
    z = pyro.sample('x', dist.MultivariateNormal(torch.tensor([0., 0]), cov))


optimizer = pyro.optim.Adam({"lr": 0.001})

#svi = pyro.infer.SVI(model2, guide2, optimizer, loss=pyro.infer.Trace_ELBO())
svi = pyro.infer.SVI(model2, guide2, optimizer, loss=wake_guide_update)

for i in range(200):
    loss = svi.step()
    if (i+1) % 1 == 0:
        print(loss)

print(torch.exp(pyro.param('sd')).detach())


# TODO: Try trad. sleep phase guide update.
