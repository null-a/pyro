import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from matplotlib import pyplot as plt

from pyro.contrib.brm import defm, makedesc, makedata
from pyro.contrib.brm.design import metadata_from_df, metadata_from_cols, RealValued, Categorical
from pyro.contrib.brm.family import Normal, HalfCauchy
from pyro.contrib.brm.formula import parse
from pyro.contrib.brm.model import model_repr
from pyro.contrib.brm.fit import fitted, Fit
from pyro.contrib.brm.priors import Prior
from pyro.contrib.brm.pyro_backend import backend as pyro_backend

def next_trial(i, formula, model_desc, data_so_far, meta, q_net):
    eps = 0.5
    N = 1000
    num_coefs = 3

    model = pyro_backend.gen(model_desc)
    data_so_far_coded = makedata(formula, data_so_far)
    data_so_far_coded_torch = {k:pyro_backend.from_numpy(v) for k,v in data_so_far_coded.items()}


    #import pdb; pdb.set_trace()

    designs = meta.levels(['x1','x2']) # e.g. [(0,0),(0,1),(1,0),(1,1)]

    new_data_cols = list(zip(*designs))
    new_data = pd.DataFrame(dict(
        x1=new_data_cols[0],
        x2=new_data_cols[1],
    ))

    # print(designs)
    # print(new_data)
    # print(data_so_far_coded)

    # Do HMC to get samples from p(theta|data_so_far)
    posterior = pyro_backend.nuts(data_so_far_coded_torch, model, iter=N)
    fit = Fit(formula, data_so_far_coded, model_desc, model, posterior, pyro_backend)
    b_samples = posterior.get_param('b')
    assert b_samples.shape == (N, num_coefs)

    # plt.clf()
    # plt.subplot(3,1,1)
    # #plt.xlim((-2,2))
    # plt.hist(b_samples[:,0].numpy(), bins=25, density=True)
    # plt.subplot(3,1,2)
    # #plt.xlim((-2,2))
    # plt.hist(b_samples[:,1].numpy(), bins=25, density=True)
    # plt.subplot(3,1,3)
    # #plt.xlim((-2,2))
    # plt.hist(b_samples[:,2].numpy(), bins=25, density=True)
    # plt.savefig('hist{}.png'.format(i))


    # Use `fitted` to get samples from p(y|theta;d)
    y_samples = torch.tensor(fitted(fit, 'sample', new_data))
    # assert y_samples.shape == (N, len(design_space))

    # Optimise q (y,d -> thetas near zero) on samples.
    #   a. Reshape data.
    inp_d_part = torch.cat([torch.tensor([float(x)*2-1 for x in d]).expand(N, -1) for d in designs])
    assert inp_d_part.shape == (N * len(designs), num_coefs-1), inp_d_part.shape
    inp_y_part = y_samples.transpose(0, 1).reshape(-1,1)
    assert inp_y_part.shape == (N * len(designs), 1), inp_y_part.shape
    inp = torch.cat([inp_d_part, inp_y_part], 1)
    assert inp.shape[0] == N * len(designs)

    out = ((-eps < b_samples) * (b_samples < eps)).float().repeat(len(designs), 1)
    assert out.shape == (N * len(designs), num_coefs)

    #   b. Optimise. (Single net on all designs.)
    #optimise(q_net, inp, out)

    # Estimate (un-normalised) EIG for each design.
    eigs = []
    for j,design in enumerate(designs):

        inp_d = inp[N*j:N*(j+1)]
        out_d = out[N*j:N*(j+1)]


        inp_d = inp_d[:,2:3] # only y, no d

        q_net = mknet()
        optimise(q_net, inp_d, out_d)

        # Make a picture of the training data.

        plt.clf()
        plt.xlabel('y')
        neg_cases = inp_d[out_d[:,1] == 0] # 2nd coef
        pos_cases = inp_d[out_d[:,1] == 1]
        plt.scatter(neg_cases.numpy(), np.random.normal(0, 0.01, neg_cases.shape), marker='.', alpha=0.15, label='coef. not nr. zero')
        plt.scatter(pos_cases.numpy(), np.random.normal(1, 0.01, pos_cases.shape), marker='.', alpha=0.15, label='coef. nr. zero')
        plt.ylim((-0.1, 1.1))

        # Vis. the function implemented by the net.
        test_in = torch.range(inp_d.min(), inp_d.max(), (inp_d.max() -inp_d.min()) / 50.)[0:-1].reshape(-1,1)
        test_out = q_net(test_in)[:,1]
        plt.plot(test_in.numpy(), test_out.detach().numpy(), color='gray', label='q(coef nr. zero|y;d)')

        plt.legend()
        plt.show()






        probs = q_net(inp_d)

        logq = torch.sum(out_d*probs + (1-out_d)*(1-probs), 1)
        eig = torch.mean(logq).item()
        eigs.append(eig)
        print('eig: {}'.format(eig))
        print('all zero would yield: {}'.format(torch.mean(torch.sum(1-out_d, 1))))
        print('====================')

    # Return argmax_d EIG(d)
    dstar = argmax(eigs)

    return dstar, eigs

def argmax(lst):
    return torch.argmax(torch.tensor(lst)).item()


def optimise(net, inp, out):

    # TODO: Mini-batches. (On shuffled inputs/outputs.)

    optimizer = optim.Adam(net.parameters(), lr=0.01)

    for i in range(1000):
        optimizer.zero_grad()
        probs = net(inp)
        logq = torch.mean(torch.sum(out*probs + (1-out)*(1-probs), 1))
        loss = -logq
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print('{} | {}'.format(i+1,logq.item()))

def get_float_input(msg):
    try:
        return float(input(msg))
    except ValueError:
        return get_float_input(msg)

def mknet():
    return nn.Sequential(nn.Linear(1, 100),
                         nn.ReLU(),
                         nn.Linear(100,50),
                         nn.ReLU(),
                         nn.Linear(50, 3),
                         nn.Sigmoid())



formula = parse('y ~ 1 + x1 + x2')

# This says (implicitly) that in the data all possible combinations of
# x1 and x2 are present. Will that fact that this can't be true
# initially trip us up? (i.e. What are the consequences of claiming
# there are data present which in fact aren't.) OTOH, is the fact that
# `Metadata` already carries this information useful for specifying a
# design space that is a subset of the full cartesian product?
meta = metadata_from_cols([
    RealValued('y'),
    Categorical('x1', [0, 1]),
    Categorical('x2', [0, 1]),
])

#print(meta.columns)
#print(meta.levels(['x1','x2']))

model_desc = makedesc(formula, meta, Normal, [Prior(('b',), Normal(0.,1.)), Prior(('resp','sigma'), HalfCauchy(.1))])

#print(model_repr(model_desc))

# If the categorical columns here don't include as least one instance
# of each level (in `meta`) then this won't be coded correctly. This
# is not what we would want. The problem is that `meta` is rederived
# from this data set. Perhaps there needs to be an option to code a
# data frame based on given meta data? Is this similar to what happens
# with new data? (Code a new data from using previous meta. How does
# this relate to the notion of compatability.)
data_so_far = pd.DataFrame(dict(
    y=[0.,0.],

    x1=pd.Categorical([0,1]),
    x2=pd.Categorical([1,0]),
))




for i in range(1000):
    print('Num results: {}'.format(len(data_so_far)))
    dstar, eigs = next_trial(i, formula, model_desc, data_so_far, meta, None)#q_net)
    print(eigs)
    design = meta.levels(['x1','x2'])[dstar]
    print('Next trial: {}'.format(design))
    result = get_float_input('Enter result: ')
    new_row = pd.DataFrame(dict(y=[result],
                                x1=pd.Categorical([design[0]]),
                                x2=pd.Categorical([design[1]])))
    data_so_far = data_so_far.append(new_row)
