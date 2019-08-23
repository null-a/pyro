import pandas as pd
from pandas.api.types import is_categorical_dtype
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from matplotlib import pyplot as plt

from pyro.contrib.brm import defm, makedesc
from pyro.contrib.brm.design import metadata_from_df, metadata_from_cols, RealValued, Categorical, makedata
from pyro.contrib.brm.family import Normal, HalfCauchy
from pyro.contrib.brm.formula import parse
from pyro.contrib.brm.model import model_repr
from pyro.contrib.brm.fit import fitted, Fit
from pyro.contrib.brm.priors import Prior
from pyro.contrib.brm.pyro_backend import backend as pyro_backend

def next_trial(formula, model_desc, data_so_far, meta):
    eps = 0.5
    N = 50#1000

    num_coefs = len(model_desc.population.coefs)

    model = pyro_backend.gen(model_desc)
    data_so_far_coded = {k:pyro_backend.from_numpy(v)
                         for k,v in makedata(formula, data_so_far, meta).items()}

    # e.g. ['x1', 'x2']
    colnames = design_space_cols(formula, meta)
    # e.g. [(0,0),(0,1),(1,0),(1,1)]
    designs = meta.levels(colnames)
    #print(designs)

    # Turn the design space into a df so that we can use `fitted` to
    # sample `y` from the model for each design.
    design_space_df_cols = list(zip(*designs))
    design_space_df = pd.DataFrame(dict((name, pd.Categorical(col)) for name,col in zip(colnames, design_space_df_cols)))
    #print(design_space_df)


    if len(data_so_far) == 0:
        print('Sampling from prior...')
        posterior = pyro_backend.prior(data_so_far_coded, model, num_samples=N)
    else:
        # Do HMC to get samples from p(theta|data_so_far)
        print('Running HMC...')
        posterior = pyro_backend.nuts(data_so_far_coded, model, iter=N)
    fit = Fit(formula, data_so_far_coded, model_desc, model, posterior, pyro_backend)
    b_samples = posterior.get_param('b')
    assert b_samples.shape == (N, num_coefs)


    # Use `fitted` to get samples from p(y|theta;d)
    y_samples = torch.tensor(fitted(fit, 'sample', design_space_df))
    # assert y_samples.shape == (N, len(design_space))

    # Optimise q (y,d -> thetas near zero) on samples.
    #   a. Reshape data.

    # Arrange sampled y in to a vector. This starts with all of the
    # samples collected for the first design, followed by samples for
    # the second design, etc.
    inputs = y_samples.transpose(0, 1).reshape(-1, 1)
    assert inputs.shape == (N * len(designs), 1)


    targets = ((-eps < b_samples) * (b_samples < eps)).float().repeat(len(designs), 1)
    assert targets.shape == (N * len(designs), num_coefs)

    #   b. Optimise. (Single net on all designs.)

    # A (nested) list used to store data which can later be used to
    # make a picture of the "training data".
    plot_data = [[None] * num_coefs for _ in range(len(designs))]

    # (Un-normalised) EIG for each design.
    eigs = []

    for j,design in enumerate(designs):

        inputs_d = inputs[N*j:N*(j+1)]
        targets_d = targets[N*j:N*(j+1)]

        q_net = mknet(num_coefs)
        optimise(q_net, inputs_d, targets_d)

        # Make a picture of the training data.

        for k in range(num_coefs):

            neg_cases = inputs_d[targets_d[:,k] == 0]
            pos_cases = inputs_d[targets_d[:,k] == 1]

            # Vis. the function implemented by the net.
            imin = inputs_d.min()
            imax = inputs_d.max()
            test_in = torch.arange(imin, imax, (imax-imin)/50.).reshape(-1, 1)
            test_out = q_net(test_in)[:,k].detach()

            plot_data[j][k] = (pos_cases.numpy(), neg_cases.numpy(), test_in.numpy(), test_out.numpy(), design)

        probs = q_net(inputs_d)

        logq = torch.sum(targets_d*probs + (1-targets_d)*(1-probs), 1)
        eig = torch.mean(logq).item()
        eigs.append(eig)
        print('eig: {}'.format(eig))
        print('all zero would yield: {}'.format(torch.mean(torch.sum(1-targets_d, 1))))
        print('====================')

    plt.show()

    # Return argmax_d EIG(d)
    dstar = argmax(eigs)

    return designs[dstar], dstar, eigs, plot_data

def make_training_data_plot(plot_data):
    plt.figure(figsize=(12,12))
    for j,row in enumerate(plot_data):
        for k,(pos_cases, neg_cases,test_in, test_out, design) in enumerate(row):
            plt.subplot(len(plot_data), len(row), (j*len(row) + k)+1)
            if j == 0:
                plt.title('coef={}'.format(k))
            if k == 0:
                plt.ylabel('q(m|y;d={})'.format(design))
            plt.xlabel('y')
            plt.scatter(neg_cases, np.random.normal(0, 0.01, neg_cases.shape), marker='.', alpha=0.15, label='coef. not nr. zero')
            plt.scatter(pos_cases, np.random.normal(1, 0.01, pos_cases.shape), marker='.', alpha=0.15, label='coef. nr. zero')
            plt.ylim((-0.1, 1.1))
            plt.plot(test_in, test_out, color='gray', label='q(m|y;d)')
            #plt.legend()

    plt.show()


def argmax(lst):
    return torch.argmax(torch.tensor(lst)).item()


def optimise(net, inputs, targets):

    # TODO: Mini-batches. (On shuffled inputs/outputs.)

    optimizer = optim.Adam(net.parameters(), lr=0.01)

    for i in range(1000):
        optimizer.zero_grad()
        probs = net(inputs)
        logq = torch.mean(torch.sum(targets*probs + (1-targets)*(1-probs), 1))
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

def mknet(outsize):
    return nn.Sequential(nn.Linear(1, 100),
                         nn.ReLU(),
                         nn.Linear(100,50),
                         nn.ReLU(),
                         nn.Linear(50, outsize),
                         nn.Sigmoid())


def df_append_row(df, row):
    assert type(row) == dict
    row_df = pd.DataFrame({k: pd.Categorical([v]) if is_categorical_dtype(df[k]) else [v]
                           for k,v in row.items()})
    out = df.append(row_df, sort=False)
    # Simply appending a new df produces a new df in which
    # (sometimes!?!) a column that was categorical in the two inputs
    # is not categorical in the output. This tweaks the result to
    # account for that. I don't know why this is happening.
    for k in row.keys():
        if is_categorical_dtype(df[k]) and not is_categorical_dtype(out[k]):
            out[k] = pd.Categorical(out[k])
    return out


# Extract the names of the columns associated with population level
# effects. (Assumes no interactions. But only requires taking union of
# all factors?)
def design_space_cols(formula, meta):
    coefs = []
    for t in formula.terms:
        if len(t.factors) == 0:
            pass#coefs.append(None) # intercept
        elif len(t.factors) == 1:
            factor = t.factors[0]
            assert type(meta.column(factor)) == Categorical
            coefs.append(factor)
        else:
            raise Exception('interactions not supported')
    return coefs

def extend_df_with_result(formula, meta, data_so_far, design, result):
    assert type(design) == tuple
    assert type(result) == float
    cols = design_space_cols(formula, meta)
    assert len(design) == len(cols)
    # This assumes that `design` is ordered following
    # `design_space_cols`.
    row = dict(zip(cols, design))
    row['y'] = result
    return df_append_row(data_so_far, row)



def main():

    formula = parse('y ~ 1 + x1 + x2')

    # This says (implicitly) that in the data all possible combinations of
    # x1 and x2 are present. Will that fact that this can't be true
    # initially trip us up? (i.e. What are the consequences of claiming
    # there are data present which in fact aren't.) OTOH, is the fact that
    # `Metadata` already carries this information useful for specifying a
    # design space that is a subset of the full cartesian product?
    meta = metadata_from_cols([
        RealValued('y'),
        Categorical('x1', ['a','b']),
        Categorical('x2', ['c','d','e']),
        #Categorical('x3', ['e', 'f']),
    ])

    #print(meta.columns)
    #print(meta.levels(['x1','x2']))

    model_desc = makedesc(formula, meta, Normal, [Prior(('b',), Normal(0.,1.)), Prior(('resp','sigma'), HalfCauchy(.1))])

    #print(model_repr(model_desc))

    data_so_far = pd.DataFrame(dict(
        y=[],
        x1=pd.Categorical([]),
        x2=pd.Categorical([]),
        #x3=pd.Categorical([]),
    ))


    for i in range(1000):
        #print('Num results: {}'.format(len(data_so_far)))
        #print(data_so_far)
        design, dstar, eigs, plot_data = next_trial(formula, model_desc, data_so_far, meta)
        make_training_data_plot(plot_data)
        print(eigs)
        print('Next trial: {}'.format(design))
        result = get_float_input('Enter result: ')
        # TODO: There's no support for group level effects here.
        data_so_far = extend_df_with_result(formula, meta, data_so_far, design, result)

if __name__ == '__main__':
    main()
