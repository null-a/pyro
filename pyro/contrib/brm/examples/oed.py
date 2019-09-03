import pandas as pd
from pandas.api.types import is_categorical_dtype
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import one_hot
import numpy as np

from matplotlib import pyplot as plt

from pyro.contrib.brm import defm, makedesc
from pyro.contrib.brm.design import metadata_from_df, metadata_from_cols, RealValued, Categorical, makedata
from pyro.contrib.brm.family import Normal, HalfCauchy, HalfNormal
from pyro.contrib.brm.formula import parse
from pyro.contrib.brm.model import model_repr
from pyro.contrib.brm.fit import fitted, Fit, marginals
from pyro.contrib.brm.priors import Prior
from pyro.contrib.brm.pyro_backend import backend as pyro_backend


def next_trial(formula, model_desc, data_so_far, meta, verbose=False):
    eps = 0.5
    N = 1000

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
        if verbose:
            print('Sampling from prior...')
        posterior = pyro_backend.prior(data_so_far_coded, model, num_samples=N)
    else:
        # Do HMC to get samples from p(theta|data_so_far)
        if verbose:
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


    targets = ((-eps < b_samples) * (b_samples < eps)).long().repeat(len(designs), 1)
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

        #q_net = QIndep(num_coefs)
        q_net = QFull(num_coefs)
        optimise(q_net, inputs_d, targets_d, verbose)

        # Make a picture of the training data.

        for k in range(num_coefs):

            neg_cases = inputs_d[targets_d[:,k] == 0]
            pos_cases = inputs_d[targets_d[:,k] == 1]

            # Vis. the function implemented by the net.
            imin = inputs_d.min()
            imax = inputs_d.max()
            test_in = torch.arange(imin, imax, (imax-imin)/50.).reshape(-1, 1)
            test_out = q_net.marginal_probs(test_in, k).detach()

            plot_data[j][k] = (pos_cases.numpy(), neg_cases.numpy(), test_in.numpy(), test_out.numpy(), design)

        eig = torch.mean(q_net.logprobs(inputs_d, targets_d)).item()
        eigs.append(eig)

    # Return argmax_d EIG(d)
    dstar = argmax(eigs)

    return designs[dstar], dstar, list(zip(designs, eigs)), fit, plot_data

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


def optimise(net, inputs, targets, verbose=False):

    # TODO: Mini-batches. (On shuffled inputs/outputs.)

    optimizer = optim.Adam(net.parameters(), lr=0.01)

    for i in range(1000):
        optimizer.zero_grad()
        loss = -torch.mean(net.logprobs(inputs, targets))
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0 and verbose:
            print('{:5d} | {:.6f}'.format(i+1,loss.item()))

    if verbose:
       print('--------------------')

def get_float_input(msg):
    try:
        return float(input(msg))
    except ValueError:
        return get_float_input(msg)


class QIndep(nn.Module):
    def __init__(self, num_coef):
        super(QIndep, self).__init__()
        assert type(num_coef) == int
        assert num_coef > 0
        self.num_coef = num_coef
        self.net = nn.Sequential(nn.Linear(1, 100),
                                 nn.ReLU(),
                                 nn.Linear(100,50),
                                 nn.ReLU(),
                                 nn.Linear(50, num_coef),
                                 nn.Sigmoid())

    def forward(self, inputs):
        assert inputs.shape[1] == 1
        # TODO: There's probably a better approach than clamping --
        # parameterize loss by logits?
        eps = 1e-6
        return self.net(inputs).clamp(eps, 1-eps)

    # Compute (vectorised, over multiple y and m) q(m|y;d).
    # m: targets
    # y: inputs
    # (;d because we make a fresh net for each design.)
    def logprobs(self, inputs, targets):
        assert inputs.shape[0] == targets.shape[0]
        N = inputs.shape[0]
        assert inputs.shape == (N, 1)
        assert targets.shape == (N, self.num_coef)
        probs = self.forward(inputs)
        targetsf = targets.float()
        return torch.sum(targetsf*torch.log(probs) + (1-targetsf)*torch.log(1-probs), 1)

    # Compute the marginal probability of a particular coefficient
    # being within [-eps,eps]. For this particular Q (which assumes
    # the joint is the product of the marginals) this only requires us
    # to pick out the appropriate marginal.
    def marginal_probs(self, inputs, coef):
        assert type(coef) == int
        assert 0 <= coef < self.num_coef
        probs = self.forward(inputs)
        return probs[:,coef]


# e.g. tensor([[0,0,1], [1,1,0]]) => tensor([1,6])
def bits2long(t):
    N, width = t.shape
    powers_of_two = torch.tensor([2**i for i in range(width-1, -1, -1)])
    out = torch.sum(t * powers_of_two, 1)
    assert out.shape == (N,)
    return out

# e.g. (3,4) => [0,0,1,1]
def int2bits(i, width):
    assert i < 2**width
    return [int(b) for b in ('{:0'+str(width)+'b}').format(i)]

# All of the target values (as bit vectors) that satisfy \theta_coef == 1
def target_values_for_marginal(coef, num_coef):
    #print(list(int2bits(i, num_coef) for i in range(2**num_coef)))
    values = [bits for bits in (int2bits(i, num_coef) for i in range(2**num_coef)) if bits[coef] == 1]
    #print(values)
    return torch.tensor(values)

# e.g. [[1,0,1],[0,0,1]] => [[0,0,0,0,0,1,0,0],[0,1,0,0,0,0,0,0]]
def bits2onehot(t):
    width = t.shape[1]
    return one_hot(bits2long(t), 2**width)

class QFull(nn.Module):
    def __init__(self, num_coef):
        super(QFull, self).__init__()
        assert type(num_coef) == int
        assert num_coef > 0
        self.num_coef = num_coef
        self.net = nn.Sequential(nn.Linear(1, 100),
                                 nn.ReLU(),
                                 nn.Linear(100, 50),
                                 nn.ReLU(),
                                 nn.Linear(50, 2**num_coef),
                                 nn.LogSoftmax(dim=1))

    def forward(self, inputs):
        assert inputs.shape[1] == 1
        return self.net(inputs)

    def logprobs(self, inputs, targets):
        assert inputs.shape[0] == targets.shape[0]
        N = inputs.shape[0]
        assert inputs.shape == (N, 1)
        assert targets.shape == (N, self.num_coef)
        logprobs = self.forward(inputs)
        assert logprobs.shape[1] == 2 ** self.num_coef
        return torch.sum(logprobs * bits2onehot(targets).float(), 1)

    def marginal_probs(self, inputs, coef):
        assert type(coef) == int
        assert 0 <= coef < self.num_coef
        logprobs = self.forward(inputs)
        cols = bits2long(target_values_for_marginal(coef, self.num_coef))
        return torch.sum(torch.exp(logprobs[:,cols]), 1)


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
        Categorical('x2', ['c','d']),
        #Categorical('x3', ['e', 'f']),
    ])

    #print(meta.columns)
    #print(meta.levels(['x1','x2']))

    model_desc = makedesc(formula, meta, Normal, [
        Prior(('b',),           Normal(0.,1.)),
        Prior(('resp','sigma'), HalfNormal(.2)),
    ])

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
        design, dstar, eigs, fit, plot_data = next_trial(formula, model_desc, data_so_far, meta, verbose=True)
        print(marginals(fit))
        make_training_data_plot(plot_data)
        print('EIGs:')
        print(eigs)
        print('Next trial: {}'.format(design))
        result = get_float_input('Enter result: ')
        # TODO: There's no support for group level effects here.
        data_so_far = extend_df_with_result(formula, meta, data_so_far, design, result)

if __name__ == '__main__':
    main()
