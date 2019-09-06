import pandas as pd
from pandas.api.types import is_categorical_dtype

import torch
import torch.optim as optim

from pyro.contrib.brm import makedesc
from pyro.contrib.brm.formula import parse
from pyro.contrib.brm.design import makedata, metadata_from_cols, RealValued, Categorical
from pyro.contrib.brm.family import Normal
from pyro.contrib.brm.backend import data_from_numpy
from pyro.contrib.brm.pyro_backend import backend as pyro_backend
from pyro.contrib.brm.fit import Fit, get_param, fitted

from pyro.contrib.brm.oed.nets import QIndep, QFull

# Provides a convenient interface for performing sequential OED.

# Also acts as a single data structure in which to store the
# components of the model definition. (Formula, family, priors, etc.)
# The constructor takes care of the boiler plate required to set-up a
# brmp model.

# Also holds the data-so-far. This data is used when computing the
# next trial. The data-so-far can be extended with the result of an
# experiment using the `add_result` method. Note that `data_so_far` is
# the only mutable state held by instances of SequentialOED. The only
# method that modifies this is `add_result`.

# There are also methods/properties for obtaining information about
# the current sequence:

# oed.design_space
# oed.data_so_far

class SequentialOED:
    def __init__(self, formula_str, cols, family=Normal, priors=[], backend=pyro_backend):
        formula = parse(formula_str)
        metadata = metadata_from_cols(cols)
        model_desc = makedesc(formula, metadata, family, priors)
        model = backend.gen(model_desc)
        data_so_far = empty_df_from_cols(cols)
        num_coefs = len(model_desc.population.coefs)

        # Build a data frame representing the design space.
        dscols = design_space_cols(formula, metadata)
        design_space = metadata.levels(dscols)
        design_space_df = pd.DataFrame(dict((name, pd.Categorical(col))
                                            for name, col in zip(dscols, list(zip(*design_space)))))

        # TODO: Prefix non-public stuff with underscores?
        self.formula = formula
        self.metadata = metadata
        self.model_desc = model_desc
        self.model = model
        self.data_so_far = data_so_far
        self.num_coefs = num_coefs
        self.design_space = design_space
        self.design_space_df = design_space_df

        self.backend = backend
        self.num_samples = 1000

    def next_trial(self, callback=None, verbose=False):

        if callback is None:
            callback = lambda *args: None

        # Code the data-so-far data frame into design matrices.
        dsf = data_from_numpy(self.backend,
                              makedata(self.formula, self.data_so_far, self.metadata))

        # Draw samples from current distribution over parameters.
        if len(self.data_so_far) == 0:
            samples = self.backend.prior(dsf, self.model, self.num_samples)
        else:
            samples = self.backend.nuts(dsf, self.model, self.num_samples)
        fit = Fit(self.formula, dsf, self.model_desc, self.model, samples, self.backend)

        b_samples = get_param(fit, 'b') # Values sampled for population-level coefs. (numpy array.)
        assert b_samples.shape == (self.num_samples, self.num_coefs)

        # Draw samples from p(y|theta;d)
        y_samples = fitted(fit, 'sample', self.design_space_df) # numpy array.
        assert y_samples.shape == (self.num_samples, len(self.design_space))

        # All ANN work is done using PyTorch, so convert samples from
        # numpy to torch ready for what follows.
        b_samples = torch.tensor(b_samples)
        y_samples = torch.tensor(y_samples)

        # Compute the targets. (These are used by all designs.)
        eps = 0.5
        # TODO: This is a long tensor for the benefit of QFull. It
        # might be worth considering whether it's possible to
        # reorganise things to avoid repeatedly converting from long
        # to float. (e.g. In `log_probs` of `QIndep`.)
        targets = ((-eps < b_samples) & (b_samples < eps)).long()
        assert targets.shape == (self.num_samples, self.num_coefs)

        # Compute the (unnormalized) EIG for each design.
        eigs = []
        cbvals = []
        for i, design in enumerate(self.design_space):
            inputs = y_samples[:,i].unsqueeze(1) # The ys for this particular design.

            # Construct and optimised the network.
            #q_net = QIndep(self.num_coefs)
            q_net = QFull(self.num_coefs)
            optimise(q_net, inputs, targets, verbose)

            eig = torch.mean(q_net.logprobs(inputs, targets)).item()
            eigs.append(eig)

            cbvals.append(callback(i, design, q_net, inputs, targets))

        dstar = argmax(eigs)
        return self.design_space[dstar], dstar, list(zip(self.design_space, eigs)), fit, cbvals

    def add_result(self, design, result):
        self.data_so_far = extend_df_with_result(self.formula, self.metadata, self.data_so_far, design, result)


def argmax(lst):
    return torch.argmax(torch.tensor(lst)).item()


def optimise(net, inputs, targets, verbose):

    # TODO: Mini-batches. (On shuffled inputs/outputs.)
    # TODO: Note: using some weight decay probably makes sense here.

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


def empty_df_from_cols(cols):
    def emptydfcol(col):
        if type(col) == Categorical:
            return pd.Categorical([])
        elif type(col) == RealValued:
            return []
        else:
            raise Exception('encountered unsupported column type')
    return pd.DataFrame({col.name: emptydfcol(col) for col in cols})


# Extract the names of the columns associated with population level
# effects. (Assumes no interactions. But only requires taking union of
# all factors?)
def design_space_cols(formula, meta):
    coefs = []
    for t in formula.terms:
        if len(t.factors) == 0:
            pass # intercept
        elif len(t.factors) == 1:
            factor = t.factors[0]
            assert type(meta.column(factor)) == Categorical
            coefs.append(factor)
        else:
            raise Exception('interactions not supported')
    return coefs


# TODO: Does it *really* take this much work to add a row to a df?
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
