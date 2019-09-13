from collections import namedtuple

import numpy as np
import pandas as pd

from pyro.contrib.brm.model import model_repr, parameter_names, scalar_parameter_map, scalar_parameter_names
from pyro.contrib.brm.family import free_param_names
from pyro.contrib.brm.design import predictors, metadata_from_df
from pyro.contrib.brm.backend import data_from_numpy

# `Fit` carries around `formula` and `contrasts` for the sole purpose
# of being able to encode any new data passed to `fitted`.

# TODO: Consider storing `formula` and `contrasts` on `ModelDescPre`
# (and then `ModelDesc`) as an alternative to storing them on `Fit`.
# (Since it seems more natural.)

# One snag is that `ModelDescPre` only sees the lengths of any custom
# coding and not the full matrix. However, while deferring having to
# give a concrete data frame is useful (because it saves having to
# make up data to tinker with a model), it's not clear that deferring
# having to give contrasts has a similar benefit.

Fit = namedtuple('Fit', 'formula contrasts data model_desc model samples backend')
Samples = namedtuple('Samples', ['raw_samples', 'get_param', 'location'])

def param_marginal(fit, parameter_name):
    assert type(fit) == Fit
    samples = fit.samples
    return fit.backend.to_numpy(samples.get_param(parameter_name))

default_quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

def format_quantiles(qs):
    return ['{:g}%'.format(q * 100) for q in qs]

# Computes statistics for an array produced by `marginal`.
def marginal_stats(arr, qs):
    assert len(arr.shape) == 2
    assert type(qs) == list
    assert all(0 <= q <= 1 for q in qs)
    mean = np.mean(arr, 0)
    sd = np.std(arr, 0)
    quantiles = np.quantile(arr, qs, 0)
    stacked = np.hstack((mean.reshape((-1, 1)), sd.reshape((-1, 1)), quantiles.T))
    return stacked

# TODO: Would it be better to replace these tables with pandas data
# frames? They also let you get at the underlying data as a numpy
# array (I assume), and have their own pretty printing.
class ArrReprWrapper:
    def __init__(self, array, row_labels, col_labels):
        assert len(array.shape) == 2
        assert row_labels is None or array.shape[0] == len(row_labels)
        assert col_labels is None or array.shape[1] == len(col_labels)
        self.array = array
        self.col_labels = col_labels
        self.row_labels = row_labels

    def __repr__(self):
        # Format a float. 2 decimal places, space for sign.
        def ff(x):
            return '{: .2f}'.format(x)
        table = [[ff(c) for c in r] for r in self.array.tolist()]
        return layout_table(add_labels(table, self.col_labels, self.row_labels))

def add_labels(table, col_labels, row_labels):
    assert type(table) == list
    assert all(type(row) == list for row in table)
    out = [col_labels] if col_labels is not None else []
    out += table
    if row_labels is not None:
        rlabels = row_labels if col_labels is None else [''] + row_labels
        assert len(out) == len(rlabels)
        out = [[name] + r for r, name in zip(out, rlabels)]
    return out

def layout_table(rows):
    out = []
    num_rows = len(rows)
    assert num_rows > 0
    num_cols = len(rows[0])
    assert all(len(row) == num_cols for row in rows)
    max_widths = [0] * num_cols
    for row in rows:
        for i, cell in enumerate(row):
            max_widths[i] = max(max_widths[i], len(cell))
    fmt = ' '.join('{{:>{}}}'.format(mw) for mw in max_widths)
    return '\n'.join(fmt.format(*row) for row in rows)

# TODO: This doesn't match the brms interface, but the deviation
# aren't improvements either. Figure out what to do about that.

# brms                                               | brmp
# -----------------------------------------------------------------------------------
# fitted(fit, summary=FALSE)                         | fitted(fit)
# fitted(dpar='mu', scale='linear', summary=FALSE)   | fitted(fit, 'linear')
# fitted(dpar='mu', scale='response', summary=FALSE) | fitted(fit, 'response')
# fitted(fit, newdata=..., summary=FALSE)            | fitted(fit, data=...)
# fitted(fit, ..., summary=TRUE)                     | summary(fitted(fit, ...))
# predict(fit, summary=FALSE)                        | fitted(fit, 'sample')
# predict(fit, summary=TRUE)                         | summary(fitted(fit, 'sample'))

# https://rdrr.io/cran/brms/man/fitted.brmsfit.html

def fitted(fit, what='expectation', data=None):
    assert type(fit) == Fit
    assert what in ['sample', 'expectation', 'linear', 'response']
    assert data is None or type(data) is pd.DataFrame

    get_param         = fit.samples.get_param
    location          = fit.samples.location
    to_numpy          = fit.backend.to_numpy
    expected_response = fit.model.expected_response_fn
    sample_response   = fit.model.sample_response_fn
    inv_link          = fit.model.inv_link_fn

    # TODO: I suspect this ought to use the metadata from the original
    # data, to ensure consistent coding of categorical
    # columns/groupings.

    mu = location(fit.data if data is None
                  else data_from_numpy(fit.backend, predictors(fit.formula, data, metadata_from_df(data), fit.contrasts)))

    if what == 'sample' or what == 'expectation':
        args = [mu if name == 'mu' else get_param(name)
                for name in free_param_names(fit.model_desc.response.family)]
        response_fn = sample_response if what == 'sample' else expected_response
        return to_numpy(response_fn(*args))
    elif what == 'linear':
        return to_numpy(mu)
    elif what == 'response':
        return to_numpy(inv_link(mu))
    else:
        raise 'Unhandled value of the `what` parameter encountered.'


# TODO: We could follow brms and make this available via a `summary`
# flag on `fitted`?
def summary(arr, qs=default_quantiles, row_labels=None):
    col_labels = ['mean', 'sd'] + format_quantiles(qs)
    return ArrReprWrapper(marginal_stats(arr, qs), row_labels, col_labels)

# Similar to the following:
# https://rdrr.io/cran/rstan/man/stanfit-method-summary.html

# TODO: This produces the same output as the old implementation of
# `marginal`, though it's less efficient. Can the previous efficiency
# be recovered? The problem is that we pull out each individual scalar
# parameter as a vector and then stack those, rather than just stack
# entire parameters as before. One thought is that such an
# optimisation might be best pushed into `get_scalar_param`. i.e. This
# might accept a list of a parameter names and return the
# corresponding scalar parameters stacked into a matrix. The aim would
# be to do this without performing any unnecessary slicing. (Though
# this sounds fiddly.)
def marginals(fit, qs=default_quantiles):
    assert type(fit) == Fit
    names = scalar_parameter_names(fit.model_desc)
    # TODO: Every call to `get_scalar_param` rebuilds the scalar
    # parameter map.
    vecs = [get_scalar_param(fit, name) for name in names]
    col_labels = ['mean', 'sd'] + format_quantiles(qs)
    arr = marginal_stats(np.stack(vecs, axis=1), qs)
    return ArrReprWrapper(arr, names, col_labels)

def print_model(fit):
    print(model_repr(fit.model_desc))

# A back end agnostic wrapper around back end specific implementations
# of `fit.samples.get_param`.
def get_param(fit, name):
    assert type(fit) == Fit
    return fit.backend.to_numpy(fit.samples.get_param(name))

# TODO: If parameter and scalar parameter names never clash, perhaps
# having a single lookup method would be convenient. Perhaps this
# could be wired up to `fit.samples[...]`?
def get_scalar_param(fit, name):
    assert type(fit) == Fit
    m = scalar_parameter_map(fit.model_desc)
    res = [p for (n, p) in m if n == name]
    assert len(res) < 2
    if len(res) == 0:
        raise KeyError('unknown parameter name: {}'.format(name))
    param_name, index = res[0]
    # Construct a slice to pick out the given index at all rows.
    slc = (slice(None, None, None),) + index
    return get_param(fit, param_name)[slc]
