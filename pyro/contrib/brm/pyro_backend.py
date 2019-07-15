import time

import numpy as np
import torch

from pyro.infer.mcmc import MCMC, NUTS

import pyro
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.contrib.autoguide import AutoDiagonalNormal
from pyro.optim import Adam

from pyro.contrib.brm.backend import Backend, Model, apply_default_hmc_args
from pyro.contrib.brm.fit import Posterior
from pyro.contrib.brm.pyro_codegen import gen


# The interface and its types are something like:

# posterior.samples :: bs
# posterior.get_param :: bs -> ps
# posterior.to_numpy :: ps -> ndarray
# backend.from_numpy :: ndarray -> ps
# model.inv_link_fn :: ps -> ps
# model.expected_response_fn :: (ps, ps, ...) -> ps

# Where:

# bs is the type of a collection of samples (Here, a list of Pyro
# traces.)
#
# ps is the type of a collection of sampled parameters (Here, a torch
# array, where first dimension ranges over samples.)
#


# The idea is that `pyro_posterior` and `pyro_get_param` capture the
# backend specific part of processing posterior samples. Alternatives
# to this approach include:

# 1. Have each back end return an iterable of samples, where each
# sample is something like a dictionary holding all of the parameters
# of interest. (Effectively the backend would be returning the result
# of mapping `get_param` over every sample for every parameter.

# 2. Have each backend implement some kind of query interface,
# allowing things like `query.marginal('b').mean()`, etc.

def posterior(run):
    return Posterior(run.exec_traces, get_param)

# Extracts a value of interest (e.g. 'b', 'r_0', 'L_1', 'sigma') from
# a single sample.

# It's expected that this should support all parameter names returned
# by `parameter_names(model)` where `model` is the `ModelDesc` from
# which samples were drawn. It should also support fetching the
# (final) value bound to `mu` in the generated code.
def get_param(samples, name):

    # TODO: Temp. disallow fetching mu this way. The correct way to do
    # this generically is to use `location`, and I'd like to know when
    # that isn't being used, hence the assertion.

    # TODO: Should the interface be that a backend's `get_param`
    # should *only* work for `parameter_names(model`)? If so, the
    # ability to grab `mu` out of samples needs preserving, perhaps in
    # some other method.

    assert not name == 'mu', '`mu` is not a param. Use `location` to fetch `mu`.'

    def getp(sample):
        if name in sample.nodes:
            return sample.nodes[name]['value']
        else:
            return sample.nodes['_RETURN']['value'][name]

    # `detach` is only necessary for SVI.

    # Creating this intermediate list is a bit unpleasant -- could
    # fill a pre-allocated array instead.
    #
    return torch.stack([getp(sample).detach() for sample in samples])


# samples :: bs
# data    :: is a dict, with values already put through `from_numpy`
#
# return  :: ps (i.e. the same kind of thing as `get_param`)
def location(modelfn, samples, data):

    # In general, we need to re-run the model, taking values from the
    # given samples at `sample` sites, and using the given data.

    # TODO: For the special case where `data` is the data used for
    # inference, we can use `get_param` to fetch `mu` from the return
    # value. (This is back-end specific.)

    # The current strategy is to patch up the trace so that we can
    # re-run the model as is. One appealing aspect of this is that it
    # might be easy to adapt to this also sample new values for y.
    # (Rather than having to add something analogous to
    # `expected_response_fn` to code gen.) OTOH, it seems like
    # `expected_response_fn` will stick around, since computing the
    # expectation as part of the model seems wasteful during
    # inference, and perhaps it's easy to adapt this to do sampling
    # with out really adding any new code gen.

    # One alternative would be to add a flag to the model that allows
    # the plate & response distribution to be skipped. (Or
    # alternatively generate two versions of the model if avoiding
    # checking the flag during inference is desirable.)

    # Another option, of course, is to codegen an extra function that
    # directly implements the requuired functionality. This function
    # could be vectorized (i.e. operate on multiple samples in
    # parallel) which is nice.

    def f(sample):
        trace = sample.copy()
        trace.remove_node('y')
        # The length of the vector of indices stored at the plate
        # will likely not match the number of data points in the
        # data. (When applying `location` to "new data".) We could
        # explicitly patch up `trace.node['obs']['value']` but
        # simply removing the node seems to work just as well.
        trace.remove_node('obs')
        return poutine.replay(modelfn, trace)(**data)['mu']

    return torch.stack([f(s) for s in samples])


# This provides a back-end specific method for turning a parameter
# samples (as returned by `get_param`) into a numpy array.

def to_numpy(param_samples):
    return param_samples.numpy()

# This gives the back-end an opportunity to convert model data from
# numpy (as generated by design.py) into its preferred representation.

# Here we convert to torch tensors. Arrays of floats use the default
# dtype.

def from_numpy(arr):
    assert type(arr) == np.ndarray
    default_dtype = torch.get_default_dtype()
    if arr.size == 0:
        # Attempting to convert an empty array using
        # `torch.from_numpy()` throws an error, so make a new
        # empty array instead. I think this can only happen when
        # `arr` holds floats, which at present will always be 64
        # bit. (See `col2numpy` in design.py.)
        assert arr.dtype == np.float64
        out = torch.empty(arr.shape)
        assert out.dtype == default_dtype
        return out
    else:
        out = torch.from_numpy(arr)
        if torch.is_floating_point(out) and not out.dtype == default_dtype:
            out = out.type(default_dtype)
        return out

def nuts(data, model, iter=None, warmup=None):
    assert type(data) == dict
    assert type(model) == Model

    iter, warmup = apply_default_hmc_args(iter, warmup)

    nuts_kernel = NUTS(model.fn, jit_compile=False, adapt_step_size=True)
    run = MCMC(nuts_kernel, num_samples=iter, warmup_steps=warmup).run(**data)

    # TODO: Optimization -- delegate to `location` only when `d` is
    # not `data`. Otherwise, fetch `mu` from the traces we've already
    # collected.
    return Posterior(run.exec_traces, get_param, lambda s, d: location(model.fn, s, d))

def svi(data, model, iter=None, num_samples=None, autoguide=None, optim=None):
    assert type(data) == dict
    assert type(model) == Model

    assert iter is None or type(iter) == int
    assert num_samples is None or type(num_samples) == int
    assert autoguide is None or callable(autoguide)

    # TODO: Fix that this interface doesn't work for
    # `AutoLaplaceApproximation`, which requires different functions
    # to be used for optimisation / collecting samples.
    autoguide = AutoDiagonalNormal if autoguide is None else autoguide
    optim = Adam({'lr': 1e-3}) if optim is None else optim

    guide = autoguide(model.fn)
    svi = SVI(model.fn, guide, optim, loss=Trace_ELBO())
    pyro.clear_param_store()

    t0 = time.time()
    max_iter_str_width = len(str(iter))
    max_out_len = 0
    for i in range(iter):
        loss = svi.step(**data)
        t1 = time.time()
        if t1 - t0 > 0.5 or (i+1) == iter:
            iter_str = str(i+1).rjust(max_iter_str_width)
            out = 'iter: {} | loss: {:.3f}'.format(iter_str, loss)
            max_out_len = max(max_out_len, len(out))
            # Sending the ANSI code to clear the line doesn't seem to
            # work in notebooks, so instead we pad the output with
            # enough spaces to ensure we overwrite all previous input.
            print('\r{}'.format(out.ljust(max_out_len)), end='')
            t0 = t1
    print()

    # We run the guide to generate traces from the (approx.)
    # posterior. We also run the model against those traces in order
    # to compute transformed parameters, such as `b`, `mu`, etc.
    def get_model_trace():
        guide_tr = poutine.trace(guide).get_trace()
        model_tr = poutine.trace(poutine.replay(model.fn, trace=guide_tr)).get_trace(**data)
        return model_tr

    # Represent the posterior as a bunch of samples, ignoring the
    # possibility that we might plausibly be able to figure out e.g.
    # posterior maginals from the variational parameters.
    samples = [get_model_trace() for _ in range(num_samples)]

    return Posterior(samples, get_param)

backend = Backend('Pyro', gen, nuts, svi, from_numpy, to_numpy)
