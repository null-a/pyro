import numpy as np
import torch

from pyro.infer.mcmc import MCMC, NUTS

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
    def getp(sample):
        if name in sample.nodes:
            return sample.nodes[name]['value']
        else:
            return sample.nodes['_RETURN']['value'][name]

    # Creating this intermediate list is a bit unpleasant -- could
    # fill a pre-allocated array instead.
    #
    return torch.stack([getp(sample) for sample in samples])

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

    return posterior(run)

def svi(*args, **kwargs):
    raise NotImplementedError

backend = Backend('Pyro', gen, nuts, svi, from_numpy, to_numpy)
