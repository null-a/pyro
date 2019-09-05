import pandas as pd

from pyro.contrib.brm.formula import parse, Formula
from pyro.contrib.brm.design import makedata, Metadata, metadata_from_df
from pyro.contrib.brm.fit import Fit
from pyro.contrib.brm.backend import Backend
from pyro.contrib.brm.family import Family, Normal
from pyro.contrib.brm.priors import build_prior_tree
from pyro.contrib.brm.model_pre import build_model_pre
from pyro.contrib.brm.model import build_model, model_repr
from pyro.contrib.brm.pyro_backend import backend as pyro_backend
from pyro.contrib.brm.backend import data_from_numpy

_default_backend = pyro_backend

def makecode(formula, df, family, priors, backend=pyro_backend):
    desc = makedesc(formula, metadata_from_df(df), family, priors)
    return backend.gen(desc).code

def makedesc(formula, metadata, family, priors):
    assert type(formula) == Formula
    assert type(metadata) == Metadata
    assert type(family) == Family
    assert type(priors) == list
    model_desc_pre = build_model_pre(formula, metadata, family)
    prior_tree = build_prior_tree(model_desc_pre, family, priors)
    return build_model(model_desc_pre, prior_tree)

def defm(formula_str, df, family=None, priors=None):
    assert type(formula_str) == str
    assert type(df) == pd.DataFrame
    assert family is None or type(family) == Family
    assert priors is None or type(priors) == list
    family = family or Normal
    priors = priors or []
    formula = parse(formula_str)
    # TODO: Both `makedata` and `designmatrices_metadata` call
    # `coding` (from design.py) internally. Instead we ought to call
    # this once and share the result. (Perhaps by having the process
    # of generating design matrices always return the metadata, while
    # retaining the ability to generate the metadata without a
    # concrete dataset.)
    #
    # Related: Perhaps design matrices ought to always have metadata
    # (i.e. column names) associated with them, as in Patsy. (This
    metadata = metadata_from_df(df)
    desc = makedesc(formula, metadata, family, priors)
    data = makedata(formula, df, metadata)
    return DefmResult(formula, desc, data)

# A wrapper around a pair of model and data. Has a friendly `repr` and
# makes it easy to fit the model.
class DefmResult:
    def __init__(self, formula, desc, data):
        self.formula = formula
        self.desc = desc
        self.data = data

    def fit(self, backend=_default_backend, algo='nuts', **kwargs):
        assert type(backend) == Backend
        assert algo in ['prior', 'nuts', 'svi']
        return getattr(self.generate(backend), algo)(**kwargs)

    # Generate model code and data from this description, using the
    # supplied backend.
    def generate(self, backend=_default_backend):
        assert type(backend) == Backend
        model = backend.gen(self.desc)
        data = data_from_numpy(backend, self.data)
        return GenerateResult(self, backend, model, data)

    # TODO: Could have `svi` and `nuts` methods do the obvious thing
    # with the default backend?

    def __repr__(self):
        return model_repr(self.desc)

# A wrapper around the result of calling DefmResult#generate. Exists
# to support the following interface:
#
# model.generate(<backend>).nuts(...)
# model.generate(<backend>).svi(...)
#
# This makes it possible to get at the code generated by a backend
# without running inference. For example:
#
# model.generate(<backend>).model.code
#
class GenerateResult():
    def __init__(self, defm_result, backend, model, data):
        self.defm_result = defm_result
        self.backend = backend
        self.model = model
        self.data = data

    def _run_algo(self, algo, *args, **kwargs):
        samples = getattr(self.backend, algo)(self.data, self.model, *args, **kwargs)
        return Fit(self.defm_result.formula, self.data, self.defm_result.desc, self.model, samples, self.backend)

    def prior(self, *args, **kwargs):
        return self._run_algo('prior', *args, **kwargs)

    def nuts(self, *args, **kwargs):
        return self._run_algo('nuts', *args, **kwargs)

    def svi(self, *args, **kwargs):
        return self._run_algo('svi', *args, **kwargs)


def brm(formula_str, df, family=None, priors=None, **kwargs):
    return defm(formula_str, df, family, priors).fit(_default_backend, **kwargs)
