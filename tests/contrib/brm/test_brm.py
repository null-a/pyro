import pytest

import torch
import pandas as pd

import pyro.poutine as poutine
from pyro.distributions import Independent, Normal, Cauchy, HalfCauchy, LKJCorrCholesky

from pyro.contrib.brm.formula import parse, Formula, _1, Term, OrderedSet
from pyro.contrib.brm.codegen import genmodel, eval_model
from pyro.contrib.brm.design import dummydata, Factor, makedata, make_metadata_lookup, designmatrices_metadata
from pyro.contrib.brm.priors import prior, Prior, PriorEdit, get_response_prior, build_prior_tree
from pyro.contrib.brm.family import getfamily, FAMILIES
from pyro.contrib.brm.model import build_model, parameters
from pyro.contrib.brm.fit import pyro_get_param


from tests.common import assert_equal

default_params = dict(
    Normal          = dict(loc=0., scale=1.),
    Cauchy          = dict(loc=0., scale=1.),
    HalfCauchy      = dict(scale=3.),
    LKJCorrCholesky = dict(eta=1.),
)

# TODO: Extend this. Could check that the response is observed?
@pytest.mark.parametrize('formula_str, metadata, family, prior_edits, expected', [
    # TODO: This (and similar examples below) can't be expressed with
    # the current parser. Is it useful to fix this (`y ~ -1`?), or can
    # these be dropped?
    #(Formula('y', [], []), [], [], ['sigma']),

    ('y ~ 1 + x', [], getfamily('Normal'), [],
     [('b_0', Cauchy, {}),
      ('sigma', HalfCauchy, {})]),

    ('y ~ 1 + x1 + x2', [], getfamily('Normal'), [],
     [('b_0', Cauchy, {}),
      ('sigma', HalfCauchy, {})]),

    #(Formula('y', [], [Group([], 'z', True)]), [Factor('z', list('ab'))], [], ['sigma', 'z_1']),
    # Groups with fewer than two terms don't sample the (Cholesky
    # decomp. of the) correlation matrix.
    #(Formula('y', [], [Group([], 'z', True)]), [Factor('z', list('ab'))], [], ['sigma', 'z_1']),
    ('y ~ 1 | z', [Factor('z', list('ab'))], getfamily('Normal'), [],
     [('sigma', HalfCauchy, {}),
      ('z_0', Normal, {}),
      ('sd_0_0', HalfCauchy, {})]),

    ('y ~ x | z', [Factor('z', list('ab'))], getfamily('Normal'), [],
     [('sigma', HalfCauchy, {}),
      ('z_0', Normal, {}),
      ('sd_0_0', HalfCauchy, {})]),

    ('y ~ 1 + x1 + x2 + (1 + x3 | z)', [Factor('z', list('ab'))], getfamily('Normal'), [],
     [('b_0', Cauchy, {}),
      ('sigma', HalfCauchy, {}),
      ('z_0', Normal, {}),
      ('sd_0_0', HalfCauchy, {}),
      ('L_0', LKJCorrCholesky, {})]),

    ('y ~ 1 + x1 + x2 + (1 + x3 || z)', [Factor('z', list('ab'))], getfamily('Normal'), [],
     [('b_0', Cauchy, {}),
      ('sigma', HalfCauchy, {}),
      ('z_0', Normal, {}),
      ('sd_0_0', HalfCauchy, {})]),

    ('y ~ 1 + x1 + x2 + (1 + x3 + x4 | z1) + (1 + x5 | z2)',
     [Factor('z1', list('ab')), Factor('z2', list('ab'))],
     getfamily('Normal'),
     [],
     [('b_0', Cauchy, {}),
      ('sigma', HalfCauchy, {}),
      ('z_0', Normal, {}),
      ('sd_0_0', HalfCauchy, {}),
      ('L_0', LKJCorrCholesky, {}),
      ('z_1', Normal, {}),
      ('sd_1_0', HalfCauchy, {}),
      ('L_1', LKJCorrCholesky, {})]),

    # Custom priors.
    ('y ~ 1 + x1 + x2',
     [],
     getfamily('Normal'),
     [PriorEdit(('b',), prior('Normal', [0., 100.]))],
     [('b_0', Normal, {'loc': 0., 'scale': 100.}),
      ('sigma', HalfCauchy, {})]),

    ('y ~ 1 + x1 + x2',
     [],
     getfamily('Normal'),
     [PriorEdit(('b', 'intercept'), prior('Normal', [0., 100.]))],
     [('b_0', Normal, {'loc': 0., 'scale': 100.}),
      ('b_1', Cauchy, {}),
      ('sigma', HalfCauchy, {})]),

    ('y ~ 1 + x1 + x2',
     [],
     getfamily('Normal'),
     [PriorEdit(('b', 'x1'), prior('Normal', [0., 100.]))],
     [('b_0', Cauchy, {}),
      ('b_1', Normal, {'loc': 0., 'scale': 100.}),
      ('b_2', Cauchy, {}),
      ('sigma', HalfCauchy, {})]),

    # Prior on coef of a factor.
    ('y ~ 1 + x',
     [Factor('x', list('ab'))],
     getfamily('Normal'),
     [PriorEdit(('b', 'x[b]'), prior('Normal', [0., 100.]))],
     [('b_0', Cauchy, {}),
      ('b_1', Normal, {'loc': 0., 'scale': 100.}),
      ('sigma', HalfCauchy, {})]),

    # Prior on group level `sd` choice.
    ('y ~ 1 + x2 + x3 | x1',
     [Factor('x1', list('ab'))],
     getfamily('Normal'),
     [PriorEdit(('sd', 'x1', 'intercept'), prior('HalfCauchy', [4.]))],
     [('sigma', HalfCauchy, {}),
      ('sd_0_0', HalfCauchy, {'scale': 4.}),
      ('sd_0_1', HalfCauchy, {}),
      ('z_0', Normal, {}),
      ('L_0', LKJCorrCholesky, {})]),

    ('y ~ 1 + x2 + x3 || x1',
     [Factor('x1', list('ab'))],
     getfamily('Normal'),
     [PriorEdit(('sd', 'x1', 'intercept'), prior('HalfCauchy', [4.]))],
     [('sigma', HalfCauchy, {}),
      ('sd_0_0', HalfCauchy, {'scale': 4.}),
      ('sd_0_1', HalfCauchy, {}),
      ('z_0', Normal, {})]),

    # Prior on L.
    ('y ~ 1 + x2 | x1',
     [Factor('x1', list('ab'))],
     getfamily('Normal'),
     [PriorEdit(('cor',), prior('LKJ', [2.]))],
     [('sigma', HalfCauchy, {}),
      ('sd_0_0', HalfCauchy, {}),
      ('z_0', Normal, {}),
      ('L_0', LKJCorrCholesky, {'eta': 2.})]),

    # Prior on parameter of response distribution.
    ('y ~ x',
     [],
     getfamily('Normal'),
     [PriorEdit(('resp', 'sigma'), prior('HalfCauchy', [4.]))],
     [('b_0', Cauchy, {}),
      ('sigma', HalfCauchy, {'scale': 4.})]),

    # Custom response family.
    ('y ~ x',
     [Factor('y', list('AB'))],
     getfamily('Bernoulli'),
     [],
     [('b_0', Cauchy, {})]),

])
def test_codegen(formula_str, metadata, family, prior_edits, expected):
    formula = parse(formula_str)
    metadata = make_metadata_lookup(metadata)
    design_metadata = designmatrices_metadata(formula, metadata)
    prior_tree = build_prior_tree(formula, design_metadata, family, prior_edits)
    model_desc = build_model(formula, prior_tree, family, metadata)
    code = genmodel(model_desc)
    #print(code)
    model = eval_model(code)
    data = dummydata(formula, metadata, 5)
    trace = poutine.trace(model).get_trace(**data)
    expected_sites = [site for (site, _, _) in expected]
    assert set(trace.stochastic_nodes) - {'obs'} == set(expected_sites)
    for (site, family, maybe_params) in expected:
        fn = unwrapfn(trace.nodes[site]['fn'])
        params = maybe_params or default_params[fn.__class__.__name__]
        assert type(fn) == family
        for (name, expected_val) in params.items():
            val = fn.__getattribute__(name)
            assert_equal(val, torch.tensor(expected_val).expand(val.shape))
    for parameter in parameters(model_desc):
        shape = pyro_get_param(trace, parameter.name).shape
        expected_shape = parameter.shape
        assert shape == expected_shape

def unwrapfn(fn):
    return unwrapfn(fn.base_dist) if type(fn) == Independent else fn


@pytest.mark.parametrize('formula_str, metadata, family', [
    ('y ~ x', [], getfamily('Bernoulli')),
    ('y ~ x', [Factor('y', list('abc'))], getfamily('Bernoulli')),
    ('y ~ x', [Factor('y', list('ab'))], getfamily('Normal')),
])
def test_family_and_response_type_checks(formula_str, metadata, family):
    formula = parse(formula_str)
    metadata = make_metadata_lookup(metadata)
    design_metadata = designmatrices_metadata(formula, metadata)
    prior_tree = build_prior_tree(formula, design_metadata, family, [])
    with pytest.raises(Exception, match='not compatible'):
        model = build_model(formula, prior_tree, family, metadata)


@pytest.mark.parametrize('formula_str, metadata, family, prior_edits', [
    ('y ~ x',
     [],
     getfamily('Normal'),
     [PriorEdit(('resp', 'sigma'), prior('Normal', [0., 1.]))]),
    ('y ~ x1 | x2',
     [Factor('x2', list('ab'))],
     getfamily('Normal'),
     [PriorEdit(('sd', 'x2'), prior('Normal', [0., 1.]))]),
    ('y ~ 1 + x1 | x2',
     [Factor('x2', list('ab'))],
     getfamily('Normal'),
     [PriorEdit(('cor', 'x2'), prior('Normal', [0., 1.]))]),
    ('y ~ x',
     [],
     getfamily('Normal'),
     [PriorEdit(('b',), prior('Bernoulli', [.5]))]),
])
def test_prior_checks(formula_str, metadata, family, prior_edits):
    formula = parse(formula_str)
    metadata = make_metadata_lookup(metadata)
    design_metadata = designmatrices_metadata(formula, metadata)
    with pytest.raises(Exception, match=r'(?i)invalid prior'):
        build_prior_tree(formula, design_metadata, family, prior_edits)


# We have to ask for float32 tensors here because the default tensor
# type is changed to float64 in conftest.py.

@pytest.mark.parametrize('formula_str, df, expected', [
    # (Formula('y', [], []),
    #  pd.DataFrame(dict(y=[1, 2, 3])),
    #  dict(X=torch.tensor([[],
    #                       [],
    #                       []]),
    #       y_obs=torch.tensor([1., 2., 3.]))),
    ('y ~ 1',
     pd.DataFrame(dict(y=[1, 2, 3])),
     dict(X=torch.tensor([[1.],
                          [1.],
                          [1.]], dtype=torch.float32),
          y_obs=torch.tensor([1., 2., 3.], dtype=torch.float32))),
    ('y ~ x',
     pd.DataFrame(dict(y=[1, 2, 3],
                       x=[4, 5, 6])),
     dict(X=torch.tensor([[4.],
                          [5.],
                          [6.]], dtype=torch.float32),
          y_obs=torch.tensor([1., 2., 3.], dtype=torch.float32))),
    ('y ~ 1 + x',
     pd.DataFrame(dict(y=[1, 2, 3],
                       x=[4, 5, 6])),
     dict(X=torch.tensor([[1., 4.],
                          [1., 5.],
                          [1., 6.]], dtype=torch.float32),
          y_obs=torch.tensor([1., 2., 3.], dtype=torch.float32))),
    ('y ~ x + 1',
     pd.DataFrame(dict(y=[1, 2, 3],
                       x=[4, 5, 6])),
     dict(X=torch.tensor([[1., 4.],
                          [1., 5.],
                          [1., 6.]], dtype=torch.float32),
          y_obs=torch.tensor([1., 2., 3.], dtype=torch.float32))),

    ('y ~ x',
     pd.DataFrame(dict(y=[1, 2, 3],
                       x=pd.Categorical(list('AAB')))),
     dict(X=torch.tensor([[1., 0.],
                          [1., 0.],
                          [0., 1.]], dtype=torch.float32),
          y_obs=torch.tensor([1., 2., 3.], dtype=torch.float32))),
    ('y ~ 1 + x',
     pd.DataFrame(dict(y=[1, 2, 3],
                       x=pd.Categorical(list('AAB')))),
     dict(X=torch.tensor([[1., 0.],
                          [1., 0.],
                          [1., 1.]], dtype=torch.float32),
          y_obs=torch.tensor([1., 2., 3.], dtype=torch.float32))),
    ('y ~ x1 + x2',
     pd.DataFrame(dict(y=[1, 2, 3],
                       x1=pd.Categorical(list('AAB')),
                       x2=pd.Categorical(list('ABC')))),
     dict(X=torch.tensor([[1., 0., 0., 0.],
                          [1., 0., 1., 0.],
                          [0., 1., 0., 1.]], dtype=torch.float32),
          y_obs=torch.tensor([1., 2., 3.], dtype=torch.float32))),

    ('y ~ 1 + x',
     pd.DataFrame(dict(y=[1, 2, 3],
                       x=pd.Categorical(list('ABC')))),
     dict(X=torch.tensor([[1., 0., 0.],
                          [1., 1., 0.],
                          [1., 0., 1.]], dtype=torch.float32),
          y_obs=torch.tensor([1., 2., 3.], dtype=torch.float32))),

    # (Formula('y', [], [Group([], 'x', True)]),
    #  pd.DataFrame(dict(y=[1, 2, 3],
    #                    x=pd.Categorical(list('ABC')))),
    #  dict(X=torch.tensor([[],
    #                       [],
    #                       []]),
    #       y_obs=torch.tensor([1., 2., 3.]),
    #       J_1=torch.tensor([0, 1, 2]),
    #       Z_1=torch.tensor([[],
    #                         [],
    #                         []]))),
    ('y ~ 1 + (1 + x1 | x2)',
     pd.DataFrame(dict(y=[1, 2, 3],
                       x1=pd.Categorical(list('AAB')),
                       x2=pd.Categorical(list('ABC')))),
     dict(X=torch.tensor([[1.],
                          [1.],
                          [1.]], dtype=torch.float32),
          y_obs=torch.tensor([1., 2., 3.], dtype=torch.float32),
          J_0=torch.tensor([0, 1, 2]),
          Z_0=torch.tensor([[1., 0.],
                            [1., 0.],
                            [1., 1.]], dtype=torch.float32))),

    ('y ~ x',
     pd.DataFrame(dict(y=pd.Categorical(list('AAB')),
                       x=[1, 2, 3])),
     dict(X=torch.tensor([[1.],
                          [2.],
                          [3.]], dtype=torch.float32),
          y_obs=torch.tensor([0, 0, 1], dtype=torch.float32))),

])
def test_designmatrix(formula_str, df, expected):
    data = makedata(parse(formula_str), df)
    assert set(data.keys()) == set(expected.keys())
    for k in expected.keys():
        assert data[k].dtype == expected[k].dtype
        assert_equal(data[k], expected[k])

def test_response_priors_is_complete():
    for family in FAMILIES:
        if family.response is not None:
            for param in family.params:
                if not param.name == family.response.param:
                    assert type(get_response_prior(family.name, param.name)) == Prior

@pytest.mark.parametrize('formula_str, expected_formula', [
    ('y ~ 1', Formula('y', OrderedSet(_1), [])),
    ('y ~ 1 + x', Formula('y', OrderedSet(_1, Term(OrderedSet('x'))), [])),
    ('y ~ x + x', Formula('y', OrderedSet(Term(OrderedSet('x'))), [])),
    ('y ~ x1 : x2', Formula('y', OrderedSet(Term(OrderedSet('x1', 'x2'))), [])),
    ('y ~ (x1 + x2) : x3',
     Formula('y',
             OrderedSet(Term(OrderedSet('x1', 'x3')),
                        Term(OrderedSet('x2', 'x3'))),
             [])),
])
def test_parser(formula_str, expected_formula):
    formula = parse(formula_str)
    assert formula == expected_formula
