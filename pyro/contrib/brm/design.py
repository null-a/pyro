from collections import namedtuple
import itertools

import torch
import numpy as np
import pandas as pd
# http://pandas.pydata.org/pandas-docs/stable/reference/general_utility_functions.html#dtype-introspection
from pandas.api.types import is_numeric_dtype, is_categorical_dtype

from pyro.contrib.brm.utils import join
from pyro.contrib.brm.formula import Formula, OrderedSet, Term, _1


# TODO: Refer to dataframe metadata as 'schema' in order to avoid
# confusion with the similarly named design matrix metadata?
def make_metadata_lookup(metadata):
    assert type(metadata) == list
    assert all(type(factor) == Factor for factor in metadata)
    # Turn a list of factors into a dictionary keyed by column name.
    return dict((factor.name, factor) for factor in metadata)


# TODO: Levels of pandas categorical columns can be any hashable type
# I think. Is our implementation flexible enough to handle the same?
# If not, we ought to check the type here and throw an error when it's
# something we can't handle. Two areas to consider are whether it's
# possible to specify priors on coefs arising from levels of a factor
# (I think this works if the user knows how the types values are
# turned into strings), and whether posterior summaries look OK. (Both
# of which have to do with whether instances of the type can be
# converted to strings in a sensible way.)

Factor = namedtuple('Factor',
                    ['name',    # column name
                     'levels']) # list of level names

# Extract metadata (as expected by `genmodel`) from a pandas
# dataframe.
def dfmetadata(df):
    return [Factor(c, list(df[c].dtype.categories))
            for c in df
            if is_categorical_dtype(df[c])]

# TODO: Use the result of `coding` to generate more realistic dummy
# data. i.e. Rather than just having X & Z matrices of the correct
# size, intercepts can be all ones, and categorical columns can be
# appropriately coded (width `codefactor`) random data.
def dummydata(formula, metadata, N):
    import torch
    assert type(metadata) == dict
    data = {}
    M = width(formula.pterms, metadata)
    data['X'] = torch.rand(N, M)
    for i, group in enumerate(formula.groups):
        M_i = width(group.gterms, metadata)
        num_levels = len(metadata[group.column].levels)
        data['Z_{}'.format(i)] = torch.rand(N, M_i)
        # Maps (indices of) data points to (indices of) levels.
        data['J_{}'.format(i)] = torch.randint(0, num_levels, size=[N])
    data['y_obs'] = torch.rand(N)
    return data

# --------------------
# Design matrix coding
# --------------------

def codenumeric(dfcol):
    assert is_numeric_dtype(dfcol)
    return [dfcol]

# Codes a categorical column/factor. When reduced==False the column is
# dummy/one-of-K coded.

# x = [A, B, C, A]

# x0 x1 x2
#  1  0  0
#  0  1  0
#  0  0  1
#  1  0  0

# When reduced==True the same coding is used, but the first column is
# dropped.

# x1 x2
#  0  0
#  1  0
#  0  1
#  0  0

def codefactor(dfcol, reduced):
    assert is_categorical_dtype(dfcol)
    factors = dfcol.cat.categories
    num_levels = len(factors)
    start = 1 if reduced else 0
    return [dfcol == factors[i] for i in range(start, num_levels)]

def col2torch(col):
    if type(col) == torch.Tensor:
        assert col.dtype == torch.float32
        return col
    else:
        # TODO: It's possible to do torch.tensor(col) here. What does
        # that do? Is it preferable to this?
        return torch.from_numpy(col.to_numpy(np.float32))

def term_order(term):
    assert type(term) == Term
    return len(term.factors)

InterceptC = namedtuple('InterceptC', [])
CategoricalC = namedtuple('CategoricalC', ['factor', 'reduced'])
NumericC = namedtuple('NumericC', ['name'])

# TODO: I do similar dispatching on type in `designmatrix` and
# `designmatrix_metadata`. It would be more Pythonic to turn
# InterceptC etc. into classes with `width` and `code` methods.

def widthC(c):
    if type(c) in [InterceptC, NumericC]:
        return 1
    elif type(c) == CategoricalC:
        return len(c.factor.levels) - (1 if c.reduced else 0)
    else:
        raise Exception('Unknown coding type.')

# Generates a description of how the given terms ought to be coded
# into a design matrix.
def coding(terms, metadata):
    assert type(terms) == OrderedSet
    assert type(metadata) == dict
    def code(i, term):
        assert type(term) == Term
        # We only know how to code the intercept and singleton terms
        # at present.
        assert len(term.factors) in [0, 1]
        if term == _1:
            assert len(term.factors) == 0
            return InterceptC()
        elif term.factors[0] in metadata:
            factor = metadata[term.factors[0]]
            return CategoricalC(factor, reduced=i>0)
        else:
            return NumericC(term.factors[0])
    sorted_terms = sorted(terms, key=term_order)
    return [code(i, term) for i, term in enumerate(sorted_terms)]


CodedFactor = namedtuple('CodedFactor', 'factor reduced')
CodedFactor.__repr__ = lambda self: '{}{}'.format(self.factor, '-' if self.reduced else '')

# Taken from the itertools documentation.
def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

# A Term represents the interaction between zero or more factors. This
# function describes how the coding for such a term can be performed
# by coding multiple interactions, each using a reduced rank coding.
# For example:
#
# Term(<'a'>) can be coded as [Intercept, a (reduced)].
# Term(<'a','b'>) can be coded as [Intercept, a (reduced), b (reduced), a:b (reduced)].

def decompose(term):
    assert type(term) == Term
    return [tuple(CodedFactor(factor, True) for factor in subset) for subset in powerset(term.factors)]


# Attempt to absorb t2 into t1. If this is possible the result of
# doing so is returned. Otherwise None is returned. This rule explains
# when absorbtion is possible.

#  t2    t1                  result
# { X , X U {x-} , ...} == { X U {x+} , ...}

def absorb(t1, t2):
    assert type(t1) == tuple
    assert all(type(p) == CodedFactor for p in t1)
    assert type(t2) == tuple
    assert all(type(p) == CodedFactor for p in t2)
    s1 = set(t1)
    s2 = set(t2)
    if s2.issubset(s1) and len(s1) - len(s2) == 1:
        diff = s1.difference(s2)
        assert len(diff) == 1
        extra_factor = list(diff)[0]
        if extra_factor.reduced:
            factor = CodedFactor(extra_factor.factor, False)
            # TODO: Is this acceptable or should we be maintaining the
            # ordering from t1? ('y ~ 1 + a + a:b' is an example where
            # that produces different results based on this choice.)
            return t2 + (factor,)

def simplify_one(termcoding):
    assert type(termcoding) == list
    assert all(type(t) == tuple and all(type(p) == CodedFactor for p in t) for t in termcoding)
    for i, j in itertools.permutations(range(len(termcoding)), 2):
        newterm = absorb(termcoding[i], termcoding[j])
        if newterm:
            out = termcoding[:]
            out[i] = newterm # Replace with absorbing interaction.
            del out[j]       # Remove absorbed interaction.
            return out

def simplify(termcoding):
    assert type(termcoding) == list
    assert all(type(t) == tuple and all(type(p) == CodedFactor for p in t) for t in termcoding)
    while True:
        maybe_termcoding = simplify_one(termcoding)
        if maybe_termcoding is None:
            return termcoding # We're done.
        termcoding = maybe_termcoding

# all_previous([['a'], ['b','c'], ['d']])
# ==           [{},    {'a'},     {'a','b','c'}]
def all_previous(xss):
    if len(xss) == 0:
        return []
    else:
        return [set()] + [set(xss[0]).union(xs) for xs in all_previous(xss[1:])]


# This is an attempt to implement the algorithm described here:
# https://patsy.readthedocs.io/en/latest/formulas.html#technical-details
def coding2(terms, metadata):
    assert type(terms) == OrderedSet
    assert type(metadata) == dict
    decomposed = [decompose(t) for t in terms]
    non_redundant = [[t for t in term if not t in previous]
                     for term, previous in zip(decomposed, all_previous(decomposed))]
    return join(simplify(t) for t in non_redundant)


def width(terms, metadata):
    assert type(metadata) == dict
    return sum(widthC(c) for c in coding(terms, metadata))

# Build a simple design matrix (as a torch tensor) from columns of a
# pandas data frame.

# TODO: There ought to be a check somewhere to ensure that all terms
# are either numeric or categorical. We're making this assumption in
# `coding`, where anything not mentioned in dfmetadata(df) is assumed
# to be numeric. But we can't check it there because we don't have a
# concreate dataframe at that point. Here we have a list of terms and
# the df, so this seems like a good place. An alternative is to do
# this in makedata, but we'd need to do so for population and group
# levels.

def designmatrix(terms, df):
    assert type(terms) == OrderedSet
    N = len(df)
    def dispatch(code):
        if type(code) == InterceptC:
            return [torch.ones(N, dtype=torch.float32)]
        elif type(code) == CategoricalC:
            return codefactor(df[code.factor.name], code.reduced)
        elif type(code) == NumericC:
            return codenumeric(df[code.name])
        else:
            raise Exception('Unknown coding type.')
    metadata = make_metadata_lookup(dfmetadata(df))
    coding_desc = coding(terms, metadata)
    coded_cols = join([dispatch(c) for c in coding_desc])
    X = torch.stack([col2torch(col) for col in coded_cols], dim=1) if coded_cols else torch.empty(N, 0)
    assert X.shape == (N, width(terms, metadata))
    #print(designmatrix_metadata(terms, df))
    return X

# --------------------------------------------------
# Experimenting with design matrix metadata:
#
# `designmatrix_metadata` computes a list of readable design matrix
# column names. The idea is that (following brms) this information has
# a number of uses:
#
# - Improve readability of e.g. `Fit` summary. e.g. Instead of just
#   showing the `b` vector, we can use this information to identify
#   each coefficient.
#
# - Users need to be able to specify their own priors for individual
#   coefficients. I think this information is used as the basis of
#   that, allowing priors to be specified by coefficient name?

def numeric_metadata(code):
    return [code.name]

def categorical_metadata(code):
    start = 1 if code.reduced else 0
    return ['{}[{}]'.format(code.factor.name, cat)
            for cat in code.factor.levels[start:]]

def designmatrix_metadata(terms, metadata):
    assert type(terms) == OrderedSet
    def dispatch(code):
        if type(code) == InterceptC:
            return ['intercept']
        elif type(code) == CategoricalC:
            return categorical_metadata(code)
        elif type(code) == NumericC:
            return numeric_metadata(code)
        else:
            raise Exception('Unknown coding type.')
    coding_desc = coding(terms, metadata)
    return join([dispatch(c) for c in coding_desc])

DesignMeta = namedtuple('DesignMeta', 'population groups')
PopulationMeta = namedtuple('PopulationMeta', 'coefs')
GroupMeta = namedtuple('GroupMeta', 'name coefs')

def designmatrices_metadata(formula, metadata):
    p = PopulationMeta(designmatrix_metadata(formula.pterms, metadata))
    gs = [GroupMeta(group.column, designmatrix_metadata(group.gterms, metadata))
          for group in formula.groups]
    return DesignMeta(p, gs)

# --------------------------------------------------

def lookupvector(column, df):
    assert type(column) == str
    assert type(df) == pd.DataFrame
    assert column in df
    assert is_categorical_dtype(df[column])
    return torch.from_numpy(df[column].cat.codes.to_numpy(np.int64))

def responsevector(column, df):
    assert type(column) == str
    assert type(df) == pd.DataFrame
    assert column in df
    dfcol = df[column]
    if is_numeric_dtype(dfcol):
        coded = codenumeric(dfcol)
    elif is_categorical_dtype(dfcol) and len(dfcol.cat.categories) == 2:
        # TODO: How does a user know how this was coded? For design
        # matrices this is revealed by the column names in the design
        # metadata, but we don't have the here.
        coded = codefactor(dfcol, reduced=True)
    else:
        raise Exception('Don\'t know how to code a response of this type.')
    assert len(coded) == 1
    return col2torch(coded[0])

def makedata(formula, df):
    assert type(formula) == Formula
    assert type(df) == pd.DataFrame
    data = {}
    data['X'] = designmatrix(formula.pterms, df)
    data['y_obs'] = responsevector(formula.response, df)
    for i, group in enumerate(formula.groups):
        data['Z_{}'.format(i)] = designmatrix(group.gterms, df)
        data['J_{}'.format(i)] = lookupvector(group.column, df)
    return data
