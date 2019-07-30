from collections import namedtuple

from pyro.contrib.brm.utils import unzip
from .formula import Formula
from .design import Metadata, RealValued, Categorical, Integral
from .family import Family, Type, nonlocparams, support_depends_on_args, args, family_repr
from .priors import select, tryselect, Node

def family_matches_response(formula, metadata, family):
    assert type(formula) == Formula
    assert type(metadata) == Metadata
    assert type(family) == Family
    # I don't think there is any way for this not to hold with the
    # present system. However, it /could/ arise if it were possible to
    # put a prior over e.g. the `num_trials` parameter of Binomial,
    # for example. Because this holds we know we can safely
    # call`family.support` with zero args below.
    assert not support_depends_on_args(family)
    factor = metadata.column(formula.response)
    if type(family.support()) == Type['Real']:
        return type(factor) == RealValued
    elif type(family.support()) == Type['Boolean']:
        if type(factor) == Categorical:
            return len(factor.levels) == 2
        elif type(factor) == Integral:
            return factor.min == 0 and factor.max == 1
        else:
            return False
    elif (type(family.support()) == Type['IntegerRange']):
        factor = metadata.column(formula.response)
        return (type(factor) == Integral and
                (family.support().lb is None or factor.min >= family.support().lb) and
                (family.support().ub is None or factor.max <= family.support().ub))
    else:
        return False

def check_family_matches_response(formula, metadata, family):
    assert type(metadata) == Metadata
    if not family_matches_response(formula, metadata, family):
        # TODO: This could be more informative. e.g. If choosing
        # Bernoulli fails, is the problem that the response is
        # numeric, or that it has more than two levels?
        error = 'The response distribution "{}" is not compatible with the type of the response column "{}".'
        raise Exception(error.format(family_repr(family), formula.response))


# Abstract model description.
ModelDesc = namedtuple('ModelDesc', 'population groups response')
Population = namedtuple('Population', 'coefs priors')
Group = namedtuple('Group', 'factor_names levels coefs sd_priors corr_prior')
Response = namedtuple('Response', 'family nonlocparams priors')

def build_model(formula, prior_tree, family, metadata):
    assert type(formula) == Formula
    assert type(prior_tree) == Node
    assert type(family) == Family
    assert type(metadata) == Metadata

    # TODO: `formula` is only used in order to perform the following
    # check. Internally, the information about the response column
    # name is used to perform the check. So, does it make sense for
    # `build_model` to take only the response column as argument?
    # Alternatively, perhaps it makes sense for this information could
    # be incorporated in design meta, or otherwise included in one of
    # the args. already received.
    check_family_matches_response(formula, metadata, family)

    # Population-level
    node = select(prior_tree, ('b',))
    b_coefs, b_priors = unzip([(n.name, n.prior_edit.prior) for n in node.children])
    population = Population(b_coefs, b_priors)
    # Assert invariant.
    assert len(population.coefs) == len(population.priors)

    # Groups
    groups = []


    for node in select(prior_tree, ('sd',)).children:

        # TODO: It's unpleasant to have to do this deserializaton.
        # (Splitting on ":".) The problem is that we only get the
        # prior tree here, and the prior tree needs to serialize the
        # factor names in order to allow priors to be specified. Once
        # design meta data is renamed to `ModelDescPre` or similar, it
        # will it seem natural to pass that in here (along with the
        # prior tree), at which point it will be more convenient to
        # grab the group's factor names from there.
        cols = node.name.split(':')
        assert all(type(metadata.column(col)) == Categorical for col in cols), 'grouping columns must be a factor'

        sd_coefs, sd_priors = unzip([(n.name, n.prior_edit.prior) for n in node.children])

        corr_node = tryselect(prior_tree, ('cor', node.name))
        corr_prior = None if corr_node is None else corr_node.prior_edit.prior

        group = Group(cols, metadata.levels(cols), sd_coefs, sd_priors, corr_prior)
        # Assert invariants.
        assert len(group.coefs) == len(group.sd_priors)
        assert group.corr_prior is None or type(group.corr_prior) == Family
        groups.append(group)

    nl_params = nonlocparams(family)
    nl_priors = [n.prior_edit.prior for n in select(prior_tree, ('resp',)).children]
    response = Response(family, nl_params, nl_priors)
    # Assert invariants.
    assert len(response.nonlocparams) == len(response.priors)

    return ModelDesc(population, groups, response)


def model_repr(model):
    assert type(model) == ModelDesc
    out = []
    def write(s):
        out.append(s)
    write('=' * 40)
    write('Population')
    write('-' * 40)
    write('Coef Priors:')
    for (coef, prior) in zip(model.population.coefs, model.population.priors):
        write('{:<15} | {}'.format(coef, family_repr(prior)))
    for i, group in enumerate(model.groups):
        write('=' * 40)
        write('Group {}'.format(i))
        write('-' * 40)
        write('Factors: {}\nNum Levels: {}'.format(', '.join(group.factor_names), len(group.levels)))
        write('Corr. Prior: {}'.format(None if group.corr_prior is None else family_repr(group.corr_prior)))
        write('S.D. Priors:')
        for (coef, sd_prior) in zip(group.coefs, group.sd_priors):
            write('{:<15} | {}'.format(coef, family_repr(sd_prior)))
    write('=' * 40)
    write('Response')
    write('-' * 40)
    write('Family: {}'.format(family_repr(model.response.family)))
    write('Link:')
    write('  Parameter: {}'.format(model.response.family.link.param))
    write('  Function:  {}'.format(model.response.family.link.fn.name))
    write('Priors:')
    for (param, prior) in zip(model.response.nonlocparams, model.response.priors):
        write('{:<15} | {}'.format(param.name, family_repr(prior)))
    write('=' * 40)
    return '\n'.join(out)


Parameter = namedtuple('Parameter', ['name', 'shape'])

def parameter_names(model):
    return [parameter.name for parameter in parameters(model)]

# This describes the set of parameters implied by a particular model.
# Any backend is expected to produce models the make this set of
# parameters available (with the described shapes) via its `get_param`
# function. (See fit.py.)
def parameters(model):
    assert type(model) == ModelDesc
    return ([Parameter('b', (len(model.population.coefs),))] +
            [Parameter('r_{}'.format(i), (len(group.levels), len(group.coefs)))
             for i, group in enumerate(model.groups)] +
            [Parameter('sd_{}'.format(i), (len(group.coefs),))
             for i, group in enumerate(model.groups)] +
            [Parameter('L_{}'.format(i), (len(group.coefs), len(group.coefs)))
             for i, group in enumerate(model.groups) if not group.corr_prior is None] +
            [Parameter(param.name, (1,)) for param in model.response.nonlocparams])

# [ (scalar_param_name, (param_name, (ix0, ix1, ...)), ... ]

# (ix0, ix1, ...) is an index into that the parameter (picked out by
# `param_name`) once put through `to_numpy`. This is a tuple because
# parameters are not necessarily vectors.

def scalar_parameter_map(model):
    assert type(model) == ModelDesc
    out = [('b_{}'.format(coef), ('b', (i,)))
           for i, coef in enumerate(model.population.coefs)]
    for ix, group in enumerate(model.groups):
        out.extend([('sd_{}__{}'.format(':'.join(group.factor_names), coef), ('sd_{}'.format(ix), (i,)))
                    for i, coef in enumerate(group.coefs)])
        out.extend([('r_{}[{},{}]'.format(':'.join(group.factor_names), '_'.join(level), coef), ('r_{}'.format(ix), (i, j)))
                    for i, level in enumerate(group.levels)
                    for j, coef in enumerate(group.coefs)])
    for param in model.response.nonlocparams:
        out.append((param.name, (param.name, (0,))))
    return out

def scalar_parameter_names(model):
    return [name for (name, _) in scalar_parameter_map(model)]
