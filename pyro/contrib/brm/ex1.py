import numpy as np
import pandas as pd

from pyro.contrib.brm import makecode
from pyro.contrib.brm.formula import Formula, Group, _1
from pyro.contrib.brm.design import makedata
from pyro.contrib.brm.priors import Prior, PriorEdit

# --------------------------------------------------
# Ex 1. Population-level effects only
# --------------------------------------------------

# Here's a simple formula:
# `y ~ 1 + x`
f1 = Formula('y', [_1, 'x'], [])

# Given a data frame:
df1a = pd.DataFrame(dict(y=[0., 1., 2.],
                         x=[10., 20., 30.]))

# ... we can generate model code and a design matrix:

print(makecode(f1, df1a, []))
# def model(X, y_obs=None):
#     assert type(X) == torch.Tensor
#     N = X.shape[0]
#     M = 2
#     assert X.shape == (N, M)
#     b_0 = pyro.sample("b_0", dist.Cauchy(torch.tensor(0.0).expand([2]), torch.tensor(1.0).expand([2])).to_event(1))
#     b = torch.cat([b_0])
#     assert b.shape == (M,)
#     mu = torch.mv(X, b)
#     sigma = pyro.sample("sigma", dist.HalfCauchy(torch.tensor(3.0).expand([1])).to_event(1))
#     with pyro.plate("obs", N):
#         y = pyro.sample("y", dist.Normal(mu, sigma.expand(N)), obs=y_obs)
#     return dict(b=b, sigma=sigma, y=y)

print(makedata(f1, df1a))
# {'X': tensor([[ 1., 10.],
#               [ 1., 20.],
#               [ 1., 30.]]),
#  'y_obs': tensor([0., 1., 2.])}

# Here we use the same formula (`y ~ 1 + x`), but this time we use a
# data frame in which the `x` column is a factor. This factor is
# appropriately coded in the design matrix, and the vector of
# coefficients (`b`) is extended to the appropriate length in the
# generated model code.

df1b = pd.DataFrame(dict(y=[0., 0., 0.],
                         x=pd.Categorical(['a', 'b', 'c'])))

print(makecode(f1, df1b, []))
# def model(X, y_obs=None):
#     assert type(X) == torch.Tensor
#     N = X.shape[0]
#     M = 3
#     assert X.shape == (N, M)
#     b_0 = pyro.sample("b_0", dist.Cauchy(torch.tensor(0.0).expand([3]), torch.tensor(1.0).expand([3])).to_event(1))
#     b = torch.cat([b_0])
#     assert b.shape == (M,)
#     mu = torch.mv(X, b)
#     sigma = pyro.sample("sigma", dist.HalfCauchy(torch.tensor(3.0).expand([1])).to_event(1))
#     with pyro.plate("obs", N):
#         y = pyro.sample("y", dist.Normal(mu, sigma.expand(N)), obs=y_obs)
#     return dict(b=b, sigma=sigma, y=y)

print(makedata(f1, df1b))
# {'X': tensor([[1., 0., 0.],
#               [1., 1., 0.],
#               [1., 0., 1.]]),
#  'y_obs': tensor([0., 0., 0.])}

# --------------------------------------------------
# Ex 2. Population and group-level effects
# --------------------------------------------------

# Here's a formula with group-level effects:
# `y ~ x1 + (1 + x2 | x3)`
f2 = Formula('y', ['x1'], [Group([_1, 'x2'], 'x3', True)])

# ... and a data frame:
df2 = pd.DataFrame(dict(y=[0., 1., 2.],
                        x1=[1., 2., 3.],
                        x2=[10., 20., 30.],
                        x3=pd.Categorical(['a', 'b', 'c'])))

# And the generated model code and design matrices etc.:

print(makecode(f2, df2, []))
# def model(X, Z_1, J_1, y_obs=None):
#     assert type(X) == torch.Tensor
#     N = X.shape[0]
#     M = 1
#     assert X.shape == (N, M)
#     b_0 = pyro.sample("b_0", dist.Cauchy(torch.tensor(0.0).expand([1]), torch.tensor(1.0).expand([1])).to_event(1))
#     b = torch.cat([b_0])
#     assert b.shape == (M,)
#     mu = torch.mv(X, b)
#
#     # [1] Group(gterms=[Intercept(), 'x2'], column='x3', corr=True)
#     M_1 = 2 # Number of coeffs
#     N_1 = 3 # Number of levels
#     assert type(Z_1) == torch.Tensor
#     assert Z_1.shape == (N, M_1) # N x 2
#     assert type(J_1) == torch.Tensor
#     assert J_1.shape == (N,)
#     sd_1_0 = pyro.sample("sd_1_0", dist.HalfCauchy(torch.tensor(3.0).expand([2])).to_event(1))
#     sd_1 = torch.cat([sd_1_0])
#     assert sd_1.shape == (M_1,) # 2
#     z_1 = pyro.sample("z_1", dist.Normal(torch.zeros([2, 3]), torch.ones([2, 3])).to_event(2))
#     assert z_1.shape == (M_1, N_1) # 2 x 3
#     L_1 = pyro.sample("L_1", dist.LKJCorrCholesky(2, torch.tensor(1.0)))
#     assert L_1.shape == (M_1, M_1) # 2 x 2
#     r_1 = torch.mm(torch.mm(torch.diag(sd_1), L_1), z_1).transpose(0, 1)
#     assert r_1.shape == (N_1, M_1) # 3 x 2
#     r_1_1 = r_1[:, 0]
#     r_1_2 = r_1[:, 1]
#     Z_1_1 = Z_1[:, 0]
#     Z_1_2 = Z_1[:, 1]
#     mu = mu + r_1_1[J_1] * Z_1_1
#     mu = mu + r_1_2[J_1] * Z_1_2
#     sigma = pyro.sample("sigma", dist.HalfCauchy(torch.tensor(3.0).expand([1])).to_event(1))
#     with pyro.plate("obs", N):
#         y = pyro.sample("y", dist.Normal(mu, sigma.expand(N)), obs=y_obs)
#     return dict(b=b, sigma=sigma, y=y)

print(makedata(f2, df2))
# {'X': tensor([[1.],
#               [2.],
#               [3.]]),
#  'y_obs': tensor([0., 1., 2.]),
#  'Z_1': tensor([[ 1., 10.],
#                 [ 1., 20.],
#                 [ 1., 30.]]),
#  'J_1': tensor([0, 1, 2])}

# (Note that modelling of correlations between group-level coefficient
# can be turned of by passing `False` as the third argument to
# `Group`.)

# --------------------------------------------------
# Ex 3. Custom priors
# --------------------------------------------------

# Users can specify custom priors for population-level coefficients
# and group-level standard deviations, at several level of
# granularity. For example it is possible to specify:

# - A prior to be used for all columns of the standard deviation
#   vector in every group.

# - A prior to be used for all columns of the standard deviation
#   vector in a particular group.

# - A prior to be used for a particular coefficient of the standard
#   deviation vector in a particular group.

# A user can combine multiple such specification to describe their
# desired priors. The system is aware of the coding used for factors
# in the data frame, allowing prior to be assigned to the coefficient
# corresponding to a particular level of the factor if necessary.

# Here's a simple example in which we specify custom priors for the
# population-level coefficients.

# `y ~ 1 + x1 + x2`
f3 = Formula('y', [_1, 'x1', 'x2'], [])
df1a = pd.DataFrame(dict(y=[0., 1., 2.],
                         x1=[1., 2., 3.],
                         x2=[10., 20., 30.]))

# Here we describe the priors:
priors3 = [
    # All population-level coefficients should use a `Normal(0, 10)`
    # prior.
    PriorEdit(['b'], Prior('Normal', [0., 10.])),
    # The intercept coefficient should use a `Cauchy(0, 100)` prior.
    # This takes precedence over the less specific request above.
    PriorEdit(['b', 'intercept'], Prior('Cauchy', [0., 100.])),
]

# Here's the code this generates:

print(makecode(f3, df1a, priors3))
# def model(X, y_obs=None):
#     assert type(X) == torch.Tensor
#     N = X.shape[0]
#     M = 3
#     assert X.shape == (N, M)
#     b_0 = pyro.sample("b_0", dist.Cauchy(torch.tensor(0.0).expand([1]), torch.tensor(100.0).expand([1])).to_event(1))
#     b_1 = pyro.sample("b_1", dist.Normal(torch.tensor(0.0).expand([2]), torch.tensor(10.0).expand([2])).to_event(1))
#     b = torch.cat([b_0, b_1])
#     assert b.shape == (M,)
#     mu = torch.mv(X, b)
#     sigma = pyro.sample("sigma", dist.HalfCauchy(torch.tensor(3.0).expand([1])).to_event(1))
#     with pyro.plate("obs", N):
#         y = pyro.sample("y", dist.Normal(mu, sigma.expand(N)), obs=y_obs)
#     return dict(b=b, sigma=sigma, y=y)
