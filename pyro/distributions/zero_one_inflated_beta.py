from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all, lazy_property

from pyro.distributions import TorchDistribution


# Beta Regression for Modelling Rates and Proportion
# https://www.ime.usp.br/~sferrari/beta.pdf

# https://cran.rstudio.com/web/packages/brms/vignettes/brms_families.html#zero-inflated-and-hurdle-models

class ZeroOneInflatedBeta(TorchDistribution):
    """
    A Zero One Inflated Beta distribution.

    :param torch.Tensor mean: mean of the beta distribution.
    :param torch.Tensor prec: precision of the beta distribution.
    :param torch.Tensor alpha: probability of a boundary outcome, i.e. of 0 or 1.
    :param torch.Tensor gamma: conditional probability of seeing a 1, given a boundary outcome.
    """
    arg_constraints = {'loc': constraints.unit_interval,
                       'prec': constraints.positive,
                       'alpha': constraints.unit_interval,
                       'gamma': constraints.unit_interval}
    support = constraints.unit_interval

    def __init__(self, loc, prec, alpha, gamma, validate_args=None):
        self.loc, self.prec, self.alpha, self.gamma = broadcast_all(loc, prec, alpha, gamma)
        batch_shape = self.loc.shape
        event_shape = torch.Size()
        super(ZeroOneInflatedBeta, self).__init__(batch_shape, event_shape, validate_args)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        loc, prec, alpha, gamma, value = broadcast_all(self.loc, self.prec, self.alpha, self.gamma, value)

        shape0 = loc * prec
        shape1 = (1 - loc) * prec

        # Avoid values close to boundary, otherwise differentiating
        # log prob will return nan. Strangely, it's necessary to do
        # this for values exactly on the boundary, even though their
        # log probs are replaced by the log prob from the discrete
        # part of the process below.
        eps = 1e-6
        cvalue = value.clamp(eps, 1-eps)

        beta_log_prob = prec.lgamma() - shape0.lgamma() - shape1.lgamma() + (shape0-1)*cvalue.log() + (shape1-1)*(1-cvalue).log()
        log_prob = (1-alpha).log() + beta_log_prob
        zeros = value == 0.
        log_prob[zeros] = alpha[zeros].log() + (1-gamma[zeros]).log()
        ones = value == 1.
        log_prob[ones] = alpha[ones].log() + gamma[ones].log()

        return log_prob

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)

        shape0 = self.loc * self.prec
        shape1 = (1 - self.loc) * self.prec

        dir_alpha = torch.stack([shape0.expand(shape), shape1.expand(shape)], -1)

        with torch.no_grad():
            mask = torch.bernoulli(self.alpha.expand(shape)) # 1 means use zero/one, 0 means use the beta
            bits = torch.bernoulli(self.gamma.expand(shape))
            # Beta samples everywhere.
            samples = torch._sample_dirichlet(dir_alpha).select(-1, 0)
            # Zero out those entries that ought to use a boundary sample.
            samples *= 1 - mask
            # Add boundary samples.
            samples += mask * bits

        return samples

    # TODO: How should this be implemented? (a la Delta, or ZeroInflatedPoisson?)
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ZeroOneInflatedBeta, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.prec = self.prec.expand(batch_shape)
        new.alpha = self.alpha.expand(batch_shape)
        new.gamma = self.gamma.expand(batch_shape)
        super(ZeroOneInflatedBeta, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

        # try:
        #     return super(ZeroOneInflatedBeta, self).expand(batch_shape)
        # except NotImplementedError:
        #     validate_args = self.__dict__.get('_validate_args')
        #     loc = self.loc.expand(batch_shape)
        #     prec = self.prec.expand(batch_shape)
        #     alpha = self.alpha.expand(batch_shape)
        #     gamma = self.gamma.expand(batch_shape)
        #     return type(self)(loc, prec, alpha, gamma, validate_args=validate_args)
