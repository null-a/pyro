import torch
import torch.nn as nn
from torch.nn.functional import one_hot

class QIndep(nn.Module):
    def __init__(self, num_coef):
        super(QIndep, self).__init__()
        assert type(num_coef) == int
        assert num_coef > 0
        self.num_coef = num_coef
        self.net = nn.Sequential(nn.Linear(1, 100),
                                 nn.ReLU(),
                                 nn.Linear(100,50),
                                 nn.ReLU(),
                                 nn.Linear(50, num_coef),
                                 nn.Sigmoid())

    def forward(self, inputs):
        assert inputs.shape[1] == 1
        # TODO: There's probably a better approach than clamping --
        # parameterize loss by logits?
        eps = 1e-6
        return self.net(inputs).clamp(eps, 1-eps)

    # Compute (vectorised, over multiple y and m) q(m|y;d).
    # m: targets
    # y: inputs
    # (;d because we make a fresh net for each design.)
    def logprobs(self, inputs, targets):
        assert inputs.shape[0] == targets.shape[0]
        N = inputs.shape[0]
        assert inputs.shape == (N, 1)
        assert targets.shape == (N, self.num_coef)
        probs = self.forward(inputs)
        targetsf = targets.float()
        return torch.sum(targetsf*torch.log(probs) + (1-targetsf)*torch.log(1-probs), 1)

    # Compute the marginal probability of a particular coefficient
    # being within [-eps,eps]. For this particular Q (which assumes
    # the joint is the product of the marginals) this only requires us
    # to pick out the appropriate marginal.
    def marginal_probs(self, inputs, coef):
        assert type(coef) == int
        assert 0 <= coef < self.num_coef
        probs = self.forward(inputs)
        return probs[:,coef]




# e.g. tensor([[0,0,1], [1,1,0]]) => tensor([1,6])
def bits2long(t):
    N, width = t.shape
    powers_of_two = torch.tensor([2**i for i in range(width-1, -1, -1)])
    out = torch.sum(t * powers_of_two, 1)
    assert out.shape == (N,)
    return out

# e.g. (3,4) => [0,0,1,1]
def int2bits(i, width):
    assert i < 2**width
    return [int(b) for b in ('{:0'+str(width)+'b}').format(i)]

# All of the target values (as bit vectors) that satisfy \theta_coef == 1
def target_values_for_marginal(coef, num_coef):
    #print(list(int2bits(i, num_coef) for i in range(2**num_coef)))
    values = [bits for bits in (int2bits(i, num_coef) for i in range(2**num_coef)) if bits[coef] == 1]
    #print(values)
    return torch.tensor(values)

# e.g. [[1,0,1],[0,0,1]] => [[0,0,0,0,0,1,0,0],[0,1,0,0,0,0,0,0]]
def bits2onehot(t):
    width = t.shape[1]
    return one_hot(bits2long(t), 2**width)

class QFull(nn.Module):
    def __init__(self, num_coef):
        super(QFull, self).__init__()
        assert type(num_coef) == int
        assert num_coef > 0
        self.num_coef = num_coef
        self.net = nn.Sequential(nn.Linear(1, 100),
                                 nn.ReLU(),
                                 nn.Linear(100, 50),
                                 nn.ReLU(),
                                 nn.Linear(50, 2**num_coef),
                                 nn.LogSoftmax(dim=1))

    def forward(self, inputs):
        assert inputs.shape[1] == 1
        return self.net(inputs)

    def logprobs(self, inputs, targets):
        assert inputs.shape[0] == targets.shape[0]
        N = inputs.shape[0]
        assert inputs.shape == (N, 1)
        assert targets.shape == (N, self.num_coef)
        logprobs = self.forward(inputs)
        assert logprobs.shape[1] == 2 ** self.num_coef
        return torch.sum(logprobs * bits2onehot(targets).float(), 1)

    def marginal_probs(self, inputs, coef):
        assert type(coef) == int
        assert 0 <= coef < self.num_coef
        logprobs = self.forward(inputs)
        cols = bits2long(target_values_for_marginal(coef, self.num_coef))
        return torch.sum(torch.exp(logprobs[:,cols]), 1)
