import math
import itertools

import torch
import torch.nn as nn
import torch.nn.init as init
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
    batch_dims = t.shape[0:-1]
    width = t.shape[-1]
    powers_of_two = torch.tensor([2**i for i in range(width-1, -1, -1)])
    out = torch.sum(t * powers_of_two, -1)
    assert out.shape == batch_dims
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
    width = t.shape[-1]
    return one_hot(bits2long(t), 2**width)

class QFull(nn.Module):
    def __init__(self, num_coef):
        super(QFull, self).__init__()
        assert type(num_coef) == int
        assert num_coef > 0
        self.num_coef = num_coef
        self.net = nn.Sequential(nn.Linear(1, 100, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(100, 50, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(50, 2**num_coef, bias=True),
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

class QFullM(nn.Module):
    def __init__(self, num_coef, num_designs):
        super(QFullM, self).__init__()
        assert type(num_coef) == int
        assert type(num_designs) == int
        assert num_coef > 0
        assert num_designs > 0
        self.num_coef = num_coef
        self.num_designs = num_designs
        self.net = nn.Sequential(MultiLinear(num_designs, 1, 100),
                                 nn.ReLU(),
                                 MultiLinear(num_designs, 100, 50),
                                 nn.ReLU(),
                                 MultiLinear(num_designs, 50, 2**num_coef),
                                 nn.LogSoftmax(dim=-1))

    def forward(self, inputs):
        assert len(inputs.shape) == 3
        N = inputs.shape[1]
        assert inputs.shape == (self.num_designs, N, 1)
        return self.net(inputs)

    def logprobs(self, inputs, targets):
        assert len(targets.shape) == 3
        assert inputs.shape[1] == targets.shape[1]
        N = inputs.shape[1]
        assert targets.shape == (self.num_designs, N, self.num_coef)
        logprobs = self.forward(inputs)
        assert logprobs.shape == (self.num_designs, N, 2 ** self.num_coef)
        return torch.sum(logprobs * bits2onehot(targets).float(), -1)

    def marginal_probs(self, inputs):
        logprobs = self.forward(inputs)
        cols = torch.stack([bits2long(target_values_for_marginal(i, self.num_coef)) for i in range(self.num_coef)])
        return torch.sum(torch.exp(logprobs[..., cols]), -1)


class MultiLinear(nn.Module):
    def __init__(self, *shape):
        super(MultiLinear, self).__init__()
        self.batch_shape   = shape[0:-2]
        self.in_features  = shape[-2]
        self.out_features = shape[-1]
        self.weight = nn.Parameter(torch.empty(*self.batch_shape, self.in_features, self.out_features))
        self.bias = nn.Parameter(torch.empty(*self.batch_shape, 1, self.out_features))
        self.reset_parameters()
        # print(self.weight)
        # print(self.bias)
        # assert False

    def reset_parameters(self):
        # Apply the init. from nn.Linear to each sub-network.
        for ix in itertools.product(*[range(i) for i in self.batch_shape]):
            init.kaiming_uniform_(self.weight[ix], a=math.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight[ix])
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias[ix], -bound, bound)

    # Given an input of shape `batch_dims + (N, in_features)`, this
    # returns a tensor with shape `batch_dims + (N, out_features)`.
    def forward(self, inp):
        return torch.matmul(inp, self.weight) + self.bias


def main():
    #net = QIndep(3)
    net1 = QFull(3)
    net2 = QFull(3)

    netM = QFullM(3,2)

    # print(net.net[0].weight.shape)
    # print(netM.net[0].weight.shape)

    netM.net[0].weight[0,0] = net1.net[0].weight
    netM.net[2].weight[0,0] = net1.net[2].weight
    netM.net[4].weight[0,0] = net1.net[4].weight

    netM.net[0].weight[1,0] = net2.net[0].weight
    netM.net[2].weight[1,0] = net2.net[2].weight
    netM.net[4].weight[1,0] = net2.net[4].weight



    netM.net[0].bias[0,0] = net1.net[0].bias
    netM.net[2].bias[0,0] = net1.net[2].bias
    netM.net[4].bias[0,0] = net1.net[4].bias

    netM.net[0].bias[1,0] = net2.net[0].bias
    netM.net[2].bias[1,0] = net2.net[2].bias
    netM.net[4].bias[1,0] = net2.net[4].bias





    inputs = torch.tensor(
        [
            [[0.],[1.],[2.],[3.]],
            [[0.],[1.],[2.],[3.]],
        ])
    targets = torch.tensor(
        [
            [[1,0,0],[0,0,1],[1,1,0],[1,1,1]],
            [[1,1,1],[1,0,1],[0,1,0],[1,0,0]],
        ])

    # out = net(inputs)#, targets)
    # outM = netM(inputs)#, targets)

    # print(out)
    # print(outM)


    lp1 = net1.logprobs(inputs[0], targets[0])
    lp2 = net2.logprobs(inputs[1], targets[1])
    lpM = netM.logprobs(inputs, targets)

    print(lp1)
    print(lp2)
    print(lpM)


    print('--------------------')

    mp1 = net1.marginal_probs(inputs[0], 0)
    print(mp1)
    mp2 = net2.marginal_probs(inputs[1], 0)
    print(mp2)
    mpM = netM.marginal_probs(inputs, 0)
    print(mpM)




def main2():
    b = torch.tensor([[1,1,1],[0,1,0]])

    print(bits2onehot(b))


    b2 = torch.tensor([[
        [0,0,0],
        [0,0,1]
    ],[
        [1,1,1],
        [0,1,0]
    ]])
    print(bits2onehot(b2))

if __name__ == '__main__':
    main()
