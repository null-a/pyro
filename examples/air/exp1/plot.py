import json
from os import path
from matplotlib import pyplot as plt

def load(fn):
    with open(fn) as f:
        return json.load(f)

def test_perf_vs_time(run):
    return list(zip(*[(run['train'][epoch][1], ll) for (epoch, ll, _) in run['test']]))

def count_acc_vs_time(run):
    return list(zip(*[(run['train'][epoch][1], acc) for (epoch, _, acc) in run['test']]))

files = [('elbo_1.json', 'elbo (data dep. baselines, funny prior)'),
         ('rws_1.json', 'reweighted wake sleep (5 samples)')]

named_runs = [(name, load(path.join('.', fn))) for (fn, name) in files]

# run = named_runs[0][1]
# print(run['train'][0]) # [loss, elapsed]
# print(run['test'][0]) # [epoch, test marginal log likelihood est., count accuracy]


for (name, run) in named_runs:
    xs, ys = test_perf_vs_time(run)
    #xs, ys = count_acc_vs_time(run)
    plt.plot(xs, ys, label=name)

plt.xlabel('inference wall time (secs)')
plt.ylabel('estimated test log marginal')
#plt.ylabel('training set count accuracy')
plt.title('AIR')
plt.legend()
plt.grid()

#plt.show()
plt.savefig('plot.pdf')
