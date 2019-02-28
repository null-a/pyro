import json
from os import path
from matplotlib import pyplot as plt

def load(fn):
    with open(fn) as f:
        return json.load(f)

def test_perf_vs_time(run):
    return list(zip(*[(run['train'][epoch][1], ll) for (epoch, ll) in run['test']]))

files = [('elbo_1.json', 'elbo (no baselines)'),
         ('elbo_avg_1.json', 'elbo (avg. baseline)'),
         ('elbo_net_1.json', 'elbo (data dep. baseline)'),
         ('rws_1.json', 'reweighted wake sleep (5 samples)')]

named_runs = [(name, load(path.join('.', fn))) for (fn, name) in files]

# print(run['train'][0]) # [loss, elapsed]
# print(run['test'][0]) # [epoch, test marginal log likelihood est.]

for (name, run) in named_runs:
    xs, ys = test_perf_vs_time(run)
    plt.plot(xs, ys, label=name)

plt.xlabel('inference wall time (secs)')
plt.ylabel('estimated test log marginal')
plt.title('SBN 50 (net hidden dim = 400)')
plt.legend()
plt.grid()

#plt.show()
plt.savefig('plot.pdf')
