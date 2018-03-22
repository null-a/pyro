import json
import sys

from matplotlib import pyplot as plt

def load(fn):
    with open(fn) as f:
        return json.load(f)
    for k,v in d.items():
        print(k)

def vis(data, prefix):
    for k, v in data.items():
        print(k)
        if k.startswith(prefix):
            plt.plot(v, label=k)
    plt.xlabel('step')
    plt.ylabel('norm')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    prefix = '' if len(sys.argv) < 3 else sys.argv[2]
    vis(load(sys.argv[1]), prefix)
