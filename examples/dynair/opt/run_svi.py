import os
import time
from functools import wraps, partial
import time
from datetime import timedelta

import torch
from pyro.infer import SVI, Trace_ELBO
import pyro.optim as optim

from opt.utils import append_line

def run_svi(mod, batches, num_epochs, optim_args, hook,
            output_path, save_period, progress_period,
            record_grad_norm, elbo_scale=1.0):
    t0 = time.time()
    num_batches = len(batches)

    grad_norm_dict = dict(value=0)
    if record_grad_norm:
        add_grad_hooks(mod, grad_norm_dict)

    throttled_report_progress = throttle(progress_period)(report_progress)

    svi = SVI(mod.model, mod.guide,
              optim.Adam(optim_args),
              loss=Trace_ELBO())

    for i in range(num_epochs):
        for j, batch in enumerate(batches):
            step = num_batches*i+j
            loss = svi.step(batch)
            grad_norm = grad_norm_dict['value']
            grad_norm_dict['value'] = 0
            elbo = -loss * elbo_scale
            throttled_report_progress(i, j, step, elbo, grad_norm, t0, output_path)
            if not hook is None:
                hook(i, j, step)

        if save_period > 0 and (i+1) % save_period == 0:
            torch.save(mod.state_dict(),
                       os.path.join(output_path, 'params-{}.pytorch'.format(i+1)))

# Based on:
# https://github.com/uber/pyro/blob/5b67518dc1ded8aac59b6dfc51d0892223e8faad/tutorial/source/gmm.ipynb
def add_grad_hooks(module, grad_norm_dict):
    def hook(name, counter, value, grad):
        count = counter['count']
        grad_norm_dict['value'] += (grad.detach() ** 2).sum().item()
        counter['count'] += 1
    for name, value in module.named_parameters():
        value.register_hook(partial(hook, name, dict(count=0), value))

class throttle(object):
    def __init__(self, period):
        self.period = period
        self.time_of_last_call = 0

    def __call__(self, f):
        @wraps(f)
        def throttled(*args, **kwargs):
            now = time.time()

            if now - self.time_of_last_call > self.period:
                self.time_of_last_call = now
                return f(*args, **kwargs)

        return throttled

def report_progress(i, j, step, elbo, grad_norm, t0, output_path):
    elapsed = timedelta(seconds=time.time() - t0)
    print('\33[2K\repoch={}, batch={}, elbo={:.2f}, elapsed={}'.format(i, j, elbo, elapsed), end='')
    append_line(os.path.join(output_path, 'elbo.csv'),
                '{:.2f},{:e},{:.1f},{}'.format(elbo, grad_norm, elapsed.total_seconds(), step))
