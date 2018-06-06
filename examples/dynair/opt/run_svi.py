import os
import time
from functools import wraps, partial
import time
from datetime import timedelta

import torch
from torch.nn.utils import clip_grad_norm_
import pyro
from pyro.infer import SVI, Trace_ELBO
import pyro.optim as optim

from opt.utils import append_line

def run_svi(mod, batches, num_epochs, optim_args, hook,
            output_path, save_period, progress_period,
            check_grad, record_grad_norm, clip_threshold, elbo_scale=1.0):
    t0 = time.time()
    num_batches = len(batches)

    throttled_report_progress = throttle(progress_period)(report_progress)

    grad_norm_dict = dict()

    svi = SVI(mod.model, mod.guide,
              optim.Adam(optim_args),
              loss=Trace_ELBO(),
              param_hook=partial(param_hook, check_grad, grad_norm_dict, record_grad_norm, clip_threshold, output_path))

    for i in range(num_epochs):
        for j, batch in enumerate(batches):
            # set every step to ensure that zero is recorded if hook
            # signals no step.
            grad_norm_dict['value'] = 0.0
            step = num_batches*i+j
            loss = svi.step(batch)
            elbo = -loss * elbo_scale
            throttled_report_progress(i, j, step, elbo, grad_norm_dict['value'], t0, output_path)
            if not hook is None:
                hook(i, j, step)

        if save_period > 0 and (i+1) % save_period == 0:
            torch.save(mod.state_dict(),
                       os.path.join(output_path, 'params-{}.pytorch'.format(i+1)))

def hasnan(t):
    return torch.isnan(t).any()

def hasinf(t):
    return (t == float('inf')).any()

def param_hook(check_grad, grad_norm_dict, record_grad_norm, clip_threshold, output_path, params):
    if check_grad:
        infected = [p for p in params if hasnan(p.grad) or hasinf(p.grad)]
        if len(infected) > 0:
            param_name = pyro.get_param_store().param_name
            def append_line_to_params(line):
                append_line(os.path.join(output_path, 'params.txt'), line)
            for p in infected:
                append_line_to_params('nan={}, inf={}, {}'.format(hasnan(p.grad), hasinf(p.grad), param_name(p)))
            append_line_to_params('')
            return True # signal that a step should not be taken

    if record_grad_norm or clip_threshold < float('inf'):
        grad_norm = clip_grad_norm_(params, clip_threshold)
        if record_grad_norm:
            grad_norm_dict['value'] = grad_norm

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
    append_line(os.path.join(output_path, 'log.csv'),
                '{:.2f},{:e},{:.4f},{}'.format(elbo, grad_norm, elapsed.total_seconds(), step))
