import os
import time
from functools import wraps
import time
from datetime import timedelta

from pyro.infer import SVI, Trace_ELBO
import pyro.optim as optim

from opt.utils import append_line

def run_svi(mod, batches, num_epochs, hook, output_path, save_period, elbo_scale=1.0):
    t0 = time.time()
    num_batches = len(batches)

    def per_param_optim_args(module_name, param_name):
        return {'lr': 1e-4}

    svi = SVI(mod.model, mod.guide,
              optim.Adam(per_param_optim_args),
              loss=Trace_ELBO())

    for i in range(num_epochs):
        for j, batch in enumerate(batches):
            step = num_batches*i+j
            loss = svi.step(batch)
            elbo = -loss * elbo_scale
            report_progress(i, j, step, elbo, t0, output_path)
            if not hook is None:
                hook(i, j, step)

        if save_period > 0 and (i+1) % save_period == 0:
            torch.save(mod.state_dict(),
                       os.path.join(output_path, 'params-{}.pytorch'.format(i+1)))

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

@throttle(1)
def report_progress(i, j, step, elbo, t0, output_path):
    elapsed = timedelta(seconds=time.time() - t0)
    print('\33[2K\repoch={}, batch={}, elbo={:.2f}, elapsed={}'.format(i, j, elbo, elapsed), end='')
    append_line(os.path.join(output_path, 'elbo.csv'),
                '{:.2f},{:.1f},{}'.format(elbo, elapsed.total_seconds(), step))
