def print_marginals(mcmc_run):
    # Looking at only the first trace is OK here because the models
    # described by lme4 have static structure.
    trace = mcmc_run.exec_traces[0]

    # Is there a more Pyronic (not poking in trace internals) way to
    # achieve this? Or is it safer the have `genmodel` return a list
    # of sample sites used by the models it generates?
    sample_sites = [k for k in trace.nodes.keys()
                    if trace.nodes[k]['type'] == 'sample' and not k in ['obs', 'y']]
    marginal = mcmc_run.marginal(sample_sites)
    for name in sample_sites:
        print('==================================================')
        print(name)
        print('-- mean ------------------------------------------')
        print(marginal.empirical[name].mean)
        print('-- stddev ----------------------------------------')
        print(marginal.empirical[name].stddev)
