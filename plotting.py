import pickle
import matplotlib.pylab as plt
import pints
import pints.plot


def plotting(old_model=False):
    if old_model:
        results_file = 'results2.pickle'
        names = ['k0', 'E0', 'Cdl', 'Ru', 'alpha', 'omega', 'sigma']
    else:
        results_file = 'results.pickle'
        names = ['k0', 'E0', 'a', 'Ru', 'Cdl', 'omega', 'sigma']

    (xs, log_posterior, log_prior, chains, mcmc_method) = pickle.load(
        open(results_file, 'rb'))

    print(
        'Found results using mcmc={} containing {} chains with {} samples'.format(
            mcmc_method, len(chains), chains[0].shape[0]
        )
    )

    print(chains.shape)
    pints.plot.trace(chains)
    plt.savefig('electrochem_pde_trace.pdf')

    pints.plot.pairwise(chains[0, :, :])
    plt.savefig('electrochem_pde_chains.pdf')

    results = pints.MCMCSummary(chains, parameter_names=names)
    print(results)


if __name__ == '__main__':
    plotting(old_model=False)
