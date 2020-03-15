import pickle
import matplotlib.pylab as plt
import pints
import pints.plot


def plotting():
    (xs, log_posterior, chains, mcmc_method) = pickle.load(open('results.pickle', 'rb'))

    print(
        'Found results using mcmc={} containing {} chains with {} samples'.format(
            mcmc_method, len(chains), chains[0].shape[0]
        )
    )

    pints.plot.trace(chains)
    plt.savefig('electrochem_pde_trace.pdf')

    pints.plot.pairwise(chains)
    plt.savefig('electrochem_pde_chains.pdf')

    results = pints.MCMCSummary(chains, parameter_names=['k0', 'E0', 'a', 'Ru', 'Cdl'])
    print(results)


if __name__ == '__main__':
    plotting()
