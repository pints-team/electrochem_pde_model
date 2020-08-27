
from model import SingleReactionSolution
import numpy as np
import pints
from data import ECTimeData
import pickle
import matplotlib.pyplot as plt
import electrochemistry



def inference2(model_raw, model_old, model, values, times):

    # Create an object with links to the model and time series
    problem = pints.SingleOutputProblem(model_old, times, values)

    # Create a log-likelihood function (adds an extra parameter!)
    log_likelihood = pints.GaussianLogLikelihood(problem)

    # Create a uniform prior over both the parameters and the new noise variable

    e0_buffer = 0.1 * (model_raw.params['Ereverse'] - model_raw.params['Estart'])
    lower_bounds = np.array([
        0.0,
        model_raw.params['Estart'] + e0_buffer,
        0.0,
        0.0,
        0.4,
        0.9* model_raw.params['omega'],
        1e-4,
    ])
    upper_bounds = np.array([
        100 * model_raw.params['k0'],
        model_raw.params['Ereverse'] - e0_buffer,
        10 * model_raw.params['Cdl'],
        10 * model_raw.params['Ru'],
        0.6,
        1.1* model_raw.params['omega'],
        0.2,
    ])
    log_prior = pints.UniformLogPrior(lower_bounds, upper_bounds)

    # Create a posterior log-likelihood (log(likelihood * prior))
    log_posterior = pints.LogPosterior(log_likelihood, log_prior)


    # Choose starting points for 3 mcmc chains
    param_names = ['k0', 'E0', 'Cdl', 'Ru', 'alpha', 'omega', 'sigma']
    start_parameters = np.array([
        model_raw.params['k0'],
        model_raw.params['E0'],
        model_raw.params['Cdl'],
        model_raw.params['Ru'],
        model_raw.params['alpha'],
        model_raw.params['omega'],
        0.01
    ])

    sigma0 = [0.5 * (h - l) for l, h in zip(lower_bounds, upper_bounds)]
    boundaries = pints.RectangularBoundaries(lower_bounds, upper_bounds)
    #found_parameters, found_value = pints.optimise(
    #            log_posterior,
    #            start_parameters,
    #            sigma0,
    #            boundaries,
    #            method=pints.CMAES
    #        )
    found_parameters = start_parameters
    print('start_parameters', start_parameters)
    print('found_parameters', found_parameters)
    xs = [
        found_parameters * 1.001,
        found_parameters * 0.999,
        found_parameters * 0.998,
    ]
    for x in xs:
        x[5] = found_parameters [5]

    # adjust Ru to something reasonable
    xs[0][3] = 1.001*5e-5
    xs[1][3] = 1.00*5e-5
    xs[2][3] = 0.999*5e-5

    transform = pints.ComposedElementWiseTransformation(
        pints.LogTransformation(1),
        pints.RectangularBoundariesTransformation(
            lower_bounds[1:], upper_bounds[1:]
        ),
    )

    # Create mcmc routine with four chains
    mcmc = pints.MCMCController(log_posterior, 3, xs, method=pints.HaarioBardenetACMC,
                                transform=transform)

    # Add stopping criterion
    mcmc.set_max_iterations(10000)

    # Run!
    chains = mcmc.run()

    # Save chains for plotting and analysis
    pickle.dump((xs, pints.GaussianLogLikelihood, log_prior,
                 chains, 'HaarioACMC'), open('results2.pickle', 'wb'))


if __name__ == '__main__':
    model = SingleReactionSolution()
    DEFAULT = {
        'reversed': True,
        'Estart': 0.5,
        'Ereverse': -0.1,
        'omega': 9.0152,
        'phase': 0,
        'dE': 0.08,
        'v': -0.08941,
        't_0': 0.001,
        'T': 297.0,
        'a': 0.07,
        'c_inf': 1 * 1e-3 * 1e-3,
        'D': 7.2e-6,
        'Ru': 8.0,
        'Cdl': 20.0 * 1e-6,
        'E0': 0.214,
        'k0': 0.0101,
        'alpha': 0.53,
    }

    names = ['k0', 'E0', 'Cdl', 'Ru', 'alpha', 'omega']
    model_old = electrochemistry.ECModel(DEFAULT)
    pints_model = electrochemistry.PintsModelAdaptor(model_old, names)

    data_old = electrochemistry.ECTimeData(
            'GC02_FeIII-1mM_1M-KCl_02a_009Hz.txt', model_old, ignore_begin_samples=5,
        ignore_end_samples=0, samples_per_period=200)

    inference2(model_old, pints_model, model, data_old.current, data_old.times)
