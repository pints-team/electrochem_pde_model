from model import SingleReactionSolution
import numpy as np
import pints
from data import ECTimeData
import pickle


def inference(model, values, times):

    # Create an object with links to the model and time series
    problem = pints.SingleOutputProblem(model, times, values)

    # Create a log-likelihood function (adds an extra parameter!)
    log_likelihood = pints.GaussianLogLikelihood(problem)



    # Create a uniform prior over both the parameters and the new noise variable
    lower_bounds = model.non_dim([1e-3, 0.0, 0.4, 0.1,   1e-6, 8.0,   1e-4])
    upper_bounds = model.non_dim([10.0, 0.4, 0.6, 100.0, 100e-6, 10.0, 0.2])
    log_prior = pints.UniformLogPrior(lower_bounds, upper_bounds)

    # Create a posterior log-likelihood (log(likelihood * prior))
    log_posterior = pints.LogPosterior(log_likelihood, log_prior)

    # Choose starting points for 3 mcmc chains
    # params =                   ['k0', 'E0', 'a', 'Ru', 'Cdl', 'freq', 'sigma']
    start_parameters = model.non_dim([0.0101, 0.214, 0.53, 8.0, 20.0e-6, 9.0152, 0.01])

    sigma0 = [0.5 * (h - l) for l, h in zip(lower_bounds, upper_bounds)]
    boundaries = pints.RectangularBoundaries(lower_bounds, upper_bounds)
    #found_parameters, found_value = pints.optimise(
    #            log_posterior,
    #            start_parameters,
    #            sigma0,
    #            boundaries,
    #            method=pints.CMAES
    #        )
    found_parameters = np.array([2.53904548e+00, 8.36714729e+00, 5.31667706e-01,
                                 1.32350288e-04, 3.35631953e-03, 1.62144064e+01, 4.15333278e-02])
    xs = [
        found_parameters * 1.001,
        found_parameters * 1.002,
        found_parameters * 1.003,
    ]
    for x in xs:
        x[5] = found_parameters[5]

    print('start_parameters', start_parameters)
    print('found_parameters', found_parameters)
    print('lower_bounds', lower_bounds)
    print('upper_bounds', upper_bounds)

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
                 chains, 'HaarioACMC'), open('results.pickle', 'wb'))


if __name__ == '__main__':
    model = SingleReactionSolution()
    data = ECTimeData('GC01_FeIII-1mM_1M-KCl_02_009Hz.txt', model,
                      ignore_begin_samples=5, ignore_end_samples=0, samples_per_period=200)
    inference(model, data.current, data.times)
