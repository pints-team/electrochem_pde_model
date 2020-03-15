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
    lower_bounds = [0.0, 0.0, 0.2, 0.0,   1e-6,   0.0]
    upper_bounds = [1.0, 0.4, 0.7, 100.0, 100e-6, 1.0]
    log_prior = pints.UniformLogPrior(lower_bounds, upper_bounds)

    # Create a posterior log-likelihood (log(likelihood * prior))
    log_posterior = pints.LogPosterior(log_likelihood, log_prior)

    # Choose starting points for 3 mcmc chains
    start_parameters = np.array([0.0101, 0.214, 0.53, 8.0, 20.0e-6, 0.01])
    xs = [
        start_parameters * 1.1,
        start_parameters * 0.9,
        start_parameters * 1.15,
    ]

    # Create mcmc routine with four chains
    mcmc = pints.MCMCController(log_posterior, 3, xs, method=pints.HaarioACMC)

    # Add stopping criterion
    mcmc.set_max_iterations(10000)

    # Run!
    chains = mcmc.run()

    # Save chains for plotting and analysis
    pickle.dump((xs, log_posterior, chains, 'HaarioACMC'), open('results.pickle', 'wb'))


if __name__ == '__main__':
    model = SingleReactionSolution()
    data = ECTimeData('GC02_FeIII-1mM_1M-KCl_02a_009Hz.txt', model,
            ignore_begin_samples=5, ignore_end_samples=0, samples_per_period=200)
    inference(model, data.current, data.times)


