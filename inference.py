from model import SingleReactionSolution
import numpy as np
import pints
from data import ECTimeData
import pickle

base_model_lower_bounds = [1e-3, 0.0, 0.4, 0.1, 1e-6, 8.0]
base_model_upper_bounds = [10.0, 0.4, 0.6, 100.0, 100e-6, 10.0]
sigma_lower_bound = 1e-4
sigma_upper_bound = 10
independent_lower_bounds = np.array(base_model_lower_bounds + [sigma_lower_bound])
independent_upper_bounds = np.array(base_model_upper_bounds + [sigma_upper_bound])
ar1_lower_bounds = np.array(base_model_lower_bounds + [-1.0, sigma_lower_bound])
ar1_upper_bounds = np.array(base_model_upper_bounds + [1.0, sigma_upper_bound])
student_t_lower_bounds = np.array(base_model_lower_bounds + [0.0, sigma_lower_bound])
student_t_upper_bounds = np.array(base_model_upper_bounds + [1000.0, sigma_upper_bound])
arma11_lower_bounds = np.array(base_model_lower_bounds + [-1.0, -1.0, sigma_lower_bound])
arma11_upper_bounds = np.array(base_model_upper_bounds + [1.0, 1.0, sigma_upper_bound])


def create_optimisation_filename(error_model_type):
    file_extra = error_model_type
    filename = 'optimisation_results_' + file_extra + ".pickle"
    return filename

def create_mcmc_filename(error_model_type):
    file_extra = error_model_type
    filename = 'mcmc_results_' + file_extra + ".pickle"
    return filename
    

def optimise(model, values, times, error_model_type):
    # Create an object with links to the model and time series
    problem = pints.SingleOutputProblem(model, times, values)
    base_model_start_parameters = [0.012, 0.214, 0.53, 0.44, 1.8e-5, 9.0152] #  close to what is obtained by CMAES in independent / autocorrelated / student-t models
    sigma_start = 0.04

    if error_model_type=="independent":
        print("Fitting independent errors model...")
        log_likelihood = pints.GaussianLogLikelihood(problem)
        lower_bounds = independent_lower_bounds
        upper_bounds = independent_upper_bounds
        log_prior = pints.UniformLogPrior(lower_bounds, upper_bounds)
        # params =                   ['k0', 'E0', 'a', 'Ru', 'Cdl', 'freq', 'sigma']
        start_parameters = np.array(base_model_start_parameters + [sigma_start])
    elif error_model_type=="ar1":
        print("Fitting AR1 model...")
        log_likelihood = pints.AR1LogLikelihood(problem)
        lower_bounds = ar1_lower_bounds
        upper_bounds = ar1_upper_bounds
        log_prior = pints.UniformLogPrior(lower_bounds, upper_bounds)
        # params =                   ['k0', 'E0', 'a', 'Ru', 'Cdl', 'freq', 'rho', 'sigma']
        start_parameters = np.array(base_model_start_parameters + [0.01, sigma_start])
    elif error_model_type=="student_t":
        print("Fitting Student-t model...")
        log_likelihood = pints.StudentTLogLikelihood(problem)
        lower_bounds = student_t_lower_bounds
        upper_bounds = student_t_upper_bounds
        log_prior = pints.UniformLogPrior(lower_bounds, upper_bounds)
        # params =                   ['k0', 'E0', 'a', 'Ru', 'Cdl', 'freq', 'nu', 'sigma']
        start_parameters = np.array(base_model_start_parameters + [400, sigma_start])
    elif error_model_type=="arma11":
        log_likelihood = pints.ARMA11LogLikelihood(problem)
        lower_bounds = arma11_lower_bounds
        upper_bounds = arma11_upper_bounds
        log_prior = pints.UniformLogPrior(lower_bounds, upper_bounds)
        # params =                   ['k0', 'E0', 'a', 'Ru', 'Cdl', 'freq', 'rho', 'phi, 'sigma']
        start_parameters = np.array(base_model_start_parameters + [0.01, 0.01, sigma_start])
    
    # Create a posterior log-likelihood (log(likelihood * prior))
    log_posterior = pints.LogPosterior(log_likelihood, log_prior)

    transform = pints.ComposedTransformation(
        pints.LogTransformation(1),
        pints.RectangularBoundariesTransformation(
            lower_bounds[1:], upper_bounds[1:]
        ),
    )
    sigma0 = [0.1 * (h - l) for l, h in zip(lower_bounds, upper_bounds)]
    boundaries = pints.RectangularBoundaries(lower_bounds, upper_bounds)
    found_parameters, found_value = pints.optimise(
                log_posterior,
                start_parameters,
                sigma0,
                boundaries,
                transformation=transform,
                method=pints.NelderMead
            )
    
    filename = create_optimisation_filename(error_model_type)
    pickle.dump((found_parameters, found_value, 'HaarioBardenetACMC'), open(filename, 'wb'))


def inference(model, values, times, error_model_type):

    problem = pints.SingleOutputProblem(model, times, values)

    if error_model_type=="independent":
        print("Fitting independent errors model...")
        log_likelihood = pints.GaussianLogLikelihood(problem)
        lower_bounds = independent_lower_bounds
        upper_bounds = independent_upper_bounds
        log_prior = pints.UniformLogPrior(lower_bounds, upper_bounds)
    elif error_model_type=="ar1":
        print("Fitting AR1 model...")
        log_likelihood = pints.AR1LogLikelihood(problem)
        lower_bounds = ar1_lower_bounds
        upper_bounds = ar1_upper_bounds
        log_prior = pints.UniformLogPrior(lower_bounds, upper_bounds)
    elif error_model_type=="student_t":
        print("Fitting Student-t model...")
        log_likelihood = pints.StudentTLogLikelihood(problem)
        lower_bounds = student_t_lower_bounds
        upper_bounds = student_t_upper_bounds
        log_prior = pints.UniformLogPrior(lower_bounds, upper_bounds)
    elif error_model_type=="arma11":
        log_likelihood = pints.ARMA11LogLikelihood(problem)
        lower_bounds = arma11_lower_bounds
        upper_bounds = arma11_upper_bounds
        log_prior = pints.UniformLogPrior(lower_bounds, upper_bounds)

    # Create a posterior log-likelihood (log(likelihood * prior))
    log_posterior = pints.LogPosterior(log_likelihood, log_prior)

    # Choose starting points for 4 mcmc chains
    filename = create_optimisation_filename(error_model_type)
    found_parameters = pickle.load(open(filename, "rb"))[0]

    transform = pints.ComposedTransformation(
        pints.LogTransformation(1),
        pints.RectangularBoundariesTransformation(
            lower_bounds[1:], upper_bounds[1:]
        ),
    )
    
    xs = [
        found_parameters,
        found_parameters,
        found_parameters,
        found_parameters
    ]
    
    for x in xs:
        print("log prob = ", log_posterior(x))
    
    print('found_parameters', found_parameters)
    print('lower_bounds', lower_bounds)
    print('upper_bounds', upper_bounds)

    # Create mcmc routine with four chains
    mcmc = pints.MCMCController(log_posterior, 4, xs, method=pints.HaarioBardenetACMC,
                                transformation=transform)

    # Add stopping criterion
    mcmc.set_max_iterations(1000)

    # Run!
    chains = mcmc.run()

    # Save chains for plotting and analysis
    filename = create_mcmc_filename(error_model_type)
    pickle.dump((xs, pints.GaussianLogLikelihood, log_prior,
                 chains, 'HaarioBardenetACMC'), open(filename, 'wb'))


if __name__ == '__main__':
    model = SingleReactionSolution()
    data = ECTimeData('GC01_FeIII-1mM_1M-KCl_02_009Hz.txt', model,
                      ignore_begin_samples=5, ignore_end_samples=0, samples_per_period=200)
    
    error_models = ['arma11', 'student_t']
    for em in error_models:
        optimise(model, data.current, data.times, em)
        inference(model, data.current, data.times, em)
