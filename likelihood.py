from model import SingleReactionSolution
import numpy as np
import pints
from data import ECTimeData
import pickle
import matplotlib.pyplot as plt
import electrochemistry


def plot_likelihood(model, values, times):

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
    param_names = ['k0', 'E0', 'a', 'Ru', 'Cdl', 'freq', 'sigma']
    start_parameters = model.non_dim([0.0101, 0.214, 0.53, 8.0, 20.0e-6, 9.0152, 0.01])

    scaling = (upper_bounds - lower_bounds)
    minx = start_parameters - scaling / 1000.0
    maxx = start_parameters + scaling / 1000.0

    fig = plt.figure()
    for i, start in enumerate(start_parameters):
        print(param_names[i])
        plt.clf()
        xgrid = np.linspace(minx[i], maxx[i], 100)
        ygrid = np.empty_like(xgrid)
        for j, x in enumerate(xgrid):
            params = np.copy(start_parameters)
            params[i] = x
            ygrid[j] = log_likelihood(params)
        plt.plot(xgrid, ygrid)
        plt.savefig('likelihood_' + param_names[i] + '.pdf')

def plot_likelihood_old(model_raw, model_old, model, values, times):

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
    start_parameters = [
        model_raw.params['k0'],
        model_raw.params['E0'],
        model_raw.params['Cdl'],
        model_raw.params['Ru'],
        model_raw.params['alpha'],
        model_raw.params['omega'],
        0.01
    ]
    start_parameters_new = model.non_dim([0.0101, 0.214, 0.53, 8.0, 20.0e-6, 9.0152, 0.01])
    fig = plt.figure()
    sim_current = model_old.simulate(start_parameters, times)
    sim_current_new = model.simulate(start_parameters_new, times)
    plt.plot(times, values, label='data')
    plt.plot(times, sim_current, label='sim')
    plt.plot(times, -sim_current_new, label='sim_new')
    plt.legend()
    print(np.linalg.norm(-sim_current_new-sim_current)/np.linalg.norm(sim_current_new))
    plt.savefig('new_versus_old_sim.pdf')

    scaling = (upper_bounds - lower_bounds)
    minx = start_parameters - scaling / 1000.0
    maxx = start_parameters + scaling / 1000.0

    for i, start in enumerate(start_parameters):
        print(param_names[i])
        plt.clf()
        xgrid = np.linspace(minx[i], maxx[i], 100)
        ygrid = np.empty_like(xgrid)
        for j, x in enumerate(xgrid):
            params = np.copy(start_parameters)
            params[i] = x
            ygrid[j] = log_likelihood(params)
        plt.plot(xgrid, ygrid)
        plt.savefig('likelihood_old_' + param_names[i] + '.pdf')





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
    data = ECTimeData('GC02_FeIII-1mM_1M-KCl_02a_009Hz.txt', model,
                      ignore_begin_samples=5, ignore_end_samples=0, samples_per_period=200)

    data_old = electrochemistry.ECTimeData(
            'GC02_FeIII-1mM_1M-KCl_02a_009Hz.txt', model_old, ignore_begin_samples=5,
        ignore_end_samples=0, samples_per_period=200)

    plot_likelihood(model, data.current, data.times)
    # plot_likelihood_old(model_old, pints_model, model, data_old.current, data_old.times)
