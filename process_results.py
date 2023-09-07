from model import SingleReactionSolution
import numpy as np
import pints
from data import ECTimeData
import pickle


def choose_filename(error_model_type, ending):
    filename = "mcmc_results_" + error_model_type + "." + ending
    return filename

def fetch_model_results(error_model_type):
    
    filename = choose_filename(error_model_type, "pickle")
    results = pickle.load(open(filename, "rb"))
    chains = results[3]
    
    return chains

def export_mcmc(error_model_type):
    
    chains = fetch_model_results(error_model_type)
    chains_stacked = np.vstack(chains)
    filename = "mcmc_draws_" + error_model_type + ".csv"
    np.savetxt(filename, chains_stacked, delimiter=",")

def model_posterior_simulation(error_model_type, n_remove):
    
    chains = fetch_model_results(error_model_type)
    chains_stacked = np.vstack(chains)
    posterior_means = np.median(chains_stacked, axis=0)
    posterior_means = posterior_means[:(len(posterior_means) - n_remove)]
    
    model = SingleReactionSolution()
    data = ECTimeData('GC01_FeIII-1mM_1M-KCl_02_009Hz.txt', model,
                      ignore_begin_samples=5, ignore_end_samples=0, samples_per_period=200)
    model_simulation = model.simulate(posterior_means, data.times)
    filename = choose_filename(error_model_type, "csv")
    np.savetxt(filename, model_simulation, delimiter=",")
    
def export_data():
    model = SingleReactionSolution()
    data = ECTimeData('GC01_FeIII-1mM_1M-KCl_02_009Hz.txt', model,
                      ignore_begin_samples=5, ignore_end_samples=0, samples_per_period=200)
    x = np.zeros((len(data.times), 2))
    x[:, 0] = data.times
    x[:, 1] = data.current
    np.savetxt("data.csv", x, delimiter=",")
    
if __name__ == '__main__':
    
    error_models = ['independent', 'ar1', 'arma11', 'student_t']
    n_noise_parameters = [1, 2, 3, 2]
    k = 0
    for em in error_models:
        print(k)
        model_posterior_simulation(em, n_noise_parameters[k])
        export_mcmc(em)
        k += 1
    export_data()