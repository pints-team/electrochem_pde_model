import electrochemistry
import numpy as np
import matplotlib.pyplot as plt

filenames = ['GC02_FeIII-1mM_1M-KCl_02a_009Hz.txt']

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

model = electrochemistry.ECModel(DEFAULT)
data0 = electrochemistry.ECTimeData(
    filenames[0], model, ignore_begin_samples=5, ignore_end_samples=0, samples_per_period=5000)
max_current = np.max(data0.current)
sim_current = model.simulate(data0.times)
plt.plot(data0.times, data0.current, label='exp')
plt.plot(data0.times, sim_current, label='sim')
plt.legend()
plt.show()
