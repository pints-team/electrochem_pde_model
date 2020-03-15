import numpy as np
from math import pi, floor, ceil


def read_cvsin_type_1(filename):
    """ read in a datafile of format svsin_type_1

    Args:
        filename (str): filename of the data file

    returns:
        time (numpy vector): vector of time samples
        current (numpy vector): vector of current samples
    """

    exp_data = np.loadtxt(filename, skiprows=19)
    if filename[-11:] == '_cv_current':
        t_index = 0
        c_index = 1
    else:
        t_index = 2
        c_index = 1
    return exp_data[:, t_index], exp_data[:, c_index]


class ECTimeData:

    def __init__(self, filename, model, ignore_begin_samples,
                 ignore_end_samples=0, samples_per_period=200, datafile_type='scsin_type_1'):
        print('ECTimeData: loading data from filename = ', filename, ' ...')
        self.times, self.current = read_cvsin_type_1(filename)

        print('\tUsing samples from ', ignore_begin_samples,
              ' to ', len(self.times) - ignore_end_samples)
        if (ignore_end_samples > 0):
            self.times = self.times[ignore_begin_samples:-ignore_end_samples]
            self.current = self.current[
                ignore_begin_samples:-ignore_end_samples]
        else:
            self.times = self.times[ignore_begin_samples:]
            self.current = self.current[ignore_begin_samples:]

        print('\tCut data to multiple of period')
        dt = self.times[100] - self.times[99]
        data_samples_per_period = 1.0 / (model._omega_d * dt)
        discard_samples = int(len(self.current) % data_samples_per_period)
        if (discard_samples > 0):
            self.times = self.times[0:-discard_samples]
            self.current = self.current[0:-discard_samples]

        downsample = int(floor(data_samples_per_period /
                               samples_per_period))
        if downsample == 0:
            downsample = 1
        print('\tBefore downsampling, have ', len(self.times), ' data points')
        print('\tDatafile has ', data_samples_per_period, ' samples per period.')
        print(
            '\tReducing number of samples by keeping every', downsample, 'point')

        self.times = self.times[::downsample]
        self.current= self.current[::downsample]

        self.current = self.current / model._I_0
        self.times = self.times / model._T_0
        # self.distance_scale = np.linalg.norm(self.current)

        print('\tAfter downsampling, have ', len(self.times), ' data points')
