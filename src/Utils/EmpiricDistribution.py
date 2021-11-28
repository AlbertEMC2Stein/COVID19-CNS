import numpy as np
from numpy.random import uniform


class EmpiricDistribution:
    def __init__(self, data_dict):
        self.keys = np.array(list(data_dict.keys()))
        self.probabilities = _normalize(data_dict.values())
        self.__cdf = np.cumsum(self.probabilities)

    def quantile(self, p):
        if not np.all((0 <= p) & (p <= 1)):
            raise ValueError("Parameter p has to be in [0; 1]")

        result = self.keys.size
        for value in self.__cdf:
            result -= p <= value

        return self.keys[result]

    def pick(self, **kwargs):
        if 'size' in kwargs.keys():
            r = uniform(size=kwargs['size'])
        else:
            r = uniform()

        return self.quantile(r)


def _normalize(dict_values):
    to_array = np.array(list(dict_values))
    return to_array / np.sum(to_array)
