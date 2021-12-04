"""
TODO Docstring Utils
"""


import numpy as np
from numpy import clip
from numpy.random import uniform, randint
from os.path import sep


class EmpiricDistribution:
    def __init__(self, data_dict: dict):
        """
        TODO Docstring EmpiricDistribution __init__
        """

        def normalize(dict_values):
            to_array = np.array(list(dict_values))
            return to_array / np.sum(to_array)

        self.keys = np.array(list(data_dict.keys()))
        self.probabilities = normalize(data_dict.values())
        self.__cdf = np.cumsum(self.probabilities)

    def quantile(self, p: float):
        """
        TODO Docstring EmpiricDistribution quantile
        """

        if not np.all((0 <= p) & (p <= 1)):
            raise ValueError("Parameter p has to be in [0; 1]")

        result = self.keys.size
        for value in self.__cdf:
            result -= p <= value

        return self.keys[result]

    def pick(self, **kwargs):
        """
        TODO Docstring pick
        """

        if 'size' in kwargs.keys():
            r = uniform(size=kwargs['size'])
        else:
            r = uniform()

        return self.quantile(r)


################################################################################################
################################################################################################
################################################################################################


class Samplers:
    """
    TODO Docstring Samplers
    """

    basic_sampler = {
        "household": lambda: randint(21),
        "age": lambda: randint(0, 90),
        "infected": lambda: uniform() < 0.1,
    }
    """
    TODO Docstring Samplers
    """

    @staticmethod
    def generate_population_data_from_samplers(property_samplers: dict, n: int):
        """
        TODO Docstring generate_population_data_from_samplers
        """

        headers = np.array(["id"] + list(property_samplers.keys()))
        rows = np.array([[i] + [sampler() for sampler in property_samplers.values()] for i in range(n)])
        timestamp = str(np.datetime64("now")).replace(':', '-')
        path = "src" + sep + "Populations" + sep + "FromSampler_" + timestamp + ".csv"

        with open(path, 'w') as f:
            f.write(','.join(headers) + '\n')
            np.savetxt(f, rows, fmt='%d', delimiter=',')


################################################################################################
################################################################################################
################################################################################################


class ProgressBar:
    def __init__(self, start_at: int, minimum: int, maximum: int):
        """
        TODO Docstring ProgressBar __init__
        """

        self.min = minimum
        self.max = maximum
        self.current = start_at
        self.printing = True

    def update(self, step: int):
        """
        TODO Docstring ProgressBar update
        """

        self.current = clip(self.current + step, self.min, self.max)
        percentage = 100 * (self.current - self.min) / (self.max - self.min)

        if int(percentage) % 5 == 0:
            if self.printing:
                self.printing = False
                p_as_int = int(percentage)
                print("Progress: %s%s (%s%%)" % (p_as_int // 5 * '#',
                                                 (20 - p_as_int // 5) * '|',
                                                 p_as_int))
        else:
            self.printing = True


################################################################################################
################################################################################################
################################################################################################


class Counter:
    def __init__(self, start):
        """
        TODO Docstring Counter __init__
        """

        self.n = start
        self.history = np.array([start])

    def get_count(self):
        """
        TODO Docstring Counter get_count
        """

        return self.n

    def step(self, mode: str, k: int = 1, return_when: str = 'after'):
        """
        TODO Docstring Counter step
        """

        if mode not in ["inc", "dec"]:
            raise ValueError(str(mode) + " is not a valid value. Try 'inc' or 'dec'.")

        old = self.n
        self.n = self.n + k if mode == "inc" else max(0, self.n - k)
        self.history = np.append(self.history, self.n)

        if return_when == 'after':
            return self.n
        elif return_when == 'before':
            return old
        else:
            raise ValueError(str(return_when) + " is not a valid value. Try 'after' or 'before'.")


################################################################################################
################################################################################################
################################################################################################
