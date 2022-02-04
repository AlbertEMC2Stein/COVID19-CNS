"""
Collection of different classes for more flexible usability.
"""

__all__ = ['EmpiricDistribution', 'Samplers', 'ProgressBar', 'Counter', 'Standalones']

import numpy as np
from numpy import clip
from numpy.random import uniform, randint
import os
from os.path import sep
import configparser


class EmpiricDistribution:
    def __init__(self, data_dict: dict):
        """
        Creates a custom probability distribution from given data.

        Parameters
        ----------
        data_dict : dict
            Dictionary with value-frequency/occurences pairs.
        """

        def normalize(dict_values):
            to_array = np.array(list(dict_values))
            return to_array / np.sum(to_array)

        self.keys = np.array(list(data_dict.keys()))
        self.probabilities = normalize(data_dict.values())
        self.__cdf = np.cumsum(self.probabilities)

    def quantile(self, p: float):
        """
        Calculates quantile(s) of p w.r.t. the given distribution.

        Parameters
        ----------
        p : float or array_like of float
            Value(s) to calculate quantile of \\(0 \\leq p \\leq 1\\).

        Returns
        ----------
        distribution element or array_like
            Quantile(s) of given values.
        """

        if not np.all((0 <= p) & (p <= 1)):
            raise ValueError("Parameter p has to be in [0; 1]")

        result = self.keys.size
        for value in self.__cdf:
            result -= p <= value

        return self.keys[result]

    def pick(self, **kwargs):
        """
        Pick a random sample that follows the given distribution.

        Other Parameters
        ----------
        size : int or tuple of int
            Number of samples to pick.

        Returns
        ----------
        distribution element or array_like
            Sample(s) following the given distribution.
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
    Class for generating populations from custom property distributions.
    """

    basic_sampler = {
        "household": lambda: randint(21),
        "age": lambda: randint(0, 90),
    }
    """
    Most basic population sampler.
    \nhousehold : uniform distribution in \\([0, \\dots, 20]\\) 
    \nage : uniform distribution in \\([0, \\dots, 89]\\)
    """

    @staticmethod
    def generate_population_data_from_samplers(property_samplers: dict, n: int):
        """
        Generates a new random population whose members properties
        follow the given distributions and saves it as
        src/Populations/FromSampler_TIMESTAMP.csv.

        Parameters
        ----------
        property_samplers : dict
            Dictionary with property-sampler pairs.

        n : int
            Size of the population generated.
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
    def __init__(self, minimum: int, maximum: int, **kwargs):
        """
        Prints a simple progressbar.

        Parameters
        ----------
        minimum : int
            Value of minimal progress.

        maximum : int
            Value of maximal progress.

        Other Parameters
        ----------
        start_at : int
            If not specified progress will start at its minimum value.
        """

        self.min = minimum
        self.max = maximum
        self.current = kwargs["start_at"] if "start_at" in kwargs.keys() else minimum
        self.printing = True

    def update(self, step: int):
        """
        Updates the current progress by the specified amount of steps.

        Parameters
        ----------
        step : int
            Amount of steps to advance progress.
        """

        self.current = clip(self.current + step, self.min, self.max)
        percentage = 100 * (self.current - self.min) / (self.max - self.min)

        if int(percentage) % 5 == 0:
            if self.printing:
                self.printing = False
                p_as_int = int(percentage)
                print("\rProgress: %s%s (%s%%)" % (p_as_int // 5 * '#',
                                                   (20 - p_as_int // 5) * '|',
                                                   p_as_int), end="")
        else:
            self.printing = True


################################################################################################
################################################################################################
################################################################################################


class Counter:
    def __init__(self, start: int):
        """
        Creates a simple counter that keeps track of its progress.
        """

        self.count = start
        self.history = np.array([start])

    def _step(self, mode: str, k: int = 1, return_when: str = 'after'):
        old = self.count
        self.count = self.count + k if mode == "inc" else max(0, self.count - k)

        if k < 0:
            raise ValueError("k has to be non-negative.")

        if return_when == 'after':
            return self.count
        elif return_when == 'before':
            return old
        else:
            raise ValueError(str(return_when) + " is not a valid value. Try 'after' or 'before'.")

    def increment(self, k: int = 1, return_when: str = 'after'):
        """
        Increments counter by the specified amount of steps.

        Parameters
        ----------
        k : int
            Amount of steps to increment.

        return_when : str
            If set to 'after' the value returned will be tho one
            after incrementing the counter. If 'before' is specified
            the value returned will be the one before incrementing.

        Returns
        ----------
        int
            Current value of counter.
        """

        return self._step('inc', k, return_when)

    def decrement(self, k: int = 1, return_when: str = 'after'):
        """
        Decrements counter by the specified amount of steps but never below 0.

        Parameters
        ----------
        k : int
            Amount of steps to decrement.

        return_when : str
            If set to 'after' the value returned will be tho one
            after decrementing the counter. If 'before' is specified
            the value returned will be the one before decrementing.

        Returns
        ----------
        int
            Current value of counter.
        """

        return self._step('dec', k, return_when)

    def save_count(self):
        """
        Appends the current counter value to the counter history.
        """

        self.history = np.append(self.history, self.count)

    def squash_history(self):
        """
        Deletes entire history but its last entry.
        """

        self.history = self.history[-1:]

    def copy(self):
        """
        Creates a copy of the counter.
        """

        c = Counter(0)
        c.count = self.count
        c.history = self.history.copy()

        return c


################################################################################################
################################################################################################
################################################################################################


class Standalones:
    """
    Class for miscellaneous functions.
    """

    @staticmethod
    def get_last_folder(path: str):
        """
        In a directory with enumerated folders, this method selects the one
        with the highest number.

        Parameters
        ----------
        path : str
            Directory to get last folder from.

        Returns
        ----------
        str
            Name of last folder.
        """

        folders = [folder for folder in os.listdir(path) if os.path.isdir(path + sep + folder)]
        if not folders: return None
        return sorted(folders)[-1]

    @staticmethod
    def check_existence(path: str):
        """
        Checks if the last folder in the given path exists and creates it if that
        is not the case.

        Parameters
        ----------
        path : str
            Path with its last element being the folder to be checked of its existence.
        """

        if not os.path.exists(path):
            os.mkdir(path)

    @staticmethod
    def make_settings(settings_file: str):
        """
        Creates a settings-dictionary which is used to initialize simulations
        from a given .cfg file.

        Parameters
        ----------
        settings_file : str
            Path to the file containing the settings.
        """

        config = configparser.RawConfigParser()
        config.read("Settings" + sep + settings_file)

        settings = {}
        for section in config.sections():
            for setting, value in config.items(section):
                try:
                    if value == str(float(value)):
                        settings[setting] = float(value)
                    else:
                        raise ValueError

                except ValueError:
                    try:
                        if value == str(int(value)):
                            settings[setting] = int(value)
                        else:
                            raise ValueError

                    except ValueError:
                        try:
                            if value in ["True", "False"]:
                                settings[setting] = value == "True"
                            else:
                                raise ValueError

                        except ValueError:
                            settings[setting] = value

        settings["file"] = settings_file

        return settings
