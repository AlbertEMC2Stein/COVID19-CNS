import numpy as np
import numpy.random as rnd
from os.path import sep


class Samplers:
    """
    TODO Docstring Samplers
    """

    basic_sampler = {
        "household": lambda: rnd.randint(21),
        "age": lambda: rnd.randint(0, 90),
        "infected": lambda: rnd.random() < 0.1,
    }


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
