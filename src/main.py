"""
TODO Docstring main \\(\\int_a^b f(x) \\ \\mathrm{d}x\\)
"""

from src.Network import Population
import numpy as np


class Simulation:
    def __init__(self, options):
        """
        TODO Docstring Simulation __init__
        """

        def check(opts):
            must_haves = ["population_name", "infection_probability_heuristic"]

            for property in must_haves:
                if property not in opts.keys():
                    raise KeyError("Options have to contain '" + property + "'.")

            return opts

        self.options = check(options)
        self.population = None

    def start(self):
        """
        TODO Docstring Simulation start
        """

        def println(*args):
            for arg in args:
                print(arg)

        self.population = Population.load_from_csv(self.options["population_name"] + ".csv")
        mask = np.array([bool(int(m.properties['infected'])) for m in self.population.members])
        println(*self.population.members[mask])

    def end(self):
        """
        TODO Docstring Simulation end
        """

        self.population.save_as_json()


if __name__ == "__main__":
    simulation_options = {
        "population_name": "FromSampler_2021-11-28T13-18-51",
        "infection_probability_heuristic": lambda mem_props: 1 - 1 / (0.05 * mem_props["age"] + 1)
    }

    sim = Simulation(simulation_options)
    sim.start()
    sim.end()





