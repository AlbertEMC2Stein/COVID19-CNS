"""
TODO Docstring main \\(\\int_a^b f(x) \\ \\mathrm{d}x\\)
"""

__all__ = ['Simulation']

import json
import os
from os.path import sep
from src.Network import Population
from src.Utils import Standalones


class Simulation:
    def __init__(self, settings):
        """
        TODO Docstring Simulation __init__
        """

        def check(opts):
            must_haves = ["population_file", "infection_probability_heuristic"]

            for property in must_haves:
                if property not in opts.keys():
                    raise KeyError("Options have to contain '" + property + "'.")

            return opts

        self.settings = check(settings)
        self.population = Population.load_from_file(self.settings["population_file"])

    def start_iteration(self):
        """
        TODO Docstring Simulation start_iteration
        """

        pass

    def end_iteration(self):
        """
        TODO Docstring Simulation end_iteration
        """

        out_path = ".." + sep + "out" + sep + self.population.name + sep
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        newest_iteration = Standalones.get_last_folder(out_path)
        override = self.settings["override_newest"] \
            if "override_newest" in self.settings.keys() \
            else newest_iteration == "9999"

        if not newest_iteration:
            out_path += "0000" + sep
            os.mkdir(out_path)
        else:
            if override:
                out_path += newest_iteration + sep
            else:
                out_path += f"{int(newest_iteration) + 1 :04d}" + sep
                os.mkdir(out_path)

        self.save_options(out_path)
        self.population.save_as_json(out_path)

    def change_options(self, settings):
        """
        TODO Docstring Simulation change_options
        """

        if self.settings["population_file"] != settings["population_file"]:
            self.population = Population.load_from_file(self.settings["population_file"])

        self.settings = settings

    def save_options(self, path: str):
        """
        TODO Docstring Simulation save_options
        """

        settings_mod = self.settings
        heuristic = settings_mod["infection_probability_heuristic"]
        settings_mod["infection_probability_heuristic"] = Standalones.serialize_function(heuristic)

        with open(path + "settings.json", 'w') as f:
            f.write(json.dumps(settings_mod, indent=4))
            f.close()


if __name__ == "__main__":
    simulation_settings = {
        "population_file": "FromSampler_2021-11-28T13-18-51.csv",
        "infection_probability_heuristic": lambda mem_props: 1 - 1 / (0.05 * mem_props["age"] + 1),
        "inner_reproduction_number": 1,
        "outer_reproduction_number": 3,
        "override_newest": True
    }

    sim = Simulation(simulation_settings)
    sim.start_iteration()
    sim.end_iteration()





