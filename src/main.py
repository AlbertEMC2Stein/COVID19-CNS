"""
TODO Docstring main \\(\\int_a^b f(x) \\ \\mathrm{d}x\\)
"""

__all__ = ['Simulation']

import json
import os
import numpy as np
import matplotlib.pylab as plt
from os.path import sep
from src.Network import Group, Population
from src.Utils import Standalones


class Simulation:
    def __init__(self, settings):
        """
        TODO Docstring Simulation __init__
        """

        def check(_settings: dict):
            must_haves = ["population_file",
                          "infection_probability_heuristic",
                          "inner_reproduction_number",
                          "outer_reproduction_number"]

            for property in must_haves:
                if property not in _settings.keys():
                    raise KeyError("Settings have to contain '" + property + "'.")

            return _settings

        self.settings = check(settings)
        self.population = Population.load_from_file(self.settings["population_file"])
        self.groups = {"Infected": Group("Infected"),
                       "Recovered": Group("Recovered")}

    def start_iteration(self):
        """
        TODO Docstring Simulation start_iteration
        """

        tick = 0
        heuristic = self.settings["infection_probability_heuristic"]
        c_inner = self.settings["inner_reproduction_number"]
        c_outer = self.settings["outer_reproduction_number"]

        # start infection
        seed = np.random.choice(self.population.members)
        seed.infect(14, seed, 0)
        self.groups["Infected"].add_member(seed)
        self.groups["Infected"].counter.save_count()
        self.groups["Infected"].counter.squash_history()

        while True:
            tick += 1

            # spread infection
            #   - inside household
            #   - outside household
            newly_infected, newly_recovered = [], []
            for member in self.groups["Infected"]:
                n_inner, n_outer = np.random.poisson(c_inner), np.random.poisson(c_outer)

                household = self.population.households[member.properties["household"]]
                newly_infected += household.spread_disease(member, n_inner, heuristic, tick)
                newly_infected += self.population.spread_disease(member, n_outer, heuristic, tick)

                # (possibly) recover
                if member.make_tick():
                    newly_recovered += [member]

            Group.move(newly_recovered, self.groups["Infected"], self.groups["Recovered"])
            for member in newly_infected:
                self.groups["Infected"].add_member(member)

            for group in self.groups.values():
                group.counter.save_count()

            # repeat till no more infectious people
            print("Day: %d, #Infected: %d" % (tick, self.groups["Infected"].size))
            if self.groups["Infected"].size == 0:
                break

    def end_iteration(self):
        """
        TODO Docstring Simulation end_iteration
        """

        def set_out_path():
            path = ".." + sep + "out" + sep + self.population.name + sep
            if not os.path.exists(path):
                os.mkdir(path)

            newest_iteration = Standalones.get_last_folder(path)
            override = self.settings["override_newest"] \
                if "override_newest" in self.settings.keys() \
                else newest_iteration == "9999"

            if not newest_iteration:
                path += "0000" + sep
                os.mkdir(path)
            else:
                if override:
                    path += newest_iteration + sep
                else:
                    path += f"{int(newest_iteration) + 1 :04d}" + sep
                    os.mkdir(path)

            return path

        def save_group_histories(path):
            header = ",".join([group.name for group in self.groups.values()])
            rows = np.array([group.history for group in self.groups.values()]).T
            np.savetxt(path + "progression.csv", rows, fmt='%d', delimiter=",", header=header, comments='')

        out_path = set_out_path()
        self.save_options(out_path)
        self.population.save_as_json(out_path)
        save_group_histories(out_path)

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
        "population_file": "DE_03_KLLand.csv",
        "infection_probability_heuristic": lambda mem_props: 1 - 1 / (0.001 * float(mem_props["age"]) + 1),
        "inner_reproduction_number": 1,
        "outer_reproduction_number": 3,
        "override_newest": True
    }

    sim = Simulation(simulation_settings)
    sim.start_iteration()
    sim.end_iteration()

    path = ".." + sep + "out" + sep + "DE_03_KLLand" + sep
    latest = Standalones.get_last_folder(path)
    data = np.genfromtxt(path + latest + sep + "progression.csv", delimiter=',', skip_header=1)
    plt.plot(106463 - data[:, 0] - data[:, 1], color='green')
    plt.plot(data[:, 0], color='red')
    plt.plot(data[:, 1], color='blue')
    plt.show()
