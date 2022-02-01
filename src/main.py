"""
TODO Docstring main \\(\\int_a^b f(x) \\ \\mathrm{d}x\\)
"""

from Simulation import *
from Utils import Standalones

if __name__ == "__main__":
    def basic_infection_heuristic(mem_props):
        return 1 - 1 / (0.001 * float(mem_props["age"]) + 1)

    def basic_mortality_heuristic(mem_props):
        return 1/10 * (float(mem_props["age"]) / 200) ** 5

    def heuristic(name):
        if name == "basic_infection_heuristic":
            return basic_infection_heuristic
        elif name == "basic_mortality_heuristic":
            return basic_mortality_heuristic
        else:
            raise ValueError("Heuristic not available")


    settings_name = "Template.cfg"
    simulation_settings = Standalones.make_settings("Settings/" + settings_name)
    simulation_settings["infection_probability_heuristic"] = heuristic(simulation_settings["infection_probability_heuristic"])
    simulation_settings["mortality_probability_heuristic"] = heuristic(simulation_settings["mortality_probability_heuristic"])

    Scenarios.single_simulation(simulation_settings)
    # Scenarios.mitigation_interval(simulation_settings, (1.5, 3), 16, 1)

    # PostProcessing.infection_graph("../out/DE_03_KLLand/0002")
