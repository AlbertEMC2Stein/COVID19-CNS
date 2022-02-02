"""
TODO Docstring main \\(\\int_a^b f(x) \\ \\mathrm{d}x\\)
"""

from Simulation import *
from Utils import Standalones

if __name__ == "__main__":
    def infection_heuristic(mem_props):
        age = int(mem_props["age"])
        if 0 <= age <= 4:
            return 0.0330660325674 / 8
        elif 5 <= age <= 14:
            return 0.148281618844 / 8
        elif 15 <= age <= 34:
            return 0.304042732216 / 8
        elif 35 <= age <= 59:
            return 0.359434902279 / 8
        elif 60 <= age <= 79:
            return 0.109905306538 / 8
        else:
            return 0.0452694075548 / 8

    def mortality_heuristic(mem_props):
        age = int(mem_props["age"])
        if 0 <= age <= 4:
            return 0.000059062155147 / 14
        elif 5 <= age <= 14:
            return 0.0000124773674418 / 14
        elif 15 <= age <= 34:
            return 0.000111900272854 / 14
        elif 35 <= age <= 59:
            return 0.00176127729351 / 14
        elif 60 <= age <= 79:
            return 0.0338030065822 / 14
        else:
            return 0.170387357522 / 14

    def heuristic(name):
        if name == "infection_heuristic":
            return infection_heuristic
        elif name == "mortality_heuristic":
            return mortality_heuristic
        else:
            raise ValueError("Heuristic not available")

    settings_name = "Template.cfg"
    simulation_settings = Standalones.make_settings("Settings/" + settings_name)
    simulation_settings["infection_probability_heuristic"] = heuristic(simulation_settings["infection_probability_heuristic"])
    simulation_settings["mortality_probability_heuristic"] = heuristic(simulation_settings["mortality_probability_heuristic"])

    Scenarios.single_simulation(simulation_settings)
    # Scenarios.mitigation_interval(simulation_settings, (1.5, 3), 16, 1)

    # PostProcessing.infection_graph("../out/DE_03_KLLand/0002")
