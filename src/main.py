"""
TODO Docstring main \\(\\int_a^b f(x) \\ \\mathrm{d}x\\)
"""

from Simulation import *
from Utils import Standalones
import numpy as np

if __name__ == "__main__":
    def infection_probability_heuristic(mem_props):
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

    def mortality_probability_heuristic(mem_props):
        age = int(mem_props["age"])
        if 0 <= age <= 4:
            result = 0.000059062155147 / 14
        elif 5 <= age <= 14:
            result = 0.0000124773674418 / 14
        elif 15 <= age <= 34:
            result = 0.000111900272854 / 14
        elif 35 <= age <= 59:
            result = 0.00176127729351 / 14
        elif 60 <= age <= 79:
            result = 0.0338030065822 / 14
        else:
            result = 0.170387357522 / 14

        if "vaccinations" in mem_props.keys():
            return result / 2
        else:
            return result

    def vaccine_failure_probability_heuristic(a, b):
        v_when = lambda member: member.properties["vaccinations"][-1][1]
        v_till = lambda member: member.properties["vaccinations"][-1][2]
        t = lambda member, tick: (tick - v_when(member)) / (v_till(member) - v_when(member))
        return lambda member, tick: 1 - a * (1 - np.exp(b * (t(member, tick) - 1))) / (1 - np.exp(-b))

    def heuristic(name):
        if name == "infection_probability_heuristic":
            return infection_probability_heuristic

        elif name == "mortality_probability_heuristic":
            return mortality_probability_heuristic

        elif name.split('-')[0] == "vaccine_failure_probability_heuristic":
            a = float(name.split('-')[1])
            b = float(name.split('-')[2])
            return vaccine_failure_probability_heuristic(a, b)

        else:
            raise ValueError("Heuristic not available")

    settings_name = "Template.cfg"
    simulation_settings = Standalones.make_settings(settings_name)
    simulation_settings["infection_probability_heuristic"] = heuristic(simulation_settings["infection_probability_heuristic"])
    simulation_settings["mortality_probability_heuristic"] = heuristic(simulation_settings["mortality_probability_heuristic"])
    simulation_settings["vaccine_failure_probability_heuristic"] = heuristic(simulation_settings["vaccine_failure_probability_heuristic"])

    #sim = Scenarios.single_simulation(simulation_settings)
    Scenarios.c_inner_vs_c_outer(simulation_settings, 3)
    #PostProcessing.infection_graph("../out/DE_03_KLLand/0005")
    #PostProcessing.compare_inner_and_outer_infection_numbers("../out/DE_03_KLLand/0005")
    #PostProcessing.progression_plots("../out/DE_03_KLLand/0005")
