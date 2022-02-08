"""
Main script for executing simulations, scenarios and post processing.
"""

import sys
import os
from os.path import sep

if os.getcwd().split(sep)[-1] == "src":
    sys.path.insert(0, sep.join(os.getcwd().split(sep)[:-1]))

import numpy as np
from src.Simulation import *
from src.Utils import Standalones

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

    def post_processing(option):
        out = ".." + sep + "out" + sep + simulation_settings["population_file"].split('.')[0] + sep
        latest = Standalones.get_last_folder(out)
        all_methods = [item[1] for item in PostProcessing.__dict__.items() if not item[0].startswith('__')]

        if option == 'All':
            for method in all_methods:
                method(out + latest)

        elif option == 'None':
            return

        else:
            specified_methods = option.split(',')
            for name in specified_methods:
                for method in all_methods:
                    if method.__name__ == name.replace(' ', ''):
                        method(out + latest)
                        break

                raise ValueError('Post processing method \'%s\' not found' % name.replace(' ', ''))

    args = sys.argv[1:]
    print(sys.path)

    settings_name = args[0] if len(args) > 0 else "Template.cfg"
    simulation_settings = Standalones.make_settings(settings_name)
    simulation_settings["infection_probability_heuristic"] = heuristic(simulation_settings["infection_probability_heuristic"])
    simulation_settings["mortality_probability_heuristic"] = heuristic(simulation_settings["mortality_probability_heuristic"])
    simulation_settings["vaccine_failure_probability_heuristic"] = heuristic(simulation_settings["vaccine_failure_probability_heuristic"])

    sim = Scenarios.single_simulation(simulation_settings)

    post_processing(simulation_settings["post_processing"])
