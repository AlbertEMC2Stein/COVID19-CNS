"""
TODO Docstring main \\(\\int_a^b f(x) \\ \\mathrm{d}x\\)
"""

from Simulation import *
import numpy as np
import threading

if __name__ == "__main__":
    def basic_infection_heuristic(mem_props):
        age = int(mem_props["age"])
        if 0 <= age <= 4:
            return 0.0330660325674 / 6
        elif 5 <= age <= 14:
            return 0.148281618844 / 6
        elif 15 <= age <= 34:
            return 0.304042732216 / 6
        elif 35 <= age <= 59:
            return 0.359434902279 / 6
        elif 60 <= age <= 79:
            return 0.109905306538 / 6
        else:
            return 0.0452694075548 / 6

    def basic_mortality_heuristic(mem_props):
        age = int(mem_props["age"])
        if 0 <= age <= 4:
            return 0.000059062155147 / 6
        elif 5 <= age <= 14:
            return 0.0000124773674418 / 6
        elif 15 <= age <= 34:
            return 0.000111900272854 / 6
        elif 35 <= age <= 59:
            return 0.00176127729351 / 6
        elif 60 <= age <= 79:
            return 0.0338030065822 / 6
        else:
            return 0.170387357522 / 6

    simulation_settings = {
        "population_file": "DE_03_KLLand.csv",
        "infection_probability_heuristic": basic_infection_heuristic,
        "mortality_probability_heuristic": basic_mortality_heuristic,
        "number_of_initially_infected": 10,
        "number_of_initially_recovered": 0,
        "number_of_initially_vaccinated": 0,
        "inner_reproduction_number": 1,
        "outer_reproduction_number": 3,
        "override_newest": True,
        "incubation_time": 2,
        "infection_time": 14,
        "recovered_immunity_time": 90,
        "vaccine_available_as_of": 365,
        "vaccination_takes_effect_time": 14,
        "vaccinations_per_day": 720,
        "vaccination_immunity_time": 90,
        "waiting_time_vaccination_until_new_vaccination": 90,
        "waiting_time_recovered_until_vaccination": 90,
        "tests_per_day": 1000,
        "test_vaccinated": True,  # FIXME make vaccinated infectable
        "quarantine_duration": 10,
        "maximal_simulation_time_interval": 2*365,
        "start_lockdown_at": 200,
        "end_lockdown_at": 50
    }

    # values = np.zeros(5)

    # def multi_sim(i):
    #     sim = Simulation(simulation_settings)
    #     sim.start_iteration()
    #     values[i] = max(sim.groups["Infected"].history)

    # T0 = threading.Thread(target=multi_sim, args=(0,)).start()
    # T1 = threading.Thread(target=multi_sim, args=(1,)).start()
    # T2 = threading.Thread(target=multi_sim, args=(2,)).start()
    # T3 = threading.Thread(target=multi_sim, args=(3,)).start()
    # T4 = threading.Thread(target=multi_sim, args=(4,)).start()

    # print(values)
    Scenarios.single_simulation(simulation_settings)
    # Scenarios.mitigation_interval(simulation_settings, (1.5, 3), 16, 1)

    folder = "../out/DE_03_KLLand/0004"
    PostProcessing.infection_graph(folder)
    PostProcessing.progression_plots(folder)
