"""
TODO Docstring main \\(\\int_a^b f(x) \\ \\mathrm{d}x\\)
"""

from Simulation import *

if __name__ == "__main__":
    def basic_infection_heuristic(mem_props):
        return 1 - 1 / (0.001 * float(mem_props["age"]) + 1)

    def basic_mortality_heuristic(mem_props):
        return 1/10 * (float(mem_props["age"]) / 200) ** 5

    simulation_settings = {
        "population_file": "DE_03_KLLand.csv",
        "infection_probability_heuristic": basic_infection_heuristic,
        "mortality_probability_heuristic": basic_mortality_heuristic,
        "number_of_initially_infected": 250,
        "number_of_initially_recovered": 2500,
        "number_of_initially_vaccinated": 10000,
        "inner_reproduction_number": 1,
        "outer_reproduction_number": 3,
        "override_newest": True,
        "incubation_time": 2,
        "infection_time": 14,
        "recovered_immunity_time": 90,
        "vaccination_takes_effect_time": 14,
        "vaccinations_per_day": 720,
        "vaccination_immunity_time": 90,
        "waiting_time_vaccination_until_new_vaccination": 90,
        "waiting_time_recovered_until_vaccination": 90,
        "maximal_simulation_time_interval": 365,
        "start_lockdown_at": 150,
        "end_lockdown_at": 50
    }

    Scenarios.single_simulation(simulation_settings)
    # Scenarios.mitigation_interval(simulation_settings, (1.5, 3), 16, 1)
