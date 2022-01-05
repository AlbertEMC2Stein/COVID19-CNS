"""
TODO Docstring main \\(\\int_a^b f(x) \\ \\mathrm{d}x\\)
"""

__all__ = ['Simulation']

import json
import os
import numpy as np
import matplotlib.pylab as plt
from os.path import sep
from Network import Group, Population
from Utils import Standalones


class Simulation:
    def __init__(self, settings):
        """
        TODO Docstring Simulation __init__
        """

        def check(_settings: dict):
            must_haves = ["population_file",
                          "infection_probability_heuristic",
                          "inner_reproduction_number",
                          "outer_reproduction_number",
                          "incubation_time",
                          "infection_time",
                          "vaccination_takes_effect_time",
                          "immunity_time",
                          "number_of_initially_infected",
                          "number_of_initially_recovered",
                          "number_of_initially_vaccinated",
                          "vaccinations_per_day",
                          "vaccination_immunity_time",
                          "waiting_time_vaccination_until_new_vaccination",
                          "waiting_time_recovered_until_vaccination",
                          "maximal_simulation_time_interval"]

            for property in must_haves:
                if property not in _settings.keys():
                    raise KeyError("Settings have to contain '" + property + "'.")

            return _settings

        self.settings = check(settings)
        self.population = Population.load_from_file(self.settings["population_file"])
        self.groups = {"Infected": Group("Infected"),
                       "Recovered": Group("Recovered"),
                       "Vaccinated": Group("Vaccinated")}
        self.stats = {"#new_infected": [],
                      "#infected": [],
                      "#recovered": [],
                      "#new_recovered": [],
                      "#new_susceptible": [],
                      "#new_vaccinated": [],
                      "seven_day_incidence": [],
                      "stats_string": "Day,Infected,"
                                      + "Newly_Infected,Recovered,Newly_Recovered,"
                                      + "Newly_Susceptible,Newly_Vaccinated,"
                                      + "Seven_Day_Incidence"}

    def start_iteration(self):
        """
        TODO Docstring Simulation start_iteration
        """
        print("Initializing simulation...")
        tick = 0
        heuristic = self.settings["infection_probability_heuristic"]
        c_inner = self.settings["inner_reproduction_number"]
        c_outer = self.settings["outer_reproduction_number"]
        c_incubation = self.settings["incubation_time"]
        c_infection = self.settings["infection_time"]
        c_immunity = self.settings["immunity_time"]
        c_vac_effect = self.settings["vaccination_takes_effect_time"]
        c_vac_immunity = self.settings["vaccination_immunity_time"]
        n_seeds = self.settings["number_of_initially_infected"]
        n_ini_recs = self.settings["number_of_initially_recovered"]
        n_ini_vacs = self.settings["number_of_initially_vaccinated"]
        c_vacs = self.settings["vaccinations_per_day"]
        t_wait_vac = self.settings["waiting_time_vaccination_until_new_vaccination"]
        t_wait_rec = self.settings["waiting_time_recovered_until_vaccination"]
        max_t = self.settings["maximal_simulation_time_interval"]
        tau = len(self.population.members)

        # start infection
        seeds_recs_vacs = np.random.choice(self.population.members,
                                           size=n_seeds + n_ini_recs + n_ini_vacs, replace=False)
        seeds = seeds_recs_vacs[:n_seeds]
        ini_recs = seeds_recs_vacs[n_seeds:n_seeds + n_ini_recs]
        ini_vacs = seeds_recs_vacs[n_seeds + n_ini_recs:]
        for seed in seeds:
            seed.infect(np.random.poisson(c_infection), seed, 0, t_immunity=np.random.poisson(c_immunity),
                        t_incubation=-1)
            self.groups["Infected"].add_member(seed)

        for ini_rec in ini_recs:
            ini_rec.make_immune(np.random.poisson(c_immunity))
            self.groups["Recovered"].add_member(ini_rec)
            ini_rec.recovered = True

        for ini_vac in ini_vacs:
            ini_vac.make_immune(np.random.poisson(c_vac_immunity))
            self.groups["Vaccinated"].add_member(ini_vac)
            ini_vac.vaccinated = True
            ini_vac._immune_in = -1

        self.groups["Infected"].counter.save_count()
        self.groups["Infected"].counter.squash_history()

        self.groups["Recovered"].counter.save_count()
        self.groups["Recovered"].counter.squash_history()

        self.groups["Vaccinated"].counter.save_count()
        self.groups["Vaccinated"].counter.squash_history()

        print("Finished initializing simulation.")
        print("Starting simulation...")

        # main simulation
        while True:
            tick += 1

            # spread infection
            #   - inside household
            #   - outside household
            newly_infected, newly_recovered = [], []
            for member in self.groups["Infected"]:
                if member._infectious_in <= 0:
                    n_inner, n_outer = np.random.poisson(c_inner), np.random.poisson(c_outer)

                    household = self.population.households[member.properties["household"]]
                    newly_infected += household.spread_disease(member, n_inner, heuristic, tick,
                                                               t_infection=np.random.poisson(c_infection),
                                                               t_incubation=np.random.poisson(c_incubation))
                    newly_infected += self.population.spread_disease(member, n_outer, heuristic, tick,
                                                                     t_infection=np.random.poisson(c_infection),
                                                                     t_incubation=np.random.poisson(c_incubation))

                    # (possibly) recover
                    if member.make_tick():
                        newly_recovered += [member]
                        member.recovered = True
                else:
                    member._infectious_in -= 1

            # possible gain/loss of immunity
            newly_susceptible_rec, newly_susceptible_vac = [], []
            for member in self.groups["Recovered"]:
                if member.make_tick_immunity():
                    newly_susceptible_rec += [member]
            for member in self.groups["Vaccinated"]:
                if member._immune_in <= 0:
                    if member.make_tick_immunity():
                        newly_susceptible_vac += [member]
                elif not member.infected:
                    member.make_tick_vac_effect()
            for member in newly_susceptible_rec:
                self.groups["Recovered"].remove_member(member)
                member.recovered = False
            for member in newly_susceptible_vac:
                self.groups["Vaccinated"].remove_member(member)
                member.vaccinated = False
            newly_susceptible = newly_susceptible_rec + newly_susceptible_vac

            Group.move(newly_recovered, self.groups["Infected"], self.groups["Recovered"])
            for member in newly_infected:
                self.groups["Infected"].add_member(member)
                if member.vaccinated:
                    self.groups["Vaccinated"].remove_member(member)

            # vaccinations
            n_vacs = np.random.poisson(c_vacs)
            new_vacs_indices = np.random.choice(range(tau), size=n_vacs, replace=False)
            newly_vaccinated = [self.population.members[i] for i in new_vacs_indices]
            no_vacs = []
            for new_vac in newly_vaccinated:
                if new_vac.vaccinate(tick, t_vac_effect=np.random.poisson(c_vac_effect),
                                     t_immunity=np.random.poisson(c_vac_immunity),
                                     t_wait_vac=t_wait_vac, t_wait_rec=t_wait_rec):
                    if new_vac.recovered:
                        self.groups["Recovered"].remove_member(new_vac)
                        new_vac.recovered = False
                        self.groups["Vaccinated"].add_member(new_vac)
                        new_vac.vaccinated = True
                    elif not new_vac.vaccinated:
                        self.groups["Vaccinated"].add_member(new_vac)
                        new_vac.vaccinated = True
                else:
                    no_vacs += [new_vac]
            n_vacs = n_vacs - len(no_vacs)

            for group in self.groups.values():
                group.counter.save_count()

            self.stats["#new_infected"] += [len(newly_infected)]
            self.stats["#infected"] += [self.groups["Infected"].size]
            self.stats["#recovered"] += [self.groups["Recovered"].size]
            self.stats["#new_recovered"] += [len(newly_recovered)]
            self.stats["#new_susceptible"] += [len(newly_susceptible)]
            self.stats["#new_vaccinated"] += [n_vacs]
            if len(self.stats["#new_infected"]) >= 7:
                self.stats["seven_day_incidence"] += [round(sum(self.stats["#new_infected"][-7:]) * 100000 / tau)]
            else:
                self.stats["seven_day_incidence"] += [round(sum(self.stats["#new_infected"]) * 100000 / tau)]

            self.stats["stats_string"] += "\n%d,%d,%d,%d,%d,%d,%d,%d" % (tick,
                                                                         self.stats["#infected"][-1],
                                                                         self.stats["#new_infected"][-1],
                                                                         self.stats["#recovered"][-1],
                                                                         self.stats["#new_recovered"][-1],
                                                                         self.stats["#new_susceptible"][-1],
                                                                         self.stats["#new_vaccinated"][-1],
                                                                         self.stats["seven_day_incidence"][-1])

            # repeat till no more infectious people or max_t has passed
            print("Day: %d, #Infected: %d, #newInf: %d, #newRec: %d, #newVac: %d, 7di: %d"
                  % (tick, self.groups["Infected"].size,
                     self.stats["#new_infected"][-1],
                     self.stats["#new_recovered"][-1],
                     self.stats["#new_vaccinated"][-1],
                     self.stats["seven_day_incidence"][-1]))
            if self.groups["Infected"].size == 0 or tick > max_t:
                break

        print("Finished simulation.")

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

        def save_stats(path: str):
            with open(path + "stats.csv", "w") as f:
                f.write(self.stats["stats_string"])
                f.close()

        def save_plots(path: str):
            def make_plot(plotname: str, title: str, datasets: iter, colors: iter):
                for i, dataset in enumerate(datasets):
                    plt.plot(dataset, color=colors[i])

                plt.xlabel("t")
                plt.ylabel("#")
                plt.title(title)
                plt.show()
                plt.savefig(path + "Plots" + sep + plotname)

            if not os.path.exists(path + "Plots"):
                os.mkdir(path + "Plots")

            data = np.genfromtxt(path + "progression.csv", delimiter=',', skip_header=1)
            make_plot("SIRV.png", "Total",
                      [self.population.size - data[:, 0] - data[:, 1] - data[:, 2], data[:, 0], data[:, 1], data[:, 2]],
                      ['green', 'red', 'blue', 'cyan'])

            make_plot("NewI.png", "New Infections",
                      [sim.stats["#new_infected"]],
                      ['red'])

            make_plot("IR.png", "Infected & Recovered",
                      [data[:, 0], data[:, 1]],
                      ['red', 'blue'])

            make_plot("I.png", "Infected",
                      [data[:, 0]],
                      ['red'])

            make_plot("V.png", "Vaccinated",
                      [data[:, 2]],
                      ['cyan'])

            make_plot("7DI.png", "Seven Day Incidence",
                      [sim.stats["seven_day_incidence"]],
                      ['red'])

        def save_options(path: str):
            settings_mod = self.settings
            heuristic = settings_mod["infection_probability_heuristic"]
            settings_mod["infection_probability_heuristic"] = Standalones.serialize_function(heuristic)

            with open(path + "settings.json", 'w') as f:
                f.write(json.dumps(settings_mod, indent=4))
                f.close()

        print("Saving simulation data...")

        out_path = set_out_path()
        self.population.save_as_json(out_path)
        save_group_histories(out_path)
        save_stats(out_path)
        save_plots(out_path)
        save_options(out_path)

        print("Finished saving simulation data.")

    def change_options(self, settings):
        """
        TODO Docstring Simulation change_options
        """

        if self.settings["population_file"] != settings["population_file"]:
            self.population = Population.load_from_file(self.settings["population_file"])

        self.settings = settings


if __name__ == "__main__":
    def basic_heuristic(mem_props):
        return 1 - 1 / (0.001 * float(mem_props["age"]) + 1)


    simulation_settings = {
        "population_file": "FromSampler_2021-12-15T12-43-56.csv",  # "DE_03_KLLand.csv",
        "infection_probability_heuristic": basic_heuristic,
        "inner_reproduction_number": 1,
        "outer_reproduction_number": 3,
        "override_newest": True,
        "incubation_time": 7,
        "infection_time": 14,
        "vaccination_takes_effect_time": 14,
        "immunity_time": 180,
        "number_of_initially_infected": 10,
        "number_of_initially_recovered": 0,
        "number_of_initially_vaccinated": 0,
        "vaccinations_per_day": 5,
        "vaccination_immunity_time": 180,
        "waiting_time_vaccination_until_new_vaccination": 90,
        "waiting_time_recovered_until_vaccination": 180,
        "maximal_simulation_time_interval": 2*365
    }

    sim = Simulation(simulation_settings)
    sim.start_iteration()
    sim.end_iteration()
