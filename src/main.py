"""
TODO Docstring main \\(\\int_a^b f(x) \\ \\mathrm{d}x\\)
"""

__all__ = ['Simulation']

import json
import os
from os.path import sep
import matplotlib.pylab as plt
import numpy as np
from Network import Group, Population
from Utils import Standalones


class Simulation:
    def __init__(self, settings):
        """
        TODO Docstring Simulation __init__
        """

        self.settings = {"population_file": "None"}
        self.change_settings(settings)

        self.population = Population.load_from_file(self.settings["population_file"])
        self._population_init = self.population.copy()
        self.groups = {"Infected": Group("Infected"),
                       "Recovered": Group("Recovered"),
                       "Vaccinated": Group("Vaccinated"),
                       "Dead": Group("Dead")}
        self.stats = {"#new_infected": [0],
                      "#new_recovered": [0],
                      "#new_susceptible": [0],
                      "#new_vaccinated": [0],
                      "#new_dead": [0],
                      "seven_day_incidence": [0]
                      }

    def start_iteration(self):
        """
        TODO Docstring Simulation start_iteration
        """

        def initialize_groups():
            def put_inits_in_respective_group():
                gen_params = lambda: {
                    "incubation_period": -1,
                    "infection_period": np.random.poisson(c_infection),
                    "immunity_period": np.random.poisson(c_immunity)
                }

                for ini_inf in ini_infs:
                    ini_inf.infect(ini_inf, 0, gen_params())
                    self.groups["Infected"].add_member(ini_inf)

                for ini_rec in ini_recs:
                    ini_rec.make_immune(np.random.poisson(c_immunity))
                    self.groups["Recovered"].add_member(ini_rec)
                    ini_rec.recovered = True

                for ini_vac in ini_vacs:
                    ini_vac.make_immune(np.random.poisson(c_vac_immunity))
                    self.groups["Vaccinated"].add_member(ini_vac)
                    ini_vac.vaccinated = True

                for group in self.groups.values():
                    group.counter.save_count()
                    group.counter.squash_history()

            infs_recs_vacs = np.random.choice(self.population.members,
                                              size=n_ini_inf + n_ini_recs + n_ini_vacs,
                                              replace=False)

            ini_infs = infs_recs_vacs[:n_ini_inf]
            ini_recs = infs_recs_vacs[n_ini_inf:n_ini_inf + n_ini_recs]
            ini_vacs = infs_recs_vacs[n_ini_inf + n_ini_recs:]

            put_inits_in_respective_group()

        def simulate_group(group: Group):
            if group.name == "Infected":
                for member in group:
                    if member.make_tick("infectious"):
                        n_inner, n_outer = np.random.poisson(c_inner), np.random.poisson(c_outer)

                        gen_params = lambda: {
                            "heuristic": infection_heuristic,
                            "incubation_period": np.random.poisson(c_incubation),
                            "infection_period": np.random.poisson(c_infection),
                            "immunity_period": np.random.poisson(c_immunity)
                        }

                        household = self.population.households[member.properties["household"]]
                        new_members["newly_infected"] += household.spread_disease(member, n_inner, tick, gen_params())
                        new_members["newly_infected"] += self.population.spread_disease(member, n_outer, tick,
                                                                                        gen_params())

                        if member.make_tick("recover"):
                            new_members["newly_recovered"] += [member]
                            member.recovered = True

                        elif np.random.uniform() < mortality_heuristic(member.properties):
                            new_members["new_dead"] += [member]
                            member.make_dead(tick)

            elif group.name == "Recovered":
                for member in group:
                    if member.make_tick("immunity"):
                        new_members["newly_susceptible_rec"] += [member]
                        new_members["newly_susceptible"] += [member]

            elif group.name == "Vaccinated":
                for member in group:
                    if member.make_tick("immunity"):
                        new_members["newly_susceptible_vac"] += [member]
                        new_members["newly_susceptible"] += [member]

                    elif not member.infected:
                        member.make_tick("vaccine")

            elif group.name == "Dead":
                pass

            else:
                raise ValueError("Group '" + group.name + "' does not have an update function")

        def move_members_to_new_groups():
            for member in new_members["newly_susceptible_rec"]:
                self.groups["Recovered"].remove_member(member)
                member.recovered = False

            for member in new_members["newly_susceptible_vac"]:
                self.groups["Vaccinated"].remove_member(member)
                member.vaccinated = False

            Group.move(new_members["newly_recovered"], self.groups["Infected"], self.groups["Recovered"])

            Group.move(new_members["new_dead"], self.groups["Infected"], self.groups["Dead"])

            for member in new_members["newly_infected"]:
                self.groups["Infected"].add_member(member)
                if member.vaccinated:
                    self.groups["Vaccinated"].remove_member(member)
                    member.vaccinated = False

        def simulate_vaccinations():
            for member in new_members["staged_vaccinated"]:
                gen_params = lambda: {
                    "t_vac_effect": np.random.poisson(c_vac_effect),
                    "t_immunity": np.random.poisson(c_vac_immunity),
                    "t_wait_vac": t_wait_vac,
                    "t_wait_rec": t_wait_rec
                }

                if member.vaccinate(tick, gen_params()):
                    if member.recovered:
                        Group.move([member], self.groups["Recovered"], self.groups["Vaccinated"])
                        member.recovered = False

                    elif not member.vaccinated:
                        self.groups["Vaccinated"].add_member(member)

                    member.vaccinated = True

                else:
                    new_members["not_vaccinated"] += [member]

        def update_stats():
            def calc_7di():
                new_inf = self.stats["#new_infected"]
                if len(self.stats["#new_infected"]) >= 7:
                    return round(sum(new_inf[-7:]) * 100000 / self.population.size)
                else:
                    return round(sum(new_inf) * 7 / len(new_inf) * 100000 / self.population.size)

            self.stats["#new_infected"] += [len(new_members["newly_infected"])]
            self.stats["#new_recovered"] += [len(new_members["newly_recovered"])]
            self.stats["#new_susceptible"] += [len(new_members["newly_susceptible"])]
            self.stats["#new_vaccinated"] += [n_vacs - len(new_members["not_vaccinated"])]
            self.stats["#new_dead"] += [len(new_members["new_dead"])]
            self.stats["seven_day_incidence"] += [calc_7di()]

        def print_stats():
            print("\rDay: %d, #Infected: %d, #Dead: %d, #newInf: %d, #newRec: %d, #newVac: %d, 7di: %d"
                  % (tick, self.groups["Infected"].size,
                     self.groups["Dead"].size,
                     self.stats["#new_infected"][-1],
                     self.stats["#new_recovered"][-1],
                     self.stats["#new_vaccinated"][-1],
                     self.stats["seven_day_incidence"][-1]), end="")

        print("\nInitializing simulation...")

        # c -> put into poisson, n -> fixed value
        tick = 0
        infection_heuristic = self.settings["infection_probability_heuristic"]
        mortality_heuristic = self.settings["mortality_probability_heuristic"]
        c_inner = self.settings["inner_reproduction_number"]
        c_outer = self.settings["outer_reproduction_number"]
        n_ini_inf = self.settings["number_of_initially_infected"]
        n_ini_recs = self.settings["number_of_initially_recovered"]
        n_ini_vacs = self.settings["number_of_initially_vaccinated"]
        c_incubation = self.settings["incubation_time"]
        c_infection = self.settings["infection_time"]
        c_immunity = self.settings["recovered_immunity_time"]
        c_vac_effect = self.settings["vaccination_takes_effect_time"]
        c_vac_immunity = self.settings["vaccination_immunity_time"]
        c_vacs = self.settings["vaccinations_per_day"]
        t_wait_vac = self.settings["waiting_time_vaccination_until_new_vaccination"]
        t_wait_rec = self.settings["waiting_time_recovered_until_vaccination"]
        max_t = self.settings["maximal_simulation_time_interval"]

        initialize_groups()

        print("Finished initializing simulation.\n\nStarting simulation...")

        print_stats()

        while True:
            tick += 1
            n_vacs = min(np.random.poisson(c_vacs), self.population.size)

            new_members = {
                "newly_susceptible": [],
                "newly_infected": [],
                "newly_recovered": [],
                "newly_susceptible_rec": [],
                "newly_susceptible_vac": [],
                "staged_vaccinated": np.random.choice(self.population.members, size=n_vacs, replace=False),
                "not_vaccinated": [],
                "new_dead": []
            }

            for group in self.groups.values():
                simulate_group(group)

            move_members_to_new_groups()
            simulate_vaccinations()

            for group in self.groups.values():
                group.counter.save_count()

            update_stats()
            print_stats()

            if self.groups["Infected"].size == 0 or tick >= max_t:
                break

        print("\nFinished simulation.")

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

        def save_disease_progression(path):
            header = ",".join([group.name for group in self.groups.values()] + list(self.stats.keys()))
            rows = np.array([group.history for group in self.groups.values()] +
                            [np.array(stat_values) for stat_values in self.stats.values()]).T

            np.savetxt(path + "progression.csv", rows, fmt='%d', delimiter=",", header=header, comments='')

        def save_plots(path: str):
            def make_plot(plotname: str, title: str, datasets: iter, colors: iter):
                for i, dataset in enumerate(datasets):
                    plt.plot(dataset, color=colors[i])

                plt.xlabel("t")
                plt.ylabel("#")
                plt.title(title)
                plt.savefig(path + "Plots" + sep + plotname)
                plt.show()

            if not os.path.exists(path + "Plots"):
                os.mkdir(path + "Plots")

            data = np.genfromtxt(path + "progression.csv", delimiter=',', skip_header=1)
            make_plot("SIRVD.png", "Total",
                      [self.population.size - data[:, 0] - data[:, 1] - data[:, 2] - data[:, 3],
                       data[:, 0], data[:, 1], data[:, 2], data[:, 3]],
                      ['green', 'red', 'blue', 'cyan', 'black'])

            make_plot("NewI.png", "New Infections",
                      [self.stats["#new_infected"]],
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
                      [self.stats["seven_day_incidence"]],
                      ['red'])

            make_plot("D.png", "Dead",
                      [data[:, 3]],
                      ['black'])

        def save_options(path: str):
            settings_mod = self.settings
            infection_heuristic = settings_mod["infection_probability_heuristic"]
            mortality_heuristic = settings_mod["mortality_probability_heuristic"]
            settings_mod["infection_probability_heuristic"] = Standalones.serialize_function(infection_heuristic)
            settings_mod["mortality_probability_heuristic"] = Standalones.serialize_function(mortality_heuristic)

            with open(path + "settings.json", 'w') as f:
                f.write(json.dumps(settings_mod, indent=4))
                f.close()

        print("\nSaving simulation data...")

        out_path = set_out_path()
        self.population.save_as_json(out_path)
        save_disease_progression(out_path)
        save_plots(out_path)
        save_options(out_path)

        print("Finished saving simulation data.")

    def change_settings(self, settings):
        """
        TODO Docstring Simulation change_settings
        """

        def check_settings():
            must_haves = ["population_file",
                          "infection_probability_heuristic",
                          "mortality_probability_heuristic",
                          "inner_reproduction_number",
                          "outer_reproduction_number",
                          "incubation_time",
                          "infection_time",
                          "vaccination_takes_effect_time",
                          "recovered_immunity_time",
                          "number_of_initially_infected",
                          "number_of_initially_recovered",
                          "number_of_initially_vaccinated",
                          "vaccinations_per_day",
                          "vaccination_immunity_time",
                          "waiting_time_vaccination_until_new_vaccination",
                          "waiting_time_recovered_until_vaccination",
                          "maximal_simulation_time_interval"]

            for property in must_haves:
                if property not in settings.keys():
                    raise KeyError("Settings have to contain '" + property + "'.")

            return settings

        self.settings = check_settings()

        if self.settings["population_file"] != settings["population_file"]:
            self.population = Population.load_from_file(self.settings["population_file"])

    def reset(self):
        self.population = self._population_init.copy()

        for group in self.groups.values():
            group.reset()

        for stat in self.stats.keys():
            self.stats[stat] = [0]


class Scenarios:
    @staticmethod
    def single_simulation(settings: dict):
        sim = Simulation(settings)
        sim.start_iteration()
        sim.end_iteration()

        return sim

    @staticmethod
    def c_inner_vs_c_outer(settings: dict, n: int = 5):
        from matplotlib.colors import LinearSegmentedColormap, LogNorm
        custom = LinearSegmentedColormap.from_list('custom', ['g', 'yellow', 'r'], N=255)

        sim = Simulation(settings)
        max_infection_values = np.zeros(shape=(n, n))
        for x, c_i in enumerate(np.linspace(0, 5, n)):
            for y, c_o in enumerate(np.linspace(0, 5, n)):
                settings["inner_reproduction_number"] = c_i
                settings["outer_reproduction_number"] = c_o

                sim.reset()
                sim.change_settings(settings)
                sim.start_iteration()

                max_infection_values[y, x] = max(sim.groups["Infected"].history)

        plt.figure(figsize=(10, 10))
        plt.imshow(max_infection_values, cmap=custom, norm=LogNorm())
        plt.title("Maximal infection numbers in\nrelation to $c_{inner}$ and $c_{outer}$", pad=10)
        plt.xlabel("$c_{inner}$")
        plt.ylabel("$c_{outer}$")
        plt.xticks(ticks=range(0, n), labels=["%.1f" % i for i in np.linspace(0, 5, n)])
        plt.yticks(ticks=range(0, n), labels=["%.1f" % i for i in np.linspace(0, 5, n)])

        for (j, i), label in np.ndenumerate(max_infection_values):
            plt.text(i, j, int(label), ha='center', va='center')

        plt.savefig("../out/general/c_inner_vs_c_outer_%dx%d.png" % n)
        plt.show()

        print(max_infection_values)

    @staticmethod
    def mitigation_interval(settings: dict, interval_boundaries: tuple, samples: int, avg_over: int = 10):
        sim = Simulation(settings)
        mitigation_interval = np.zeros(samples)
        interval = np.linspace(interval_boundaries[0], interval_boundaries[1], samples)

        for run in range(1, avg_over + 1):
            print("\n" * 25 + "Run %d" % run)
            for i, c_o in enumerate(interval):
                settings["outer_reproduction_number"] = c_o

                sim.reset()
                sim.change_settings(settings)
                sim.start_iteration()

                mitigation_interval[i] = (run - 1) / run * mitigation_interval[i] \
                                         + 1 / run * max(sim.groups["Infected"].history)

        plt.plot(interval, mitigation_interval, color='r')
        plt.title("Maximal infection numbers in relation to $c_{outer}$.")
        plt.xlabel("$c_{outer}$")
        plt.ylabel("maximal infections")
        plt.xticks(ticks=interval, labels=["%.2f" % i for i in interval])
        plt.xlim(interval_boundaries)
        plt.grid()
        plt.show()


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
        "inner_reproduction_number": 1.5,
        "outer_reproduction_number": 2.5,
        "override_newest": False,
        "incubation_time": 2,
        "infection_time": 14,
        "recovered_immunity_time": 90,
        "vaccination_takes_effect_time": 14,
        "vaccinations_per_day": 720,
        "vaccination_immunity_time": 90,
        "waiting_time_vaccination_until_new_vaccination": 90,
        "waiting_time_recovered_until_vaccination": 90,
        "maximal_simulation_time_interval": 2*365
    }

    sim = Scenarios.single_simulation(simulation_settings)
    # Scenarios.mitigation_interval(simulation_settings, (1.5, 3), 16, 1)
