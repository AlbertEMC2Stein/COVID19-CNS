"""
Allows user to run single simulations or whole scenarios
with different custom settings, which then can be analysed
with various postprocessing methods.
"""

__all__ = ['Simulation', 'Scenarios', 'PostProcessing']

import json
import csv
import os
import shutil
from os.path import sep
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
from src.Network import Group, Population
from src.Utils import ProgressBar, Standalones


################################################################################################
################################################################################################
################################################################################################


class Simulation:
    def __init__(self, settings: dict):
        """
        Create a Simulation object with the given settings.

        Parameters
        ----------
        settings : dict
            The dictionary containing all relevant settings for
            the simulation.
        """

        self.settings = None
        self.change_settings(settings)

        self.population = Population.load_from_file(self.settings["population_file"])
        self._population_init = self.population.copy()
        self.groups = {"Infected": Group("Infected"),
                       "Recovered": Group("Recovered"),
                       "Vaccinated": Group("Vaccinated"),
                       "Dead": Group("Dead"),
                       "Quarantined": Group("Quarantined")}
        self.stats = {"new_infected": [0],
                      "new_recovered": [0],
                      "new_susceptible": [0],
                      "new_vaccinated": [0],
                      "new_dead": [0],
                      "ill_vaccinated": [0],
                      "test_results_-": [0],
                      "test_results_+": [0],
                      "seven_day_incidence": [0],
                      "in_lockdown": [0]
                      }
        self.arrange_lockdown = False
        self.lockdown_duration = 0
        self.lockdown_ended = -np.inf

    def start_iteration(self):
        """
        Starts the simulation and runs it until infection numbers
        drop to zero or time given in 'maximum_simulation_time_interval'
        is reached.
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

                        if self.arrange_lockdown:
                            n_inner, n_outer = np.random.poisson(3/2 * c_inner), np.random.poisson(1/2 * c_outer)

                        if member.quarantined:
                            n_inner, n_outer = np.random.poisson(2 * c_inner), 0

                        gen_params = lambda: {
                            "infection_probability_heuristic": infection_heuristic,
                            "vaccine_failure_probability_heuristic": vaccine_heuristic,
                            "incubation_period": np.random.poisson(c_incubation),
                            "infection_period": np.random.poisson(c_infection),
                            "immunity_period": np.random.poisson(c_immunity),
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

                    if member.vaccinated:
                        member.make_tick("vaccine")

                new_members["newly_infected_vac"] = [m for m in new_members["newly_infected"] if m.vaccinated]

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

            elif group.name == "Quarantined":
                for member in group:
                    if member.make_tick("quarantine", tick):
                        group.remove_member(member)

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

            for member in new_members["new_dead"]:
                self.groups["Quarantined"].remove_member(member)

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

        def simulate_tests():
            def test_and_quarantine_procedure(member):
                result = False
                if not member.quarantined and tick != member._last_tested:
                    if (not self.settings["test_vaccinated"]) and "vaccinations" in member.properties.keys() and \
                            tick < member.properties["vaccinations"][-1][0] + self.settings["vaccination_immunity_time"]:
                        return False

                    result = member.test(tick)
                    results[result] += 1
                    if result:
                        member.quarantine(self.settings["quarantine_duration"])
                        self.groups["Quarantined"].add_member(member)

                return result

            def backtrack(member, depth):
                if depth <= 0 or np.random.uniform() > self.settings["backtracking_probability"]:
                    return

                for contact in member.recent_contacts:
                    if test_and_quarantine_procedure(contact):
                        backtrack(contact, depth - 1)

            n_tests = min(np.random.poisson(c_tests), self.population.size)
            results = [0, 0]
            for member in np.random.choice(self.population.members, size=n_tests, replace=False):
                if test_and_quarantine_procedure(member):
                    backtrack(member, self.settings["backtracking_depth"])

            self.stats["test_results_-"] += [results[0]]
            self.stats["test_results_+"] += [results[1]]

        def decide_measure(measure: str):
            if measure == "lockdown":
                # Would a lockdown be arranged/prolonged?
                if self.settings["start_lockdown_at"] <= self.stats["seven_day_incidence"][-1]:
                    result = True

                elif self.settings["end_lockdown_at"] >= self.stats["seven_day_incidence"][-1]:
                    result = False

                else:
                    result = self.arrange_lockdown

                # If it is (not) prolonged, is it still in the allowed duration interval?
                if result:
                    if self.lockdown_duration >= self.settings["maximum_lockdown_duration"]:
                        self.lockdown_duration = 0
                        self.lockdown_ended = tick
                        result = False

                else:
                    if self.arrange_lockdown and self.lockdown_duration < self.settings["minimum_lockdown_duration"]:
                        result = True

                    else:
                        if self.arrange_lockdown:
                            self.lockdown_ended = tick

                        self.lockdown_duration = 0

                # If a new lockdown is arranged, is it sufficiently spaced from the last?
                if result and not self.arrange_lockdown:
                    result = tick - self.lockdown_ended > self.settings["lockdown_gap"]

                # Are enough people vaccinated?
                if self.groups["Vaccinated"].size >= self.settings["heard_immunity"] * self.population.size:
                    result = self.arrange_lockdown

                self.lockdown_duration += result

                return result

            else:
                raise ValueError("Measure not available")

        def update_stats():
            def calc_7di():
                positive = self.stats["test_results_+"]
                if len(positive) >= 7:
                    return round(sum(positive[-7:]) * 100000 / self.population.size)
                else:
                    return round(sum(positive) * 7 / len(positive) * 100000 / self.population.size)

            self.stats["new_infected"] += [len(new_members["newly_infected"])]
            self.stats["new_recovered"] += [len(new_members["newly_recovered"])]
            self.stats["new_susceptible"] += [len(new_members["newly_susceptible"])]
            self.stats["new_vaccinated"] += [n_vacs - len(new_members["not_vaccinated"])]
            self.stats["new_dead"] += [len(new_members["new_dead"])]
            self.stats["ill_vaccinated"] += [len(new_members["newly_infected_vac"])]
            self.stats["seven_day_incidence"] += [calc_7di()]
            self.stats["in_lockdown"] += [1 if self.arrange_lockdown else 0]

        def print_stats():
            color = bcolors.FAIL if self.arrange_lockdown else bcolors.OKGREEN
            print(color + "\rDay: %04d, #Infected: %d, #Vaccinated: %d, #Quarantined: %d, #Dead: %d, 7di: %d\033[K"
                  % (tick,
                     self.groups["Infected"].size,
                     self.groups["Vaccinated"].size,
                     self.groups["Quarantined"].size,
                     self.groups["Dead"].size,
                     self.stats["seven_day_incidence"][-1]), end="")

        print("\nInitializing simulation...")

        # c -> put into poisson, n -> fixed value
        tick = 0
        infection_heuristic = self.settings["infection_probability_heuristic"]
        mortality_heuristic = self.settings["mortality_probability_heuristic"]
        vaccine_heuristic = self.settings["vaccine_failure_probability_heuristic"]
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
        c_tests = self.settings["tests_per_day"]
        t_wait_vac = self.settings["waiting_time_vaccinated_until_new_vaccination"]
        t_wait_rec = self.settings["waiting_time_recovered_until_vaccination"]
        t_vac_available = self.settings["vaccine_available_as_of"]
        max_t = self.settings["maximum_simulation_time_interval"]

        initialize_groups()

        print("Finished initializing simulation.\n\nStarting simulation...")

        print_stats()

        while True:
            tick += 1
            n_vacs = min(np.random.poisson(c_vacs), self.population.size)
            n_vacs = round(n_vacs * np.exp(-100 / (tick - t_vac_available)) if tick > t_vac_available else 0)

            self.arrange_lockdown = decide_measure("lockdown")

            new_members = {
                "newly_susceptible": [],
                "newly_infected": [],
                "newly_infected_vac": [],
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
            simulate_tests()

            for group in self.groups.values():
                group.counter.save_count()

            update_stats()
            print_stats()

            if self.groups["Infected"].size == 0 or tick >= max_t:
                break

        print(bcolors.ENDC + "\nFinished simulation.")

    def end_iteration(self):
        """
        Ends the simulation and saves the population data as
        population.json, the progression data as progression.csv
        and the used settings as settings.cfg in out/POPULATION_NAME/xxxx/.

        If 'override_newest' is set to True, then the latest saved simulation
        is overridden.
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

        def save_options(path: str):
            shutil.copyfile("Settings" + sep + self.settings["file"], path + "settings.cfg")

        print("\nSaving simulation data...")

        out_path = set_out_path()
        self.population.save_as_json(out_path)
        save_disease_progression(out_path)
        save_options(out_path)

        if self.settings["override_newest"] and os.path.exists(out_path + "Plots"):
            shutil.rmtree(out_path + "Plots")

        print("Finished saving simulation data.")

    def change_settings(self, settings: dict):
        """
        Change the settings of the current simulation.

        If 'population_file' does not change, loading it is omitted.

        Parameters
        ----------
        settings : dict
            The dictionary containing all relevant settings for
            the simulation.
        """

        def check_settings():
            all_settings = Standalones.make_settings("Template.cfg")

            for property in all_settings.keys():
                if property not in settings.keys():
                    settings[property] = all_settings[property]

            if settings["start_lockdown_at"] < settings["end_lockdown_at"]:
                raise ValueError("end_lockdown_at must be smaller than start_lockdown_at")

            return settings

        self.settings = check_settings()

        if self.settings["population_file"] != settings["population_file"]:
            self.population = Population.load_from_file(self.settings["population_file"])

    def reset(self):
        """
        Resets simulation to its state of initialization.
        """

        self.population = self._population_init.copy()

        for group in self.groups.values():
            group.reset()

        for stat in self.stats.keys():
            self.stats[stat] = [0]

        self.arrange_lockdown = False
        self.lockdown_duration = 0
        self.lockdown_ended = -np.inf


################################################################################################
################################################################################################
################################################################################################


class Scenarios:
    """
    Class for simulating various scenarios
    """

    @staticmethod
    def single_simulation(settings: dict):
        """
        Run a single simulation with given settings.

        Parameters
        ----------
        settings : dict
            The dictionary containing all relevant settings for
            the simulation.

        Returns
        ----------
        Simulation
            The simulation object.
        """

        sim = Simulation(settings)
        sim.start_iteration()
        sim.end_iteration()

        return sim

    @staticmethod
    def c_inner_vs_c_outer(settings: dict, n: int = 5):
        """
        Creates a \\(n \\times n\\)-heatmap containing the peak infection values for
        each simulation in relation to the inner- and outer- reproduction numbers in
        the range from 0 to 5. Plot is saved as out/general/c_inner_vs_c_outer_nxn.png.

        Parameters
        ----------
        settings : dict
            The dictionary containing all relevant settings for
            the simulation.

        n : int
            Number of simulations along each axis.
            Defaulted to 5.
        """

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
        plt.title("Maximum infection numbers in\nrelation to $c_{inner}$ and $c_{outer}$", pad=10)
        plt.xlabel("$c_{inner}$")
        plt.ylabel("$c_{outer}$")
        plt.xticks(ticks=range(0, n), labels=["%.1f" % i for i in np.linspace(0, 5, n)])
        plt.yticks(ticks=range(0, n), labels=["%.1f" % i for i in np.linspace(0, 5, n)])

        for (j, i), label in np.ndenumerate(max_infection_values):
            plt.text(i, j, int(label), ha='center', va='center')

        Standalones.check_existence(".." + sep + "out" + sep + "general")

        plt.savefig("../out/general/c_inner_vs_c_outer_%dx%d.png" % (n, n))
        plt.show()

        print(max_infection_values)

    @staticmethod
    def mitigation_interval(settings: dict, interval_boundaries: tuple = (1.5, 3), samples: int = 16, avg_over: int = 10):
        """
        Creates a plot containing the (average) peak infection values for simulations
        with varying outer reproduction numbers in the given range. Plot is saved as
        out/general/c_inner_vs_c_outer_nxn.png.

        Parameters
        ----------
        settings : dict
            The dictionary containing all relevant settings for
            the simulation.

        interval_boundaries : tuple
            Range of outer reproduction numbers to simulate.
            Defaulted to (1.5, 3).

        samples: int
            Number of (equidistant) values for outer reproduction number to
            be simulated.
            Defaulted to 16.

        avg_over: int
            Number of runs to average peak values over.
            Defaulted to 10.
        """

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

        Standalones.check_existence(".." + sep + "out" + sep + "general")

        plt.plot(interval, mitigation_interval, color='r')
        plt.title("Maximum infection numbers in relation to $c_{outer}$.")
        plt.xlabel("$c_{outer}$")
        plt.ylabel("maximum infections")
        plt.xticks(ticks=interval, labels=["%.2f" % i for i in interval])
        plt.xlim(interval_boundaries)
        plt.grid()
        plt.savefig("../out/general/mitigation%.2f-%.2f_%d.png" % (interval_boundaries[0], interval_boundaries[1], avg_over))
        plt.show()


################################################################################################
################################################################################################
################################################################################################


class PostProcessing:
    """
    Class for post processing generated simulation data
    """

    @staticmethod
    def infection_graph(folder: str):
        """
        Creates the infection-graph for the given simulation data and
        saves it as folder/Plots/Infection_graph.pdf.

        Parameters
        ----------
        folder : Folder with simulation data to process
        """

        def get_plot_elements():
            f = json.load(open(folder + "population.json"))
            member_id_dict = {member["id"]: i for i, member in enumerate(f["members"])}

            p = ProgressBar(0, len(f["members"]))
            for member in f["members"]:
                self = member_id_dict[member["id"]]
                if "infections" not in member.keys():
                    p.update(1)
                    continue

                for infection in member["infections"]:
                    infectant = member_id_dict[infection[0]]
                    first_day = infection[2]
                    last_day = infection[4]

                    plot_elements["Lines"] += [[[self, first_day], [self, last_day]]]

                    if infectant != self:
                        plot_elements["Lines"] += [[[self, first_day], [infectant, first_day - 1]]]
                        plot_elements["Infected"] += [[self, first_day]]

                    else:
                        plot_elements["Initials"] += [[self, 0]]

                p.update(1)

        if folder[-1] != sep:
            folder += sep

        Standalones.check_existence(folder + "Plots")

        plot_elements = {"Initials": [], "Infected": [], "Lines": []}
        get_plot_elements()
        plot_elements["Initials"] = np.array(plot_elements["Initials"])
        plot_elements["Infected"] = np.array(plot_elements["Infected"])
        plot_elements["Lines"] = np.array(plot_elements["Lines"])

        ax = plt.gca()
        ax.add_collection(LineCollection(plot_elements["Lines"], color='r', alpha=0.2, linewidth=0.01))
        ax.plot(*plot_elements["Infected"][:, [0, 1]].T, color='r', marker='x', linestyle='None', markersize=0.01)
        ax.plot(*plot_elements["Initials"][:, [0, 1]].T, color='b', marker='x', linestyle='None', markersize=0.01)
        plt.xlabel('Population')
        plt.ylabel('Day')
        plt.savefig(folder + "Plots" + sep + "Infection_graph.pdf")
        plt.show()

    @staticmethod
    def progression_plots(folder: str):
        """
        Creates various plots showing the disease progression for the given simulation
        data and saves them as folder/Plots/PLOT_NAME.png.

        Parameters
        ----------
        folder : Folder with simulation data to process
        """

        def make_plot(plotname: str, title: str, datasets: iter, colors: iter, labels: iter = []):
            _, ax = plt.subplots()
            days = np.arange(0, len(datasets[0]), 1)

            for i, dataset in enumerate(datasets):
                ax.plot(dataset, color=colors[i])

            ax.fill_between(days, 0, 1, where=data["in_lockdown"], color='red', alpha=0.25,
                            transform=ax.get_xaxis_transform())
            ax.set_xlabel("t")
            ax.set_ylabel("#")
            ax.set_xlim(0, days[-1])
            ax.set_title(title)

            if labels:
                ax.legend(labels)

            plt.savefig(folder + "Plots" + sep + plotname)
            plt.show()

        if folder[-1] != sep:
            folder += sep

        Standalones.check_existence(folder + "Plots")

        population_size = json.load(open(folder + "population.json"))["size"]
        data_stream = csv.DictReader(open(folder + "progression.csv"))
        data = {}
        for row in data_stream:
            for key, value in row.items():
                if key not in data.keys():
                    data[key] = []

                data[key] += [int(value)]

        data = {column: np.array(data[column]) for column in data.keys()}
        make_plot("SIRV.png", "Total",
                  [population_size - data["Infected"] - data["Recovered"] - data["Vaccinated"] - data["Dead"],
                   data["Infected"], data["Recovered"], data["Vaccinated"]],
                  ['green', 'red', 'blue', 'cyan'],
                  ["Susceptible", "Infected", "Recovered", "Vaccinated"])

        make_plot("NewI.png", "New Infections",
                  [data["new_infected"], data["ill_vaccinated"]],
                  ['red', 'purple'],
                  ["Regular", "Vaccinated"])

        make_plot("IR.png", "Infected & Recovered",
                  [data["Infected"], data["Recovered"]],
                  ['red', 'blue'],
                  ["Infected", "Recovered"])

        make_plot("V.png", "Vaccinated",
                  [data["Vaccinated"]],
                  ['cyan'],
                  ["Vaccinated"])

        make_plot("7DI.png", "Seven Day Incidence",
                  [data["seven_day_incidence"]],
                  ['red'])

        make_plot("D.png", "Dead",
                  [data["Dead"]],
                  ['black'])

    @staticmethod
    def compare_inner_and_outer_infection_numbers(folder: str):
        """
        Creates a bar plot comparing the relative numbers of infections that
        took place inside and outside of households and saves it as
        folder/Plots/inner_vs_outer.png.

        Parameters
        ----------
        folder : Folder with simulation data to process
        """

        def get_infection_data():
            f = json.load(open(folder + "population.json"))

            p = ProgressBar(0, len(f["members"]))
            for member in f["members"]:
                if "infections" not in member.keys():
                    p.update(1)
                    continue

                for infection in member["infections"]:
                    if infection[1]:
                        infection_data["inside"] += 1

                    else:
                        infection_data["outside"] += 1

                p.update(1)

        if folder[-1] != sep:
            folder += sep

        Standalones.check_existence(folder + "Plots")

        infection_data = {"inside": 0, "outside": 0}
        get_infection_data()

        p_inside = infection_data["inside"] / sum(infection_data.values())

        plt.bar(*zip(*infection_data.items()))
        for i, v in enumerate([p_inside, 1 - p_inside]):
            plt.text(i, list(infection_data.values())[i] / 2, "%.1f%%" % (100 * v), color='black', ha='center',
                     va='center', fontsize=32)

        plt.ylabel('Total')
        plt.savefig(folder + "Plots" + sep + "inner_vs_outer.png")
        plt.show()

    @staticmethod
    def mean_latency_period(folder: str):
        """
        Creates a bar plot comparing the latency periods and
        evaluating the overall mean latency time and saves it as
        folder/Plots/latency_periods.png.

        Parameters
        ----------
        folder : Folder with simulation data to process
        """

        def get_infection_data():
            f = json.load(open(folder + "population.json"))

            p = ProgressBar(0, len(f["members"]))
            for member in f["members"]:
                if "infections" not in member.keys():
                    p.update(1)
                    continue

                for infection in member["infections"]:
                    if infection[0] != member["id"]:
                        if infection[0] not in infection_data.keys():
                            infection_data[infection[0]] = []

                        infection_data[infection[0]] += [int(infection[2])]

                p.update(1)

        if folder[-1] != sep:
            folder += sep

        Standalones.check_existence(folder + "Plots")

        infection_data = {}
        get_infection_data()

        latency_periods = {}
        mean = 0.0
        for i, infections in enumerate(infection_data.values()):
            infections = sorted(infections)
            for j in range(len(infections) - 1):
                period = infections[j+1] - infections[j]

                if period < 30:
                    if period not in latency_periods.keys():
                        latency_periods[period] = 0

                    latency_periods[period] += 1
                    mean = i / (i + 1) * mean + period / (i + 1)

        plt.bar(*zip(*latency_periods.items()))
        plt.axvline(x=mean, color='r')
        plt.title("mean = %.2f" % mean)
        plt.xlabel("Latency period (days)")
        plt.ylabel("#")
        plt.savefig(folder + "Plots" + sep + "latency_periods.png")
        plt.show()

    @staticmethod
    def vaccine_quotas(folder: str):
        """
        Creates a bar plot showing how many people have received how
        many shots of the vaccine. Plot is saved as folder/Plots/vaccine_quotas.png.

        Parameters
        ----------
        folder : Folder with simulation data to process
        """

        def get_vaccination_data():
            f = json.load(open(folder + "population.json"))

            p = ProgressBar(0, len(f["members"]))
            for member in f["members"]:
                if "vaccinations" not in member.keys():
                    p.update(1)
                    continue

                n = len(member["vaccinations"])
                if n not in vaccination_data.keys():
                    vaccination_data[n] = 0

                vaccination_data[n] += 1
                p.update(1)

        if folder[-1] != sep:
            folder += sep

        Standalones.check_existence(folder + "Plots")

        vaccination_data = {}
        get_vaccination_data()

        cumsum = np.sum(list(vaccination_data.values()))
        vaccination_data = {n: 100 * vs / cumsum for n, vs in vaccination_data.items()}

        if cumsum != 0:
            plt.bar(*zip(*vaccination_data.items()))
            plt.xlabel("Vaccinations")
            plt.ylabel("%")
            plt.savefig(folder + "Plots" + sep + "vaccine_quotas.png")
            plt.show()
        else:
            print("\rNo vaccinations were tracked.        ", end="")

    @staticmethod
    def death_distribution(folder: str):
        """
        Creates a bar plot showing the age distribution of people
        who died saves the resulting plot as folder/Plots/death_distribution.png.

        Parameters
        ----------
        folder : Folder with simulation data to process
        """

        def get_death_data():
            f = json.load(open(folder + "population.json"))

            p = ProgressBar(0, len(f["members"]))
            for member in f["members"]:
                if "Death" not in member.keys():
                    p.update(1)
                    continue

                else:
                    age = int(member["age"])
                    if age not in death_data.keys():
                        death_data[age] = 0

                    death_data[age] += 1

                p.update(1)

        if folder[-1] != sep:
            folder += sep

        Standalones.check_existence(folder + "Plots")

        death_data = {}
        get_death_data()

        if death_data != {}:
            plt.bar(*zip(*death_data.items()))
            plt.title("Deaths")
            plt.xlim(left=-1)
            plt.xlabel("age")
            plt.ylabel("#")
            plt.savefig(folder + "Plots" + sep + "death_distribution.png")
            plt.show()
        else:
            print("\rNo deaths were tracked.              ", end="")

    @staticmethod
    def effective_reproduction_number(folder: str):
        """
        Creates a plot showing the effective reproduction number over time
        and saves it as folder/Plots/latency_periods.png.

        Parameters
        ----------
        folder : Folder with simulation data to process
        """

        def get_infection_data():
            data_stream = csv.DictReader(open(folder + "progression.csv"))
            data = {}
            for row in data_stream:
                for key, value in row.items():
                    if key not in data.keys():
                        data[key] = []

                    data[key] += [int(value)]

            settings = Standalones.make_settings("settings.cfg", path=folder)

            return settings["incubation_time"] + settings["infection_time"], data["Infected"], data["new_infected"], len(data["Infected"])

        if folder[-1] != sep:
            folder += sep

        Standalones.check_existence(folder + "Plots")

        mean_infection_time, infection_numbers, new_infection_numbers, duration = get_infection_data()

        effective_r_values = np.zeros(duration)
        for day in range(duration):
            maxd = min(day + mean_infection_time, duration)
            for timestamp in range(day, maxd):
                effective_r_values[day] += new_infection_numbers[timestamp]

        effective_r_values /= np.array(infection_numbers)
        max_r = max(effective_r_values)

        points = np.array([np.arange(duration), effective_r_values]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        cmap = ListedColormap(['g', 'orange', 'r'])
        norm = BoundaryNorm([0, 0.8, 1.2, max(1.2, max_r)], cmap.N)
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(effective_r_values)

        plt.gca().add_collection(lc)
        plt.xlim([0, duration-1])
        plt.ylim([0, max_r + 0.25])
        plt.xlabel("Day")
        plt.ylabel("Effective Reproduction number")
        plt.savefig(folder + "Plots" + sep + "effective_reproduction_number.png")
        plt.show()


################################################################################################
################################################################################################
################################################################################################


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
