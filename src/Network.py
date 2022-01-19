"""
TODO Docstring Network
"""

__all__ = ['Member', 'Household', 'Population', 'Group']

import textwrap
import numpy as np
import csv
import json
from os.path import sep
from src.Utils import ProgressBar, Counter


################################################################################################
################################################################################################
################################################################################################


class Member:
    def __init__(self, properties: dict):
        """
        TODO Docstring Member __init__
        """

        def check(_properties: dict):
            must_haves = ["id", "household"]

            for property in must_haves:
                if property not in _properties.keys():
                    raise KeyError("Properties have to contain '" + property + "'.")

            return _properties

        self.properties = check(properties)
        self.infected = False
        self.recovered = False
        self.vaccinated = False
        self.dead = False
        self._susceptible_in = -1
        self._recovers_in = -1
        self._infectious_in = -1
        self._immune_in = -1

    def __str__(self):
        return "\n".join(["%s = %s" % (attr, val) for attr, val in self.__dict__.items()]) #str(self.properties)

    def infect(self, infectant: 'Member', timestamp: int, disease_parameters: dict):
        """
        TODO Docstring Member infect
        """

        if "immune" in self.properties.keys() and self.properties["immune"] or self.infected or self.dead:
            return False

        n_incubation = disease_parameters["incubation_period"]
        n_infection = disease_parameters["infection_period"]
        shared_household = self.properties["household"] == infectant.properties["household"]
        infection_data = [(infectant.properties["id"],
                           shared_household,
                           timestamp,
                           timestamp + n_incubation,
                           timestamp + n_incubation + n_infection)]

        if "infections" not in self.properties.keys():
            self.properties["infections"] = infection_data
        else:
            self.properties["infections"] += infection_data

        self.infected = True
        self._infectious_in = n_incubation
        self._recovers_in = n_infection
        self._susceptible_in = disease_parameters["immunity_period"]
        return True

    def make_immune(self, immunity_duration: int):
        """
        TODO Docstring Member make_immune
        """
        if self.infected:
            raise RuntimeError
            self.infected = False
            self._infectious_in = -1
            self._recovers_in = -1
            self._susceptible_in = immunity_duration
            self.properties["immune"] = True
            return True
        else:
            self._infectious_in = -1
            self._recovers_in = -1
            self._susceptible_in = immunity_duration
            self.properties["immune"] = True
            return False

    def vaccinate(self, timestamp: int, vaccine_parameters: dict):
        """
        TODO Docstring Member vaccinate
        """

        vaccine_unavailable = self.infected or self.dead or \
                              "vaccinations" in self.properties.keys() and \
                              timestamp < self.properties["vaccinations"][-1][1] + vaccine_parameters["t_wait_vac"] or \
                              "infections" in self.properties.keys() and \
                              timestamp < self.properties["infections"][-1][4] + vaccine_parameters["t_wait_rec"]
        if vaccine_unavailable:
            return False

        else:
            vaccination_data = [(timestamp,
                                 timestamp + vaccine_parameters["t_vac_effect"],
                                 timestamp + vaccine_parameters["t_immunity"])]

            if "vaccinations" in self.properties.keys():
                self.properties["vaccinations"] += vaccination_data

            else:
                self.properties["vaccinations"] = vaccination_data

            if "immune" in self.properties.keys() and self.properties["immune"]:
                self._immune_in = 0
                self._susceptible_in = max(vaccine_parameters["t_immunity"], self._susceptible_in)

            else:
                self._immune_in = vaccine_parameters["t_vac_effect"]
                self._susceptible_in = vaccine_parameters["t_immunity"]

            return True

    def make_dead(self, timestamp: int):
        """
        TODO Docstring Member make_dead
        """

        self.infected = False
        self.recovered = False
        self.vaccinated = False
        self._susceptible_in = -1
        self._recovers_in = -1
        self._infectious_in = -1
        self._immune_in = -1

        self.dead = True
        self.properties["Death"] = timestamp

    def make_tick(self, option: str):
        """
        TODO Docstring Member make_tick
        """

        if option == "infectious":
            result = self._infectious_in <= 0
            if self._infectious_in > 0:
                self._infectious_in -= 1

            return result

        elif option == "immunity":
            if self._immune_in <= 0:
                self._susceptible_in -= 1
                if self._susceptible_in == 0:
                    self._susceptible_in = -1
                    self.properties["immune"] = False
                    return True

            return False

        elif option == "vaccine":
            self._immune_in -= 1
            if self._immune_in == 0:
                self.properties["immune"] = True

        elif option == "default":
            if self.infected:
                self._recovers_in -= 1
                if self._recovers_in == 0:
                    self._recovers_in = -1
                    self.infected = False
                    self.properties["immune"] = True
                    return True

            return False

        else:
            raise ValueError("option not supported")

    def copy(self):
        """
        TODO Docstring Member copy
        """

        m = Member(self.properties.copy())
        m.infected = self.infected
        m.recovered = self.recovered
        m.vaccinated = self.vaccinated
        m.dead = self.dead
        m._susceptible_in = self._susceptible_in
        m._recovers_in = self._recovers_in
        m._infectious_in = self._infectious_in
        m._immune_in = self._immune_in

        return m

################################################################################################
################################################################################################
################################################################################################


class Group:
    def __init__(self, name: str):
        """
        TODO Docstring Group __init__
        """

        self.name = name
        self.members = np.array([])
        self.counter = Counter(0)

    def __str__(self):
        result = self.__class__.__name__ + ": " + self.name + "\nMembers: "
        for member in self:
            result += str(member) + "\n\t\t "

        return result[:-1]

    def __iter__(self):
        return iter(self.members)

    def add_member(self, member: Member):
        """
        TODO Docstring Group add_member
        """

        self.members = np.append(self.members, member)
        self.counter.increment()

    def remove_member(self, member: Member):
        """
        TODO Docstring Group remove_member
        """

        old_size = self.members.size
        self.members = self.members[self.members != member]
        self.counter.decrement(old_size - self.members.size)

    def spread_disease(self, infectant: Member, n: int, timestamp: int, disease_parameters: dict):
        """
        TODO Docstring Group spread_disease
        """

        result = []
        for other in np.random.choice(self.members, n):
            if np.random.uniform() < disease_parameters["heuristic"](other.properties):
                if other.infect(infectant, timestamp, disease_parameters):
                    result += [other]

        return result

    def reset(self):
        """
        TODO Docstring Group reset
        """

        self.members = np.array([])
        self.counter = Counter(0)

    # def copy(self):
    #     """
    #     TODO Docstring Group copy
    #     """

    #     g = Group(self.name)
    #     g.members = np.array([member.copy() for member in self.members])
    #     g.counter = self.counter.copy()

    #     return g

    @staticmethod
    def move(members: iter, origin: 'Group', destination: 'Group'):
        """
        TODO docstring Group move
        """

        for member in members:
            origin.remove_member(member)
            destination.add_member(member)

    @property
    def history(self):
        return self.counter.history

    @property
    def size(self):
        return self.members.size


################################################################################################
################################################################################################
################################################################################################


class Household(Group):
    def __init__(self, identifier: int):
        """
        TODO Docstring Household __init__
        """

        super().__init__(str(identifier))

    @property
    def id(self):
        return self.name


################################################################################################
################################################################################################
################################################################################################


class Population(Group):
    def __init__(self, name: str):
        """
        TODO Docstring Population __init__
        """

        super().__init__(name)
        self.households = {}

    def add_member(self, member: Member, count: int = True):
        """
        Adds member as a Member to the Population
        and to its Household within the Population.

        Parameters
        ----------
        member : Member
            The Member to be added to the Population.
        count :
            Option whether to increase population counter or not.

        Raises
        ------
        KeyError
            member is expected to have the property 'household'.
            If member does not meet the expectations,
            a KeyError will be raised.

        Returns
        -------
        None.
        """

        household_id = member.properties["household"]
        if household_id in self.households.keys():
            self.households[household_id].add_member(member)
        else:
            household = Household(household_id)
            self.households[household_id] = household
            self.households[household_id].add_member(member)

        if type(self.members) == np.ndarray:
            self.members = np.append(self.members, member)
        else:
            self.members += [member]

        self.counter.increment()

    def save_as_json(self, path: str) -> None:
        """
        TODO Docstring Population save_as_json
        """

        if len(self.members) == 0:
            raise ValueError("Population can't be empty.")

        with open(path + "population.json", 'w') as f:
            wrapper = "{\n\t\"name\": \"" + self.name + "\",\n\t\"members\": [\n"
            inner = ""
            for member in self:
                json_str = json.dumps(member.properties, indent=4)
                inner += json_str + ', \n'

            inner = textwrap.indent(inner[:-3] + '\n', '\t\t')

            f.write(wrapper + inner + "\t]\n}")
            f.close()

    def copy(self):
        """
        TODO Docstring Population copy
        """

        p = Population(self.name)
        p.members = list(p.members)
        for member in self.members:
            p.add_member(member.copy())

        p.members = np.array(p.members)
        p.counter = self.counter.copy()

        return p

    @staticmethod
    def load_from_csv(file_name: str, path: str = "Populations" + sep):
        """
        Create a Population object according to the data given in file_name.

        Parameters
        ----------
        file_name : str
            The name of the file containing the data of the population.
            Expected to end in .csv.
        path : str, optional
            The path to the file named file_name.
            The default is "Populations" + sep.

        Returns
        -------
        p : Population
            A Population object containing the data for the given file.
        """

        p = Population(file_name[:-4])
        p.members = []
        with open(path + file_name, newline='') as f:
            progress = ProgressBar(1, 1, sum(1 for _ in f) - 1)

        print("Loading population data...")

        progress.update(0)
        with open(path + file_name, newline='') as f:
            for m_dict in csv.DictReader(f):
                progress.update(1)
                p.add_member(Member(m_dict))

        print("Finished loading.")

        p.members = np.array(p.members)
        p.counter.squash_history()
        return p

    @staticmethod
    def load_from_json(file_name: str, path: str = "Populations" + sep):
        """
        Create a Population object according to the data given in file_name.

        Parameters
        ----------
        file_name : str
            The name of the file containing the data of the population.
            Expected to end in .json.
        path : str, optional
            The path to the file named file_name.
            The default is "Populations" + sep.

        Returns
        -------
        p : Population
            A Population object containing the data for the given file.
        """

        p = Population(file_name[:-5])
        p.members = []
        with open(path + file_name, "r") as f:
            print("Load json...")
            data = json.load(f)
            print("Finished loading.\nAdding members to population...")

            progress = ProgressBar(1, 1, len(data["members"]))
            progress.update(0)
            for member in data["members"]:
                progress.update(1)
                p.add_member(Member(member))

            print("Finished adding members.")

        p.members = np.array(p.members)
        p.counter.squash_history()
        return p

    @staticmethod
    def load_from_file(file_name: str, path: str = "Populations" + sep):
        """
        Create a Population object according to the data given in file_name.

        Parameters
        ----------
        file_name : str
            The name of the file containing the data of the population.
            Expected to end in .csv or .json.
        path : str, optional
            The path to the file named file_name.
            The default is "Populations" + sep.

        Raises
        ------
        ValueError
            file_name is expected to end in .csv or .json.
            If file_name does not meet the expectation,
            a ValueError will be raised.

        Returns
        -------
        Population
            A Population object containing the data for the given file.
        """

        if file_name[-4:] == ".csv":
            return Population.load_from_csv(file_name, path)
        elif file_name[-5:] == ".json":
            return Population.load_from_json(file_name, path)
        else:
            raise ValueError("file_name must end in .csv or .json.")

################################################################################################
################################################################################################
################################################################################################
