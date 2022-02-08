"""
Collection of the fundamental classes of the network-model of infection.
These are:

    Member:
        A representation of a person. Consists of a dict 'properties'
        containing all relevant data on this person, several attributes used
        only within the simulation and some private counters used only in
        functions in the 'Member'-class called during simulation.

    Group:
        A base class for sets of several persons. Consists of a name-string,
        an array containing its members and a counter saving the amounts of
        members in the group.

    Household:
        A representation of a household of people. A subclass of Group, where
        the name is the string of the household-id.

    Population:
        A representation of a larger set of people from several households.
        A subclass of Group, additionally consisting of a list of households.
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
        Create a new member with the given attributes in 'properties'.

        Parameters
        ----------
        properties : dict
            Dictionary containing all pre-defined attributes of the new member.

        Raises
        ------
        KeyError
            The properties-dict is expected to contain the keys 'id' and
            'household'. If properties does not meet the expectation,
            a KeyError will be raised.
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
        self.immune = False
        self.quarantined = False
        self.recent_contacts = []
        self._susceptible_in = -1
        self._infectious_in = -1
        self._recovers_in = -1
        self._immune_in = -1
        self._released_in = -1
        self._last_tested = -1

    def __str__(self):
        return "\n".join(["%s = %s" % (attr, val) for attr, val in self.__dict__.items()])

    def infect(self, infectant: 'Member', timestamp: int, disease_parameters: dict):
        """
        Infect the member, if it is infectable.

        Parameters
        ----------
        infectant : 'Member'
            The infectious member that infects the member on which 'infect' is
            applied.

        timestamp : int
            Current day in the simulation.

        disease_parameters : dict
            Dictionary containing all relevant infection parameters.
            Is expected to contain the keys
            'incubation_period',
            'infection_period',
            'vaccine_failure_probability_heuristic',
            'immunity_period'.

        Returns
        -------
        bool
            Whether the member has been infected.
        """

        n_incubation = disease_parameters["incubation_period"]
        n_infection = disease_parameters["infection_period"]
        shared_household = self.properties["household"] == infectant.properties["household"]

        if (self.immune and not self.vaccinated) or self.infected or \
                self.dead or (self.quarantined and not shared_household):
            return False

        if self.vaccinated and self._immune_in <= 0:
            p_failure = disease_parameters["vaccine_failure_probability_heuristic"](self, timestamp)
            if np.random.uniform() > p_failure:
                return False

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

    def vaccinate(self, timestamp: int, vaccine_parameters: dict):
        """
        Vaccinate the member, if it is allowed to be vaccinated.

        Parameters
        ----------
        timestamp : int
            Current day in the simulation.

        vaccine_parameters : dict
            Dictionary containing all relevant vaccination parameters.
            Is expected to contain the keys
            't_wait_vac',
            't_wait_rec',
            't_vac_effect',
            't_immunity'.

        Returns
        -------
        bool
            Whether the member has been vaccinated.
        """

        vaccine_unavailable = self.infected or self.dead or self.quarantined or \
                              "vaccinations" in self.properties.keys() and \
                              timestamp < self.properties["vaccinations"][-1][1] + vaccine_parameters["t_wait_vac"] or \
                              "infections" in self.properties.keys() and \
                              timestamp < self.properties["infections"][-1][4] + vaccine_parameters["t_wait_rec"]

        if not vaccine_unavailable:
            vaccination_data = [(timestamp,
                                 timestamp + vaccine_parameters["t_vac_effect"],
                                 timestamp + vaccine_parameters["t_vac_effect"] + vaccine_parameters["t_immunity"])]

            if "vaccinations" in self.properties.keys():
                self.properties["vaccinations"] += vaccination_data

            else:
                self.properties["vaccinations"] = vaccination_data

            if self.immune:
                self._immune_in = 0
                self._susceptible_in = max(vaccine_parameters["t_immunity"], self._susceptible_in)

            else:
                self._immune_in = vaccine_parameters["t_vac_effect"]
                self._susceptible_in = vaccine_parameters["t_immunity"]

            return True

        return False

    def test(self, timestamp: int):
        """
        Test the member for an infection.

        Parameters
        ----------
        timestamp : int
            Current day in the simulation.

        Returns
        -------
        bool
            Whether the test is positive.
        """

        result = self.infected * (np.random.uniform() < 0.99)

        if "tests" not in self.properties.keys():
            self.properties["tests"] = [0, 0]

        self.properties["tests"][result] += 1
        self._last_tested = timestamp

        return result

    def quarantine(self, days: int):
        """
        Place the member in quarantine.

        Parameters
        ----------
        days : int
            Amount of days (in the simulation) the member has to stay in
            quarantine.
        """

        self.quarantined = True
        self._released_in = days

    def add_to_contacts(self, other: 'Member'):
        """
        Add 'other' to the list of recent contacts of the member.

        Parameters
        ----------
        other : 'Member'
            The member which is to be added to the list of recent contacts of
            the member on which 'add_to_contacts' is called.
        """

        if len(self.recent_contacts) < 5:
            self.recent_contacts += [other]

        else:
            self.recent_contacts = self.recent_contacts[1:] + [other]

    def make_immune(self, immunity_duration: int):
        """
        Make the member immune.
        Not to be used in the simulation other than in the initialization of
        groups ('put_inits_in_respective_group' in 'initialize_groups' in
        'start_iteration'in the Simulation-class).

        Parameters
        ----------
        immunity_duration : int
            Amount of days (in the simulation) the member is to be immune.

        Returns
        -------
        bool
            Whether the member was infected before it has been made immune.
            This should never be the case.
        """

        if self.infected: # This should never be the case.
            print("THIS SHOULD NOT APPEAR ON CONSOLE")
            self.infected = False
            self._infectious_in = -1
            self._recovers_in = -1
            self._susceptible_in = immunity_duration
            self.immune = True
            return True

        else:
            self._infectious_in = -1
            self._recovers_in = -1
            self._susceptible_in = immunity_duration
            self.immune = True
            return False

    def make_dead(self, timestamp: int):
        """
        Kill the member.

        Parameters
        ----------
        timestamp : int
            Current day in the simulation.
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

    def make_tick(self, option: str, timestamp: int = None):
        """
        Update the private counters of the member depending on the group or the
        stage the member is in.

        Parameters
        ----------
        option : str
            The type of update to be made.
            
        timestamp : int, optional
            Current day in the simulation.
            The default is None.

        Raises
        ------
        ValueError
            The 'option'-string is expected to be one of the following strings:
            'infectious',
            'immunity',
            'vaccine',
            'recover',
            'quarantine'.
            If 'option' does not meet the expectation,
            a ValueError will be raised.

        Returns
        -------
        bool
            Whether a change of attributes has been reached.
            No returns if option is 'vaccine'.
        """

        if option == "infectious":
            if self._infectious_in > 0:
                self._infectious_in -= 1

            return self._infectious_in <= 0

        elif option == "immunity":
            if self._immune_in <= 0:
                self._susceptible_in -= 1
                if self._susceptible_in == 0:
                    self._susceptible_in = -1
                    self.immune = False
                    return True

            return False

        elif option == "vaccine":
            self._immune_in -= 1
            if self._immune_in == 0:
                self.immune = True

        elif option == "recover":
            self._recovers_in -= 1
            if self._recovers_in == 0:
                self._recovers_in = -1
                self.infected = False
                self.immune = True
                return True

            return False

        elif option == "quarantine":
            self._released_in -= 1

            if "days_in_quarantine" not in self.properties.keys():
                self.properties["days_in_quarantine"] = 0

            self.properties["days_in_quarantine"] += 1

            if self._released_in == 0:
                if self.test(timestamp):
                    self._released_in += 5
                else:
                    self._released_in = -1
                    self.quarantined = False
                    return True

            return False

        else:
            raise ValueError("option not supported")

    def copy(self):
        """
        Create a member as a copy of the member.
        (Deep copy)

        Returns
        -------
        Member
            The copy of the member.
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
        Create a new group named 'name'.

        Parameters
        ----------
        name : str
            The name of the new group.
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

    def add_member(self, member: 'Member'):
        """
        Add 'member' to the group.

        Parameters
        ----------
        member : Member
            The member to be added to the group.
        """

        self.members = np.append(self.members, member)
        self.counter.increment()

    def remove_member(self, member: 'Member'):
        """
        Remove 'member' from the group.

        Parameters
        ----------
        member : Member
            The member to be removed from the group.
        """

        old_size = self.members.size
        self.members = self.members[self.members != member]
        self.counter.decrement(old_size - self.members.size)

    def spread_disease(self, infectant: 'Member', n: int, timestamp: int, disease_parameters: dict):
        """
        Let 'infectant' infect randomly chosen members of the group.

        Parameters
        ----------
        infectant : 'Member'
            The infectious member to be infecting members of the group.

        n : int
            The amount of members to be infected.

        timestamp : int
            Current day in the simulation.

        disease_parameters : dict
            Dictionary containing all relevant disease parameters.
            Is expected to contain the keys
            'infection_probability_heuristic',
            'incubation_period',
            'infection_period',
            'vaccine_failure_probability_heuristic',
            'immunity_period'.

        Returns
        -------
        list
            List of members which have been infected by 'infectant'.
        """

        result = []
        for other in np.random.choice(self.members, n):
            if infectant.properties["id"] == other.properties["id"]:
                continue

            infectant.add_to_contacts(other)

            if np.random.uniform() < disease_parameters["infection_probability_heuristic"](other.properties):
                if other.infect(infectant, timestamp, disease_parameters):
                    result += [other]

        return result

    def reset(self):
        """
        Reset the group.
        """

        self.members = np.array([])
        self.counter = Counter(0)

    @staticmethod
    def move(members: iter, origin: 'Group', destination: 'Group'):
        """
        Remove all members in 'members' from 'origin' and add them to
        'destination'.

        Parameters
        ----------
        members : iter
            Iterable of members to be moved from 'origin' to 'destination'.

        origin : Group
            Group from which the members in 'members' are to be removed.

        destination : Group
            Group to which the members i 'members' are to be added.
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
        Create a new household with household-id 'identifier'.

        Parameters
        ----------
        identifier : int
            The id of the new household.
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
        Create a new population named 'name'.

        Parameters
        ----------
        name : str
            The name of the new population.
        """

        super().__init__(name)
        self.households = {}

    def add_member(self, member: 'Member'):
        """
        Add 'member' as a Member to the Population
        and to its Household within the Population.

        Parameters
        ----------
        member : Member
            The member to be added to the Population.

        Raises
        ------
        KeyError
            The member 'member' is expected to have the property 'household'.
            If 'member' does not meet the expectations,
            a KeyError will be raised.
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

    def save_as_json(self, path: str):
        """
        Save the population as population.json at the given 'path'.

        Parameters
        ----------
        path : str
            The path at which the population-file is to be saved.

        Raises
        ------
        ValueError
            The population is expected to contain members.
            If the population does not meet the expectation,
            a ValueError will be raised.
        """

        if len(self.members) == 0:
            raise ValueError("Population can't be empty.")

        with open(path + "population.json", 'w') as f:
            wrapper = "{\n\t\"name\": \"" + self.name + "\",\n\t\"size\": " + str(self.size) + ",\n\t\"members\": [\n"
            inner = ""
            for member in self:
                json_str = json.dumps(member.properties, indent=4)
                inner += json_str + ', \n'

            inner = textwrap.indent(inner[:-3] + '\n', '\t\t')

            f.write(wrapper + inner + "\t]\n}")
            f.close()

    def copy(self):
        """
        Create a population as a copy of the population.
        (deep copy)

        Returns
        -------
        Population
            The copy of the population.
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
        Population
            A Population object containing the data for the given file.
        """

        p = Population(file_name[:-4])
        p.members = []
        with open(path + file_name, newline='') as f:
            progress = ProgressBar(1, sum(1 for _ in f) - 1)

        print("Loading population data...")

        progress.update(0)
        with open(path + file_name, newline='') as f:
            for m_dict in csv.DictReader(f):
                progress.update(1)
                p.add_member(Member(m_dict))

        print("\nFinished loading.")

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
        Population
            A Population object containing the data for the given file.
        """

        p = Population(file_name[:-5])
        p.members = []
        with open(path + file_name, "r") as f:
            print("Load json...")
            data = json.load(f)
            print("Finished loading.\n\nAdding members to population...")

            progress = ProgressBar(1, len(data["members"]))
            progress.update(0)
            for member in data["members"]:
                progress.update(1)
                p.add_member(Member(member))

            print("\nFinished adding members.")

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
