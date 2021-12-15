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

    def __str__(self):
        return str(self.properties)


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

        self.members = self.members[self.members != member]
        self.counter.decrement()

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

    def add_member(self, member: Member):
        """
        Adds member as a Member to the Population
        and to its Household within the Population.

        Parameters
        ----------
        member : Member
            The Member to be added to the Population.

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


################################################################################################
################################################################################################
################################################################################################
