"""
TODO Docstring Network
"""

__all__ = ['Member', 'Household', 'Population']

import textwrap

import numpy as np
import csv
import json
from os.path import sep
from src.Utils import ProgressBar


class Member:
    def __init__(self, properties: dict):
        """
        TODO Docstring Member __init__
        """

        self.properties = {}
        for property, value in properties.items():
            self.properties[property] = value

    def __str__(self):
        return str(self.properties)


################################################################################################
################################################################################################
################################################################################################


class Household:
    internal_reproduction_number = 1

    def __init__(self, id: int):
        """
        TODO Docstring Household __init__
        """

        self.members = np.array([])
        self.id = id
        self.internal_reproduction_number = 1

    def __str__(self):
        result = "Household " + str(self.id) + ":\n"
        for member in self.members:
            result += str(member) + "\n"

        return result[:-1]

    def __len__(self):
        return self.members.size

    def add_member(self, member: Member):
        """
        TODO Docstring Household add_member
        """

        self.members = np.append(self.members, member)


################################################################################################
################################################################################################
################################################################################################


class Population:
    def __init__(self, name: str):
        """
        TODO Docstring Population __init__
        """

        self.name = name
        self.members = np.array([])
        self.households = {}

    def __str__(self):
        result = "Population: " + self.name + "\n"
        for household in self.households.values():
            result += str(household) + "\n"

        return result[:-1]

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

        try:
            household_id = member.properties["household"]
        except KeyError:
            raise KeyError("Every individual is expected to have a 'household' property.")

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

    @staticmethod
    def load_from_csv(file_name: str, path: str = "Populations" + sep, progress: bool = False):
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
        progress : bool, optional
            Enables or disables the printing of a progress bar.
            The default is False.

        Returns
        -------
        p : Population
            A Population object containing the data for the given file.
        """

        p = Population(file_name[:-4])
        p.members = []

        if progress:
            with open(path + file_name, newline='') as f:
                progress = ProgressBar(1, 1, sum(1 for _ in f) - 1)

            progress.update(0)
        with open(path + file_name, newline='') as f:
            for m_dict in csv.DictReader(f):
                if progress:
                    progress.update(1)
                p.add_member(Member(m_dict))

        p.members = np.array(p.members)
        return p

    @staticmethod
    def load_from_json(file_name: str, path: str = "Populations" + sep, progress: bool = False):
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
        progress : bool, optional
            Enables or disables the printing of a progress bar.
            The default is False.

        Returns
        -------
        p : Population
            A Population object containing the data for the given file.
        """

        p = Population(file_name[:-5])
        p.members = []
        with open(path + file_name, "r") as f:
            print("Load json.")
            data = json.load(f)
            print("Finished loading.")

            progress = ProgressBar(1, 1, len(data["members"]))
            progress.update(0)
            for member in data["members"]:
                progress.update(1)
                p.add_member(Member(member))

        p.members = np.array(p.members)
        return p

    @staticmethod
    def load_from_file(file_name: str, path: str = "Populations" + sep, progress: bool = False):
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
        progress : bool, optional
            Enables or disables the printing of a progress bar.
            The default is False.

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
            return Population.load_from_csv(file_name, path, progress)
        elif file_name[-5:] == ".json":
            return Population.load_from_json(file_name, path, progress)
        else:
            raise ValueError("file_name must end in .csv or .json.")

    def save_as_json(self, path: str = ".." + sep + "out" + sep + "Simulated" + sep):
        """
        TODO Docstring Population save_as_json
        """

        if len(self.members) == 0:
            raise ValueError("Population can't be empty.")

        with open(path + self.name + ".json", 'w') as f:
            wrapper = "{\n\t\"name\": \"" + self.name + "\",\n\t\"members\": [\n"
            inner = ""
            for member in self.members:
                json_str = json.dumps(member.properties, indent=4)
                inner += json_str + ', \n'

            inner = textwrap.indent(inner[:-3] + '\n', '\t\t')

            f.write(wrapper + inner + "\t]\n}")
            f.close()


################################################################################################
################################################################################################
################################################################################################
