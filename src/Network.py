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
        TODO Docstring Population add_member
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
    def load_from_csv(file_name: str, path: str = "Populations" + sep):
        """
        TODO Docstring Population load_from_csv
        """

        p = Population(file_name[:-4])
        p.members = []

        with open(path + file_name, newline='') as f:
            progress = ProgressBar(1, 1, sum(1 for _ in f) - 1)

        progress.update(0)
        with open(path + file_name, newline='') as f:
            for m_dict in csv.DictReader(f):
                progress.update(1)
                p.add_member(Member(m_dict))

        p.members = np.array(p.members)
        return p

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
