import csv
import json
import numpy as np
from os.path import sep
from src.Utils.ProgressBar import ProgressBar
from .Member import Member
from .Household import Household


class Population:
    def __init__(self, name: str):
        self.name = name
        self.members = np.array([])
        self.households = {}

    def __str__(self):
        result = "Population: " + self.name + "\n"
        for household in self.households.values():
            result += str(household) + "\n"

        return result[:-1]

    def add_member(self, member: Member):
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
        p = Population(file_name[:-4])
        p.members = []

        with open(path + file_name, newline='') as f:
            progress = ProgressBar(1, 1, sum(1 for _ in f) - 1)

        with open(path + file_name, newline='') as f:
            for m_dict in csv.DictReader(f):
                progress.update(1)
                p.add_member(Member(m_dict))

        p.members = np.array(p.members)
        return p

    def save_as_json(self, path: str = ".." + sep + "out" + sep + "Simulated" + sep):
        if len(self.members) == 0:
            raise ValueError("Population can't be empty.")

        with open(path + self.name + ".csv", 'w') as f:
            headers = self.members[0].properties.keys()
            rows = np.array([list(member.properties.values()) for member in self.members], dtype=int)

            f.write(','.join(headers) + '\n')
            np.savetxt(f, rows, fmt='%d', delimiter=',')


