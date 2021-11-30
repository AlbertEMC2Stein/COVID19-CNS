import numpy as np
from os.path import sep
import csv
from Individual import Individual
from Household import Household


class Population:
    def __init__(self, name):
        self.name = name
        self.members = np.array([])
        self.households = {}


    def add_member(self, member, from_csv = False):
        person = member
        if not type(person) == Individual:
            person = Individual(person)
        try:
            household_id = person.properties["household"]
        except:
            if from_csv:
                raise KeyError("The csv-file is expected to have a \
                               'household' column.")
            else:
                raise KeyError("Every member or person is expected to have a \
                               'household' key.")
        if household_id in self.households.keys():
            self.households[household_id].add_member(person)
        else:
            household = Household(household_id)
            self.households[household_id] = household
            self.households[household_id].add_member(person)
        self.members = np.append(self.members, person)


    def load_from_csv(self, csv_file):
        with open(csv_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                person = Individual(row)
                self.add_member(person, from_csv = True)


    def save_as_csv(self, path="out" + sep + "Simulated" + sep):
        if len(self.members) == 0:
            raise ValueError("Population can't be empty.")

        with open(path + self.name + ".csv", 'w') as f:
            headers = self.members[0].properties.keys()
            rows = np.array([list(member.properties.values) for member in self.members])

            f.write(','.join(headers) + '\n')
            np.savetxt(f, rows, fmt='%d', delimiter=',')

