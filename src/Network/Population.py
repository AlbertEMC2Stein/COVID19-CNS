import numpy as np
from os.path import sep


class Population:
    def __init__(self, name):
        self.name = name
        self.members = np.array([])
        self.households = {}

    def add_member(self, member):
        self.members = np.append(self.members, member)
        

    @staticmethod
    def load_from_csv(file_name, path="src" + sep + "Populations" + sep):
        data = np.genfromtxt(path + file_name, delimiter=',', dtype=str)
        return Population(file_name[:-4])

    def save_as_csv(self, path="out" + sep + "Simulated" + sep):
        if len(self.members) == 0:
            raise ValueError("Population can't be empty.")

        with open(path + self.name + ".csv", 'w') as f:
            headers = self.members[0].properties.keys()
            rows = np.array([list(member.properties.values) for member in self.members])

            f.write(','.join(headers) + '\n')
            np.savetxt(f, rows, fmt='%d', delimiter=',')
