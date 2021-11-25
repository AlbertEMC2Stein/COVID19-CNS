import numpy as np


class Household:
    internal_reproduction_number = 1

    def __init__(self):
        self.members = np.array([])

    def __str__(self):
        result = "Members:\n"
        for member in self.members:
            result += str(member) + "\n"

        return result

    def add_member(self, member):
        self.members = np.append(self.members, member)