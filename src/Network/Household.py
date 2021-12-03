import numpy as np
from .Member import Member


class Household:
    """
    TODO Docstring Household
    """

    internal_reproduction_number = 1

    def __init__(self, id: int):
        """
        TODO Docstring Household __init__
        """
        self.members = np.array([])
        self.id = id
        self.internal_reproduction_number = 1

    def __str__(self):
        result = "Household " + self.id + ":\n"
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
