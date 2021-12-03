import numpy as np


class Counter:
    """
    TODO Docstring Counter
    """

    def __init__(self, start):
        """
        TODO Docstring Counter __init__
        """
        self.n = start
        self.history = np.array([start])

    def get_count(self):
        """
        TODO Docstring Counter get_count
        """
        return self.n

    def step(self, mode: str, k: int = 1, return_when: str = 'after'):
        """
        TODO Docstring Counter step
        """
        if mode not in ["inc", "dec"]:
            raise ValueError(str(mode) + " is not a valid value. Try 'inc' or 'dec'.")

        old = self.n
        self.n = self.n + k if mode == "inc" else max(0, self.n - k)
        self.history = np.append(self.history, self.n)

        if return_when == 'after':
            return self.n
        elif return_when == 'before':
            return old
        else:
            raise ValueError(str(return_when) + " is not a valid value. Try 'after' or 'before'.")
