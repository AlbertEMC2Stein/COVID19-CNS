"""
TODO Docstring main \\(\\int_a^b f(x) \\ \\mathrm{d}x\\)
"""

from src.Network import Population
import numpy as np


if __name__ == "__main__":
    p = Population.load_from_csv("FromSampler_2021-11-28T13-18-51.csv")
    mask = np.array([bool(int(m.properties['infected'])) for m in p.members])

    p.save_as_json()

    print(*p.members[mask])

