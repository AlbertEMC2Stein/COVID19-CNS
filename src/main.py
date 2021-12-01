from src.Network.Population import Population
import numpy as np

if __name__ == "__main__":
    p = Population.load_from_csv("DE_03_KLLand.csv")
    mask = np.array([bool(int(m.properties['gender'])) for m in p.members])

    print(*p.members[mask])

