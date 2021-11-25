import numpy.random as rnd

simple_dist = {
    "age": lambda: rnd.randint(0, 90),
    "infected": lambda: True if rnd.random() < 0.1 else False,
}