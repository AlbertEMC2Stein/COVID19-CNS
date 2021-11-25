from unittest import TestCase

from src.Utils.EmpiricDistribution import EmpiricDistribution


class TestEmpiricDistribution(TestCase):
    data = {'red': 13,
            'blue': 34,
            'yellow': 16,
            'white': 27,
            'black': 10}

    dist = EmpiricDistribution(data)

    def test_cdf(self):
        self.assertAlmostEqual(TestEmpiricDistribution.dist._EmpiricDistribution__cdf[-1], 1)

    def test_quantile(self):
        self.fail()

    def test_pick(self):
        self.fail()