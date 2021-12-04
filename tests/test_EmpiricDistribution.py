from unittest import TestCase
from src.Utils import EmpiricDistribution
import numpy as np


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
        def _quantile(p):
            c = lambda k: sum(TestEmpiricDistribution.dist.probabilities[i] for i in range(k + 1))
            return max([i * (c(i - 1) < p <= c(i)) for i in range(5)])

        for _ in range(100):
            p = np.random.uniform()
            q = TestEmpiricDistribution.dist.quantile(p)
            _q = TestEmpiricDistribution.dist.keys[_quantile(p)]
            self.assertEqual(q, _q)

    def test_pick(self):
        for _ in range(100):
            shp = np.random.randint(0, 11, size=4)
            p = TestEmpiricDistribution.dist.pick(size=shp)
            self.assertTrue(np.all(np.array(p.shape) == shp))
