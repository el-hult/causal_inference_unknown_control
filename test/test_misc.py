import unittest
import itertools

import numpy as np

import lib.misc


class TestCrossMoment(unittest.TestCase):
    def test_std_normal(self):
        n = 2_000_000
        np.random.seed(0)
        data = np.random.multivariate_normal(mean=np.zeros(2), cov=np.eye(2), size=n)
        cm4 = lib.misc.cross_moment_4(data)
        self.check_independent(cm4, 0, 1, 0, 3)

    def test_normal(self):
        n = 10_000_000
        np.random.seed(0)
        data = np.random.multivariate_normal(mean=np.ones(2), cov=np.eye(2), size=n)
        cm4 = lib.misc.cross_moment_4(data)
        self.check_independent(cm4, 1, 1, 0, 3)

    def test_normal_2(self):
        n = 10_000_000
        np.random.seed(0)
        data = np.random.multivariate_normal(
            mean=0.1 * np.ones(2), cov=0.5 * np.eye(2), size=n
        )
        cm4 = lib.misc.cross_moment_4(data)
        self.check_independent(cm4, 0.1, 0.5, 0, 3)

    @staticmethod
    def check_independent(cm4, mean, var, skew, kurt):
        """Check cross moment is as expected, given iid data
        If X and Y are iid, and you draw (x,y) pairs, check  kurtosis is as expected
        have really high tolerance, since these things converge slowly....

        notation is as on wikiepdia:
        - mu is central moment
        - mu_prime is moment around origin
        - mu_tilde is standardized moments

        and cm4 is the cross moment around the origin.
        So we need to convert all properties to mu_prime-variables

        Formulas from
        https://en.wikipedia.org/wiki/Central_moment
        https://en.wikipedia.org/wiki/Standardized_moment
        """
        mu_tilde_1 = 0
        mu_tilde_2 = 1
        mu_tilde_3 = skew
        mu_tilde_4 = kurt

        std = np.sqrt(var)
        mu_1 = mu_tilde_1 * std ** 1
        mu_2 = mu_tilde_2 * std ** 2
        mu_3 = mu_tilde_3 * std ** 3
        mu_4 = mu_tilde_4 * std ** 4

        mu_prime_1 = mean + mu_1
        mu_prime_2 = mu_2 + mu_prime_1 ** 2
        mu_prime_3 = mu_3 + 3 * mu_prime_1 * mu_prime_2 - 2 * mu_prime_1 ** 3
        mu_prime_4 = (
            mu_4
            + 4 * mu_prime_1 ** 1 * mu_prime_3
            - 6 * mu_prime_1 ** 2 * mu_prime_2
            + 3 * mu_prime_1 ** 3 * mu_prime_1
        )

        for i, j, k, l in itertools.product([0, 1], repeat=4):
            tot = i + j + k + l
            actual = cm4[i, j, k, l]
            if tot == 4 or tot == 0:
                desired = mu_prime_4
            elif tot == 2:
                desired = mu_prime_2 * mu_prime_2
            elif tot == 3 or tot == 1:
                desired = mu_prime_1 * mu_prime_3
            else:
                raise RuntimeError
            np.testing.assert_almost_equal(actual, desired, decimal=2)


if __name__ == "__main__":
    unittest.main()
