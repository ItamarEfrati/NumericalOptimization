import unittest

import numpy as np

from constrained_min import InteriorPT
from tests.examples import QPExample, LPExample
from utils import plot_qp_example, plot_lp_example


class TestUnconstrained(unittest.TestCase):

    def test_qp(self):
        func = QPExample.func
        ineq_constraints = [QPExample.ineq_1, QPExample.ineq_2, QPExample.ineq_3]
        x_0 = np.array([0.1, 0.2, 0.7]).reshape(-1, 1)
        iterior_pt = InteriorPT(func, ineq_constraints, QPExample.eq_constraints_mat, QPExample.eq_constraints_rhs, x_0)
        iterior_pt.solve()
        plot_qp_example(iterior_pt.locations, iterior_pt.f_values, iterior_pt.ineq_values)

    def test_lp(self):
        func = LPExample.func
        ineq_constraints = [LPExample.ineq_1, LPExample.ineq_2, LPExample.ineq_3, LPExample.ineq_4]
        x_0 = np.array([0.5, 0.75]).reshape(-1, 1)
        iterior_pt = InteriorPT(func, ineq_constraints, LPExample.eq_constraints_mat, LPExample.eq_constraints_rhs, x_0)
        iterior_pt.solve()
        plot_lp_example(iterior_pt.locations, iterior_pt.f_values, iterior_pt.ineq_values)


if __name__ == '__main__':
    unittest.main()
