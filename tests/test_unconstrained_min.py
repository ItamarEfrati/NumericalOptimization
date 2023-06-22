import unittest

import numpy as np

from tests.examples import example_1, example_2, example_3, rosenbrock_function
from unconstrained_min import GD, Newton, SR1, BFGS
from utils import plot_example_1, plot_example_2, plot_example_3, plot_rosenbrock_function


class TestStringMethods(unittest.TestCase):

    def test_example_1(self):
        x_0 = np.array([1, 1])
        methods = {
            'Gradient Decent': GD(example_1, x_0, obj_tol=10e-8, param_tol=10e-12, max_iter=100),
            'Newton': Newton(example_1, x_0, obj_tol=10e-8, param_tol=10e-12, max_iter=100),
            'SR1': SR1(example_1, x_0, obj_tol=10e-8, param_tol=10e-12, max_iter=100),
            'BFGS': BFGS(example_1, x_0, obj_tol=10e-8, param_tol=10e-12, max_iter=100)
        }
        for optimizer in methods.values():
            optimizer.solve()
        plot_example_1(methods)

    def test_example_2(self):
        x_0 = np.array([1, 1])
        methods = {
            'Gradient Decent': GD(example_2, x_0, obj_tol=10e-8, param_tol=10e-12, max_iter=100),
            'Newton': Newton(example_2, x_0, obj_tol=10e-8, param_tol=10e-12, max_iter=100),
            'SR1': SR1(example_2, x_0, obj_tol=10e-8, param_tol=10e-12, max_iter=100),
            'BFGS': BFGS(example_2, x_0, obj_tol=10e-8, param_tol=10e-12, max_iter=100)
        }
        for optimizer in methods.values():
            optimizer.solve()
        plot_example_2(methods)

    def test_example_3(self):
        x_0 = np.array([1, 1])
        methods = {
            'Gradient Decent': GD(example_3, x_0, obj_tol=10e-8, param_tol=10e-12, max_iter=100),
            'Newton': Newton(example_3, x_0, obj_tol=10e-8, param_tol=10e-12, max_iter=100),
            'SR1': SR1(example_3, x_0, obj_tol=10e-8, param_tol=10e-12, max_iter=100),
            'BFGS': BFGS(example_3, x_0, obj_tol=10e-8, param_tol=10e-12, max_iter=100)
        }
        for optimizer in methods.values():
            optimizer.solve()
        plot_example_3(methods)

    def test_rosenbrock_function(self):
        x_0 = np.array([-1, 2])
        methods = {
            'Gradient Decent': GD(rosenbrock_function, x_0, obj_tol=10e-8, param_tol=10e-12, max_iter=10_000),
            'Newton': Newton(rosenbrock_function, x_0, obj_tol=10e-8, param_tol=10e-12, max_iter=100),
            'SR1': SR1(rosenbrock_function, x_0, obj_tol=10e-8, param_tol=10e-12, max_iter=100),
            'BFGS': BFGS(rosenbrock_function, x_0, obj_tol=10e-8, param_tol=10e-12, max_iter=100)
        }
        for optimizer in methods.values():
            optimizer.solve()
        plot_rosenbrock_function(methods)


if __name__ == '__main__':
    unittest.main()
