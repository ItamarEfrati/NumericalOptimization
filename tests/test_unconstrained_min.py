import unittest

import numpy as np

from tests.examples import example_1, example_2, example_3, rosenbrock_function, example_5, example_6
from unconstrained_min import GD, Newton, SR1, BFGS
from utils import plot_example


class TestStringMethods(unittest.TestCase):
    x_0 = np.array([1, 1]).reshape(-1, 1)

    def test_examples(self):

        for i, example in enumerate([example_1, example_2, example_3, example_5, example_6]):
            n = i + 1 if i < 3 else i + 2
            print(f"Example {n}")
            methods = {
                'Gradient Decent': GD(example, self.x_0, obj_tol=10e-8, param_tol=10e-12, max_iter=100),
                'Newton': Newton(example, self.x_0, obj_tol=10e-8, param_tol=10e-12, max_iter=100),
                'SR1': SR1(example, self.x_0, obj_tol=10e-8, param_tol=10e-12, max_iter=100),
                'BFGS': BFGS(example, self.x_0, obj_tol=10e-8, param_tol=10e-12, max_iter=100)
            }
            for optimizer in methods.values():
                optimizer.solve()

            plot_example(methods, example, f'example_{n}')

    def test_rosenbrock_function(self):
        print(f"Example Rosenbrock")
        x_0 = np.array([-1, 2]).reshape(-1, 1)
        methods = {
            'Gradient Decent': GD(rosenbrock_function, x_0, obj_tol=10e-8, param_tol=10e-12, max_iter=10_000),
            'Newton': Newton(rosenbrock_function, x_0, obj_tol=10e-8, param_tol=10e-12, max_iter=100),
            'SR1': SR1(rosenbrock_function, x_0, obj_tol=10e-8, param_tol=10e-12, max_iter=100),
            'BFGS': BFGS(rosenbrock_function, x_0, obj_tol=10e-8, param_tol=10e-12, max_iter=100)
        }
        for optimizer in methods.values():
            optimizer.solve()
        plot_example(methods, rosenbrock_function, 'rosenback')


if __name__ == '__main__':
    unittest.main()
