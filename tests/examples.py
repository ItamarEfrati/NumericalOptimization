import numpy as np


def example_1(x: np.array, evaluate_hessian):
    Q = np.array([[1, 0], [0, 1]])
    f = x @ Q @ x
    g = 2 * Q @ x
    h = 2 * Q if evaluate_hessian else None

    return f, g, h


def example_2(x, evaluate_hessian):
    Q = np.array([[1, 0], [0, 100]])
    f = x @ Q @ x
    g = 2 * Q @ x
    h = 2 * Q if evaluate_hessian else None

    return f, g, h


def example_3(x, evaluate_hessian):
    A = np.array([[np.sqrt(3) / 2, -0.5], [0.5, np.sqrt(3) / 2]])
    Q = A.T @ np.array([[100, 0], [0, 1]]) @ A
    f = x @ Q @ x
    g = 2 * Q @ x
    h = 2 * Q if evaluate_hessian else None

    return f, g, h


def rosenbrock_function(x, evaluate_hessian):
    f = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    g = np.array([-400 * (x[0] * x[1] - x[0] ** 3) - 2 * (1 - x[0]),
                  200 * (x[1] - x[0] ** 2)])

    h = np.array([[-400 * x[0] + 1200 * x[1] ** 2 + 2, -400 * x[0]],
                  [-400 * x[0], 200]]) \
        if evaluate_hessian else None

    return f, g, h

def example_5(x, evaluate_hessian):
    pass
