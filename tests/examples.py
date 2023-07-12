import numpy as np


def example_1(x: np.array, evaluate_hessian):
    Q = np.array([[1, 0], [0, 1]])
    f = (x.T @ Q @ x).item()
    g = (2 * Q @ x).reshape(-1, 1)
    h = 2 * Q if evaluate_hessian else None

    return f, g, h


def example_2(x, evaluate_hessian):
    Q = np.array([[1, 0], [0, 100]])
    f = (x.T @ Q @ x).item()
    g = (2 * Q @ x).reshape(-1, 1)
    h = 2 * Q if evaluate_hessian else None

    return f, g, h


def example_3(x, evaluate_hessian):
    A = np.array([[np.sqrt(3) / 2, -0.5], [0.5, np.sqrt(3) / 2]])
    Q = A.T @ np.array([[100, 0], [0, 1]]) @ A
    f = (x.T @ Q @ x).item()
    g = (2 * Q @ x).reshape(-1, 1)
    h = 2 * Q if evaluate_hessian else None

    return f, g, h


def rosenbrock_function(x, evaluate_hessian):
    f = (100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2).item()
    g = np.array([-400 * (x[0] * x[1] - x[0] ** 3) - 2 * (1 - x[0]),
                  200 * (x[1] - x[0] ** 2)]).reshape(-1, 1)

    h = np.array([[-400 * x[0].item() + 1200 * x[1].item() ** 2 + 2, -400 * x[0].item()],
                  [-400 * x[0].item(), 200]]).reshape(2, 2) \
        if evaluate_hessian else None

    return f, g, h


def example_5(x, evaluate_hessian):
    a = np.array([2, -3]).reshape(-1, 1)
    f = (a.T @ x).item()
    g = a.reshape(-1, 1)
    h = np.array([[0, 0], [0, 0]])
    return f, g, h


def example_6(x, evaluate_hessian):
    f = (np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) + np.exp(-x[0] - 0.1)).item()

    x_0_g = np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) - np.exp(-x[0] - 0.1)
    x_1_g = 3 * np.exp(x[0] + 3 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1)
    g = np.array([x_0_g, x_1_g]).reshape(-1, 1)
    h = None
    if evaluate_hessian:
        x_00_h = np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) + np.exp(-x[0] - 0.1)
        x_01_h = 3 * np.exp(x[0] + 3 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1)
        x_10_h = 3 * np.exp(x[0] + 3 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1)
        x_11_h = 9 * np.exp(x[0] + 3 * x[1] - 0.1) + 9 * np.exp(x[0] - 3 * x[1] - 0.1)

        h = np.array([[x_00_h, x_01_h], [x_10_h, x_11_h]]).reshape(2, 2)
    return f, g, h


def log_barrier_function(x, ineq_constraints):
    functions_values = list(map(lambda func: func(x), ineq_constraints))
    f = sum(map(lambda a: -np.log(a), functions_values))
    g = sum(map(lambda a: -1 / a, functions_values))


class QPExample:
    eq_constraints_mat = np.array([1, 1, 1]).reshape(1, 3)
    eq_constraints_rhs = np.array([1]).reshape(-1, 1)

    @staticmethod
    def ineq_1(x, return_desc=False):
        description = '-x <= 0'
        f = -x[0]
        g = np.array([-1, 0, 0]).reshape(-1, 1)
        h = np.zeros((3, 3))
        if return_desc:
            return f, g, h, description
        return f, g, h

    @staticmethod
    def ineq_2(x, return_desc=False):
        description = '-y <= 0'
        f = -x[1]
        g = np.array([0, -1, 0]).reshape(-1, 1)
        h = np.zeros((3, 3))
        if return_desc:
            return f, g, h, description
        return f, g, h

    @staticmethod
    def ineq_3(x, return_desc=False):
        description = '-z <= 0'
        f = -x[2]
        g = np.array([0, 0, -1]).reshape(-1, 1)
        h = np.zeros((3, 3))
        if return_desc:
            return f, g, h, description
        return f, g, h

    @staticmethod
    def func(x, t, return_desc=False):
        description = 'min x^2 + y^2 + (z+1)^2'
        f = t * (x[0] ** 2 + x[1] ** 2 + (x[2] + 1) ** 2)
        g = (t * np.array([2 * x[0], 2 * x[1], 2 * (x[2] + 1)])).reshape(-1, 1)
        h = t * np.diag([2, 2, 2])
        if return_desc:
            return f, g, h, description
        return f, g, h


class LPExample:
    eq_constraints_mat = None
    eq_constraints_rhs = None

    @staticmethod
    def ineq_1(x, return_desc=False):
        description = 'x - 2 <= 0'
        f = x[0] - 2
        g = np.array([1, 0]).reshape(-1, 1)
        h = np.zeros((2, 2))
        if return_desc:
            return f, g, h, description
        return f, g, h

    @staticmethod
    def ineq_2(x, return_desc=False):
        description = 'y - 1 <= 0'
        f = x[1] - 1
        g = np.array([0, 1]).reshape(-1, 1)
        h = np.zeros((2, 2))
        if return_desc:
            return f, g, h, description
        return f, g, h

    @staticmethod
    def ineq_3(x, return_desc=False):
        description = '-y <= 0'
        f = -x[1]
        g = np.array([0, -1]).reshape(-1, 1)
        h = np.zeros((2, 2))
        if return_desc:
            return f, g, h, description
        return f, g, h

    @staticmethod
    def ineq_4(x, return_desc=False):
        description = "-y -x + 1  <= 0"
        f = -x[1] - x[0] + 1
        g = np.array([-1, -1]).reshape(-1, 1)
        h = np.zeros((2, 2))
        if return_desc:
            return f, g, h, description
        return f, g, h

    @staticmethod
    def func(x, t, return_desc=False):
        description = 'max x + y'
        f = t * (-x[0] - x[1])
        g = (t * np.array([-1, -1])).reshape(-1, 1)
        h = t * np.zeros((2, 2))
        if return_desc:
            return f, g, h, description
        return f, g, h
