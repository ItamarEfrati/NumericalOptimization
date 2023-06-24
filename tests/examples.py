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
