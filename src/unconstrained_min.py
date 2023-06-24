from abc import ABC, abstractmethod

import numpy as np


class LineSearch(ABC):
    def __init__(self,
                 f,
                 x0,
                 obj_tol,
                 param_tol,
                 max_iter,
                 name):
        """

        :param f: is the function minimized
        :param x0: is the starting point
        :param obj_tol: is the numeric tolerance for successful termination in terms of small enough change in
                        objective function values,
        :param param_tol: is the numeric tolerance for successful termination in terms of small enough distance
                          between two consecutive iterations iteration locations
        :param max_iter: is the maximum allowed number of iterations
        """

        self.f = f
        self.obj_tol = obj_tol
        self.param_tol = param_tol
        self.max_iter = max_iter
        self.locations = [x0]
        self.f_values = []
        self.success = False
        self.name = name

    def solve(self):
        print(f"Solving using {self.name}")
        f, g, _ = self.f(self.locations[-1], False)
        self.f_values.append(f)
        print(f"Iteration 0, current location {self.locations[-1]}, current value {self.f_values[-1]}")
        for i in range(self.max_iter):
            p_k = -g
            p_k = self.get_p_k(p_k)
            alpha = self.get_step_size(p_k, self.locations[-1], f, g)
            x_k_1 = self.locations[-1] + alpha * p_k
            self.locations.append(x_k_1)
            f, g, h = self.f(self.locations[-1].reshape(-1), False)
            self.f_values.append(f)
            if np.linalg.norm(self.locations[-2] - self.locations[-1]) < self.param_tol:
                self.success = True
                break
            if np.abs(self.f_values[-2] - self.f_values[-1]) < self.obj_tol:
                self.success = True
                break
            print(f"Iteration {i + 1}, current location {self.locations[-1]}, current value {self.f_values[-1]}")
        if self.success:
            print(f"Iteration {i + 1}, current location {self.locations[-1]}, current value {self.f_values[-1]}")

    def get_step_size(self, p_k, x_k, f_l, g_l):
        alpha = 2
        is_condition_satisfied = False
        c = 0.01
        while not is_condition_satisfied:
            alpha *= 0.5
            left_side, _, _ = self.f(x_k + alpha * p_k, False)
            right_size = (f_l + c * alpha * g_l.T @ p_k).item()
            is_condition_satisfied = left_side <= right_size

        return alpha

    @abstractmethod
    def get_p_k(self, p_k):
        pass


class GD(LineSearch):

    def __init__(self, f, x0, obj_tol, param_tol, max_iter):
        super().__init__(f, x0, obj_tol, param_tol, max_iter, 'Gradient Decent')

    def get_p_k(self, p_k):
        return p_k


class Newton(LineSearch):
    def __init__(self, f, x0, obj_tol, param_tol, max_iter):
        super().__init__(f, x0, obj_tol, param_tol, max_iter, 'Newton')

    def get_p_k(self, p_k):
        f, g, h = self.f(self.locations[-1], True)
        return np.linalg.pinv(h) @ p_k


class SR1(LineSearch):
    def __init__(self, f, x0, obj_tol, param_tol, max_iter):
        super().__init__(f, x0, obj_tol, param_tol, max_iter, 'SR1')
        self.b_k_list = [np.eye(x0.shape[0])]

    def get_p_k(self, p_k):
        if len(self.locations) == 1:
            return p_k
        x_k_1, x_k = self.locations[-1], self.locations[-2]
        y_k = (self.f(x_k_1, False)[1] - self.f(x_k, False)[1]).reshape(-1, 1)
        if np.all(y_k == 0):
            return p_k
        s_k = (x_k_1 - x_k).reshape(-1, 1)
        b_k = self.b_k_list[-1]
        exp_1 = y_k - b_k @ s_k
        numerator = exp_1 @ exp_1.T
        denominator = exp_1.T @ s_k

        b_k_1 = b_k + numerator / denominator
        self.b_k_list.append(b_k_1)

        return np.linalg.inv(b_k_1) @ p_k


class BFGS(LineSearch):
    def __init__(self, f, x0, obj_tol, param_tol, max_iter):
        super().__init__(f, x0, obj_tol, param_tol, max_iter, 'BFGS')
        self.b_k_list = [np.eye(x0.shape[0])]

    def get_p_k(self, p_k):
        if len(self.locations) == 1:
            return p_k
        x_k_1, x_k = self.locations[-1], self.locations[-2]
        y_k = (self.f(x_k_1, False)[1] - self.f(x_k, False)[1]).reshape(-1, 1)
        if np.all(y_k == 0):
            return p_k
        s_k = (x_k_1 - x_k).reshape(-1, 1)
        b_k = self.b_k_list[-1]
        exp_1 = (b_k @ s_k @ s_k.T @ b_k.T) / (s_k.T @ b_k @ s_k)
        exp_2 = (y_k @ y_k.T) / (y_k.T @ s_k)

        b_k_1 = b_k - exp_1 + exp_2

        self.b_k_list.append(b_k_1)

        return np.linalg.inv(b_k_1) @ p_k
