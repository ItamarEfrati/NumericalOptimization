import numpy as np

from unconstrained_min import Newton


class InteriorPT:

    def __init__(self,
                 func,
                 ineq_constraints,
                 eq_constraints_mat,
                 eq_constraints_rhs,
                 x0,
                 epsilon=1e-5,
                 newton_termination_threshold=1e-5,
                 max_iter=1000):
        """
        minimizes the function func subject to the list of inequality constraints specified by the Python list of
        functions ineq_constraints, and to the affine equality constraints 𝐴𝑥=𝑏 that are specified by the matrix
        eq_constraints_mat, and the right hand side vector eq_constraints_rhs. The outer iterations start at x0.
        :param func:
        :param ineq_constraints:
        :param eq_constraints_mat:
        :param eq_constraints_rhs:
        :param x0:
        """

        self.max_iter = max_iter
        self.func = func
        self.ineq_constraints = ineq_constraints
        self.eq_constraints_mat = eq_constraints_mat
        self.eq_constraints_rhs = eq_constraints_rhs
        self.mu = 10
        self.epsilon = epsilon
        self.newton_termination_threshold = newton_termination_threshold
        self.locations = [x0]
        self.f_values = []
        self.ineq_values = None
        self.success = False

    def calculate_ineq_constraints(self, x, return_desc=False):
        f_l, g_l, h_l = [], [], []
        for ineq in self.ineq_constraints:
            if return_desc:
                f, g, h, desc = ineq(x, return_desc)
                f = (f, desc)
            else:
                f, g, h = ineq(x, return_desc)
            f_l.append(f)
            g_l.append(g)
            h_l.append(h)

        return f_l, g_l, h_l

    def phi(self, x):
        f_p, g_p, h_p = self.calculate_ineq_constraints(x)
        f = - np.sum(np.log(np.multiply(-1, f_p)))
        f = np.inf if np.isnan(f) else f
        g = np.zeros((g_p[0].shape[0], 1))
        for i in range(len(g_p)):
            g += (-1 / f_p[i]) * g_p[i]

        h = np.zeros((g.shape[0], g.shape[0]))
        for i in range(len(h_p)):
            h += (1 / f_p[i] ** 2) * g_p[i] @ g_p[i].T
            h += (-1 / f_p[i]) * h_p[i]

        return f, g, h

    def solve_with_newton(self, t):
        def is_terminate():
            return (0.5 * p_k.T @ (h + h_p) @ p_k).item() < self.newton_termination_threshold

        def get_step_size():
            alpha = 2
            is_condition_satisfied = False
            c = 0.01
            f_l = f + f_p
            g_l = g + g_p
            while not is_condition_satisfied:
                alpha *= 0.5
                left_side, _, _ = self.func(x + alpha * p_k, t)
                l, _, _ = self.phi(x + alpha * p_k)
                left_side += l
                right_size = (f_l + c * alpha * g_l.T @ p_k).item()
                is_condition_satisfied = left_side <= right_size

            return alpha

        for _ in range(self.max_iter):
            x = self.locations[-1]
            f, g, h = self.func(x, t)
            f_p, g_p, h_p = self.phi(x)
            self.f_values.append((f, f + f_p, t))
            kkt_rhs = -(g + g_p)
            kkt_mat = h + h_p
            if self.eq_constraints_mat is not None:
                constraints_zero_rhs = np.zeros_like(self.eq_constraints_mat)[:, 0].reshape(-1, 1)
                kkt_rhs = np.concatenate([kkt_rhs, constraints_zero_rhs])
                kkt_mat = np.vstack([np.hstack((kkt_mat, self.eq_constraints_mat.T)),
                                     np.hstack((self.eq_constraints_mat, constraints_zero_rhs))])

            p_k = np.linalg.solve(kkt_mat, kkt_rhs)[:x.shape[0]]
            if is_terminate():
                self.ineq_values, _, _ = self.calculate_ineq_constraints(self.locations[-1], return_desc=True)
                break
            alpha = get_step_size()
            self.locations.append(x + p_k * alpha)

    def solve(self):
        t = 1 / self.mu
        m = max(len(self.ineq_constraints), 1)
        while (m / t) > self.epsilon:
            t = t * self.mu
            self.solve_with_newton(t)
