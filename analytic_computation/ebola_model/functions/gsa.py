from scipy.optimize import root
import numpy as np

from ebola_model.functions.probability import PMO

model = PMO(0.7, 0.4/0.7, 173/27, 28/173, 5.9, 0.25, 4, 0.2)

class Model:
    def __init__(self):
        self.q_c_values = []
        self.q_h_values = []
        self.p_c_values = []
        self.p_h_values = []

    def find_p_q(self, solution):
        self.q_c_values.append(solution[0])
        self.q_h_values.append(solution[2])
        if 1 - solution[0] < 0:
            self.p_c_values.append(0.0)
        else:
            self.p_c_values.append(1.0 - solution[0])
        if 1 - solution[2] < 0:
            self.p_h_values.append(0.0)
        else:
            self.p_h_values.append(1.0 - solution[2])
        return self.p_c_values, self.p_h_values

    @staticmethod
    def evaluate(X, r_f=5.9):
        Y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            def equations(vars):
                x, y, z = vars
                eq1 = (X[i, 0] * (1 - X[i, 4])) / (X[i, 0] + 1) * x**2 +\
                      (X[i, 0] * X[i, 4]) / (X[i, 0] + 1) * x * z +\
                      0.7*X[i, 1] / (X[i, 0] + 1) * y +\
                      (1 - 0.7*X[i, 1]) / (X[i, 0] + 1) - x
                eq2 = (r_f * (1 - X[i, 4])) / (r_f + 1) * x * y +\
                      (r_f * X[i, 4]) / (r_f + 1) * y * z + 1 / (r_f + 1) - y
                eq3 = (X[i, 2] * (1 - X[i, 4])) / (X[i, 2] + X[i, 3] + 1) *x*z\
                      + (X[i, 4] * X[i, 2] + X[i, 3]) /\
                        (X[i, 2] + X[i, 3] + 1) * z**2 +\
                      1 / (X[i, 2] + X[i, 3] + 1) - z
                return [eq1, eq2, eq3]

            solution = root(equations, [0.5, 0.5, 0.5], method='lm')
            p_c_values, p_h_values = model.find_p_q(solution=solution.x)
            Y[i] = p_c_values[i]
        return Y
    
    def evaluate_h(self, h, X, r_f=5.9):
        Y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            def equations(vars):
                x, y, z = vars
                eq1 = (X[i, 0] * (1 - h)) / (X[i, 0] + 1) * x**2 +\
                      (X[i, 0] * h) / (X[i, 0] + 1) * x * z +\
                      0.7*X[i, 1] / (X[i, 0] + 1) * y +\
                      (1 - 0.7*X[i, 1]) / (X[i, 0] + 1) - x
                eq2 = (r_f * (1 - h)) / (r_f + 1) * x * y +\
                      (r_f * h) / (r_f + 1) * y * z + 1 / (r_f + 1) - y
                eq3 = (X[i, 2] * (1 - h)) / (X[i, 2] + X[i, 3] + 1) * x * z +\
                      (h * X[i, 2] + X[i, 3]) / (X[i, 2] + X[i, 3] + 1) * z**2\
                      + 1 / (X[i, 2] + X[i, 3] + 1) - z
                return [eq1, eq2, eq3]

            solution = root(equations, [0.5, 0.5, 0.5], method='lm')
            self.find_p_q(solution=solution.x)
            Y[i] = self.p_c_values[i]  # Use index i instead of 0
        return Y