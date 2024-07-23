from scipy.optimize import root
import numpy as np

class PMO:
    def __init__(self, d, f, N, q, phi, lambda_h, beta, alpha):
        self.d = d
        self.f = f
        self.N = N
        self.q = q
        self.phi = phi
        self.lambda_h = lambda_h
        self.beta = beta
        self.alpha = alpha
        self.q_c_values = []
        self.q_h_values = []
        self.p_c_values = []
        self.p_h_values = []

    def find_p_q(self, solution):
        """Method calculates a list of probabilities of a major outbreak given
        the probability that an outbreak does not occur.

        Parameters
        ----------
        solution : array
            Probabilities that an outbreak occurs and is treated
            initially in the community and in a healthcare facility

        Returns
        -------
        tuple[list, list]
            List of probabilities that an outbreak occurs and is treated
            initially in the community and in a healthcare facility
        """
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
    
    def find_p_q_combination(self, solution):
        """Method calculates the probability of a major outbreak given the
        probability that an outbreak does not occur.

        Parameters
        ----------
        solution : array
            Probabilities that an outbreak does not occur after being treated
            initially in the community and in a healthcare facility

        Returns
        -------
        tuple[float, float]
            Probabilities that an outbreak occurs and is treated
            initially in the community and in a healthcare facility
        """
        self.q_c_values.append(solution[0])
        self.q_h_values.append(solution[2])
        if 1 - solution[0] < 0:
            self.p_c_value = 0.0
        else:
            self.p_c_value = 1.0 - solution[0]
        if 1 - solution[2] < 0:
            self.p_h_value = 0.0
        else:
            self.p_h_value= 1.0 - solution[2]
        return self.p_c_value, self.p_h_value

    def variables(self):
        """Method calculates the parameters of the model.
        """
        r_c = self.N * self.q
        p_b = self.d * self.f
        r_f = self.phi
        r_v = self.lambda_h
        r_w = self.N * self.q * self.beta * self.alpha
        return r_c, p_b, r_f, r_v, r_w

    def pmo(self, r_c, p_b, r_f, r_v, r_w, h):
        """Method calculates the probability of a major outbreak given the
        variables of the model.

        Parameters
        ----------
        r_c : float
            Average expected number of infections within the community
        p_b : float
            Probability of an unsafe burial
        r_f : float
            Average expected number of infections from an unsafe burial
        r_v : float
            Average expected number of infections of healthcare facility visitors
        r_w : float
            Average expected number of infections of healthcare workers
        h : float
            Probability of hospitalisation
        """
        def equations(vars):
            x, y, z = vars
            eq1 = (r_c * (1 - h)) / (r_c + 1) * x ** 2 +\
                  (r_c * h) / (r_c + 1) * x * z + p_b / (r_c + 1) * y +\
                  (1 - p_b) / (r_c + 1) - x
            eq2 = (r_f * (1 - h)) / (r_f + 1) * x * y +\
                  (r_f * h) / (r_f + 1) * y * z + 1 / (r_f + 1) - y
            eq3 = (r_v * (1 - h)) / (r_v + r_w + 1) * x * z +\
                  (h * r_v + r_w) / (r_v + r_w + 1) * z ** 2 +\
                  1 / (r_v + r_w + 1) - z
            return [eq1, eq2, eq3]

        solution = root(equations, [0.5, 0.5, 0.5], method='lm')
        self.find_p_q(solution.x)
        return self.p_c_values, self.p_h_values
    
    def pmo_burial_compliance(self, alpha, r, h=0.6):
        """Method calculates the probability of a major outbreak for varying
        levels of burial compliance and given effectiveness of barrier nursing.

        Parameters
        ----------
        alpha : float
            Effectiveness of barrier nursing
        r : tuple
            Parameters that define the model
        h : float
            Probability of hospitalisation
        """
        self.q_c_values = []
        self.q_h_values = []
        self.p_c_values = []
        self.p_h_values = []
        for p_f in np.linspace(0.4/0.7, 0, 1000):
            def equations(vars):
                x, y, z = vars
                eq1 = (r[0]*(1 - h)) / (r[0] + 1) *x**2 +\
                          (r[0]*h) / (r[0] + 1) *x*z +\
                          0.7*p_f / (r[0] + 1) *y +\
                          (1 - 0.7*p_f) / (r[0] + 1) - x
                eq2 = (r[2]*(1 - h)) / (r[2] + 1) *x*y +\
                        (r[2]*h) / (r[2] + 1) *y*z + 1 / (r[2] + 1) - y
                eq3 = (r[3]*(1 - h)) / (r[3] + r[4]/0.2*alpha + 1) *x*z +\
                        (h*r[3] + r[4]/0.2*alpha) / (r[3] + r[4]/0.2*alpha + 1) * z**2 +\
                        1 / (r[3] + r[4]/0.2*alpha + 1) - z
                return [eq1, eq2, eq3]

            solution = root(equations, [0.5, 0.5, 0.5], method='lm')
            self.find_p_q(solution.x)
        return self.p_c_values, self.p_h_values
