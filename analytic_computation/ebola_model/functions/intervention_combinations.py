import numpy as np
from scipy.optimize import root

class Combination:
     
    def funeral_worker(r_c, p_b, r_f, r_v, r_w, h, model):
        """Method calculates the probability of a major outbreak for varying
        average expected number of infections of healthcare workers, and
        probability of unsafe burials given death.
        
        Parameters
        ----------
        r_c : float
            Average expected number of infections within the community
        p_b : float
            probability of unsafe burial given death
        r_f : float
            Average expected number of infections from an unsafe burial
        r_v : float
            Average expected number of infections of healthcare facility
            visitors
        r_w : float
            Average expected number of infections of healthcare workers
        h : float
            Probability of hospitalisation
        model : class
            Instance of the class for the model

        Returns
        -------
        tuple[np.array, np.array]
            Probabilities that an outbreak occurs and is treated
            initially in the community and in a healthcare facility
        """
        solution_c_matrix = np.zeros((300, 300))
        solution_h_matrix = np.zeros((300, 300))
        p_f_ = np.linspace(0, 1, 300)
        r_w_ = np.linspace(0*r_w, 2*r_w, 300)

        for i in range(len(r_w_)):
            for j in range(len(p_f_)):
                def equations(vars):
                    x, y, z = vars
                    eq1 = (r_c*(1 - h)) / (r_c + 1) *x**2 +\
                          (r_c*h) / (r_c + 1) *x*z +\
                          0.7*p_f_[j] / (r_c + 1) *y +\
                          (1 - 0.7*p_f_[j]) / (r_c + 1) - x
                    eq2 = (r_f*(1 - h)) / (r_f + 1) *x*y +\
                          (r_f*h) / (r_f + 1) *y*z + 1 / (r_f + 1) - y
                    eq3 = (r_v*(1 - h)) / (r_v + r_w_[i] + 1) *x*z +\
                          (h*r_v + r_w_[i]) / (r_v + r_w_[i] + 1) * z**2 +\
                          1 / (r_v + r_w_[i] + 1) - z
                    return [eq1, eq2, eq3]

                solution = root(equations, [0.5, 0.5, 0.5], method='lm')
                p_c_value, p_h_value = model.find_p_q_combination(solution.x)
                if p_c_value < 1e-11:
                    p_c_value = 0
                if p_h_value < 1e-11:
                    p_h_value = 0
                solution_c_matrix[i, j] = p_c_value
                solution_h_matrix[i, j] = p_h_value
        return solution_c_matrix, solution_h_matrix

    def visitor_worker(r_c, p_b, r_f, r_v, r_w, h, model):
        """Method calculates the probability of a major outbreak for varying
        average expected number of infections of healthcare facility visitors,
        and average expected number of infections of healthcare workers.
        
        Parameters
        ----------
        r_c : float
            Average expected number of infections within the community
        p_b : float
            probability of unsafe burial given death
        r_f : float
            Average expected number of infections from an unsafe burial
        r_v : float
            Average expected number of infections of healthcare facility
            visitors
        r_w : float
            Average expected number of infections of healthcare workers
        h : float
            Probability of hospitalisation
        model : class
            Instance of the class for the model
        
        Returns
        -------
        tuple[np.array, np.array]
            Probabilities that an outbreak occurs and is treated
            initially in the community and in a healthcare facility
        """
        solution_c_matrix = np.zeros((300, 300))
        solution_h_matrix = np.zeros((300, 300))
        r_v_ = np.linspace(0*r_v, 2*r_v, 300)
        r_w_ = np.linspace(0*r_w, 2*r_w, 300)

        for i in range(len(r_w_)):
            for j in range(len(r_v_)):
                def equations(vars):
                    x, y, z = vars
                    eq1 = (r_c*(1 - h)) / (r_c + 1) *x**2 +\
                        (r_c*h) / (r_c + 1) *x*z +\
                        p_b / (r_c + 1) *y +\
                        (1 - p_b) / (r_c + 1) - x
                    eq2 = (r_f*(1 - h)) / (r_f + 1) *x*y +\
                        (r_f*h) / (r_f + 1) *y*z + 1 / (r_f + 1) - y
                    eq3 = (r_v_[j]*(1 - h)) / (r_v_[j] + r_w_[i] + 1) *x*z +\
                        (h*r_v_[j] + r_w_[i]) / (r_v_[j] + r_w_[i] + 1) * z**2 +\
                        1 / (r_v_[j] + r_w_[i] + 1) - z
                    return [eq1, eq2, eq3]

                solution = root(equations, [0.5, 0.5, 0.5], method='lm')
                p_c_value, p_h_value = model.find_p_q_combination(solution.x)
                if p_c_value < 1e-11:
                    p_c_value = 0
                if p_h_value < 1e-11:
                    p_h_value = 0
                solution_c_matrix[i, j] = p_c_value
                solution_h_matrix[i, j] = p_h_value
        return solution_c_matrix, solution_h_matrix
    
    def hospitalisation_community(r_c, p_b, r_f, r_v, r_w, model):
        """Method calculates the probability of a major outbreak for varying
        average expected number of infections within the community, and
        probability of hospitalisation.
        
        Parameters
        ----------
        r_c : float
            Average expected number of infections within the community
        p_b : float
            probability of unsafe burial given death
        r_f : float
            Average expected number of infections from an unsafe burial
        r_v : float
            Average expected number of infections of healthcare facility
            visitors
        r_w : float
            Average expected number of infections of healthcare workers
        model : class
            Instance of the class for the model
            
        Returns
        -------
        tuple[np.array, np.array]
            Probabilities that an outbreak occurs and is treated
            initially in the community and in a healthcare facility
        """
        solution_c_matrix = np.zeros((300, 300))
        solution_h_matrix = np.zeros((300, 300))
        h = np.linspace(0, 1, 300)
        r_c_ = np.linspace(0*r_c, 2*r_c, 300)

        for i in range(len(r_c_)):
            for j in range(len(h)):
                def equations(vars):
                    x, y, z = vars
                    eq1 = (r_c_[i]*(1 - h[j])) / (r_c_[i] + 1) *x**2 +\
                          (r_c_[i]*h[j]) / (r_c_[i] + 1) *x*z +\
                          p_b / (r_c_[i] + 1) *y +\
                          (1 - p_b) / (r_c_[i] + 1) - x
                    eq2 = (r_f*(1 - h[j])) / (r_f + 1) *x*y +\
                          (r_f*h[j]) / (r_f + 1) *y*z + 1 / (r_f + 1) - y
                    eq3 = (r_v*(1 - h[j])) / (r_v + r_w + 1) *x*z +\
                          (h[j]*r_v + r_w) / (r_v + r_w + 1) * z**2 +\
                          1 / (r_v + r_w + 1) - z
                    return [eq1, eq2, eq3]

                solution = root(equations, [0.5, 0.5, 0.5], method='lm')
                p_c_value, p_h_value = model.find_p_q_combination(solution.x)
                solution_c_matrix[i, j] = p_c_value
                solution_h_matrix[i, j] = p_h_value
        return solution_c_matrix, solution_h_matrix
