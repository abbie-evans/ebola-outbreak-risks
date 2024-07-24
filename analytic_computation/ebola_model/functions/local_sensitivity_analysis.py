import numpy as np
import matplotlib.pyplot as plt
from ebola_model.functions import probability as p

class LSA:
    def __init__(self, model):
        p.PMO = model
        self.gradients_c = []
        self.gradients_h = []

    def community_infections(self, r, h):
        """Method calculates the probability of a major outbreak given the
        average expected number of infections in the community.

        Parameters
        ----------
        r : tuple
            Parameters that define the model
        h : float
            Probability of treatment in a healthcare facility

        Returns
        -------
        tuple[list, list, array]
            List of probabilities that an outbreak occurs and is treated
            initially in the community and in a healthcare facility and the
            array of expected number of community infections.
        """
        p.PMO.q_c_values = []
        p.PMO.q_h_values = []
        p.PMO.p_c_values = []
        p.PMO.p_h_values = []
        x = np.linspace(0*r[0], 2*r[0], 10000)
        for r_c_ in x:
            p_c_values, p_h_values = p.PMO.pmo(r_c_, r[1], r[2], r[3], r[4], h)
        gradient_c, gradient_h = np.gradient(p_c_values, x), np.gradient(p_h_values, x)
        self.gradients_c.append(gradient_c[5000]*r[0])
        self.gradients_h.append(gradient_h[5000]*r[0])
        return p_c_values, p_h_values, x

    def plot_community_infections(self, r, h):
        """Method plots the probability of a major outbreak against the average
        expected number of infections in the community.
        
        Parameters
        ----------
        r : tuple
            Parameters that define the model
        h : float
            Probability of treatment in a healthcare facility
        """
        p_c_values, p_h_values, x = self.community_infections(r, h)

        plt.figure(figsize = [8, 6])
        plt.plot(x, p_c_values, label='First case C', color='red')
        plt.plot(x, p_h_values, label='First case H', color='blue')
        plt.axvline(x=r[0], linestyle='--', color='black')
        plt.xlabel(r'Number of community infections ($R_C$)',
                fontsize=20, labelpad=10)
        plt.ylabel('Probability of a major outbreak', fontsize=20, labelpad=10)
        plt.xlim(x[0], x[-1])
        plt.ylim(0, 1)
        plt.legend(loc='upper left', fontsize=18)
        ax = plt.gca()
        ax.set_xticks(np.linspace(x[0], np.round(x[-1], 1), 6),
                    [f'{label:.0f}' if label == 0 else f'{label:.1f}' for
                    label in np.linspace(x[0], np.round(x[-1], 1), 6)],
                    fontsize=18)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=18)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.show()

    def funeral_infections(self, r, h):
        """Method calculates the probability of a major outbreak given the
        probability of an unsafe burial given death.
        
        Parameters
        ----------
        r : tuple
            Parameters that define the model
        h : float
            Probability of treatment in a healthcare facility
        
        Returns
        -------
        tuple[list, list, array]
            List of probabilities that an outbreak occurs and is treated
            initially in the community and in a healthcare facility and the
            array of probabilities of an unsafe burial given death.
        """
        p.PMO.q_c_values = []
        p.PMO.q_h_values = []
        p.PMO.p_c_values = []
        p.PMO.p_h_values = []
        x = np.linspace(0, 1, 10000)
        for p_f in x:
            p_c_values, p_h_values = p.PMO.pmo(r[0], r[1]/(0.4/0.7)*p_f, r[2], r[3], r[4], h)
        gradient_c, gradient_h = np.gradient(p_c_values, x), np.gradient(p_h_values, x)
        self.gradients_c.append(gradient_c[5714]*(0.4/0.7))
        self.gradients_h.append(gradient_h[5714]*(0.4/0.7))
        return p_c_values, p_h_values, x

    def plot_funeral_infections(self, r, h):
        """Method plots the probability of a major outbreak against the
        probability of an unsafe burial given death.
        
        Parameters
        ----------
        r : tuple
            Parameters that define the model
        h : float
            Probability of treatment in a healthcare facility
        """
        p_c_values, p_h_values, x = self.funeral_infections(r, h)

        plt.figure(figsize = [8, 6])
        plt.plot(x, p_c_values, label='First case C', color='red')
        plt.plot(x, p_h_values, label='First case H', color='blue')
        plt.axvline(x=0.4/0.7, linestyle='--', color='black')
        plt.xlabel(r'Probability of unsafe burial given death ($p_f$)',
                fontsize=20, labelpad=10)
        plt.ylabel('Probability of a major outbreak', fontsize=20, labelpad=10)
        plt.xlim(x[0], x[-1])
        plt.ylim(0, 1)
        plt.legend(loc='upper left', fontsize=18)
        ax = plt.gca()
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=18)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=18)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.show()

    def hospital_visitors(self, r, h):
        """Method calculates the probability of a major outbreak given the
        average expected number of infections of healthcare facility visitors.
        
        Parameters
        ----------
        r : tuple
            Parameters that define the model
        h : float
            Probability of treatment in a healthcare facility
            
        Returns
        -------
        tuple[list, list, array]
            List of probabilities that an outbreak occurs and is treated
            initially in the community and in a healthcare facility and the
            array of expected number of healthcare facility visitor infections.
        """
        p.PMO.q_c_values = []
        p.PMO.q_h_values = []
        p.PMO.p_c_values = []
        p.PMO.p_h_values = []
        x = np.linspace(0*r[3], 2*r[3], 10000)
        for r_v_ in x:
            p_c_values, p_h_values = p.PMO.pmo(r[0], r[1], r[2], r_v_, r[4], h)
        gradient_c, gradient_h = np.gradient(p_c_values, x), np.gradient(p_h_values, x)
        self.gradients_c.append(gradient_c[5000]*r[3])
        self.gradients_h.append(gradient_h[5000]*r[3])
        return p_c_values, p_h_values, x

    def plot_hospital_visitors(self, r, h):
        """Method plots the probability of a major outbreak against the average
        expected number of healthcare facility visitor infections.

        Parameters
        ----------
        r : tuple
            Parameters that define the model
        h : float
            Probability of treatment in a healthcare facility
        """
        p_c_values, p_h_values, x = self.hospital_visitors(r, h)

        plt.figure(figsize = [8, 6])
        plt.plot(x, p_c_values, label='First case C', color='red')
        plt.plot(x, p_h_values, label='First case H', color='blue')
        plt.axvline(x=r[3], linestyle='--', color='black')
        plt.xlabel(r'Number of visitor infections ($R_V$)',
                fontsize=20, labelpad=10)
        plt.ylabel('Probability of a major outbreak', fontsize=20, labelpad=10)
        plt.xlim(x[0], x[-1])
        plt.ylim(0, 1)
        plt.legend(loc='upper left', fontsize=18)
        ax = plt.gca()
        ax.set_xticks(np.linspace(x[0], np.round(x[-1], 1), 6),
                    [f'{label:.0f}' if label == 0 else f'{label:.1f}' for
                    label in np.linspace(x[0], np.round(x[-1], 1), 6)],
                    fontsize=18)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=18)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.show()

    def hcw_infections(self, r, h):
        """Method calculates the probability of a major outbreak given the
        average expected number of healthcare worker infections.
        
        Parameters
        ----------
        r : tuple
            Parameters that define the model
        h : float
            Probability of treatment in a healthcare facility
            
        Returns
        -------
        tuple[list, list, array]
            List of probabilities that an outbreak occurs and is treated
            initially in the community and in a healthcare facility and the
            array of expected number of healthcare worker infections.
        """
        p.PMO.q_c_values = []
        p.PMO.q_h_values = []
        p.PMO.p_c_values = []
        p.PMO.p_h_values = []
        x = np.linspace(0*r[4], 2*r[4], 10000)
        for r_w_ in x:
            p_c_values, p_h_values = p.PMO.pmo(r[0], r[1], r[2], r[3], r_w_, h)
        gradient_c, gradient_h = np.gradient(p_c_values, x), np.gradient(p_h_values, x)
        self.gradients_c.append(gradient_c[5000]*r[4])
        self.gradients_h.append(gradient_h[5000]*r[4])
        return p_c_values, p_h_values, x

    def plot_hcw_infections(self, r, h):
        """Method plots the probability of a major outbreak against the average
        expected number of healthcare worker infections.
        
        Parameters
        ----------
        r : tuple
            Parameters that define the model
        h : float
            Probability of treatment in a healthcare facility
        """
        p_c_values, p_h_values, x = self.hcw_infections(r, h)

        plt.figure(figsize = [8, 6])
        plt.plot(x, p_c_values, label='First case C', color='red')
        plt.plot(x, p_h_values, label='First case H', color='blue')
        plt.axvline(x=r[4], linestyle='--', color='black')
        plt.xlabel(r'Number of healthcare worker infections ($R_W$)',
                fontsize=20, labelpad=10)
        plt.ylabel('Probability of a major outbreak', fontsize=20, labelpad=10)
        plt.xlim(x[0], x[-1])
        plt.ylim(0, 1)
        plt.legend(loc='upper left', fontsize=18)
        ax = plt.gca()
        ax.set_xticks(np.linspace(x[0], np.round(x[-1], 1), 6),
                    [f'{label:.0f}' if label == 0 else f'{label:.1f}' for
                    label in np.linspace(x[0], np.round(x[-1], 1), 6)],
                    fontsize=18)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=18)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.show()

    def hosp_pmo(self, r, h):
        """Method calculates the probability of a major outbreak given the
        probability of treatment in a healthcare facility.
        
        Parameters
        ----------
        r : tuple
            Parameters that define the model
        h : float
            Probability of treatment in a healthcare facility
        
        Returns
        -------
        tuple[list, list, array]
            List of probabilities that an outbreak occurs and is treated
            initially in the community and in a healthcare facility and the
            array of probabilities of treatment in a healthcare facility.
        """
        p.PMO.q_c_values = []
        p.PMO.q_h_values = []
        p.PMO.p_c_values = []
        p.PMO.p_h_values = []
        x = np.linspace(0, 1, 10000)
        for h_ in np.linspace(1, 0, 10000):
            p_c_values, p_h_values = p.PMO.pmo(r[0], r[1], r[2], r[3], r[4], h_)
        gradient_c, gradient_h = np.gradient(p_c_values, x), np.gradient(p_h_values, x)
        self.gradients_c.append(gradient_c[4000]*0.4)
        self.gradients_h.append(gradient_h[4000]*0.4)
        return p_c_values, p_h_values, x

    def plot_hosp_pmo(self, r, h):
        """Method plots the probability of a major outbreak against the
        probability of treatment in the community.
        
        Parameters
        ----------
        r : tuple
            Parameters that define the model
        h : float
            Probability of treatment in a healthcare facility
        """
        p_c_values, p_h_values, x = self.hosp_pmo(r, h)

        plt.figure(figsize = [8, 6])
        plt.plot(x, p_c_values, label='First case C', color='red')
        plt.plot(x, p_h_values, label='First case H', color='blue')
        plt.axvline(x=1-h, linestyle='--', color='black')
        plt.xlabel(r'Probability of treatment in the community ($1-p_h$)',
                fontsize=20, labelpad=10)
        plt.ylabel('Probability of a major outbreak', fontsize=20, labelpad=10)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend(loc='upper left', fontsize=18)
        ax = plt.gca()
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=18)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=18)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.show()
