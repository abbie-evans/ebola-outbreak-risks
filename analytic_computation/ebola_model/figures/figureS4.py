import matplotlib.pyplot as plt
import numpy as np

from ebola_model.functions import probability as p
from ebola_model.functions import intervention_combinations as ic

def plot_results_c(matrix):
        matrix = np.flip(matrix, axis=0)

        plt.figure(figsize = [8, 6])
        plt.imshow(matrix, cmap='viridis_r')
        cbar = plt.colorbar()
        cbar.set_label(r'Probability of major outbreak ($\pi_C$)', fontsize=18, labelpad=10)
        cbar.ax.tick_params(labelsize=18)

        plt.ylim(top=0, bottom=len(matrix) - 1)
        plt.xlim(left=0, right=len(matrix) - 1)
        plt.xlabel('Probability of treatment in a' + '\n' +
                   r'healthcare facility ($p_h$)', fontsize=20, labelpad=10)
        plt.ylabel('Number of community' + '\n' + r'infections ($R_C$)',
                   fontsize=20, multialignment='center', labelpad=10)

        ax = plt.gca()
        ax.set_xticks(np.linspace(0, len(c) - 1, 6),
                      [f'{label:.0f}' if label == 0 else f'{label:.1f}' for
                       label in np.linspace(0, 1, 6)], fontsize=18)
        xtick_labels = [0, 0.2, 0.4, 0.6, 0.8, 1]
        ax.set_xticklabels(xtick_labels, fontsize=18)
        ax.set_yticks(np.linspace(0, len(c) - 1, 6),
                      [f'{label:.0f}' if label == 0 else f'{label:.2f}' for
                       label in np.linspace(r[0]*2, 0, 6)], fontsize=18)
        plt.tight_layout()
        plt.show()

def plot_results_h(matrix):
        matrix = np.flip(matrix, axis=0)

        plt.figure(figsize = [8, 6])
        plt.imshow(matrix, cmap='viridis_r')
        cbar = plt.colorbar()
        cbar.set_label(r'Probability of major outbreak ($\pi_H$)', fontsize=18, labelpad=10)
        cbar.ax.set_yticks(np.linspace(0.1, 0.3, 5))
        cbar.ax.set_yticklabels([0.1, 0.15, 0.2, 0.25, 0.3], fontsize=18)

        plt.ylim(top=0, bottom=len(matrix) - 1)
        plt.xlim(left=0, right=len(matrix) - 1)
        plt.xlabel('Probability of treatment in a' + '\n' +
                   r'healthcare facility ($p_h$)', fontsize=20, labelpad=10)
        plt.ylabel('Number of community' + '\n' + r'infections ($R_C$)',
                   fontsize=20, multialignment='center', labelpad=10)

        ax = plt.gca()
        ax.set_xticks(np.linspace(0, len(c) - 1, 6))
        xtick_labels = [0, 0.2, 0.4, 0.6, 0.8, 1]
        ax.set_xticklabels(xtick_labels, fontsize=18)
        ax.set_yticks(np.linspace(0, len(c) - 1, 6),
                      [f'{label:.0f}' if label == 0 else f'{label:.2f}' for
                       label in np.linspace(r[0]*2, 0, 6)], fontsize=18)
        plt.tight_layout()
        plt.show()

model = p.PMO(0.7, 0.4/0.7, 173/27, 28/173, 5.9, 0.25, 4, 0.2)
r = model.variables()
c, h = ic.Combination.hospitalisation_community(*r, model=model)
plot_results_c(c)
plot_results_h(h)
