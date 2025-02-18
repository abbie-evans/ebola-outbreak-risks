import matplotlib.pyplot as plt
import numpy as np

from ebola_model.functions import probability as p
from ebola_model.functions import intervention_combinations as ic

def plot_results(matrix):
        matrix = np.flip(matrix, axis=0)

        plt.figure(figsize = [8, 6])
        plt.imshow(matrix, cmap='viridis_r')
        cbar = plt.colorbar()
        cbar.set_label(r'Probability of major outbreak $(\pi_H)$', fontsize=18, labelpad=10)
        cbar.ax.tick_params(labelsize=18)
        cbar.ax.set_yticks(np.linspace(0, 0.5, 6))
        cbar.ax.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=18)

        # Plot the contour lines
        levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        plt.contour(matrix, levels=levels, colors='w', linestyles=':', linewidths=1.5)

        plt.ylim(top=0, bottom=len(matrix) - 1)
        plt.xlim(left=0, right=len(matrix) - 1)
        plt.xlabel(r'Number of visitor infections ($R_V$)', fontsize=20,
                   labelpad=10)
        plt.ylabel('Number of healthcare worker' + '\n' + r'infections ($R_W$)',
                   fontsize=20, multialignment='center', labelpad=10)

        ax = plt.gca()
        ax.set_xticks(np.linspace(0, len(c) - 1, 6),
                      [f'{label:.0f}' if label == 0 else f'{label:.1f}' for
                       label in np.linspace(0, r[3]*2, 6)], fontsize=18)
        ax.set_yticks(np.linspace(0, len(c) - 1, 6),
                      [f'{label:.0f}' if label == 0 else f'{label:.2f}' for
                       label in np.linspace(r[4]*2, 0, 6)], fontsize=18)
        plt.tight_layout()
        plt.show()

model = p.PMO(0.7, 0.4/0.7, 173/27, 28/173, 5.9, 0.25, 4, 0.2)
r = model.variables()
c, h = ic.Combination.visitor_worker(*r, h=0.6, model=model)
plot_results(h)
