import matplotlib.pyplot as plt
import numpy as np

from ebola_model.functions import probability as p
from ebola_model.functions import intervention_combinations as ic

def plot_results(matrix):
        matrix = np.flip(matrix, axis=0)

        plt.figure(figsize = [8, 6])
        plt.imshow(matrix, cmap='viridis_r')
        cbar = plt.colorbar()
        cbar.set_label(r'Probability of major outbreak ($\pi_C$)', fontsize=18, labelpad=10)
        cbar.ax.tick_params(labelsize=18)
        cbar.ax.set_yticks(np.linspace(0, 0.7, 8))
        cbar.ax.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], fontsize=18)

        # Plot white markers
        plt.plot(0.4/0.7*len(matrix), len(matrix)//2, 'wo')
        plt.plot(0.4/0.7*len(matrix), 286, 'wo') # Plot point where alpha=0.019
        plt.arrow(0.4/0.7*len(matrix), 150, 0, 120, head_width=5, head_length=5, fc='w', ec='w')
        plt.text(180, 220, 'Barrier' + '\n' + 'nursing', color='white', fontsize=18)
        plt.arrow(0.4/0.7*len(matrix), 286, -50, 0, head_width=5, head_length=5, fc='w', ec='w')
        plt.text(110, 275, 'Safe' + '\n' + 'burials', color='white', fontsize=18)

        plt.ylim(top=0, bottom=len(matrix) - 1)
        plt.xlim(left=0, right=len(matrix) - 1)
        plt.xlabel('Probability of unsafe burial' + '\n' + r'given death ($p_f$)', fontsize=20,
                   labelpad=10)
        plt.ylabel('Number of healthcare worker' + '\n' + r'infections ($R_W$)',
                   fontsize=20, multialignment='center', labelpad=10)

        ax = plt.gca()
        ax.set_xticks(np.linspace(0, len(c) - 1, 6))
        ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=18)
        ax.set_yticks(np.linspace(0, len(c) - 1, 6),
                      [f'{label:.0f}' if label == 0 else f'{label:.2f}' for
                       label in np.linspace(r[4]*2, 0, 6)], fontsize=18)
        plt.tight_layout()
        plt.show()

model = p.PMO(0.7, 0.4/0.7, 173/27, 28/173, 5.9, 0.25, 4, 0.2)
r = model.variables()
c, h = ic.Combination.funeral_worker(*r, h=0.6, model=model)
plot_results(c)
