import matplotlib.pyplot as plt
import numpy as np

from ebola_model.functions import probability as p

def plot_pmo_param(p_values, parameter, color, linestyle, label):
    """Plot the probabilty of a major outbreak against the uptake of safe
    burial practices (g)
    """
    plt.plot(parameter, p_values[0], color=color, linestyle=linestyle, label=label)
    plt.xlabel('Additional compliance with safe burial practices (%)', fontsize=20,
               labelpad=10)
    plt.ylabel(r'Probability of a major outbreak ($\pi_C$)', fontsize=20, labelpad=10)
    plt.xlim(0, parameter[-1])
    plt.ylim(bottom=0)
    ax = plt.gca()
    ax.set_xticks(np.linspace(0, parameter[-1], 11),
                  np.linspace(0, 100, 11).astype(int), fontsize=18)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

model = p.PMO(0.7, 0.4/0.7, 173/27, 28/173, 5.9, 0.25, 4, 0.2)
r = model.variables()

# Plot with increased effectiveness of barrier nursing
plot_pmo_param(model.pmo_burial_compliance(0.5, r),
               np.linspace(0, 0.4/0.7, 1000), color='green', linestyle='-', label='50%')
plot_pmo_param(model.pmo_burial_compliance(0.3, r),
               np.linspace(0, 0.4/0.7, 1000), color='red', linestyle='-', label='70%')
plot_pmo_param(model.pmo_burial_compliance(0.1, r),
               np.linspace(0, 0.4/0.7, 1000), color='blue', linestyle='-', label='90%')

plt.legend(title='Effectiveness of' + '\n' + 'barrier nursing',
           fontsize=18, title_fontsize=18)
plt.show()