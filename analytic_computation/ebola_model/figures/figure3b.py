import matplotlib.pyplot as plt
import numpy as np

from ebola_model.functions import probability as p

def plot_pmo_param(p_values, parameter, color, linestyle, label):
    """Plot the probabilty of a major outbreak against the uptake of safe
    burial practices (g)
    """
    plt.plot(parameter, p_values[0], color=color, linestyle=linestyle, label=label)
    #plt.plot(parameter, p_values[1], label='First case H', color='blue')
    plt.xlabel('Compliance with safe burial practices (%)', fontsize=20,
               labelpad=10)
    plt.ylabel('Probability of a major outbreak', fontsize=20, labelpad=10)
    plt.xlim(0, parameter[-1])
    ax = plt.gca()
    ax.set_xticks(np.linspace(0, parameter[-1], 11),
                  np.linspace(0, 100, 11).astype(int), fontsize=18)
    ax.set_yticks(np.linspace(0, 1, 6), [f'{label:.0f}' if label == 0
                                         else f'{label:.1f}' for label in
                                         np.linspace(0, 1, 6)], fontsize=18)
    plt.ylim(0, 0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

model = p.PMO(0.7, 0.4/0.7, 173/27, 28/173, 5.9, 0.25, 4, 0.2)
r = model.variables()

plt.figure(figsize=[8, 6])

# Plot with reduced effectiveness
plot_pmo_param(model.pmo_burial_compliance(0.18, r), np.linspace(0, 5.9, 1000), color='green', linestyle='-', label='10%')
plot_pmo_param(model.pmo_burial_compliance(0.1, r), np.linspace(0, 5.9, 1000), color='red', linestyle='-', label='50%')
# Plot with known barrier nursing parameter and confidence interval
plot_pmo_param(model.pmo_burial_compliance(0.047, r),
               np.linspace(0, 5.9, 1000), color='blue', linestyle='--', label=None)

plot_pmo_param(model.pmo_burial_compliance(0.023, r),
               np.linspace(0, 5.9, 1000), color='blue', linestyle='-', label='88.5%')

plot_pmo_param(model.pmo_burial_compliance(0.0079, r),
               np.linspace(0, 5.9, 1000), color='blue', linestyle='--', label=None)

plt.legend(title='Effectiveness of' + '\n' + 'barrier nursing',
           fontsize=18, title_fontsize=18)
plt.show()