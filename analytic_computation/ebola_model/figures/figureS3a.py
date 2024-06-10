import numpy as np
import matplotlib.pyplot as plt

from ebola_model.functions import probability as p
from ebola_model.functions import local_sensitivity_analysis as lsa

model = p.PMO(0.7, 0.4/0.7, 173/27, 28/173, 5.9, 0.25, 4, 0.2)
lsa = lsa.LSA(model)
r = model.variables()

normalised_matrix = lsa.lsa_varying_h(r)
plt.figure(figsize=(8, 6))
plt.plot(np.linspace(0, 1, 22), normalised_matrix[0],
            marker='o', label=r'$R_C$')
plt.plot(np.linspace(0, 1, 22), normalised_matrix[1],
            marker='s', label=r'$p_f$')
plt.plot(np.linspace(0, 1, 22), normalised_matrix[2],
            marker='^', label=r'$R_V$')
plt.plot(np.linspace(0, 1, 22), normalised_matrix[3],
            marker='*', label=r'$R_W$')
plt.xlabel('Probability of treatment in a healthcare' + '\n' +
            r'facility ($p_h$)', fontsize=20, labelpad=10, ha='center')
plt.ylabel('Rate of change index', fontsize=20, labelpad=10)
plt.legend(fontsize=18, loc='upper left')
plt.xlim(0, 1)
plt.ylim(0, 1)
ax = plt.gca()
ax.set_xticks(np.linspace(0, 1, 6), [f'{label:.0f}' if label == 0 else
                                        f'{label:.1f}' for label in
                                        np.linspace(0, 1, 6)],
                                        fontsize=18)
ax.set_yticks(np.linspace(0, 1, 6), [f'{label:.0f}' if label == 0 else
                                        f'{label:.1f}' for label in
                                        np.linspace(0, 1, 6)],
                                        fontsize=18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()
