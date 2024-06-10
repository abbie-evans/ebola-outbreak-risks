import numpy as np
import matplotlib.pyplot as plt

from ebola_model.functions.probability import PMO
from ebola_model.functions.local_sensitivity_analysis import LSA

model = PMO(0.7, 0.4/0.7, 173/27, 28/173, 5.9, 0.25, 4, 0.2)
lsa = LSA(model)
r = model.variables()

# Run functions to calculate the gradients
lsa.community_infections(r, h=0.6)
lsa.funeral_infections(r, h=0.6)
lsa.hospital_visitors(r, h=0.6)
lsa.hcw_infections(r, h=0.6)
lsa.hosp_pmo(r, h=0.6)

plt.figure(figsize = [8, 6])
plt.bar([r'$R_{C}$', r'$p_{f}$', r'$R_V$', r'$R_W$', r'$1-p_h$'],
        lsa.gradients_h, edgecolor='black')
plt.ylabel('Rate of change (PMO)', fontsize=20, labelpad=10)
plt.xticks(fontsize=20)
plt.yticks(np.linspace(0, 0.7, 8), [f'{label:.0f}' if label == 0 else
                                    f'{label:.1f}' for label in
                                    np.linspace(0, 0.7, 8)], fontsize=18)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()