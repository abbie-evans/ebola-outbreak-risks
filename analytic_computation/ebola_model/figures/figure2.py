import numpy as np
import matplotlib.pyplot as plt

from ebola_model.functions.probability import PMO
from ebola_model.functions.local_sensitivity_analysis import LSA

model = PMO(0.7, 0.4/0.7, 173/27, 28/173, 5.9, 0.25, 4, 0.2)
lsa = LSA(model)
r = model.variables()

# Figure 2A
lsa.plot_community_infections(r, h=0.6)

# Figure 2B
lsa.plot_funeral_infections(r, h=0.6)

# Figure 2C
lsa.plot_hospital_visitors(r, h=0.6)

# Figure 2D
lsa.plot_hcw_infections(r, h=0.6)

# Figure 2E
lsa.plot_hosp_pmo(r, h=0.6)

# Figure 2F
plt.figure(figsize = [7, 5])
plt.bar([r'$R_{C}$', r'$p_{f}$', r'$R_V$', r'$R_W$', r'$1-p_h$'],
        lsa.gradients_c, edgecolor='black')
plt.ylabel('Rate of change (PMO)', fontsize=18, labelpad=10)
plt.xticks(fontsize=18)
plt.yticks(np.linspace(0, 0.7, 8), [f'{label:.0f}' if label == 0 else
                                    f'{label:.1f}' for label in
                                    np.linspace(0, 0.7, 8)], fontsize=16)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()
