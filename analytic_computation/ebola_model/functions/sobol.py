import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from SALib.analyze.sobol import analyze
from SALib.sample.sobol import sample
from ebola_model.functions import gsa

def finding_indices():
    """Function to find the first-order and total-order sensitivity indices of
    the model and to plot the results.
    """
    # Define model parameters
    problem = {'num_vars': 5,
               'names': [r'$R_C$', r'$p_f$', r'$R_V$', r'$R_W$', r'$p_h$'],
               'bounds': [[0, 56/27], [0, 1], [0, 0.5], [0, 1.6*28/27], [0, 1]]}

    # Generate samples
    param_values = sample(problem, 8192, calc_second_order=False)

    # Run the model
    m = gsa.Model()
    Y = m.evaluate(X=param_values)

    # Perform analysis
    Si = analyze(problem, Y, calc_second_order=False, print_to_console=False)

    # Plot the sensitivity indices with error bars
    plt.figure(figsize = [8, 6])
    plt.bar(problem['names'], Si['S1'], yerr=Si['S1_conf'], capsize=5)
    plt.ylabel('First-order Sobol\' indices', fontsize=20, labelpad=10)
    plt.tick_params(labelsize=18, axis='both')
    plt.ylim(0, 0.4)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize = [8, 6])
    plt.bar(problem['names'], Si['ST'], yerr=Si['ST_conf'], capsize=5)
    plt.ylabel('Total-order Sobol\' indices', fontsize=20, labelpad=10)
    plt.tick_params(labelsize=18, axis='both')
    plt.ylim(0, 0.4)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

def varying_h():
    """Function to vary the probability of treatment in a healthcare facility
    and to find the first-order and total-order sensitivity indices of the
    model for each value of h. The results are then plotted.
    """
    problem = {'num_vars': 4,
               'names': [r'$R_C$', r'$p_f$', r'$R_V$', r'$R_W$'],
               'bounds': [[0, 56/27], [0, 1], [0, 0.5], [0, 1.6*28/27]]}

    # # Generate samples
    param_values = sample(problem, 8192, calc_second_order=False)

    # Run the model
    h = np.linspace(0, 1, 22)
    df = pd.DataFrame() # Dataframe to store first-order indices
    st = pd.DataFrame() # Dataframe to store total-order indices
    for i in range(len(h)):
        m = gsa.Model()
        Y = m.evaluate_h(h[i], X=param_values)
        Si = analyze(problem, Y, calc_second_order=False,
                     print_to_console=False)
        # Update the dataframe for each loop
        df = pd.concat([df, pd.DataFrame([Si['S1']],
                                         columns=problem['names'])], axis=0)
        st = pd.concat([st, pd.DataFrame([Si['ST']],
                                         columns=problem['names'])], axis=0)

    # Plot the sensitivity indices against h
    plt.figure(figsize = [8, 6])
    plt.plot(h, df['$R_C$'], marker='o', label=r'$R_C$')
    plt.plot(h, df['$p_f$'], marker='s', label=r'$p_f$')
    plt.plot(h, df['$R_V$'], marker='^', label=r'$R_V$')
    plt.plot(h, df['$R_W$'], marker='*', label=r'$R_W$')
    plt.xlabel('Probability of treatment in a healthcare' + '\n' +
               r'facility ($p_h$)', fontsize=20, labelpad=10)
    plt.ylabel('First-order Sobol\' indices', fontsize=20, labelpad=10)
    plt.legend(fontsize=18, loc='upper left')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ax = plt.gca()
    ax.set_xticks(np.linspace(0, 1, 6), [f'{label:.0f}' if label == 0 else
                                            f'{label:.1f}' for label in
                                            np.linspace(0, 1, 6)], fontsize=18)
    ax.set_yticks(np.linspace(0, 1, 6), [f'{label:.0f}' if label == 0 else
                                            f'{label:.1f}' for label in
                                            np.linspace(0, 1, 6)], fontsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize = [8, 6])
    plt.plot(h, st['$R_C$'], marker='o', label=r'$R_C$')
    plt.plot(h, st['$p_f$'], marker='s', label=r'$p_f$')
    plt.plot(h, st['$R_V$'], marker='^', label=r'$R_V$')
    plt.plot(h, st['$R_W$'], marker='*', label=r'$R_W$')
    plt.xlabel('Probability of treatment in a healthcare' + '\n' +
               r'facility ($p_h$)', fontsize=20, labelpad=10)
    plt.ylabel('Total-order Sobol\' indices', fontsize=20, labelpad=10)
    plt.legend(fontsize=18, loc='upper left')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ax = plt.gca()
    ax.set_xticks(np.linspace(0, 1, 6), [f'{label:.0f}' if label == 0 else
                                            f'{label:.1f}' for label in
                                            np.linspace(0, 1, 6)], fontsize=18)
    ax.set_yticks(np.linspace(0, 1, 6), [f'{label:.0f}' if label == 0 else
                                            f'{label:.1f}' for label in
                                            np.linspace(0, 1, 6)], fontsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()
