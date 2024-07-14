# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 11:05:11 2024

@author: holli
"""

import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm

# Datei laden
df_full = pd.read_pickle(r"D:\Daten\Uni\Bachelor_Arbeit_old\eta.pkl")




#<0|0>,<1|1>,<2|2>,<0|1>,<1|0>,<0|2>,<2|0>,<1|2>,<2|1>,a,a_dagger


# Wähle die zu plottenden Datenpunkte aus und filtere V > 0
df_filtered = df_full[df_full['V'] < 0]
psi_00_all = df_filtered['<0|0>'].to_numpy()
psi_11_all = df_filtered['<1|1>'].to_numpy()
psi_22_all = df_filtered['<2|2>'].to_numpy()
purity = df_filtered['purity'].to_numpy()
values = df_filtered['additional params'].to_numpy()
variances = df_filtered['Variances']

# Extrahiere alle Parameter aus den 'additional params'
kappa = np.array([params[0] for params in values])
gamma = np.array([params[1] for params in values])
Gamma = np.array([params[2] for params in values])
Omega = np.array([params[3] for params in values])
delta_1 = np.array([params[4] for params in values])
delta_2 = np.array([params[5] for params in values])
eta = np.array([params[6] for params in values])
V_vals = np.array([params[7] for params in values])



def get_real_value(value):
    if isinstance(value, sp.Expr):
        return float(sp.re(value))
    return float(np.real(value))

# Erstellen einer Funktion zur Auswahl der Achsenvariablen
def create_heatmap(x_var, y_var, z_var, z_title):
    x_unique = np.unique(x_var)
    y_unique = np.unique(y_var)
    
    heatmap_data = np.full((len(y_unique), len(x_unique)), np.nan)
    
    
    for i, y in enumerate(y_unique):
        for j, x in enumerate(x_unique):
            indices = np.where((y_var == y) & (x_var == x))[0]
            if indices.size > 0:
                heatmap_data[i, j] = get_real_value(z_var[indices][0])
    
    plot_heatmap_adjusted(heatmap_data, z_title, xlabel='x', ylabel='y', x_unique=x_unique, y_unique=y_unique)

def plot_heatmap_adjusted(data, title, xlabel='x', ylabel='y', cmap='viridis', x_unique=None, y_unique=None, log_scale=False):
    plt.figure(figsize=(10, 8))
    if log_scale:
        ax = sns.heatmap(data, cmap=cmap, cbar_kws={'label': title}, norm=LogNorm())
    else:
        ax = sns.heatmap(data, cmap=cmap, cbar_kws={'label': title})
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set linear scale for ticks
    num_x_ticks = 20
    num_y_ticks = 20

    if x_unique is not None:
        ax.set_xticks(np.linspace(0, len(x_unique) - 1, num_x_ticks))
        ax.set_xticklabels(np.round(np.linspace(x_unique.min(), x_unique.max(), num_x_ticks), 2))
    if y_unique is not None:
        ax.set_yticks(np.linspace(0, len(y_unique) - 1, num_y_ticks))
        ax.set_yticklabels(np.round(np.linspace(y_unique.min(), y_unique.max(), num_y_ticks), 2))

    plt.show()



# plot V over Omega
# create_heatmap(Omega, V_vals, psi_00_all, '<0|0>')
# create_heatmap(Omega, V_vals, psi_11_all, '<1|1>')
# create_heatmap(Omega, V_vals, psi_22_all, '<2|2>')
# create_heatmap(Omega, V_vals, purity, 'Purity')


# Plot V over Eta
# create_heatmap(eta, V_vals, psi_00_all, '<0|0>')
# create_heatmap(eta, V_vals, psi_11_all, '<1|1>')
# create_heatmap(eta, V_vals, psi_22_all, '<2|2>')
# create_heatmap(eta, V_vals, purity, 'Purity')


