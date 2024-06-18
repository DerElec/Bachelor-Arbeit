# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 10:53:27 2024

@author: Paul
"""

import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import seaborn as sns

# Lade die Daten aus der .pkl Datei
df_full = pd.read_pickle("results_full_random_without_V_with_delta.pkl")

# Wähle die zu plottenden Datenpunkte aus und filtere V < 0
df_filtered = df_full[df_full['V'] < 0]
V = df_filtered['V'].to_numpy()
purity = df_filtered['purity'].to_numpy()
Omega = np.array([params[3] for params in df_filtered['additional params']])

# Create a meshgrid
Omega_unique = np.unique(Omega)
V_unique = np.unique(V)

# Initialize matrix
heatmap_data_purity = np.full((len(V_unique), len(Omega_unique)), np.nan)

def get_real_value(value):
    if isinstance(value, sp.Expr):
        return float(sp.re(value))
    return float(np.real(value))

# Fill the matrix
for i, v in enumerate(V_unique):
    for j, o in enumerate(Omega_unique):
        indices = np.where((V == v) & (Omega == o))[0]
        if indices.size > 0:
            heatmap_data_purity[i, j] = get_real_value(purity[indices][0])

# Plot the heatmap for the full range
plt.figure(figsize=(10, 8))
ax = sns.heatmap(heatmap_data_purity, cmap='viridis', cbar_kws={'label': 'Purity'})
ax.set_xlabel('Omega')
ax.set_ylabel('V')

# Set linear scale for ticks
num_x_ticks = 20
num_y_ticks = 20

ax.set_xticks(np.linspace(0, len(Omega_unique) - 1, num_x_ticks))
ax.set_xticklabels(np.round(np.linspace(Omega_unique.min(), Omega_unique.max(), num_x_ticks), 2))
ax.set_yticks(np.linspace(0, len(V_unique) - 1, num_y_ticks))
ax.set_yticklabels(np.round(np.linspace(V_unique.min(), V_unique.max(), num_y_ticks), 2))

plt.show()

# Filter data for specific ranges
omega_filter = (Omega >= 4.5) & (Omega <= 5.5)
v_filter = (V >= -3) & (V <= -2)

omega_filter = (Omega >= 4) & (Omega <= 6)
v_filter = (V >= -3) & (V <= -1)



filtered_indices = omega_filter & v_filter

# Extract filtered values
Omega_filtered = Omega[filtered_indices]
V_filtered = V[filtered_indices]
purity_filtered = purity[filtered_indices]

# Create a meshgrid for the filtered values
Omega_unique_filtered = np.unique(Omega_filtered)
V_unique_filtered = np.unique(V_filtered)

# Initialize matrix for filtered values
heatmap_data_purity_filtered = np.full((len(V_unique_filtered), len(Omega_unique_filtered)), np.nan)

# Fill the matrix for filtered values
for i, v in enumerate(V_unique_filtered):
    for j, o in enumerate(Omega_unique_filtered):
        indices = np.where((V_filtered == v) & (Omega_filtered == o))[0]
        if indices.size > 0:
            heatmap_data_purity_filtered[i, j] = get_real_value(purity_filtered[indices][0])

# Plot the heatmap for the filtered range
plt.figure(figsize=(10, 8))
ax = sns.heatmap(heatmap_data_purity_filtered, cmap='viridis', cbar_kws={'label': 'Purity'})
ax.set_xlabel('Omega')
ax.set_ylabel('V')

# Set linear scale for ticks
num_x_ticks = 10
num_y_ticks = 10

ax.set_xticks(np.linspace(0, len(Omega_unique_filtered) - 1, num_x_ticks))
ax.set_xticklabels(np.round(np.linspace(Omega_unique_filtered.min(), Omega_unique_filtered.max(), num_x_ticks), 2))
ax.set_yticks(np.linspace(0, len(V_unique_filtered) - 1, num_y_ticks))
ax.set_yticklabels(np.round(np.linspace(V_unique_filtered.min(), V_unique_filtered.max(), num_y_ticks), 2))

plt.show()


omega_filter = (Omega >= 0) & (Omega <= 1)
v_filter = (V >= -1.5) & (V <= 0)



filtered_indices = omega_filter & v_filter

# Extract filtered values
Omega_filtered = Omega[filtered_indices]
V_filtered = V[filtered_indices]
purity_filtered = purity[filtered_indices]

# Create a meshgrid for the filtered values
Omega_unique_filtered = np.unique(Omega_filtered)
V_unique_filtered = np.unique(V_filtered)

# Initialize matrix for filtered values
heatmap_data_purity_filtered = np.full((len(V_unique_filtered), len(Omega_unique_filtered)), np.nan)

# Fill the matrix for filtered values
for i, v in enumerate(V_unique_filtered):
    for j, o in enumerate(Omega_unique_filtered):
        indices = np.where((V_filtered == v) & (Omega_filtered == o))[0]
        if indices.size > 0:
            heatmap_data_purity_filtered[i, j] = get_real_value(purity_filtered[indices][0])

# Plot the heatmap for the filtered range
plt.figure(figsize=(10, 8))
ax = sns.heatmap(heatmap_data_purity_filtered, cmap='viridis', cbar_kws={'label': 'Purity'})
ax.set_xlabel('Omega')
ax.set_ylabel('V')

# Set linear scale for ticks
num_x_ticks = 10
num_y_ticks = 10

ax.set_xticks(np.linspace(0, len(Omega_unique_filtered) - 1, num_x_ticks))
ax.set_xticklabels(np.round(np.linspace(Omega_unique_filtered.min(), Omega_unique_filtered.max(), num_x_ticks), 2))
ax.set_yticks(np.linspace(0, len(V_unique_filtered) - 1, num_y_ticks))
ax.set_yticklabels(np.round(np.linspace(V_unique_filtered.min(), V_unique_filtered.max(), num_y_ticks), 2))

plt.show()
