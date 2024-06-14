import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
import numpy as np
import seaborn as sns

# Lade die Daten aus der .pkl Datei
df_full = pd.read_pickle("results_full_random_without_V_2.pkl")

# Wähle die zu plottenden Datenpunkte aus
V = df_full['V'].to_numpy()
psi_00_all = df_full['<0|0>'].to_numpy()
psi_11_all = df_full['<1|1>'].to_numpy()
psi_22_all = df_full['<2|2>'].to_numpy()
eigenvals = df_full['eigenvals']    
purity = df_full['purity'].to_numpy()              
values = df_full['additional params'].to_numpy()    
start_cond = df_full['startcond']   
variances = df_full['Variances']

# Extract Omega and V values from additional params
#vals = [kappa, gamma, Gamma, Omega, delta_1, delta_2, eta, V]
Omega = np.array([params[3] for params in values])
V_vals = np.array([params[7] for params in values])

# Create a meshgrid for Omega and V
Omega_unique = np.unique(Omega)
V_unique = np.unique(V_vals)

# Initialize matrices for the heatmap
heatmap_data_00 = np.full((len(V_unique), len(Omega_unique)), np.nan)
heatmap_data_11 = np.full((len(V_unique), len(Omega_unique)), np.nan)
heatmap_data_22 = np.full((len(V_unique), len(Omega_unique)), np.nan)
heatmap_data_purity = np.full((len(V_unique), len(Omega_unique)), np.nan)
variance_matrices = [np.full((len(V_unique), len(Omega_unique)), np.nan) for _ in range(len(variances[0]))]

# Helper function to extract the real part of a value
def get_real_value(value):
    if isinstance(value, sp.Expr):
        return float(sp.re(value))
    return float(np.real(value))

# Fill the matrices
for i, v in enumerate(V_unique):
    for j, o in enumerate(Omega_unique):
        indices = np.where((V_vals == v) & (Omega == o))[0]
        if indices.size > 0:
            heatmap_data_00[i, j] = get_real_value(psi_00_all[indices][0])
            heatmap_data_11[i, j] = get_real_value(psi_11_all[indices][0])
            heatmap_data_22[i, j] = get_real_value(psi_22_all[indices][0])
            heatmap_data_purity[i, j] = get_real_value(purity[indices][0])
            for k in range(len(variances[0])):
                variance_matrices[k][i, j] = get_real_value(variances[indices[0]][k])

# Funktion zur Erstellung der Heatmap mit angepasster Achsenskalierung
def plot_heatmap_adjusted(data, title, xlabel='Omega', ylabel='V', cmap='viridis'):
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(data, cmap=cmap, cbar_kws={'label': title})
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Set linear scale for ticks
    num_x_ticks = 20  # Erhöhen der Anzahl der Ticks auf der x-Achse
    num_y_ticks = 20  # Erhöhen der Anzahl der Ticks auf der y-Achse

    ax.set_xticks(np.linspace(0, len(Omega_unique) - 1, num_x_ticks))
    ax.set_xticklabels(np.round(np.linspace(Omega_unique.min(), Omega_unique.max(), num_x_ticks), 2))
    ax.set_yticks(np.linspace(0, len(V_unique) - 1, num_y_ticks))
    ax.set_yticklabels(np.round(np.linspace(V_unique.min(), V_unique.max(), num_y_ticks), 2))
    # Plot horizontal line at y=0
    # x_zero_index = np.argmin(np.abs(Omega_unique))
    # plt.axvline(x=0, color='red', linestyle='--')
    # y_zero_index = np.argmin(np.abs(V_unique))
    # plt.axhline(y=0, color='red', linestyle='--')
    plt.show()

# Plot each heatmap separately
plot_heatmap_adjusted(heatmap_data_00, '<0|0>', 'Omega', 'V')
plot_heatmap_adjusted(heatmap_data_11, '<1|1>', 'Omega', 'V')
plot_heatmap_adjusted(heatmap_data_22, '<2|2>', 'Omega', 'V')
plot_heatmap_adjusted(heatmap_data_purity, 'Purity', 'Omega', 'V')

# Plot variances heatmap
for i, variance_matrix in enumerate(variance_matrices):
    plot_heatmap_adjusted(variance_matrix, f'Variance {i+1}', 'Omega', 'V')