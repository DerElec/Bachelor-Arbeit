import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
import numpy as np
import seaborn as sns
# Lade die Daten aus der .pkl Datei
df_full = pd.read_pickle("results_full_random_without_V.pkl")

# Wähle die zu plottenden Datenpunkte aus
V = df_full['V'].to_numpy()
psi_00_all = df_full['<0|0>'].to_numpy()
psi_11_all = df_full['<1|1>'].to_numpy()
psi_22_all = df_full['<2|2>'].to_numpy()
eigenvals = df_full['eigenvals']    
purity = df_full['purity'].to_numpy()              
values = df_full['additional params'].to_numpy()    
start_cond = df_full['startcond']   
variances=df_full['Variances']
#print(varianes)    
#psi_00_all = np.array([arr.real for arr in df_full['<0|0>']])  # Konvertiere komplexe Zahlen in reale Zahlen

# Values = [kappa, gamma, Gamma, Omega, delta_1, delta_2, eta, V]

# Extract Omega and V values from additional params
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

# Fill the matrices
for i, v in enumerate(V_unique):
    for j, o in enumerate(Omega_unique):
        indices = np.where((V_vals == v) & (Omega == o))[0]
        if indices.size > 0:
            heatmap_data_00[i, j] = np.real(psi_00_all[indices][0])
            heatmap_data_11[i, j] = np.real(psi_11_all[indices][0])
            heatmap_data_22[i, j] = np.real(psi_22_all[indices][0])
            heatmap_data_purity[i, j] = np.real(purity[indices][0])
            for k in range(len(variances[0])):
                variance_matrices[k][i, j] = np.real(variances[indices[0]][k])

# Function to plot each heatmap with linear scale
def plot_heatmap(data, title, xlabel='Omega', ylabel='V', cmap='viridis'):
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(data, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Set linear scale for ticks
    num_x_ticks = 10  # Set the number of ticks you want on the x-axis
    num_y_ticks = 10  # Set the number of ticks you want on the y-axis

    ax.set_xticks(np.linspace(0, len(Omega_unique) - 1, num_x_ticks))
    ax.set_xticklabels(np.round(np.linspace(Omega_unique.min(), Omega_unique.max(), num_x_ticks), 2))
    ax.set_yticks(np.linspace(0, len(V_unique) - 1, num_y_ticks))
    ax.set_yticklabels(np.round(np.linspace(V_unique.min(), V_unique.max(), num_y_ticks), 2))
    # Plot horizontal line at y=0
    x_zero_index = np.argmin(np.abs(Omega_unique))
    plt.axvline(x=x_zero_index, color='red', linestyle='--')
    y_zero_index = np.argmin(np.abs(V_unique))
    plt.axhline(y=y_zero_index, color='red', linestyle='--')
    plt.show()

# Plot each heatmap separately
plot_heatmap(heatmap_data_00, '<0|0>')
plot_heatmap(heatmap_data_11, '<1|1>')
plot_heatmap(heatmap_data_22, '<2|2>')
plot_heatmap(heatmap_data_purity, 'Purity')

# Plot variances heatmap
for i, variance_matrix in enumerate(variance_matrices):
    plot_heatmap(variance_matrix, f'Variance {i+1}')


