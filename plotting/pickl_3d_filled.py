import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm
import os

# Parameter zum Steuern der Plotausgabe
show_plots = False

# Dateipfad zu den Daten
#data_file_path = r"D:\Daten\Uni\Bachelor_Arbeit_old\daten\results_full_random_with_V_2_stationary.pkl"

data_file_path =r"D:\Daten\Uni\Bachelor_Arbeit_old\generativ\eta.pkl"



df_full = pd.read_pickle(data_file_path)

# Extrahiere den Dateinamen ohne Erweiterung für den Ordnernamen
data_filename = os.path.splitext(os.path.basename(data_file_path))[0]
output_directory = os.path.join(os.path.dirname(data_file_path), data_filename)

# Erstelle den Ordner, falls er nicht existiert
if not show_plots and not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Wähle die zu plottenden Datenpunkte aus und filtere V > 0
df_filtered = df_full[df_full['V'] < 0]
V = df_filtered['V'].to_numpy()
psi_00_all = df_filtered['<0|0>'].to_numpy()
psi_11_all = df_filtered['<1|1>'].to_numpy()
psi_22_all = df_filtered['<2|2>'].to_numpy()
eigenvals = df_filtered['eigenvals']
purity = df_filtered['purity'].to_numpy()
values = df_filtered['additional params'].to_numpy()
start_cond = df_filtered['startcond']
variances = df_filtered['Variances']

# Extract Omega and V values from additional params
Omega = np.array([params[3] for params in values])
V_vals = np.array([params[7] for params in values])
delta_2 = np.array([params[5] for params in values])
delta = np.array([params[4] for params in values])

# Create a meshgrid 
Omega_unique = np.unique(Omega)
V_unique = np.unique(V_vals)
delta_2_unique = np.unique(delta_2)

# Initialize matrices 
heatmap_data_00 = np.full((len(V_unique), len(Omega_unique)), np.nan)
heatmap_data_11 = np.full((len(V_unique), len(Omega_unique)), np.nan)
heatmap_data_22 = np.full((len(V_unique), len(Omega_unique)), np.nan)
heatmap_data_purity = np.full((len(V_unique), len(Omega_unique)), np.nan)
variance_matrices = [np.full((len(V_unique), len(Omega_unique)), np.nan) for _ in range(len(variances.iloc[0]))]

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
            for k in range(len(variances.iloc[indices[0]])):
                variance_matrices[k][i, j] = get_real_value(variances.iloc[indices[0]][k])

def plot_heatmap_adjusted(data, title, xlabel='Omega', ylabel='V', cmap='viridis', omega_lines=None, log_scale=False, save_path=None):
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

    ax.set_xticks(np.linspace(0, len(Omega_unique) - 1, num_x_ticks))
    ax.set_xticklabels(np.round(np.linspace(Omega_unique.min(), Omega_unique.max(), num_x_ticks), 2))
    ax.set_yticks(np.linspace(0, len(V_unique) - 1, num_y_ticks))
    ax.set_yticklabels(np.round(np.linspace(V_unique.min(), V_unique.max(), num_y_ticks), 2))

    if omega_lines:
        for omega in omega_lines:
            omega_idx = np.where(np.isclose(Omega_unique, omega))[0]
            if omega_idx.size > 0:
                plt.axvline(x=omega_idx[0], color='red', linewidth=2, linestyle='--')

    if show_plots:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()

# Plot each heatmap separately with phase transition line
omega_lines = False

plot_heatmap_adjusted(heatmap_data_00, '<0|0>', 'Omega', 'V', omega_lines=omega_lines,
                      save_path=os.path.join(output_directory, '<0|0>.png'))
plot_heatmap_adjusted(heatmap_data_11, '<1|1>', 'Omega', 'V', omega_lines=omega_lines,
                      save_path=os.path.join(output_directory, '<1|1>.png'))
plot_heatmap_adjusted(heatmap_data_22, '<2|2>', 'Omega', 'V', omega_lines=omega_lines,
                      save_path=os.path.join(output_directory, '<2|2>.png'))
plot_heatmap_adjusted(heatmap_data_purity, 'Purity', 'Omega', 'V',
                      save_path=os.path.join(output_directory, 'Purity.png'))

# Plot variances heatmap with logarithmic scale
for i, variance_matrix in enumerate(variance_matrices):
    plot_heatmap_adjusted(variance_matrix, f'Variance {i + 1}', 'Omega', 'V', log_scale=True,
                          save_path=os.path.join(output_directory, f'Variance_{i + 1}.png'))

# 2D plot of variance over V
variance_over_V = [get_real_value(variance[0]) + get_real_value(variance[1]) + get_real_value(variance[2]) for variance in variances]

plt.figure(figsize=(10, 8))
plt.plot(V, variance_over_V, 'o-', linewidth=0)
plt.xlabel('V')
plt.ylabel('Variances sum')
plt.title('Variance over V')
plt.grid(True)
if show_plots:
    plt.show()
else:
    plt.savefig(os.path.join(output_directory, 'Variance_over_V.png'))
    plt.close()

# 2D plot of variance over Omega
variance_over_Omega = [get_real_value(variance[0]) + get_real_value(variance[1]) + get_real_value(variance[2]) for variance in variances]

plt.figure(figsize=(10, 8))
plt.plot(Omega, variance_over_Omega, 'o', linewidth=0)
plt.xlabel('Omega')
plt.ylabel('Variances sum')
plt.title('Variance over Omega')
plt.grid(True)
if show_plots:
    plt.show()
else:
    plt.savefig(os.path.join(output_directory, 'Variance_over_Omega.png'))
    plt.close()

# 2D plot for purity over Omega and Delta_2
heatmap_data_purity_delta2 = np.full((len(delta_2_unique), len(Omega_unique)), np.nan)

def plot_heatmap_delta2(data, title, xlabel='Omega', ylabel='Delta_2', cmap='viridis', phase_transition=None, log_scale=False, save_path=None):
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

    ax.set_xticks(np.linspace(0, len(Omega_unique) - 1, num_x_ticks))
    ax.set_xticklabels(np.round(np.linspace(Omega_unique.min(), Omega_unique.max(), num_x_ticks), 2))
    ax.set_yticks(np.linspace(0, len(delta_2_unique) - 1, num_y_ticks))
    ax.set_yticklabels(np.round(np.linspace(delta_2_unique.min(), delta_2_unique.max(), num_y_ticks), 2))

    if phase_transition:
        plt.axhline(y=phase_transition, color='red', linewidth=2, linestyle='--')

    if show_plots:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()

# Fill the new matrix
for i, d in enumerate(delta_2_unique):
    for j, o in enumerate(Omega_unique):
        indices = np.where((delta_2 == d) & (Omega == o))[0]
        if indices.size > 0:
            heatmap_data_purity_delta2[i, j] = get_real_value(purity[indices][0])

plot_heatmap_delta2(heatmap_data_purity_delta2, 'Purity over Omega and Delta_2', 'Omega', 'Delta_2',
                    save_path=os.path.join(output_directory, 'Purity_over_Omega_and_Delta_2.png'))

unique_Omega_values = np.unique(Omega)
print("Verfügbare Omega-Werte:", unique_Omega_values)

# Plot Purity and psi_22 over V for all unique Omega values
for fixed_Omega in unique_Omega_values:
    indices_fixed_Omega = np.where(Omega == fixed_Omega)[0]

    if indices_fixed_Omega.size > 0:
        V_fixed_Omega = V[indices_fixed_Omega]
        psi_22_fixed_Omega = np.array([get_real_value(val) for val in psi_22_all[indices_fixed_Omega]])
        purity_fixed_Omega = np.array([get_real_value(val) for val in purity[indices_fixed_Omega]])

        plt.figure(figsize=(10, 8))
        plt.plot(V_fixed_Omega, psi_22_fixed_Omega, 'o-', label='psi_22', linewidth=0)
        plt.plot(V_fixed_Omega, purity_fixed_Omega, 's-', label='Purity', linewidth=0)
        plt.xlabel('V')
        plt.ylabel('Values')
        plt.title(f'Purity and psi_22 over V for Omega = {fixed_Omega}')
        plt.legend()
        plt.grid(True)
        if show_plots:
            plt.show()
        else:
            plt.savefig(os.path.join(output_directory, f'Purity_and_psi_22_over_V_for_Omega_{fixed_Omega}.png'))
            plt.close()
