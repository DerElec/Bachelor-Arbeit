import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import seaborn as sns

# Lade die Daten aus der .pkl Datei
df_full = pd.read_pickle("results_full_random_without_V_with_delta.pkl")

# Wähle die zu plottenden Datenpunkte aus
V = df_full['V'].to_numpy()
purity = df_full['purity'].to_numpy()
variances = df_full['Variances']
values = df_full['additional params'].to_numpy()

# Extract Omega and V values from additional params
Omega = np.array([params[3] for params in values])
V_vals = np.array([params[7] for params in values])

# Create a meshgrid 
Omega_unique = np.unique(Omega)
V_unique = np.unique(V_vals)

# Initialize matrices 
heatmap_data_purity = np.full((len(V_unique), len(Omega_unique)), np.nan)
variance_sum_matrix = np.full((len(V_unique), len(Omega_unique)), np.nan)

def get_real_value(value):
    if isinstance(value, sp.Expr):
        return float(sp.re(value))
    return float(np.real(value))

# Fill the matrices
for i, v in enumerate(V_unique):
    for j, o in enumerate(Omega_unique):
        indices = np.where((V_vals == v) & (Omega == o))[0]
        if indices.size > 0:
            heatmap_data_purity[i, j] = get_real_value(purity[indices][0])
            variance_sum_matrix[i, j] = sum(get_real_value(var) for var in variances[indices[0]])

# Plot the heatmap with overlaid variances
plt.figure(figsize=(10, 8))
ax = sns.heatmap(heatmap_data_purity, cmap='viridis', cbar_kws={'label': 'Purity'})

# Overlay the variances as contour lines
contour = plt.contour(variance_sum_matrix, levels=10, colors='white', linestyles='dashed')
plt.clabel(contour, inline=True, fontsize=8, fmt='%.2f')

# Set axis labels
ax.set_xlabel('Omega')
ax.set_ylabel('V')

# Set linear scale for ticks
num_x_ticks = 20
num_y_ticks = 20
ax.set_xticks(np.linspace(0, len(Omega_unique) - 1, num_x_ticks))
ax.set_xticklabels(np.round(np.linspace(Omega_unique.min(), Omega_unique.max(), num_x_ticks), 2))
ax.set_yticks(np.linspace(0, len(V_unique) - 1, num_y_ticks))
ax.set_yticklabels(np.round(np.linspace(V_unique.min(), V_unique.max(), num_y_ticks), 2))

plt.title('Purity Heatmap with Variance Sum Contours')
plt.show()
