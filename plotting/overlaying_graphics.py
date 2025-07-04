import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import seaborn as sns

# df_full = pd.read_pickle("results_full_random_without_V_2.pkl")
# df_full = pd.read_pickle("results_full_random_with_V.pkl")
# df_full = pd.read_pickle("results_full_random_with_V_2_stationary.pkl")
#df_full = pd.read_pickle("results_full_random_without_V_with_delta.pkl")

#df_full = pd.read_pickle("results_without_V_start_0_dense.pkl")
df_full = pd.read_pickle(r"D:\Daten\Uni\Bachelor_Arbeit_old\daten\results_full_random_without_V_3.pkl")

# Datei laden
#df_full = pd.read_pickle(r"D:\Daten\Uni\Bachelor_Arbeit_old\eta.pkl")

#df_full = pd.read_pickle("results_without_V_start_0.pkl")
# Wähle die zu plottenden Datenpunkte aus
df_filtered = df_full[df_full['V'] < 0]
V = df_filtered['V'].to_numpy()
purity = df_full['purity'].to_numpy()
variances = df_full['Variances']
values = df_full['additional params'].to_numpy()

# Extract parameters from additional params
kappa = np.array([params[0] for params in values])
gamma = np.array([params[1] for params in values])
# Gamma = np.array([params[2] for params in values])  # Not used in the formula
Omega = np.array([params[3] for params in values])
# delta_1 = np.array([params[4] for params in values])  # Not used in the formula
delta_2 = np.array([params[5] for params in values])
eta = np.array([params[6] for params in values])
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

# Calculate the line points for the condition V = -delta_2/2 * ((Omega * kappa)^2 / (16 * (eta * gamma)^2) + 1)
Omega_line = np.linspace(Omega_unique.min(), Omega_unique.max(), 500)
V_line = np.mean(-delta_2 / 2 * ((Omega_line[:, np.newaxis] * kappa[np.newaxis, :])**2 / (16 * (eta[np.newaxis, :] * gamma[np.newaxis, :])**2) + 1), axis=1)

# Plot the heatmap with overlaid variances
plt.figure(figsize=(10, 8))
ax = sns.heatmap(heatmap_data_purity, cmap='viridis', cbar_kws={'label': 'Purity'})

# Overlay the variances as contour lines
contour = plt.contour(variance_sum_matrix, levels=10, colors='white', linestyles='dashed')
plt.clabel(contour, inline=True, fontsize=8, fmt='%.2f')

# Plot the line for the given condition
plt.plot((Omega_line - Omega_unique.min()) / (Omega_unique.max() - Omega_unique.min()) * (len(Omega_unique) - 1),
         (V_line - V_unique.min()) / (V_unique.max() - V_unique.min()) * (len(V_unique) - 1),
         'r-', label='V condition line')

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

plt.title('Purity Heatmap with Variance Sum Contours and V Condition Line')
plt.legend()
plt.show()
