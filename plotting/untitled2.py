import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import seaborn as sns

# Load the file
df_full = pd.read_pickle(r"D:\Daten\Uni\Bachelor_Arbeit_old\eta.pkl")

# Select data points for plotting
df_filtered = df_full[df_full['V'] < 0]
V = df_filtered['V'].to_numpy()
purity = df_filtered['purity'].to_numpy()
variances = df_filtered['Variances']
values = df_filtered['additional params'].to_numpy()

# Extract parameters from 'additional params'
kappa = np.array([params[0] for params in values])
gamma = np.array([params[1] for params in values])
Omega = np.array([params[3] for params in values])
delta_2 = np.array([params[5] for params in values])
eta = np.array([params[6] for params in values])
V_vals = np.array([params[7] for params in values])

# Use the first value of kappa, gamma, Omega, delta_2
Omega = Omega[0]
delta_2 = delta_2[0]
kappa = kappa[0]
gamma=gamma[0]
# Create a meshgrid
eta_unique = np.unique(eta)
V_unique = np.unique(V_vals)

# Initialize matrices
heatmap_data_purity = np.full((len(V_unique), len(eta_unique)), np.nan)
variance_sum_matrix = np.full((len(V_unique), len(eta_unique)), np.nan)

def get_real_value(value):
    if isinstance(value, sp.Expr):
        return float(sp.re(value))
    return float(np.real(value))

# Fill the matrices
for i, v in enumerate(V_unique):
    for j, e in enumerate(eta_unique):
        indices = np.where((V_vals == v) & (eta == e))[0]
        if indices.size > 0:
            heatmap_data_purity[i, j] = get_real_value(purity[indices][0])
            variance_sum_matrix[i, j] = sum(get_real_value(var) for var in variances[indices[0]])

# Calculate the line points for the condition V = -delta_2/2 * ((Omega * kappa)^2 / (16 * (eta * gamma)^2) + 1)
eta_line = np.linspace(eta_unique.min(), eta_unique.max(), 500)

V_line = np.mean(-delta_2 / 2 * ((Omega * kappa)**2 / (16 * (eta_line[:, np.newaxis] * gamma)**2) + 1), axis=1)

# Plot the heatmap with overlaid variances
plt.figure(figsize=(10, 8))
ax = sns.heatmap(heatmap_data_purity, cmap='viridis', cbar_kws={'label': 'Purity'})

# Overlay the variances as contour lines
contour = plt.contour(variance_sum_matrix, levels=10, colors='white', linestyles='dashed')
plt.clabel(contour, inline=True, fontsize=8, fmt='%.2f')

# Plot the line for the given condition
plt.plot((eta_line - eta_unique.min()) / (eta_unique.max() - eta_unique.min()) * (len(eta_unique) - 1),
         (V_line - V_unique.min()) / (V_unique.max() - V_unique.min()) * (len(V_unique) - 1),
         'r-', label='V condition line')

# Set axis labels
ax.set_xlabel('eta')
ax.set_ylabel('V')

# Set linear scale for ticks
num_x_ticks = 20
num_y_ticks = 20
ax.set_xticks(np.linspace(0, len(eta_unique) - 1, num_x_ticks))
ax.set_xticklabels(np.round(np.linspace(eta_unique.min(), eta_unique.max(), num_x_ticks), 2))
ax.set_yticks(np.linspace(0, len(V_unique) - 1, num_y_ticks))
ax.set_yticklabels(np.round(np.linspace(V_unique.min(), V_unique.max(), num_y_ticks), 2))

plt.title('Purity Heatmap with Variance Sum Contours and V Condition Line')
plt.legend()
plt.show()
