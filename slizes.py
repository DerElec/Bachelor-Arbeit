import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import seaborn as sns
from matplotlib.colors import LogNorm

# Laden der Daten
df_full = pd.read_pickle("results_without_V_start_0.pkl")
#df_full = pd.read_pickle("results_without_V_start_0_dense.pkl")
# Wähle die zu plottenden Datenpunkte aus und filtere V < 0
df_filtered = df_full[df_full['V'] < 0]
V = df_filtered['V'].to_numpy()
psi_00_all = df_filtered['<0|0>'].to_numpy()
psi_11_all = df_filtered['<1|1>'].to_numpy()
psi_22_all = df_filtered['<2|2>'].to_numpy()
eigenvals = df_filtered['eigenvals']
purity = np.real(df_filtered['purity'].to_numpy())  # Verwenden Sie np.real
values = df_filtered['additional params'].to_numpy()
start_cond = df_filtered['startcond']
variances = df_filtered['Variances']

# Extract Omega and V values from additional params
Omega = np.array([params[3] for params in values])
V_vals = np.array([params[7] for params in values])
delta_2 = np.array([params[5] for params in values])
delta = np.array([params[4] for params in values])

unique_Omega_values = np.unique(Omega)
print("Verfügbare Omega-Werte:", unique_Omega_values)

def get_real_value(value):
    if isinstance(value, sp.Expr):
        return float(sp.re(value))
    return float(np.real(value))

# Plot Purity und psi_00, psi_11, psi_22 über V für alle einzigartigen Omega-Werte
for fixed_Omega in unique_Omega_values:
    indices_fixed_Omega = np.where(Omega == fixed_Omega)[0]

    if indices_fixed_Omega.size > 0:
        V_fixed_Omega = V[indices_fixed_Omega]
        psi_00_fixed_Omega = np.array([get_real_value(val) for val in psi_00_all[indices_fixed_Omega]])
        psi_11_fixed_Omega = np.array([get_real_value(val) for val in psi_11_all[indices_fixed_Omega]])
        psi_22_fixed_Omega = np.array([get_real_value(val) for val in psi_22_all[indices_fixed_Omega]])
        purity_fixed_Omega = np.array([get_real_value(val) for val in purity[indices_fixed_Omega]])
        
        # Berechne die Summe der Varianzen für die gegebenen Indizes
        variance_sums = np.array([np.sum([get_real_value(variances.iloc[idx][k]) for k in range(len(variances.iloc[idx]))]) for idx in indices_fixed_Omega])

        # Sortiere die Daten nach V in absteigender Reihenfolge
        sorted_indices = np.argsort(V_fixed_Omega)[::-1]
        V_fixed_Omega = V_fixed_Omega[sorted_indices]
        psi_00_fixed_Omega = psi_00_fixed_Omega[sorted_indices]
        psi_11_fixed_Omega = psi_11_fixed_Omega[sorted_indices]
        psi_22_fixed_Omega = psi_22_fixed_Omega[sorted_indices]
        purity_fixed_Omega = purity_fixed_Omega[sorted_indices]
        variance_sums = variance_sums[sorted_indices]

        plt.figure(figsize=(10, 8))
        plt.plot(V_fixed_Omega, psi_00_fixed_Omega, 'o-', label='psi_00', linewidth=0)
        plt.plot(V_fixed_Omega, psi_11_fixed_Omega, 's-', label='psi_11', linewidth=0)
        plt.plot(V_fixed_Omega, psi_22_fixed_Omega, '^-', label='psi_22', linewidth=0)
        #plt.plot(V_fixed_Omega, purity_fixed_Omega, 'x-', label='Purity', linewidth=0)
        #plt.plot(V_fixed_Omega, variance_sums, 'd-', label='Sum of Variances', linewidth=0)
        plt.xlabel('V')
        plt.ylabel('Values')
        plt.title(f'Psi Values over V for Omega = {fixed_Omega}')
        plt.legend()
        plt.grid(True)
        plt.show()
