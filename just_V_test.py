import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
import numpy as np
# Lade die Daten aus der .pkl Datei
df_full = pd.read_pickle(r"C:\Users\hphha\Desktop\BA Stuff\Bachelor-Arbeit\results_full_3.pkl")

# Wähle die zu plottenden Datenpunkte aus
V = df_full['V'].to_numpy()
psi_00_all = df_full['<0|0>'].to_numpy()
psi_11_all = df_full['<1|1>'].to_numpy()
psi_22_all = df_full['<2|2>'].to_numpy()
eigenvals = df_full['eigenvals']    
traces = df_full['purity']              
values = df_full['additional params']   
start_cond = df_full['startcond']   
varianes=df_full['Variances']
#print(varianes)    
psi_00_all = np.array([arr.real for arr in df_full['<0|0>']])  # Konvertiere komplexe Zahlen in reale Zahlen

# Values = [kappa, gamma, Gamma, Omega, delta_1, delta_2, eta, V]

# Extrahiere Omega aus den zusätzlichen Parametern
Omega = values.apply(lambda x: x[3]).to_numpy()

# Konvertiere die komplexen Zahlen in reale Zahlen und nimm den letzten Wert


# Erstelle ein 2D-Gitter
V_unique = np.unique(V)
Omega_unique = np.unique(Omega)

# Initialisiere eine Matrix für die Erwartungswerte
heatmap_data = np.full((len(V_unique), len(Omega_unique)), np.nan)

# Fülle die Matrix mit den Erwartungswerten
for i, v in enumerate(V_unique):
    for j, omega in enumerate(Omega_unique):
        mask = np.where((V == v) & (Omega == omega))[0]
        if len(mask) > 0:
            heatmap_data[i, j] = psi_00_all[mask[-1]]  # Nimm den letzten Wert

# Erstelle die Heatmap
fig, ax = plt.subplots()

c = ax.pcolormesh(Omega_unique, V_unique, heatmap_data, cmap='RdBu', shading='auto')
ax.set_title('Heatmap of <0|0> Expectation Values over Omega and V')
ax.set_xlabel('Omega')
ax.set_ylabel('V')
fig.colorbar(c, ax=ax)

plt.show()