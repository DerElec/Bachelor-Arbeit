import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
# Lade die Daten aus der .pkl Datei
df_full = pd.read_pickle('results_full.pkl')

# Wähle die zu plottenden Datenpunkte aus
V = df_full['V']
psi_00_all = df_full['<0|0>']
psi_11_all = df_full['<1|1>']
psi_22_all = df_full['<2|2>']
eigenvals = df_full['eigenvals']    
traces = df_full['purity']                                  

# Plot für jeden V-Wert erstellen
for i, v in enumerate(V):
    plt.figure(figsize=(10, 6))
    plt.plot(psi_00_all[i], label='<0|0>')
    plt.plot(psi_11_all[i], label='<1|1>')
    plt.plot(psi_22_all[i], label='<2|2>')
    
    # Labels und Titel hinzufügen
    plt.xlabel('Time Steps')
    plt.ylabel('Expectation Values')
    plt.legend()
    for j in eigenvals[i]:
        if sp.re(j)<0:
            plt.title(f'Expectation Values over Time for V = {v}, Purity = {traces[i]}, but problem')
            break
        else:
            plt.title(f'Expectation Values over Time for V = {v}, Purity = {traces[i]}')
    
    # Zeige den Plot
    plt.show()
