# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 18:37:54 2024

@author: holli
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

kappa = 1  # cavity loss rate
gamma = 1  # rate from cavity and atom coupling
Gamma = 2  # Decay rate from first excited state to ground
Delta_1 = 1  # Detuning between first excited state and cavity-Pump detuning
eta = 1
g_0 = 1
Delta_2 = 1

Omega_values = np.arange(0.1, 12, 0.1)
V_values = np.arange(-8, -0.5, 0.01)
Omega_grid, V_grid = np.meshgrid(Omega_values, V_values)
positive_eigenvalues_count = np.zeros_like(Omega_grid)
problematic_eigenvalues = np.zeros_like(Omega_grid)

for i, Omega in enumerate(Omega_values):
    for j, V in enumerate(V_values):
        # a = -2 * eta / kappa + 0j
        # a_dagger = -2 * eta / kappa + 0j
        # psi00 = (Delta_2 / (2 * V) + 1)
        # psi11 = 0 + 0j
        # psi22 = -Delta_2 / (2 * V)
        # V_cond= -Delta_2 / 2 * ((Omega * kappa)**2 / (16 * (eta * gamma)**2) + 1)

        # if V<V_cond:    
        #     psi20 = -(Omega * kappa * Delta_2 / (8 * eta * gamma * V))
        #     psi02 = np.conj(psi20)
        # else:
        #     psi02 = (4 * eta * gamma / (Omega * kappa) * (Delta_2 / (2 * V) + 1))#np.conj(psi20 )
        #     psi20 = np.conj(psi02)
        # psi10 = 0.0 + 0j
        # psi01 = 0.0 + 0j
        # psi21 = 0.0 + 0j
        # psi12 = 0.0 + 0j
        
        ####################
        # a = 0
        # a_dagger = 0
        # psi00 = 0
        # psi11 = 0 + 0j
        # psi22 = 1
        # psi20 = 0
        # psi02 = 0
        # psi10 = 0.0 + 0j
        # psi01 = 0.0 + 0j
        # psi21 = 0.0 + 0j
        # psi12 = 0.0 + 0j
        ########################
        a = 0
        a_dagger = 0
        ##############################################
        # psi00 = 0.16884275124707740+0.00000000000000000j
        # psi01 = 0.05285495678456895+0.06126674337862835j
        # psi02 = 0.00893939129055982+0.12284872741208981j
        # psi10 = 0.05285495678456896-0.06126674337862835j
        # psi11 = 0.49539487582407621-0.00000000000000000j
        # psi12 = -0.09613467423691810+0.02460639371916040j
        # psi20 = 0.00893939129055981-0.12284872741208981j
        # psi21 = -0.09613467423691806-0.02460639371916042j
        # psi22 = 0.33576237292884648+0.00000000000000000j
        #####################################
        psi00 = 0.42004291175716757-0.00000000000000000j
        psi01 = -0.00129269880313136-0.02430937710709920j
        psi02 = 0.00679350037220931-0.09547257190626260j
        psi10 = -0.00129269880313136+0.02430937710709921j
        psi11 = 0.52040871362966168+0.00000000000000000j
        psi12 = -0.08027973365285912-0.01822614412039732j
        psi20 = 0.00679350037220931+0.09547257190626257j
        psi21 = -0.08027973365285912+0.01822614412039732j
        psi22 = 0.05954837461317056-0.00000000000000000j
                

        rho_stationary = np.array([
            [psi00, psi10,psi20],
            [psi01, psi11,psi21],
            [psi02,psi12, psi22],
        ])
        eigenvals_rho = np.linalg.eigvals(rho_stationary)
        if np.trace(rho_stationary) != 1:
            print(np.trace(rho_stationary))
            #print("error not correct trace")
        for ev in eigenvals_rho:
            if np.real(ev) < -10**(-12):
                #print(f"Problematic, eigenvalue is {ev}")
                problematic_eigenvalues[j, i] = 1

        M = np.array([
            [-kappa / 2, 0, 0, 0, 0, 0, -1j * gamma * g_0, 0, 0, 0, 0],
            [0, -kappa / 2, 0, 0, 0, 1j * gamma * g_0, 0, 0, 0, 0, 0],
            [1j * gamma * g_0 * psi10, -1j * gamma * g_0 * psi01, 0, Gamma, 0, 1j * g_0 * gamma * a, -1j * g_0 * gamma * a_dagger, 0, 0, 0, 0],
            [-1j * gamma * g_0 * psi10, 1j * gamma * g_0 * psi01, 0, -Gamma, 0, -1j * gamma * g_0 * a, 1j * gamma * g_0 * a_dagger, 1j * Omega / 2, -1j * Omega / 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, -1j * Omega / 2, 1j * Omega / 2, 0, 0],
            [0, -1j * gamma * g_0 * (psi11 - psi00), 1j * gamma * g_0 * a_dagger, -1j * gamma * g_0 * a_dagger, 0, -Gamma / 2 + 1j * Delta_1, 0, 0, 0, 1j * Omega / 2, 0],
            [1j * gamma * g_0 * (psi11 - psi00), 0, -1j * gamma * g_0 * a, 1j * gamma * g_0 * a, 0, 0, -Gamma / 2 - 1j * Delta_1, 0, 0, 0, -1j * Omega / 2],
            [-1j * gamma * g_0 * psi20, 0, 0, 1j * Omega / 2, 1j * (2 * V * psi21 - Omega / 2), 0, 0, -Gamma / 2 + 1j * (Delta_2 - Delta_1 + 2 * V * psi22), 0, -1j * gamma * g_0 * a, 0],
            [0, 1j * gamma * g_0 * psi02, 0, -1j * Omega / 2, -2j * V * psi12 + 1j * Omega / 2, 0, 0, 0, -Gamma / 2 + 1j * (Delta_1 - Delta_2 - 2 * V * psi22), 0, 1j * gamma * g_0 * a_dagger],
            [0, -1j * gamma * g_0 * psi21, 0, 0, 2j * V * psi20, 1j * Omega / 2, 0, -1j * gamma * g_0 * a_dagger, 0, 1j * (Delta_2 + 2 * V * psi22), 0],
            [1j * g_0 * gamma * psi12, 0, 0, 0, -2j * V * psi02, 0, -1j * Omega / 2, 0, 1j * gamma * g_0 * a, 0, -1j * (Delta_2 + 2 * V * psi22)]
        ])

        # Berechne die Eigenwerte der Matrix M
        eigenvalues = np.linalg.eigvals(M)
        # Zähle die Anzahl der positiven Eigenwerte größer als 10**(-13)
        count_positive_eigenvalues = np.sum(np.real(eigenvalues) > 10**(-12))
        positive_eigenvalues_count[j, i] = count_positive_eigenvalues

# Heatmap erstellen
plt.figure(figsize=(10, 8))
plt.pcolormesh(Omega_grid, V_grid, positive_eigenvalues_count, shading='auto', cmap='viridis')
plt.colorbar(label='Number of Positive Eigenvalues > 10^(-12)')
plt.xlabel('Omega')
plt.ylabel('V')
plt.title('Heatmap of Positive Eigenvalues over Omega and V')
plt.gca().invert_yaxis()  # Invertiere die y-Achse

# Linie für die Bedingung V = -Delta_2 / 2 * ((Omega * kappa)**2 / (16 * (eta * gamma)**2) + 1)
V_line = -Delta_2 / 2 * ((Omega_values * kappa)**2 / (16 * (eta * gamma)**2) + 1)
plt.plot(Omega_values, V_line, color='blue', label='Condition Line')
plt.legend()

plt.show()

# Heatmap für problematische Eigenwerte erstellen
plt.figure(figsize=(10, 8))
plt.plot(Omega_values, V_line, color='red', label='Condition Line')
plt.pcolormesh(Omega_grid, V_grid, problematic_eigenvalues, shading='auto', cmap='Reds')
plt.colorbar(label='Problematic Eigenvalues (1 = Problematic)')
plt.xlabel('Omega')
plt.ylabel('V')
plt.title('Heatmap of Problematic Eigenvalues over Omega and V')
plt.gca().invert_yaxis()  # Invertiere die y-Achse

plt.show()
