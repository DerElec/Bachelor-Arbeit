# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 18:37:54 2024

@author: holli
"""

import numpy as np

kappa = 1  # cavity loss rate
gamma = 1  # rate from cavity and atom coupling
Gamma = 2  # Decay rate from first excited state to ground
Delta_1 = 1  # Detuning between first excited state and cavity-Pump detuning
eta = 1
g_0 = 1
Delta_2 = 1
#Omega=1
counter=0
for Omega in np.arange(0.1,12,0.1):
    for V in np.arange(-8,-0.1,0.1):

        # a = -2 * eta / kappa + 0j
        # a_dagger = -2 * eta / kappa + 0j
        # psi00 = (Delta_2 / (2 * V) + 1)
        # psi11 = 0 + 0j
        # psi22 = -Delta_2 / (2 * V)
        # psi20 = -(Omega * kappa * Delta_2 / (8 * eta * gamma * V))
        # psi02 = np.conj(psi20)
        # psi10 = 0.0 + 0j
        # psi01 = 0.0 + 0j
        # psi21 = 0.0 + 0j
        # psi12 = 0.0 + 0j
        a = 0
        a_dagger = 0
        psi00 = 0
        psi11 = 0 + 0j
        psi22 = 1
        psi20 = 0
        psi02 = 0
        psi10 = 0.0 + 0j
        psi01 = 0.0 + 0j
        psi21 = 0.0 + 0j
        psi12 = 0.0 + 0j
        
        if np.round(psi20)!= np.round(psi02):
            print(psi02,psi20)
        
        
        M = np.array([
            [-kappa / 2, 0, 0, 0, 0, 0, -1j * gamma * g_0, 0, 0, 0, 0],
            [0, -kappa / 2, 0, 0, 0, 1j * gamma * g_0, 0, 0, 0, 0, 0],
            [1j * gamma * g_0 * psi10, -1j * gamma * g_0 * psi01, 0, Gamma, 0, 1j * g_0 * gamma * a , -1j * g_0 * gamma * a_dagger, 0, 0, 0, 0],
            [-1j * gamma * g_0 * psi10, 1j * gamma * g_0 * psi01, 0, -Gamma, 0, -1j * gamma * g_0 * a, 1j * gamma * g_0 * a_dagger, 1j * Omega / 2, -1j * Omega / 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, -1j * Omega / 2, 1j * Omega / 2, 0, 0],
            [0, -1j * gamma * g_0 * (psi11 - psi00), 1j * gamma * g_0 * a_dagger, -1j * gamma * g_0 * a_dagger, 0, -Gamma / 2 + 1j*Delta_1, 0, 0, 0, 1j * Omega / 2, 0],
            [1j * gamma * g_0 * (psi11 - psi00), 0, -1j * gamma * g_0 * a, 1j * gamma * g_0 * a, 0, 0, -Gamma / 2 - 1j * Delta_1, 0, 0, 0, -1j * Omega / 2],
            [-1j * gamma * g_0 * psi20, 0, 0, 1j * Omega / 2, 1j*(2 * V * psi21 - Omega / 2), 0, 0, -Gamma / 2 + 1j * (Delta_2 - Delta_1 + 2 * V * psi22), 0, -1j * gamma * g_0 * a, 0],
            [0, 1j * gamma * g_0 * psi02, 0, -1j * Omega / 2, -2j * V * psi12 + 1j * Omega / 2, 0, 0, 0, -Gamma / 2 + 1j * (Delta_1 - Delta_2 - 2 * V * psi22), 0, 1j * gamma * g_0 * a_dagger],
            [0, -1j * gamma * g_0 * psi21, 0, 0, 2j * V * psi20, 1j * Omega / 2, 0, -1j * gamma * g_0 * a_dagger, 0, 1j * (Delta_2 + 2 * V * psi22), 0],
            [1j * g_0 * gamma * psi12, 0, 0, 0, -2j * V * psi02, 0, -1j * Omega / 2, 0, 1j * gamma * g_0 * a, 0, -1j * (Delta_2 + 2 * V * psi22)]
        ])
        
        # Berechne die Eigenwerte der Matrix M
        eigenvalues = np.linalg.eigvals(M)
        
        # Gebe die Eigenwerte aus
        #print(eigenvalues)
        
        for i in eigenvalues:
            if i > 10**(-13):
                counter+=1
print(counter)