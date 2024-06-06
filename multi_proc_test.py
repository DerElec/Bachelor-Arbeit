# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 16:55:47 2024

@author: Paul
"""

import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt 
import time
import pandas as pd
import tracing as tr
import sympy as sp
from multiprocessing import Pool



def compute(Omega):
    # Parameter für DGL
    
    kappa = 1 #cavity loss rate0
    gamma = 1 #rate from cavity and atom coupling
    Gamma = 2 #Decay rate from first excited state to ground

    delta_1 = 1 #Detuning between first excited state and cavity-Pump detuning
    delta_2 = 2 #Detuning between second excited state and Laser-Pump detuning (Pump meaning the pumping field )
    eta=1

    V=-delta_2/2*((Omega*kappa)**2/(16*(eta*gamma)**2)+1)
    print(V)
    print(Omega)
    T=2000
     #Time 

    # stationärer Zustand
    def get_constants(kappa, gamma, Omega, eta):
        c_1 = (-Omega * kappa) / (4 * eta * gamma) * np.sqrt(1 / ((Omega**2 * kappa**2) / (16 * eta**2 * gamma**2) + 1))
        c_2 = np.sqrt(1 - c_1 * np.conj(c_1))
        return c_1, c_2

    c_1, c_2 = get_constants(kappa, gamma, Omega, eta)

     #Random testing
    a0= 0
    a_dagger_0=0
    psi00 = 0
    psi11 = 0.0 + 0j
    psi22 = 1
    psi20 = 0
    psi02 = 0
    psi10 = 0.0 + 0j
    psi01 = 0.0 + 0j
    psi21=0.0+0j
    psi12=0.0+0j
    ##################################################
    #Random testing
    # a0= 2*eta/kappa+0j
    # a_dagger_0=2*eta/kappa+0j
    # psi00 = 0
    # psi11 = 0.0 + 0j
    # psi22 = 1
    # psi20 = -Omega*kappa*delta_2/(8*eta*gamma*V)
    # psi02 = 4*eta*gamma/(Omega*kappa)*(delta_2/(2*V)+1)
    # psi10 = 0.0 + 0j
    # psi01 = 0.0 + 0j
    # psi21=0.0+0j
    # psi12=0.0+0j
    ####################################
    #Stationary state# perturbed
    # eps=0.05
    # other=1-eps
    # a0= -2*eta/kappa+0j
    # a_dagger_0=-2*eta/kappa+0j
    # psi00 = other*(delta_2/(2*V)+1)+eps/2
    # psi11 = 0 +0j
    # psi22 = -other*delta_2/(2*V)+eps/2
    # psi20 = other*(-Omega*kappa*delta_2/(8*eta*gamma*V))
    # psi02 = np.conj(psi20)
    # #other*(4*eta*gamma/(Omega*kappa)*(delta_2/(2*V)+1))
    # psi10 = 0.0 + 0j
    # psi01 = 0.0 + 0j
    # psi21=0.0+0j
    # psi12=0.0+0j
    #####################################
    #Stationary state# unperturbed
    # a0= -2*eta/kappa+0j
    # a_dagger_0=-2*eta/kappa+0j
    # psi00 = (delta_2/(2*V)+1)
    # psi11 = 0 +0j
    # psi22 = -delta_2/(2*V)
    # psi20 = (-Omega*kappa*delta_2/(8*eta*gamma*V))
    # psi02 = (4*eta*gamma/(Omega*kappa)*(delta_2/(2*V)+1))
    # psi10 = 0.0 + 0j
    # psi01 = 0.0 + 0j
    # psi21=0.0+0j
    # psi12=0.0+0j
    #print(psi20,psi02)
    # Define the elements of the density matrix rho

    def dydt(t, y):
        # Zerlegung der Zustandsvariablen
        a, a_dagger, ket00, ket01, ket10, ket11, ket22, ket21, ket12, ket20, ket02 = y
    
        #Differentialgleichungen für die Mittelwerte der Operatoren
        da_dt = -kappa/2 *a - 1j*( gamma * ket01)+eta
        #da_dagger_dt = -kappa/2 * a_dagger + 1j * gamma * ket10
        da_dagger_dt =np.conj(da_dt)
        
        dket00_dt = +Gamma * ket11 + 1j * gamma * (ket10 * a - ket01 * a_dagger)
        
        dket01_dt = -Gamma/2 * ket01 + 1j * (-delta_1 * ket01 + gamma * (ket11 * a - ket00 * a) - Omega/2 * ket02)
        #dket10_dt = -Gamma/2 * ket10 - 1j * (- (omega_1 - omega_c) * ket10 + gamma * (ket11_k * a_dagger - ket00 * a) - Omega/2 * ket20)
        dket10_dt =np.conj(dket01_dt)
        
        dket11_dt = -Gamma * ket11 + 1j * gamma * (ket01 * a_dagger - ket10 * a) + 1j * Omega/2 * (ket21 - ket12)
        dket22_dt = 1j * Omega / 2 * (ket12 - ket21)
        
        dket21_dt = -Gamma/2 * ket21 + 1j * (delta_2 * ket21 - delta_1 * ket21 - gamma * ket20 * a + Omega/2 * (ket11 - ket22) + 2*V * ket21*ket22)
        dket12_dt = np.conj(dket21_dt)
        
        dket02_dt= 1j*(-delta_2*ket02-Omega/2*ket01-2*V*ket02*ket22+gamma *ket12 *a)
        dket20_dt=np.conj(dket02_dt)
        

        return [da_dt, da_dagger_dt, dket00_dt, dket01_dt, dket10_dt, dket11_dt, dket22_dt, dket21_dt, dket12_dt,dket20_dt,dket02_dt]
    
    

    t_eval = np.linspace(0, T, 10000)
    y0 = [a0, a_dagger_0, psi00, psi01, psi10, psi11, psi22, psi21, psi12, psi20, psi02]

    sol = solve_ivp(dydt, (0, T), y0, t_eval=t_eval, method='DOP853', rtol=1e-13, atol=1e-16)
    
    final_values = [sol.y[2][-1], sol.y[5][-1], sol.y[6][-1]]
    final_values_full = [sol.y[2], sol.y[5], sol.y[6]]
    result = [V] + final_values
    result_full = [V] + final_values_full
    
    rho_unstationary = sp.Matrix([
        [sol.y[6][-1], sol.y[7][-1], sol.y[9][-1]],
        [sol.y[8][-1], sol.y[5][-1], sol.y[4][-1]],
        [sol.y[10][-1], sol.y[3][-1], sol.y[2][-1]]
    ])
    trace, eigenvals = tr.get_trace_eigenvals(rho_unstationary)
    for i in eigenvals:
        if sp.re(i) < 0:
            print(f"problematic, eigenvalue is {i}")
    print(f"Purity is {trace}")
    
    return result, result_full
# Multiprocessing zur Berechnung der Ergebnisse
if __name__ == '__main__':
    with Pool(processes=4) as pool:  # Anzahl der Prozesse angeben
        Omega_values = np.arange(-2, 10, 0.25)
        results = pool.map(compute, Omega_values)

    # Ergebnisse trennen
    results, results_full = zip(*results)
    
    # Daten in DataFrames speichern
    df = pd.DataFrame(results, columns=['V', '<0|0>', '<1|1>', '<2|2>'])
    df_full = pd.DataFrame(results_full, columns=['V', '<0|0>', '<1|1>', '<2|2>'])

    # Ergebnisse speichern
    df.to_excel('results.xlsx', index=False, engine='openpyxl')
    df_full.to_excel('results_full.xlsx', index=False, engine='openpyxl')
    df.to_pickle('results.pkl')
    df_full.to_pickle('results_full.pkl')
    
    
    