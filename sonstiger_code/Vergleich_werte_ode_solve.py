# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 12:54:49 2024

@author: holli
"""

import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt 
import time 

def time_trac(function):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # Start der Zeitmessung
        result = function(*args, **kwargs)  # Funktion ausführen
        end_time = time.perf_counter()  # Ende der Zeitmessung
        elapsed_time = end_time - start_time  # Berechnung der verstrichenen Zeit
        print(f"Die Ausführungsdauer von '{function.__name__}' betrug: {elapsed_time:.4f} Sekunden ")
        return result
    return wrapper



for lt in np.arange(-300,-299.5,0.5):
    V=-0.45
    # Parameter for dgl
    print(V)
    kappa = 2 #cavity loss rate
    gamma = 1 #rate from cavity and atom coupling
    Gamma = 2 #Decay rate from first excited state to ground
    Omega = 2 #Laser atom coupling constant
    delta_1 = 1 #Detuning between first excited state and cavity-Pump detuning
    delta_2 = 1 #Detuning between second excited state and Laser-Pump detuning (Pump meaning the pumping field )
    eta=2
    #V = 0.2 #Atom-Atom coupling constant
    # V=-delta_2*((Omega*kappa/(4*eta*gamma))**2+1)/2
    #V=-50
    T=1000
     #Time 
    
    ########################################
    #stationary state 
    def get_constants(kappa,gamma,Omega,eta):
        #c_2 = 1 / np.sqrt((Omega**2 * kappa**2) / (16 * eta**2 * gamma**2 ) + 1)
        
        c_1 = (-Omega * kappa) / (4 * eta *  gamma) * np.sqrt(1 / ((Omega**2 * kappa**2) / (16 * eta**2 * gamma**2 ) + 1))
        c_2=np.sqrt(1-c_1*np.conj(c_1))
        return c_1,c_2
        
    c_1,c_2= get_constants(kappa,gamma,Omega,eta)
    
    #Random testing
    # a0= 2*eta/kappa+0j
    # a_dagger_0=2*eta/kappa+0j
    # psi00 = 0
    # psi11 = 0.0 + 0j
    # psi22 = 1
    # psi20 = 0
    # psi02 = 0
    # psi10 = 0.0 + 0j
    # psi01 = 0.0 + 0j
    # psi21=0.0+0j
    # psi12=0.0+0j
    ####################################
    #Stationary state
    a0= -2*eta/kappa+0j
    a_dagger_0=-2*eta/kappa+0j
    psi00 = delta_2/(2*V)+1
    psi11 = 0 +0j
    psi22 = -delta_2/(2*V)
    psi20 = -Omega*kappa*delta_2/(8*eta*gamma*V)
    psi02 = 4*eta*gamma/(Omega*kappa)*(delta_2/(2*V)+1)
    psi10 = 0.0 + 0j
    psi01 = 0.0 + 0j
    psi21=0.0+0j
    psi12=0.0+0j
    
    
    print(psi00+psi11+psi22)
    
    def dydt(t, y):
        # Zerlegung der Zustandsvariablen
        a, a_dagger, ket00, ket01, ket10, ket11, ket22, ket21, ket12, ket20, ket02 = y
    
        #Differentialgleichungen für die Mittelwerte der Operatoren
        da_dt = -kappa/2 *a - 1j*( gamma * ket01)-eta
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
        
        # print("da_dt:", da_dt)
        # print("dadagger_dt:", da_dagger_dt)
        # print("dket01_dt:", dket01_dt)
        # print("dket10_dt:", dket10_dt)
        # print("dket02_dt:", dket02_dt)
        # print("dket20_dt:", dket20_dt)
        # print("dket21_dt:", dket21_dt)
        # print("dket12_dt:", dket12_dt)
        # print(t)
        # if dket22_dt>0.1:
        #     print("stop")
        #     return 
        return [da_dt, da_dagger_dt, dket00_dt, dket01_dt, dket10_dt, dket11_dt, dket22_dt, dket21_dt, dket12_dt,dket20_dt,dket02_dt]
    
    
    t_eval = np.linspace(0, T, 10000)  
    y0 = [a0, a_dagger_0, psi00, psi01, psi10, psi11, psi22, psi21, psi12,psi20,psi02]  # Anfangsbedingungen, vereinfachte Anfangszustände für Nicht-Diagonalelemente
    
    #@time_trac
    def solver(y0):
        # sol = solve_ivp(dydt, (0, T), y0, t_eval=t_eval#, method='RK45', rtol=1e-12, atol=1e-15)
        #                 , method='RK45', rtol=1e-12, atol=1e-15)
        sol = solve_ivp(dydt, (0, T), y0, t_eval=t_eval#, method='RK45', rtol=1e-12, atol=1e-15)
                        , method='DOP853', rtol=1e-13, atol=1e-16)
        return sol
    sol=solver(y0)

    
    
    # plt.plot(sol.t, abs(sol.y[0]), label='a')
    # plt.plot(sol.t, abs(sol.y[1]), label='a^dagger')
    
    # #übergänge
    # plt.plot(sol.t, abs(sol.y[3]), label='<0|1>')
    # plt.plot(sol.t, abs(sol.y[4]), label='<1|0>')
    # plt.plot(sol.t, abs(sol.y[7]), label='<2|1>')
    # plt.plot(sol.t, abs(sol.y[8]), label='<1|2>')
    # plt.plot(sol.t, abs(sol.y[9]), label='<2|0>')
    # plt.plot(sol.t, abs(sol.y[10]), label='<0|2>')
    
    #adjoint zustände 
    plt.plot(sol.t, sol.y[2], label='<0|0>')
    plt.plot(sol.t, sol.y[5], label='<1|1>')
    plt.plot(sol.t, sol.y[6], label='<2|2>')
    print(sol.y[2][0],sol.y[5][0],sol.y[6][0])
    
    
    plt.plot(sol.t, sol.y[2]+sol.y[5]+sol.y[6])
    
    plt.xlabel('Time in arbitrary units')
    plt.ylabel('Value of expectation value')
    plt.legend()
    plt.title(f'Dynamic of the Expectation value of the operators for V={V}')
    plt.show()