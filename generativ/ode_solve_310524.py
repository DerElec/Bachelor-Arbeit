import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt 
import time
import pandas as pd
import sympy as sp
import os
import tracing as tr
Omega_start = 4
Omega_end = 12
Omega_step = 1000

Delta_start = 1
Delta_end = 4
Delta_step = 0.1

start_time = time.time()

def time_wrapper(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # Start der Zeitmessung
        result = func(*args, **kwargs)  # Funktion ausführen
        end_time = time.perf_counter()  # Ende der Zeitmessung
        elapsed_time = end_time - start_time  # Berechnung der verstrichenen Zeit
        
        # Speichern der Zeit in einer .txt Datei
        with open('execution_times.txt', 'a') as file:
            file.write(f"Die Ausführungsdauer von '{func.__name__}' betrug: {elapsed_time:.4f} Sekunden\n")
        
        return result
    return wrapper

results = []
results_full = []

for Omega in np.arange(Omega_start, Omega_end, Omega_step):    
    for V in np.arange(-4, -0.5, 111):
        kappa = 1  # cavity loss rate
        gamma = 1  # rate from cavity and atom coupling
        Gamma = 2  # Decay rate from first excited state to ground
        delta_1 = 1  # Detuning between first excited state and cavity-Pump detuning
        delta_2 = 1  # Detuning between second excited state and Laser-Pump detuning (Pump meaning the pumping field)
        eta = 1
        
        vals = [kappa, gamma, Gamma, Omega, delta_1, delta_2, eta, V]
        
        #print(f'kappa={kappa}, gamma={gamma}, Gamma={Gamma}, V={V}, Omega={Omega}, eta={eta}, delta_1,2={delta_1},{delta_2}')
        
        T = 2000
        def dydt(t, y):
            a, a_dagger, ket00, ket01, ket10, ket11, ket22, ket21, ket12, ket20, ket02 = y
            
            da_dt = -kappa/2 * a - 1j * (gamma * ket01) + eta
            da_dagger_dt = np.conj(da_dt)
            dket00_dt = +Gamma * ket11 + 1j * gamma * (ket10 * a - ket01 * a_dagger)
            dket01_dt = -Gamma/2 * ket01 + 1j * (-delta_1 * ket01 + gamma * (ket11 * a - ket00 * a) - Omega/2 * ket02)
            dket10_dt = np.conj(dket01_dt)
            dket11_dt = -Gamma * ket11 + 1j * gamma * (ket01 * a_dagger - ket10 * a) + 1j * Omega/2 * (ket21 - ket12)
            dket22_dt = 1j * Omega / 2 * (ket12 - ket21)
            dket21_dt = -Gamma/2 * ket21 + 1j * (delta_2 * ket21 - delta_1 * ket21 - gamma * ket20 * a + Omega/2 * (ket11 - ket22) + 2 * V * ket21 * ket22)
            dket12_dt = np.conj(dket21_dt)
            dket02_dt = 1j * (-delta_2 * ket02 - Omega/2 * ket01 - 2 * V * ket02 * ket22 + gamma * ket12 * a)
            dket20_dt = np.conj(dket02_dt)
            
            return [da_dt, da_dagger_dt, dket00_dt, dket01_dt, dket10_dt, dket11_dt, dket22_dt, dket21_dt, dket12_dt, dket20_dt, dket02_dt]
        
        def solver(y0):
            sol = solve_ivp(dydt, (0, T), y0, t_eval=t_eval, method='DOP853', rtol=1e-13, atol=1e-16)
            return sol


        
        
        
        
        
        
        #stationary state
        #V=-delta_2 / 2 * ((Omega * kappa)**2 / (16 * (eta * gamma)**2) + 1)
        a0 = 2 * eta / kappa + 0j
        a_dagger_0 = 2 * eta / kappa + 0j
        psi00 = (delta_2 / (2 * V) + 1)
        
        psi22 = -delta_2 / (2 * V)
        
        V_cond= -delta_2 / 2 * ((Omega * kappa)**2 / (16 * (eta * gamma)**2) + 1)

        if V<V_cond:    
            psi20 = (Omega * kappa * delta_2 / (8 * eta * gamma * V))
            psi02 = np.conj(psi20)
        else:
            psi02 = -(4 * eta * gamma / (Omega * kappa) * (delta_2 / (2 * V) + 1))#np.conj(psi20 )
            psi20 = np.conj(psi02)
        
        psi11 = 0 + 0j
        psi10 = 0.0 + 0j
        psi01 = 0.0 + 0j
        psi21 = 0.0 + 0j
        psi12 = 0.0 + 0j
        
        
        ##########################
        
        # a0= 0
        # a_dagger_0=0
        # psi00 = 0
        # psi11 = 0.0 + 0j
        # psi22 = 1
        # psi20 = 0
        # psi02 = 0
        # psi10 = 0.0 + 0j
        # psi01 = 0.0 + 0j
        # psi21=0.0+0j
        # psi12=0.0+0j
        
        
        rho_start = sp.Matrix([
            [psi00, psi10, psi20],
            [psi01, psi11, psi21],
            [psi02, psi12, psi22]
        ])
        
        trace, eigenvals = tr.get_trace_eigenvals(rho_start)
        for i in eigenvals:
            if sp.re(i) < -np.e**(-12):
                print(f"Problematic, eigenvalue is {i}")
                break
        
        
        
        
        startcond = [a0, a_dagger_0, psi00, psi11, psi22, psi10, psi01, psi21, psi12, psi20, psi02]
        
        t_eval = np.linspace(0, T, 10000)
        y0 = [a0, a_dagger_0, psi00, psi01, psi10, psi11, psi22, psi21, psi12, psi20, psi02]
        
        sol = solver(y0)
        
        plt.plot(sol.t, sol.y[2].real, label='<0|0>')
        plt.plot(sol.t, sol.y[5].real, label='<1|1>')
        plt.plot(sol.t, sol.y[6].real, label='<2|2>')
        
        # plt.plot(sol.t, sol.y[2].imag, label='<0|0> im')
        # plt.plot(sol.t, sol.y[5].imag, label='<1|1> im')
        # plt.plot(sol.t, sol.y[6].imag, label='<2|2> im')
        
        
        
        
        plt.plot(sol.t, (sol.y[2] + sol.y[5] + sol.y[6]).real)
        
        plt.xlabel('Time in arbitrary units')
        plt.ylabel('Value of expectation value')
        plt.legend()
        plt.title(f'kappa={kappa}, gamma={gamma}, Gamma={Gamma}, V={V}, Omega={Omega}, eta={eta}, delta_1,2={delta_1},{delta_2}')
        plt.show()
        
        final_values = [sol.y[2][-1], sol.y[5][-1], sol.y[6][-1]]
        final_values_full = [sol.y[2], sol.y[5], sol.y[6]]
        
        rho_unstationary = sp.Matrix([
            [sol.y[6][-1], sol.y[7][-1], sol.y[9][-1]],
            [sol.y[8][-1], sol.y[5][-1], sol.y[4][-1]],
            [sol.y[10][-1], sol.y[3][-1], sol.y[2][-1]]
        ])
        
        trace, eigenvals = tr.get_trace_eigenvals(rho_unstationary)
        for i in eigenvals:
            if sp.re(i) < -np.e**(-12):
                print(f"Problematic, eigenvalue is {i}")
                break
        
        Averiging_rate = 500
        t_last_x_values = sol.t[-Averiging_rate:]
        averaged_vals = tr.calculate_avrg(Averiging_rate, sol.y[2], sol.y[5], sol.y[6], t_last_x_values)
        variances = tr.calculate_variance(Averiging_rate, sol.y[2], sol.y[5], sol.y[6], averaged_vals, t_last_x_values)
        
        results_full.append([V] + averaged_vals + [eigenvals] + [trace] + [vals] + [startcond] + [variances])
        
    df_full_new = pd.DataFrame(results_full, columns=['V', '<0|0>', '<1|1>', '<2|2>', 'eigenvals', 'purity', 'additional params', 'startcond', 'Variances'])
    
    if os.path.exists('results_full_3.pkl') and os.path.getsize('results_full_3.pkl') > 0:
        df_full_existing = pd.read_pickle('results_full_3.pkl')
        df_full_combined = pd.concat([df_full_existing, df_full_new], ignore_index=True)
    else:
        df_full_combined = df_full_new
    
    df_full_combined.to_pickle('results_full_3.pkl')
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Die Laufzeit der main-Funktion beträgt: {elapsed_time} Sekunden")
