import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt 
import time
import pandas as pd
import tracing as tr  # Assuming tracing is a valid custom module
import sympy as sp
from multiprocessing import Pool
import os


Omega_start=0
Omega_end=5
Omega_step=0.1

Delta_start=0.1
Delta_end=6
Delta_step=0.01


def compute(Omega):
    result_full = []
    for delta_2 in np.arange(Delta_start, Delta_end, Delta_step):
        
        kappa = 1 # cavity loss rate
        gamma = 1 # rate from cavity and atom coupling
        Gamma = 2 # Decay rate from first excited state to ground
    
        delta_1 = 1 # Detuning between first excited state and cavity-Pump detuning
        #delta_2 = 2 # Detuning between second excited state and Laser-Pump detuning (Pump meaning the pumping field)
        eta = 1
        
        V = -delta_2 / 2 * ((Omega * kappa)**2 / (16 * (eta * gamma)**2) + 1)
        #print(f'kappa={kappa},gamma={gamma},Gamma={Gamma},V={V},Omega={Omega},eta={eta},delta_1,2={delta_1},{delta_2}')
        vals = [kappa, gamma, Gamma, Omega, delta_1, delta_2, eta, V]
        T = 2000 # Time 
        #print(V)
        def dydt(t, y):
            # Decompose state variables
            a, a_dagger, ket00, ket01, ket10, ket11, ket22, ket21, ket12, ket20, ket02 = y

            # Differential equations for the mean values of the operators
            da_dt = -kappa / 2 * a - 1j * (gamma * ket01) + eta
            da_dagger_dt = np.conj(da_dt)  # Hermitian conjugate of da_dt

            dket00_dt = Gamma * ket11 + 1j * gamma * (ket10 * a - ket01 * a_dagger)
            dket01_dt = -Gamma / 2 * ket01 + 1j * (-delta_1 * ket01 + gamma * (ket11 * a - ket00 * a) - Omega / 2 * ket02)
            dket10_dt = np.conj(dket01_dt)  # Hermitian conjugate of dket01_dt

            dket11_dt = -Gamma * ket11 + 1j * gamma * (ket01 * a_dagger - ket10 * a) + 1j * Omega / 2 * (ket21 - ket12)
            dket22_dt = 1j * Omega / 2 * (ket12 - ket21)

            dket21_dt = -Gamma / 2 * ket21 + 1j * (delta_2 * ket21 - delta_1 * ket21 - gamma * ket20 * a + Omega / 2 * (ket11 - ket22) + 2 * V * ket21 * ket22)
            dket12_dt = np.conj(dket21_dt)  # Hermitian conjugate of dket21_dt

            dket02_dt = 1j * (-delta_2 * ket02 - Omega / 2 * ket01 - 2 * V * ket02 * ket22 + gamma * ket12 * a)
            dket20_dt = np.conj(dket02_dt)  # Hermitian conjugate of dket02_dt

            return [da_dt, da_dagger_dt, dket00_dt, dket01_dt, dket10_dt, dket11_dt, dket22_dt, dket21_dt, dket12_dt, dket20_dt, dket02_dt]

        # Initial conditions
        a0 = 0
        a_dagger_0 = 0
        psi00 = 0
        psi11 = 0.0 + 0j
        psi22 = 1
        psi20 = 0
        psi02 = 0
        psi10 = 0.0 + 0j
        psi01 = 0.0 + 0j
        psi21 = 0.0 + 0j
        psi12 = 0.0 + 0j
        
        startcond = [a0, a_dagger_0, psi00, psi11, psi22, psi10, psi01, psi21, psi12, psi20, psi02]

        t_eval = np.linspace(0, T, 10000)
        y0 = [a0, a_dagger_0, psi00, psi01, psi10, psi11, psi22, psi21, psi12, psi20, psi02]

        sol = solve_ivp(dydt, (0, T), y0, t_eval=t_eval, method='DOP853', rtol=1e-13, atol=1e-16)



        final_values = [sol.y[2][-1], sol.y[5][-1], sol.y[6][-1]]
        final_values_full = [sol.y[2], sol.y[5], sol.y[6]]
        result = [V] + final_values
        
        rho_unstationary = sp.Matrix([
            [sol.y[6][-1], sol.y[7][-1], sol.y[9][-1]],
            [sol.y[8][-1], sol.y[5][-1], sol.y[4][-1]],
            [sol.y[10][-1], sol.y[3][-1], sol.y[2][-1]]
        ])
        
        trace, eigenvals = tr.get_trace_eigenvals(rho_unstationary)
        
        for i in eigenvals:
            if sp.re(i) < -np.exp(-12):
                print(f"Problematic, eigenvalue is {i}")
                break
        
        Averiging_rate = 50
        averaged_vals = tr.calculate_avrg(Averiging_rate, sol.y[2], sol.y[5], sol.y[6])
        variances = tr.calculate_variance(Averiging_rate, sol.y[2], sol.y[5], sol.y[6], averaged_vals)
        #print([V] + averaged_vals + [eigenvals] + [trace] + [vals] + [startcond] + [variances])
        result_full.append([V] + averaged_vals + [eigenvals] + [trace] + [vals] + [startcond] + [variances])
        
    return result_full

# Multiprocessing zur Berechnung der Ergebnisse
if __name__ == '__main__':
    start_time = time.time()
    with Pool(processes=22) as pool:  # Anzahl der Prozesse angeben
        Omega_values = np.arange(Omega_start, Omega_end, Omega_step)
        results_full = pool.map(compute, Omega_values)

    # Flatten the results list
    results_flat = [item for sublist in results_full for item in sublist]

    # Create DataFrame
    df_full_new = pd.DataFrame(results_flat, columns=['V', '<0|0>', '<1|1>', '<2|2>', 'eigenvals', 'purity', 'additional params', 'startcond', 'Variances'])

    if os.path.exists('results_full_3.pkl'):
        df_full_existing = pd.read_pickle('results_full_3.pkl')
        df_full_combined = pd.concat([df_full_existing, df_full_new], ignore_index=True)
    else:
        df_full_combined = df_full_new
    
    # Save the combined DataFrames
    df_full_combined.to_pickle('results_full_3.pkl')

    print(f"Execution time: {time.time() - start_time} seconds")