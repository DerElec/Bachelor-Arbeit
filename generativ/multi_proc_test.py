import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt 
import time
import pandas as pd
import sympy as sp
from multiprocessing import Pool
import os
from filelock import FileLock
import logging
import tracing as tr

logging.basicConfig(filename='error.log', level=logging.ERROR)

Omega_start = 0
Omega_end = 12
Omega_step = 1

def is_positive_semidefinite(matrix):
    """Check if a matrix is positive semidefinite."""
    return np.all(np.linalg.eigvals(matrix) >= 0)

def is_density_matrix_physical(rho):
    """Check if the density matrix is physical (trace 1 and positive semidefinite)."""
    trace_one = np.isclose(np.trace(rho), 1.0, atol=1e-10)
    positive_semidefinite = is_positive_semidefinite(rho)
    return trace_one and positive_semidefinite

def compute(Omega):
    result_full = []
    for V in np.arange(-8, -0.01, 1): 
        delta_2 = 1
        kappa = 1  # cavity loss rate
        gamma = 1  # rate from cavity and atom coupling
        Gamma = 2  # Decay rate from first excited state to ground
        delta_1 = 1  # Detuning between first excited state and cavity-Pump detuning
        eta = 1
        vals = [kappa, gamma, Gamma, Omega, delta_1, delta_2, eta, V]
        T = 20000  # Time 
        T_auflösung = 100 * T  # this gives how many datapoints are between 0 and T or in other words resolution
        
        def dydt(t, y):
            a, a_dagger, ket00, ket01, ket10, ket11, ket22, ket21, ket12, ket20, ket02 = y

            da_dt = -kappa / 2 * a - 1j * (gamma * ket01) + eta
            da_dagger_dt = np.conj(da_dt)

            dket00_dt = Gamma * ket11 + 1j * gamma * (ket10 * a - ket01 * a_dagger)
            dket01_dt = -Gamma / 2 * ket01 + 1j * (-delta_1 * ket01 + gamma * (ket11 * a - ket00 * a) - Omega / 2 * ket02)
            dket10_dt = np.conj(dket01_dt)

            dket11_dt = -Gamma * ket11 + 1j * gamma * (ket01 * a_dagger - ket10 * a) + 1j * Omega / 2 * (ket21 - ket12)
            dket22_dt = 1j * Omega / 2 * (ket12 - ket21)

            dket21_dt = -Gamma / 2 * ket21 + 1j * (delta_2 * ket21 - delta_1 * ket21 - gamma * ket20 * a + Omega / 2 * (ket11 - ket22) + 2 * V * ket21 * ket22)
            dket12_dt = np.conj(dket21_dt)

            dket02_dt = 1j * (-delta_2 * ket02 - Omega / 2 * ket01 - 2 * V * ket02 * ket22 + gamma * ket12 * a)
            dket20_dt = np.conj(dket02_dt)

            return [da_dt, da_dagger_dt, dket00_dt, dket01_dt, dket10_dt, dket11_dt, dket22_dt, dket21_dt, dket12_dt, dket20_dt, dket02_dt]

        a0 = 2 * eta / kappa + 0j
        a_dagger_0 = 2 * eta / kappa + 0j
        psi00 = (delta_2 / (2 * V) + 1)
        psi22 = -delta_2 / (2 * V)
        
        V_cond = -delta_2 / 2 * ((Omega * kappa)**2 / (16 * (eta * gamma)**2) + 1)

        if V < V_cond:    
            psi20 = (Omega * kappa * delta_2 / (8 * eta * gamma * V))
            psi02 = np.conj(psi20)
        else:
            psi02 = -(4 * eta * gamma / (Omega * kappa) * (delta_2 / (2 * V) + 1)) # np.conj(psi20 )
            psi20 = np.conj(psi02)
        
        psi11 = 0 + 0j
        psi10 = 0.0 + 0j
        psi01 = 0.0 + 0j
        psi21 = 0.0 + 0j
        psi12 = 0.0 + 0j
        
        startcond = [a0, a_dagger_0, psi00, psi01, psi02, psi10, psi11, psi12, psi20, psi21, psi22]
        
        rho_start = np.array([
            [psi00, psi10, psi20],
            [psi01, psi11, psi21],
            [psi02, psi12, psi22]
        ], dtype=np.complex128)
        
        if not is_density_matrix_physical(rho_start):
            logging.error(f"Initial density matrix is not physical: {rho_start}")
            return []

        t_eval = np.linspace(0, T, T_auflösung)
        y0 = [a0, a_dagger_0, psi00, psi01, psi10, psi11, psi22, psi21, psi12, psi20, psi02]

        sol = solve_ivp(dydt, (0, T), y0, t_eval=t_eval, method='DOP853', rtol=1e-8, atol=1e-10)

        final_values = [sol.y[2][-1], sol.y[5][-1], sol.y[6][-1]]
        rho_unstationary = np.array([
            [sol.y[6][-1], sol.y[7][-1], sol.y[9][-1]],
            [sol.y[8][-1], sol.y[5][-1], sol.y[4][-1]],
            [sol.y[10][-1], sol.y[3][-1], sol.y[2][-1]]
        ], dtype=np.complex128)
        
        purity, eigenvals = tr.get_trace_eigenvals(sp.Matrix(rho_unstationary))
        
        min_array1, max_array1 = np.min(sol.y[2]), np.max(sol.y[2])
        min_array2, max_array2 = np.min(sol.y[5]), np.max(sol.y[5])
        min_array3, max_array3 = np.min(sol.y[6]), np.max(sol.y[6])
        
        overall_min, overall_max = min(min_array1, min_array2, min_array3), max(max_array1, max_array2, max_array3)
        try:
            if sp.re(overall_min) < -10**(-10):
                logging.error(f"Error, negative value for either 00, 11, 22: {overall_min}")
            if sp.re(overall_max) > 1 + 10**(-10):
                logging.error(f"Error, expect value too big: {overall_max}")
            for i in eigenvals:
                if sp.re(i) < -10**(-10):
                    logging.error(f"Problematic, eigenvalue is {i}")
            if sp.re(purity) < -10**(-10):
                logging.error("Purity isn't positive")
        except Exception as e:
            logging.error("Error occurred during eigenvalue checks", exc_info=True)

        Averiging_rate = 500
        
        t_last_x_values = sol.t[-Averiging_rate:]
      
        #<0|0>,<1|1>,<2|2>,<0|1>,<1|0>,<0|2>,<2|0>,<1|2>,<2|1>,a,a_dagger
        array_of_sols = [sol.y[2], sol.y[5], sol.y[6], sol.y[3], sol.y[4], sol.y[10], sol.y[9], sol.y[8], sol.y[7], sol.y[0], sol.y[1]]
        
        averaged_array, variance_array = tr.calculate_avrg_vars(Averiging_rate, array_of_sols, t_last_x_values)

        result_full.append([V] + averaged_array + [eigenvals] + [purity] + [vals] + [startcond] + variance_array)
        
        df_step = pd.DataFrame(result_full, columns=['V', '<0|0>', '<1|1>', '<2|2>','<0|1>','<1|0>','<0|2>','<2|0>','<1|2>','<2|1>','a','a_dagger',
                                                     'eigenvals', 'purity', 'additional params', 'startcond', 
                                                     'var_<0|0>', 'var_<1|1>', 'var_<2|2>','var_<0|1>','var_<1|0>','var_<0|2>','var_<2|0>','var_<1|2>','var_<2|1>','var_a','var_a_dagger',]
                               )
        try:
            lock = FileLock("stationary_140724.pkl.lock")
            with lock:
                if os.path.exists(f'stationary_140724.pkl'):
                    df_existing = pd.read_pickle(f'stationary_140724.pkl')
                    df_combined = pd.concat([df_existing, df_step], ignore_index=True)
                else:
                    df_combined = df_step
                df_combined.to_pickle(f'stationary_140724.pkl')
        except Exception as e:
            logging.error("Error occurred while saving results", exc_info=True)
        
    return result_full

if __name__ == '__main__':
    start_time = time.time()
    with Pool(os.cpu_count()) as pool:
        Omega_values = np.arange(Omega_start, Omega_end, Omega_step)
        results_full = pool.map(compute, Omega_values)

    print(f"Execution time: {time.time() - start_time} seconds")
