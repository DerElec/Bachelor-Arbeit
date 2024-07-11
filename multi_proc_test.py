import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt 
import time
import pandas as pd
import tracing as tr  # Assuming tracing is a valid custom module
import sympy as sp
from multiprocessing import Pool
import os
from filelock import FileLock
import logging

logging.basicConfig(filename='error.log', level=logging.ERROR)
Omega_start = 1
Omega_end = 10
Omega_step = 1





def compute(eta):
    result_full = []
    for V in np.arange(-8, -0.1, 0.5): 
        delta_2 = 1
        kappa = 1  # cavity loss rate
        gamma = 1  # rate from cavity and atom coupling
        Gamma = 2  # Decay rate from first excited state to ground
        delta_1 = 1  # Detuning between first excited state and cavity-Pump detuning
        Omega = 1
        
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
        ################################
        # a0 = 0
        # a_dagger_0 = 0
        # psi00 = 0.32114059488639229-0.00000000000000000j
        # psi01 = -0.09934076258338194+0.11840399957296362j
        # psi02 = -0.09268208075981116-0.01307119588540335j
        # psi10 = -0.09934076258338195-0.11840399957296362j
        # psi11 = 0.35867514453638044+0.00000000000000001j
        # psi12 = 0.10886730973070291+0.06873136581451349j
        # psi20 = -0.09268208075981113+0.01307119588540335j
        # psi21 = 0.10886730973070288-0.06873136581451347j
        # psi22 = 0.32018426057722715-0.00000000000000001j
        startcond=[a0,a_dagger_0 ,psi00 ,psi01 ,psi02 ,psi10,psi11 ,psi12 ,
        psi20,psi21 ,psi22]
        
        
        rho_start = sp.Matrix([
            [psi00, psi10, psi20],
            [psi01, psi11, psi21],
            [psi02, psi12, psi22]
        ])
        
        trace, eigenvals = tr.get_trace_eigenvals(rho_start)
        for i in eigenvals:
            if sp.re(i) < -10**(-12):
                print(f"Problematic, eigenvalue is {i}")
                break
        
        
        
        
        
        t_eval = np.linspace(0, T, T_auflösung)
        y0 = [a0, a_dagger_0, psi00, psi01, psi10, psi11, psi22, psi21, psi12, psi20, psi02]

        sol = solve_ivp(dydt, (0, T), y0, t_eval=t_eval, method='DOP853', rtol=1e-8, atol=1e-10)

        final_values = [sol.y[2][-1], sol.y[5][-1], sol.y[6][-1]]
        rho_unstationary = sp.Matrix([
            [sol.y[6][-1], sol.y[7][-1], sol.y[9][-1]],
            [sol.y[8][-1], sol.y[5][-1], sol.y[4][-1]],
            [sol.y[10][-1], sol.y[3][-1], sol.y[2][-1]]
        ])
        
        purity, eigenvals = tr.get_trace_eigenvals(rho_unstationary)
        
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
        averaged_vals = tr.calculate_avrg(Averiging_rate, sol.y[2], sol.y[5], sol.y[6], t_last_x_values)
        variances = tr.calculate_variance(Averiging_rate, sol.y[2], sol.y[5], sol.y[6], averaged_vals, t_last_x_values)
        result_full.append([V] + averaged_vals + [eigenvals] + [purity] + [vals] + [startcond] + [variances])
        
        df_step = pd.DataFrame(result_full, columns=['V', '<0|0>', '<1|1>', '<2|2>', 'eigenvals', 'purity', 'additional params', 'startcond', 'Variances'])
        try:
            lock = FileLock("eta.pkl.lock")
            with lock:
                if os.path.exists(f'eta.pkl'):
                    df_existing = pd.read_pickle(f'eta.pkl')
                    df_combined = pd.concat([df_existing, df_step], ignore_index=True)
                else:
                    df_combined = df_step
                df_combined.to_pickle(f'eta.pkl')
        except Exception as e:
            logging.error("Error occurred while saving results", exc_info=True)
        
    return result_full

if __name__ == '__main__':
    start_time = time.time()
    with Pool(os.cpu_count()) as pool:
        Omega_values = np.arange(Omega_start, Omega_end, Omega_step)
        results_full = pool.map(compute, Omega_values)

    results_flat = [item for sublist in results_full for item in sublist]

    df_full_new = pd.DataFrame(results_flat, columns=['V', '<0|0>', '<1|1>', '<2|2>', 'eigenvals', 'purity', 'additional params', 'startcond', 'Variances'])


    print(f"Execution time: {time.time() - start_time} seconds")











# def compute(Omega):
#     result_full = []
#     for V in np.arange(-7, -0.1, 0.5): 
#         delta_2 = 1
#         kappa = 1  # cavity loss rate
#         gamma = 1  # rate from cavity and atom coupling
#         Gamma = 2  # Decay rate from first excited state to ground
#         delta_1 = 1  # Detuning between first excited state and cavity-Pump detuning
#         eta = 1
        
#         vals = [kappa, gamma, Gamma, Omega, delta_1, delta_2, eta, V]
#         T = 20000  # Time 
#         T_auflösung = 100 * T  # this gives how many datapoints are between 0 and T or in other words resolution
        
#         def dydt(t, y):
#             a, a_dagger, ket00, ket01, ket10, ket11, ket22, ket21, ket12, ket20, ket02 = y

#             da_dt = -kappa / 2 * a - 1j * (gamma * ket01) + eta
#             da_dagger_dt = np.conj(da_dt)

#             dket00_dt = Gamma * ket11 + 1j * gamma * (ket10 * a - ket01 * a_dagger)
#             dket01_dt = -Gamma / 2 * ket01 + 1j * (-delta_1 * ket01 + gamma * (ket11 * a - ket00 * a) - Omega / 2 * ket02)
#             dket10_dt = np.conj(dket01_dt)

#             dket11_dt = -Gamma * ket11 + 1j * gamma * (ket01 * a_dagger - ket10 * a) + 1j * Omega / 2 * (ket21 - ket12)
#             dket22_dt = 1j * Omega / 2 * (ket12 - ket21)

#             dket21_dt = -Gamma / 2 * ket21 + 1j * (delta_2 * ket21 - delta_1 * ket21 - gamma * ket20 * a + Omega / 2 * (ket11 - ket22) + 2 * V * ket21 * ket22)
#             dket12_dt = np.conj(dket21_dt)

#             dket02_dt = 1j * (-delta_2 * ket02 - Omega / 2 * ket01 - 2 * V * ket02 * ket22 + gamma * ket12 * a)
#             dket20_dt = np.conj(dket02_dt)

#             return [da_dt, da_dagger_dt, dket00_dt, dket01_dt, dket10_dt, dket11_dt, dket22_dt, dket21_dt, dket12_dt, dket20_dt, dket02_dt]

#         # a0 = 0
#         # a_dagger_0 = 0
#         # psi00 = 0
#         # psi11 = 0.0 + 0j
#         # psi22 = 1
#         # psi20 = 0
#         # psi02 = 0
#         # psi10 = 0.0 + 0j
#         # psi01 = 0.0 + 0j
#         # psi21 = 0.0 + 0j
#         # psi12 = 0.0 + 0j
#         ################################
#         a0 = 0
#         a_dagger_0 = 0
#         psi00 = 0.32114059488639229-0.00000000000000000j
#         psi01 = -0.09934076258338194+0.11840399957296362j
#         psi02 = -0.09268208075981116-0.01307119588540335j
#         psi10 = -0.09934076258338195-0.11840399957296362j
#         psi11 = 0.35867514453638044+0.00000000000000001j
#         psi12 = 0.10886730973070291+0.06873136581451349j
#         psi20 = -0.09268208075981113+0.01307119588540335j
#         psi21 = 0.10886730973070288-0.06873136581451347j
#         psi22 = 0.32018426057722715-0.00000000000000001j
#         startcond=[a0,a_dagger_0 ,psi00 ,psi01 ,psi02 ,psi10,psi11 ,psi12 ,
#         psi20,psi21 ,psi22]
        
#         t_eval = np.linspace(0, T, T_auflösung)
#         y0 = [a0, a_dagger_0, psi00, psi01, psi10, psi11, psi22, psi21, psi12, psi20, psi02]

#         sol = solve_ivp(dydt, (0, T), y0, t_eval=t_eval, method='DOP853', rtol=1e-8, atol=1e-10)

#         final_values = [sol.y[2][-1], sol.y[5][-1], sol.y[6][-1]]
#         rho_unstationary = sp.Matrix([
#             [sol.y[6][-1], sol.y[7][-1], sol.y[9][-1]],
#             [sol.y[8][-1], sol.y[5][-1], sol.y[4][-1]],
#             [sol.y[10][-1], sol.y[3][-1], sol.y[2][-1]]
#         ])
        
#         purity, eigenvals = tr.get_trace_eigenvals(rho_unstationary)
        
#         min_array1, max_array1 = np.min(sol.y[2]), np.max(sol.y[2])
#         min_array2, max_array2 = np.min(sol.y[5]), np.max(sol.y[5])
#         min_array3, max_array3 = np.min(sol.y[6]), np.max(sol.y[6])
        
#         overall_min, overall_max = min(min_array1, min_array2, min_array3), max(max_array1, max_array2, max_array3)
#         try:
#             if sp.re(overall_min) < -10**(-10):
#                 logging.error(f"Error, negative value for either 00, 11, 22: {overall_min}")
#             if sp.re(overall_max) > 1 + 10**(-10):
#                 logging.error(f"Error, expect value too big: {overall_max}")
#             for i in eigenvals:
#                 if sp.re(i) < -10**(-10):
#                     logging.error(f"Problematic, eigenvalue is {i}")
#             if sp.re(purity) < -10**(-10):
#                 logging.error("Purity isn't positive")
#         except Exception as e:
#             logging.error("Error occurred during eigenvalue checks", exc_info=True)

#         Averiging_rate = 500
#         t_last_x_values = sol.t[-Averiging_rate:]
#         averaged_vals = tr.calculate_avrg(Averiging_rate, sol.y[2], sol.y[5], sol.y[6], t_last_x_values)
#         variances = tr.calculate_variance(Averiging_rate, sol.y[2], sol.y[5], sol.y[6], averaged_vals, t_last_x_values)
#         result_full.append([V] + averaged_vals + [eigenvals] + [purity] + [vals] + [startcond] + [variances])
        
#         df_step = pd.DataFrame(result_full, columns=['V', '<0|0>', '<1|1>', '<2|2>', 'eigenvals', 'purity', 'additional params', 'startcond', 'Variances'])
#         try:
#             lock = FileLock("eta.pkl.lock")
#             with lock:
#                 if os.path.exists(f'eta.pkl'):
#                     df_existing = pd.read_pickle(f'eta.pkl')
#                     df_combined = pd.concat([df_existing, df_step], ignore_index=True)
#                 else:
#                     df_combined = df_step
#                 df_combined.to_pickle(f'eta.pkl')
#         except Exception as e:
#             logging.error("Error occurred while saving results", exc_info=True)
        
#     return result_full

# if __name__ == '__main__':
#     start_time = time.time()
#     with Pool(os.cpu_count()) as pool:
#         Omega_values = np.arange(Omega_start, Omega_end, Omega_step)
#         results_full = pool.map(compute, Omega_values)

#     results_flat = [item for sublist in results_full for item in sublist]

#     df_full_new = pd.DataFrame(results_flat, columns=['V', '<0|0>', '<1|1>', '<2|2>', 'eigenvals', 'purity', 'additional params', 'startcond', 'Variances'])


#     print(f"Execution time: {time.time() - start_time} seconds")
