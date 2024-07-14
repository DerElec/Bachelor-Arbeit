# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 21:53:22 2024

@author: holli
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Systemparameter
kappa = 1
gamma = 1
Gamma = 2
delta_1 = 1
delta_2 = 1
Omega = 6
V = -4
eta = 1
g_0 = 1

# Anfangswerte basierend auf den gegebenen Bedingungen
a0 = 2 * eta / kappa + 0j
a_dagger_0 = 2 * eta / kappa + 0j
psi00 = (delta_2 / (2 * V) + 1)
psi22 = -delta_2 / (2 * V)

V_cond = -delta_2 / 2 * ((Omega * kappa)**2 / (16 * (eta * gamma)**2) + 1)

if V < V_cond:    
    psi20 = (Omega * kappa * delta_2 / (8 * eta * gamma * V))
    psi02 = np.conj(psi20)
else:
    psi02 = -(4 * eta * gamma / (Omega * kappa) * (delta_2 / (2 * V) + 1))
    psi20 = np.conj(psi02)

psi11 = 0 + 0j
psi10 = 0.0 + 0j
psi01 = 0.0 + 0j
psi21 = 0.0 + 0j
psi12 = 0.0 + 0j

startcond = [a0, a_dagger_0, psi00, psi01, psi10, psi11, psi22, psi21, psi12, psi20, psi02]

# Differentialgleichungssystem
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

# Numerische Integration mit DOP853
t_span = (0, 5000)  # Längere Simulationszeit
t_eval = np.linspace(0, 5000, 500000)  # Höhere Auflösung
sol = solve_ivp(dydt, t_span, startcond, method='DOP853', t_eval=t_eval, dense_output=True, atol=1e-12, rtol=1e-12)

# Definition von t und y
t = sol.t
y = sol.y

# Maxima finden (nach den ersten 500 Sekunden)
def find_maxima(y, t, ignore_time=500):
    maxima_indices = []
    ignore_index = np.searchsorted(t, ignore_time)
    for i in range(ignore_index, len(y) - 1):
        if y[i - 1] < y[i] and y[i + 1] < y[i]:
            maxima_indices.append(i)
    return maxima_indices

# Maxima für jede Lösung finden
maxima_indices_list = []
periods = []

for i in range(len(y)):
    maxima_indices = find_maxima(np.real(y[i]), t)
    maxima_indices_list.append(maxima_indices)
    if len(maxima_indices) > 1:
        period = np.mean(np.diff(t[maxima_indices]))
        periods.append(period)
    else:
        periods.append(np.nan)

# Abstände der ersten und letzten 5 Maxima vergleichen
for i in range(len(y)):
    if len(maxima_indices_list[i]) >= 10:
        first_five_maxima = y[i, maxima_indices_list[i][:5]]
        last_five_maxima = y[i, maxima_indices_list[i][-5:]]
        print(f"Lösung {i}:")
        print(f"  Erste 5 Maxima: {first_five_maxima}")
        print(f"  Letzte 5 Maxima: {last_five_maxima}")

        # Abstände der Maxima berechnen
        first_distances = np.diff(t[maxima_indices_list[i][:5]])
        last_distances = np.diff(t[maxima_indices_list[i][-5:]])
        print(f"  Abstände der ersten 5 Maxima: {first_distances}")
        print(f"  Abstände der letzten 5 Maxima: {last_distances}")

# Definition der Jacobi-Matrix des linearisierten Systems
def jacobian_linearized(a, a_dagger, psi00, psi01, psi10, psi11, psi22, psi21, psi12, psi20, psi02, g_0=1, Delta_1=1, Delta_2=1):
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
    return M

# Floquet-Analyse über mehrere Perioden mit linearisiertem System
def floquet_analysis(sol, periods, num_periods=5):
    t_fin = sol.t[-1]
    t_start = t_fin - periods[-1] * num_periods
    y0 = sol.sol(t_start)
    n = len(y0)
    J0 = np.eye(n, dtype=complex)  # Initial matrix is identity

    def variational_eq(t, Y):
        Y = Y.reshape((n, n + 1))
        dy = dydt(t, Y[:, 0])
        a, a_dagger, psi00, psi01, psi10, psi11, psi22, psi21, psi12, psi20, psi02 = Y[:, 0]
        A = jacobian_linearized(a, a_dagger, psi00, psi01, psi10, psi11, psi22, psi21, psi12, psi20, psi02)
        
        dYdt = np.zeros_like(Y)
        dYdt[:, 0] = dy
        dYdt[:, 1:] = A @ Y[:, 1:]
        return dYdt.flatten()

    t_eval = np.linspace(t_start, t_fin, num_periods * 100)
    Y0 = np.hstack([y0[:, None], J0]).flatten()
    sol_var = solve_ivp(variational_eq, [t_start, t_fin], Y0, t_eval=t_eval, method='DOP853', dense_output=True)

    YT = sol_var.sol(t_fin).reshape((n, n + 1))
    monodromy_matrix = YT[:, 1:]
    floquet_multipliers = np.linalg.eigvals(monodromy_matrix)

    # Skalieren der Floquet-Multiplikatoren für eine Periode
    floquet_multipliers_single_period = np.power(floquet_multipliers, 1/num_periods)

    return floquet_multipliers_single_period

# Berechnung der Floquet-Multiplikatoren für jede Periode
floquet_results = []
for i, period in enumerate(periods):
    if not np.isnan(period):
        print(f"Lösung {i}: Periode = {period}")
        floquet_multipliers = floquet_analysis(sol, periods, num_periods=5)
        floquet_results.append((i, floquet_multipliers))
        print(f"Floquet-Multiplikatoren für Lösung {i}:")
        for j in floquet_multipliers:
            print(abs(j))

# Floquet-Ergebnisse anzeigen
import math 
for result in floquet_results:
    i, multipliers = result
    print(f"Lösung {i}:")
    for multiplier in multipliers:
        if math.isclose(multiplier, 1.0, rel_tol=10**(-8)):
            
            print(f" stable:{abs(multiplier)}")
            
        elif abs(multiplier)<1-10**(-8):
            print(f" decaying:{abs(multiplier)}")
        elif abs(multiplier)>1+10**(-8):
            print(f" increasing:{abs(multiplier)}")
            
            
            
            
            
            
