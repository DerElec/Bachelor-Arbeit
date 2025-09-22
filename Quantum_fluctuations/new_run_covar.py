import numpy as np 
import sympy as sp
from sympy import Matrix, simplify, pprint
from scipy.integrate import solve_ivp
import run_script as singlets
from run_script import convert_state, rhs_gellmann_qp_from_ket, rhs_gellmann_qp_from_x
import matplotlib.pyplot as plt
from typing import Sequence, Optional, Tuple, Dict
import covar_everything as covar_everything
import covar_everything as covar
import symplectic_matrix as symplect
# Python code (comments in English; console/output in German)

import numpy as np


# Python code (comments in English; console/output in German)

import numpy as np
def get_initial_covariance_matrix(y0_ket):
    """
    Calculates the initial covariance matrix Sigma(0) for a given
    initial state y0_ket.
    """
    if len(y0_ket) != 11:
        raise ValueError("The vector y0_ket must have 11 elements.")
    alpha = y0_ket[0]
    rho00, rho01, rho10, rho11, rho22, rho21, rho12, rho20, rho02 = y0_ket[2:]
    rho_3level = np.array([
        [rho00, rho01, rho02],
        [rho10, rho11, rho12],
        [rho20, rho21, rho22]
    ], dtype=complex)

    l1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    l2 = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex)
    l3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex)
    l4 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex)
    l5 = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=complex)
    l6 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex)
    l7 = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex)
    l8 = (1/np.sqrt(3)) * np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=complex)
    lambdas = [l1, l2, l3, l4, l5, l6, l7, l8]

    Sigma0 = np.zeros((10, 10))
    for i in range(8):
        for j in range(8):
            Li, Lj = lambdas[i], lambdas[j]
            anticommutator = Li @ Lj + Lj @ Li
            exp_val = np.trace(rho_3level @ anticommutator)
            Sigma0[i, j] = np.real(exp_val) / 2
            
    re_a, im_a = np.real(alpha), np.imag(alpha)
    Sigma0[8, 8] = 2 * re_a**2 + 0.5
    Sigma0[9, 9] = 2 * im_a**2 + 0.5
    Sigma0[8, 9] = Sigma0[9, 8] = 2 * re_a * im_a
    
    m_x = np.array([np.real(np.trace(rho_3level @ L)) for L in lambdas])
    m_q = np.sqrt(2) * re_a
    m_p = np.sqrt(2) * im_a
    
    for i in range(8):
        Sigma0[i, 8] = Sigma0[8, i] = m_x[i] * m_q
        Sigma0[i, 9] = Sigma0[9, i] = m_x[i] * m_p

    return Sigma0

def build_symplectic_matrix(populations):
    """
    Creates a 10x10 symplectic matrix for ONE set of population values.
    """
    if len(populations) != 8:
        raise ValueError("The 'populations' vector must have exactly 8 elements.")
    f = np.zeros((8, 8, 8))
    f[0, 1, 2] = 1; f[0, 3, 6] = 0.5; f[0, 4, 5] = 0.5
    f[1, 3, 5] = -0.5; f[1, 4, 6] = 0.5
    f[2, 3, 4] = 0.5; f[2, 5, 6] = -0.5
    f[3, 4, 7] = np.sqrt(3) / 2; f[5, 6, 7] = np.sqrt(3) / 2
    for i in range(8):
        for j in range(i + 1, 8):
            for k in range(j + 1, 8):
                if f[i, j, k] != 0:
                    f[j, i, k] = -f[i, j, k]; f[i, k, j] = -f[i, j, k]
                    f[k, i, j] = f[i, j, k]; f[k, j, i] = -f[i, j, k]
                    f[j, k, i] = f[i, j, k]
    m_8x8 = np.zeros((8, 8), dtype=float) # s is real
    for a in range(8):
        for b in range(8):
            m_8x8[a, b] = 2 * np.dot(f[a, b, :], populations)
    m_10x10 = np.zeros((10, 10), dtype=float)
    m_10x10[:8, :8] = m_8x8
    m_10x10[8, 9] = 1
    m_10x10[9, 8] = -1
    return m_10x10

def build_symplectic_matrix_ts(sol):
    """
    Erstellt eine Zeitreihe von symplektischen Matrizen aus einer DGL-Lösung.
    """
    matrix_timeseries = []
    
    # Die Erwartungswerte m(t) sind die ersten 10 Zeilen von sol.y
    m_timeseries = sol.y[:10, :] 

    # Iteriere über die Zeitschritte
    for i in range(m_timeseries.shape[1]):
        # Extrahiere x1...x8 aus den m-Werten für den Zeitschritt i
        gellman_values_at_t = m_timeseries[2:, i]
        
        # Erstelle die Matrix für diesen Zeitschritt
        s_at_t = build_symplectic_matrix(gellman_values_at_t)
        matrix_timeseries.append(s_at_t)
        
    return matrix_timeseries

def get_and_check_matrices_at_time(t_target, sol):
    """
    Findet den nächstgelegenen Zeitpunkt, prüft ob Sigma PSD ist,
    gibt die Matrizen aus und gibt sie zurück.

    Args:
        t_target (float): Der gewünschte Zeitpunkt.
        sol: Das von solve_ivp zurückgegebene Lösungsobjekt.
    
    Returns:
        tuple[sp.Matrix, sp.Matrix]: Ein Tupel mit der Kovarianzmatrix Sigma
                                     und der symplektischen Matrix s als SymPy-Matrizen.
    """
    # Finde den Index des Zeitpunkts, der t_target am nächsten kommt
    time_index = np.argmin(np.abs(sol.t - t_target))
    actual_time = sol.t[time_index]
    
    print("─" * 70)
    print(f"Matrizen für den Zeitpunkt t = {actual_time:.3f} (nächstgelegen zu t_target = {t_target})")
    print("─" * 70)
    
    # Extrahiere den vollständigen Zustandsvektor Y zu diesem Zeitpunkt
    Y_at_t = sol.y[:, time_index]
    
    # 1. Rekonstruiere Sigma
    sigma_flat = Y_at_t[10:]
    sigma_np = sigma_flat.reshape((10, 10))
    sigma_sym = sp.Matrix(sigma_np)

    # 2. Führe den PSD-Check für Sigma durch
    # Berechne die Eigenwerte der reellen, symmetrischen Matrix Sigma
    eigenvalues_sigma = np.linalg.eigvalsh(sigma_np)
    # Prüfe, ob alle Eigenwerte nicht-negativ sind (mit einer kleinen Toleranz)
    is_psd = np.all(eigenvalues_sigma >= -1e-9)
    
    print("\n--- PSD-Check für Kovarianzmatrix ---")
    print(f"Ist die Kovarianzmatrix Σ(t) positiv semidefinit? {'Ja' if is_psd else 'Nein'}")
    if not is_psd:
        print(f"Kleinster Eigenwert von Σ(t): {np.min(eigenvalues_sigma):.4f}")
    print("─" * 37)

    # 3. Gib Sigma aus
    print("\nKovarianzmatrix Σ(t):")
    pprint(sigma_sym)
    
    # 4. Rekonstruiere und drucke die symplektische Matrix s
    m_at_t = Y_at_t[:10]
    gellman_values = m_at_t[2:] # x1 bis x8
    s_np = build_symplectic_matrix(gellman_values)
    s_sym = sp.Matrix(s_np)
    
    print("\nSymplektische Matrix s(t):")
    pprint(s_sym)

    # 5. Gib die SymPy-Matrizen zurück
    return sigma_sym, s_sym


def is_PDS(x):
    if np.all(np.linalg.eigvals(x) > 0):
        return True
    else:
        return eigenvals(x)
    

# ========================================================================
# HAUPTSKRIPT
# ========================================================================
if __name__ == "__main__":
    # 1. SETUP: Parameter und Anfangsbedingungen
    print("1. System wird eingerichtet...")
    g0, Delta1, Delta2, V, Gamma, Omega, kappa, eta = sp.symbols("g0 Delta1 Delta2 V Gamma Omega kappa eta")
    Omega_val = 8.0
    Gamma_val = 2.0
    V_val = -0.5 * ((Omega_val / 4)**2 + 1)
    numeric_params = { 
        g0: 1, Delta1: 1, Delta2: 1, V: V_val, Gamma: Gamma_val, 
        Omega: Omega_val, kappa: 1, eta: 1 
    }
    params = {
        'g0':1, 'kappa': 1.0, 'gamma': 1.0, 'Gamma': Gamma_val,
        'Omega': Omega_val, 'delta1': 1.0, 'delta2': 1.0,
        'eta': 1.0, 'V': V_val
    }
    
    y0_ket = np.array([0+0j, 0+0j, 1+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j])
    m0 = convert_state(y0_ket)
    Sigma0 = get_initial_covariance_matrix(y0_ket)



    




    # 2. SYMBOLIK: Umwandlung in schnelle numerische Funktionen
    print("2. Symbolische Matrizen werden in numerische Funktionen umgewandelt...")
    m_syms = sp.symbols('m1:11')
    G_num_params, _, _, _, _, _, _, W_num_params, _, _, _ = covar_everything.get_important_matricies(numeric_params)
    print("Alle Parameter wurden erfolgreich ersetzt.")
    g_func = sp.lambdify(m_syms, G_num_params, 'numpy')
    w_func = sp.lambdify(m_syms, W_num_params, 'numpy')
    
    # 3. DGL: Definition der kombinierten RHS-Funktion
    def rhs_combined(t, Y, params, g_func, w_func):
        m = Y[:10]
        Sigma = Y[10:].reshape((10, 10))
        dm_dt = rhs_gellmann_qp_from_x(t, m, params)
        m_reordered = np.concatenate([m[2:], m[:2]])
        G = g_func(*m_reordered)
        W = w_func(*m_reordered)
        dSigma_dt = G @ Sigma + Sigma @ G.T + W
        return np.concatenate([dm_dt.astype(np.float64), dSigma_dt.flatten()])

    G_sym, sDs_sym, Z_sym, P_sym, Q_sym,sE_sym, Z_prime_sym, W_sym, Sigma_dt_sym, Sigma_sym, K_sym = covar.get_important_matricies_symbol()
    #pprint(G_sym)
    #pprint(W_sym)
    pprint(P_sym)




    solve=True
    # 4. LÖSEN
    if solve:
        print("3. Differentialgleichungssystem wird gelöst...")
        t_span = (0.0, 50.0)##############################################################
        t_eval = np.linspace(*t_span, 501)
        Y0 = np.concatenate([np.real(m0), Sigma0.flatten()])
        sol = solve_ivp(
            fun=lambda t, y: rhs_combined(t, y, params, g_func, w_func),
            t_span=t_span, y0=Y0, t_eval=t_eval, method='RK45'
        )
        print("Lösung erfolgreich berechnet.")

        # 5. POST-PROCESSING
        print("4. Erstelle s(t) Zeitreihe...")
        s_timeseries = build_symplectic_matrix_ts(sol)
        print("5. Führe PSD-Check durch und plotte kleinsten Eigenwert...")
        eigenvalue_trajectories = []
        for i in range(len(sol.t)):
            sigma_t = sol.y[10:, i].reshape((10, 10))
            s_t = s_timeseries[i]
            M_t = sigma_t + (1j / 2) * s_t
            eigenvals = np.linalg.eigvalsh(M_t)
            eigenvalue_trajectories.append(eigenvals)
        eigenvalue_trajectories = np.array(eigenvalue_trajectories)

        # Finde den kleinsten Eigenwert für jeden Zeitschritt
        min_eigenvalue_trajectory = np.min(eigenvalue_trajectories, axis=1)

        # 6. VISUALISIERUNG
        plt.figure(figsize=(12, 7))
        plt.plot(sol.t, min_eigenvalue_trajectory, label='Kleinster Eigenwert')
        plt.axhline(0, color='red', linestyle='--', linewidth=2, label='Positivitäts-Grenze (y=0)')
        plt.title('PSD Check: Zeitentwicklung des kleinsten Eigenwerts von Σ + i/2 s')
        plt.xlabel('Zeit')
        plt.ylabel('Eigenwert')
        plt.grid(True)

        plt.legend()
        plt.tight_layout()
        #print("\nGrafik 'psd_check_min_eigenvalue.png' wurde gespeichert.")
        plt.show()


        # --- Ändern Sie die Aufrufe am Ende Ihres `if __name__ == "__main__"`-Blocks zu diesem ---

        # Beispiel 1: Prüfe und erhalte die Matrizen am Ende der Simulation
        print("\n\n--- Matrizen am Ende der Simulation ---")
        Sigma_ende, s_ende = get_and_check_matrices_at_time(t_target=sol.t[-1], sol=sol)

        # Beispiel 2: Prüfe und erhalte die Matrizen in der Mitte der Simulation
        print("\n\n--- Matrizen zur Mitte der Simulation ---")
        Sigma_mitte, s_mitte = get_and_check_matrices_at_time(t_target=25.0, sol=sol)






        # --- Plot 1: Alle Elemente der Kovarianzmatrix ---
        print("Erstelle Plot 1: Zeitentwicklung der Kovarianzmatrix-Elemente...")
        sigma_trajectories = sol.y[10:, :]
        plt.figure(figsize=(12, 7))
        plt.plot(sol.t, sigma_trajectories.T, alpha=0.5) # alpha für bessere Sichtbarkeit
        plt.title('Zeitentwicklung aller 100 Elemente der Kovarianzmatrix Σ(t)')
        plt.xlabel('Zeit')
        plt.ylabel('Wert des Matrixelements')
        plt.grid(True)
        plt.show()

        # --- Plot 2: Populationen rho_ii ---
        print("Erstelle Plot 2: Zeitentwicklung der Populationen...")
        
        def reconstruct_populations(sol_obj):
            # Extrahiere x3 und x8 aus den Erwartungswerten
            # Indizes: 0=Q, 1=P, 2=x1, 3=x2, 4=x3, ..., 9=x8
            x3_t = sol_obj.y[4, :]
            x8_t = sol_obj.y[9, :]
            
            # Rekonstruiere die Populationen gemäß den Formeln
            sum00_11 = (2 + np.sqrt(3) * x8_t) / 3
            rho00_t = np.real((sum00_11 + x3_t) / 2)
            rho11_t = np.real((sum00_11 - x3_t) / 2)
            rho22_t = 1 - rho00_t - rho11_t
            
            return rho00_t, rho11_t, rho22_t

        rho00, rho11, rho22 = reconstruct_populations(sol)
        trace = rho00 + rho11 + rho22 # Die Summe sollte immer 1 sein

        plt.figure(figsize=(12, 7))
        plt.plot(sol.t, rho00, label=r'$\rho_{00}(t)$')
        plt.plot(sol.t, rho11, label=r'$\rho_{11}(t)$')
        plt.plot(sol.t, rho22, label=r'$\rho_{22}(t)$')
        plt.plot(sol.t, trace, '--', label='Summe (Spur)', linewidth=2)
        
        plt.title('Zeitentwicklung der Populationen')
        plt.xlabel('Zeit')
        plt.ylabel('Population')
        plt.legend()
        plt.grid(True)
        plt.show()
