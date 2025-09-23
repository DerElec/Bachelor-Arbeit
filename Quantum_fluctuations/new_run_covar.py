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

# Fügen Sie diese Funktion zu den anderen Hilfsfunktionen am Anfang hinzu
def check_symplectic_properties(s_matrix_np: np.ndarray, n: int, tol: float = 1e-9):
    """
    Überprüft eine gegebene Matrix auf ihre symplektischen Eigenschaften.

    Args:
        s_matrix_np (np.ndarray): Die zu prüfende (2n x 2n) Matrix.
        n (int): Die halbe Dimension der Matrix.
        tol (float): Toleranz für die numerischen Vergleiche.
    """
    dim = 2 * n
    if s_matrix_np.shape != (dim, dim):
        raise ValueError(f"Matrix muss die Dimension ({dim}, {dim}) haben.")

    # 1. Determinanten-Check
    det_s = np.linalg.det(s_matrix_np)
    is_det_one = np.isclose(det_s, 1.0, atol=tol)
    
    print("\n--- Symplektischer Check für Matrix s(t) ---")
    print(f"Determinante von s(t): {det_s:.6f}")
    print(f"Ist die Determinante ≈ 1? {'Ja' if is_det_one else 'Nein'}")

    # 2. Symplektische Bedingung: S^T * J * S = J
    # Erstelle die Standard-symplektische-Blockmatrix J
    I_n = np.identity(n)
    zero_n = np.zeros((n, n))
    J = np.block([[zero_n, I_n], [-I_n, zero_n]])
    
    # Berechne die linke Seite der Gleichung
    left_side = s_matrix_np.T @ J @ s_matrix_np
    
    # Prüfe, ob das Ergebnis nahe an J liegt
    is_stjs_j = np.allclose(left_side, J, atol=tol)
    
    print(f"Wird die Bedingung S^T J S = J erfüllt? {'Ja' if is_stjs_j else 'Nein'}")
    if not is_stjs_j:
        # Zeige die Abweichung, wenn die Bedingung verletzt ist
        diff_matrix = sp.Matrix(np.round(left_side - J, 4))
        print("Differenz (S^T J S - J):")
        pprint(diff_matrix)
    print("─" * 42)


# Modifizieren Sie diese bestehende Funktion
def get_and_check_matrices_at_time(t_target, sol):
    """
    Findet den nächstgelegenen Zeitpunkt, prüft Matrizen,
    gibt sie aus und gibt sie zurück.
    """
    # Finde den Index des Zeitpunkts, der t_target am nächsten kommt
    time_index = np.argmin(np.abs(sol.t - t_target))
    actual_time = sol.t[time_index]
    
    print("─" * 70)
    print(f"Matrizen für den Zeitpunkt t = {actual_time:.3f} (nächstgelegen zu t_target = {t_target})")
    print("─" * 70)
    
    # Extrahiere den vollständigen Zustandsvektor Y zu diesem Zeitpunkt
    Y_at_t = sol.y[:, time_index]
    
    # 1. Rekonstruiere Sigma und führe den PSD-Check durch
    sigma_flat = Y_at_t[10:]
    sigma_np = sigma_flat.reshape((10, 10))
    sigma_sym = sp.Matrix(sigma_np)
    eigenvalues_sigma = np.linalg.eigvalsh(sigma_np)
    is_psd = np.all(eigenvalues_sigma >= -1e-9)
    
    print("\n--- PSD-Check für Kovarianzmatrix Σ(t) ---")
    print(f"Ist Σ(t) positiv semidefinit? {'Ja' if is_psd else 'Nein'}")
    if not is_psd:
        print(f"Kleinster Eigenwert von Σ(t): {np.min(eigenvalues_sigma):.4f}")
    print("─" * 43)
    
    print("\nKovarianzmatrix Σ(t):")
    pprint(sigma_sym)
    
    # 2. Rekonstruiere die symplektische Matrix s
    m_at_t = Y_at_t[:10]
    gellman_values = m_at_t[2:]
    s_np = build_symplectic_matrix(gellman_values)
    s_sym = sp.Matrix(s_np)
    
    print("\nSymplektische Matrix s(t):")
    pprint(s_sym)

    # 3. FÜHRE DEN NEUEN SYMPLEKTISCHEN CHECK DURCH
    # Da unsere Matrix 10x10 ist, ist n = 5
    check_symplectic_properties(s_np, n=5)

    return sigma_sym, s_sym



def is_PDS(x):
    if np.all(np.linalg.eigvals(x) > 0):
        return True
    else:
        return np.linalg.eigenvals(x)
    

# ========================================================================
# HAUPTSKRIPT
# ========================================================================

if __name__ == "__main__":
    # 1. SETUP: Parameter und Anfangsbedingungen
    print("1. System wird eingerichtet...")
    g0, Delta1, Delta2, V, Gamma, Omega, kappa, eta = sp.symbols("g0 Delta1 Delta2 V Gamma Omega kappa eta")
    Omega_val = 8.0
    Gamma_val = 2.0
    V_val = -6.0 
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


    pprint(G_sym)





    solve = True
   # 4. LÖSEN
    if solve:
        print("3. Differentialgleichungssystem wird gelöst...")
        t_span = (0.0, 200.0)
        t_eval = np.linspace(*t_span, 1001)
        Y0 = np.concatenate([np.real(m0), Sigma0.flatten()])
        sol = solve_ivp(
            fun=lambda t, y: rhs_combined(t, y, params, g_func, w_func),
            t_span=t_span, y0=Y0, t_eval=t_eval, method='RK45'
        )
        print("Lösung erfolgreich berechnet.")

        # ========================================================================
        # 5. POST-PROCESSING & KOMBINIERTER PLOT
        # ========================================================================
        print("4. Berechne relevante Eigenwerte für G(t), W(t), M(t) und Sigma(t)...")

        # --- Berechnung für G(t) und W(t) ---
        min_eigenvalues_W = []
        max_real_eigenvalues_G = [] # NEUE LISTE für G-Eigenwerte

        for i in range(len(sol.t)):
            m_t = sol.y[:10, i]
            m_reordered = np.concatenate([m_t[2:], m_t[:2]])
            
            # Kleinster Eigenwert von W(t)
            W_t = w_func(*m_reordered)
            min_eigenvalues_W.append(np.min(np.linalg.eigvalsh(W_t)))

            # NEU: Größter Realteil der Eigenwerte von G(t)
            G_t = g_func(*m_reordered)
            eigenvalues_G = np.linalg.eigvals(G_t) # G ist nicht symmetrisch -> eigvals
            max_real_eigenvalues_G.append(np.max(np.real(eigenvalues_G)))

        # --- Berechnung für M(t) und Sigma(t) ---
        s_timeseries = build_symplectic_matrix_ts(sol)
        min_eigenvalue_trajectory_M = []
        min_eigenvalues_Sigma = []

        for i in range(len(sol.t)):
            sigma_t = sol.y[10:, i].reshape((10, 10))
            min_eigenvalues_Sigma.append(np.min(np.linalg.eigvalsh(sigma_t)))
            s_t = s_timeseries[i]
            M_t = sigma_t + (1j / 2) * s_t
            min_eigenvalue_trajectory_M.append(np.min(np.linalg.eigvalsh(M_t)))
        
        # --- Kombinierter Plot für alle vier Verläufe ---
        print("5. Erstelle kombinierten Plot...")
        plt.figure(figsize=(14, 8))
        
        plt.plot(sol.t, max_real_eigenvalues_G, label='Größter Realteil Eig(G(t)) (Stabilität)', color='green', linestyle='-.') # NEUER PLOT
        plt.plot(sol.t, min_eigenvalues_W, label='Kleinster Eig(W(t))', color='darkcyan', linestyle='--')
        plt.plot(sol.t, min_eigenvalues_Sigma, label=r'Kleinster Eig($\Sigma(t)$)', color='orange')
        plt.plot(sol.t, min_eigenvalue_trajectory_M, label=r'Kleinster Eig(M(t) = $\Sigma + \frac{i}{2}s$)', color='purple', linewidth=2)

        plt.axhline(0, color='red', linestyle=':', linewidth=2, label='Referenzlinie (y=0)')
        plt.title('Zeitentwicklung der relevanten Eigenwerte')
        plt.xlabel('Zeit')
        plt.ylabel('Wert des Eigenwerts / Realteils')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # ========================================================================
        # 6. WEITERE PLOTS (unverändert)
        # ========================================================================
        # (Die Plots für die Kovarianz-Elemente und Populationen bleiben hier unverändert)

        # ========================================================================
        # 6. WEITERE PLOTS (unverändert)
        # ========================================================================

        # --- Plot 1: Alle Elemente der Kovarianzmatrix ---
        print("Erstelle Plot: Zeitentwicklung der Kovarianzmatrix-Elemente...")
        sigma_trajectories = sol.y[10:, :]
        plt.figure(figsize=(12, 7))
        plt.plot(sol.t, sigma_trajectories.T, alpha=0.5) 
        plt.title('Zeitentwicklung aller 100 Elemente der Kovarianzmatrix Σ(t)')
        plt.xlabel('Zeit')
        plt.ylabel('Wert des Matrixelements')
        plt.grid(True)
        plt.show()

        # --- Plot 2: Populationen rho_ii ---
        print("Erstelle Plot: Zeitentwicklung der Populationen...")
        def reconstruct_populations(sol_obj):
            x3_t = sol_obj.y[4, :]
            x8_t = sol_obj.y[9, :]
            sum00_11 = (2 + np.sqrt(3) * x8_t) / 3
            rho00_t = np.real((sum00_11 + x3_t) / 2)
            rho11_t = np.real((sum00_11 - x3_t) / 2)
            rho22_t = 1 - rho00_t - rho11_t
            return rho00_t, rho11_t, rho22_t

        rho00, rho11, rho22 = reconstruct_populations(sol)
        trace = rho00 + rho11 + rho22

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

        print("\n\n" + "="*70)
        print("DETAILLIERTE MATRIX-ANALYSE FÜR EINEN ZEITPUNKT")
        print("="*70)
        # Überprüfe die Matrizen am Ende der Simulation
        get_and_check_matrices_at_time(t_target=sol.t[-1], sol=sol)
