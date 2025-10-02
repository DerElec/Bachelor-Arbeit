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
# Python code (comments in English; console/output in German)
import scipy
import numpy as np

sp.init_printing(use_unicode=True, wrap_line=False)#, num_columns=200)

# Python code (comments in English; console/output in German)

import numpy as np
# def get_initial_covariance_matrix(y0_ket):
#     """
#     Calculates the initial covariance matrix Sigma(0) for a given
#     initial state y0_ket.
#     """
#     if len(y0_ket) != 11:
#         raise ValueError("The vector y0_ket must have 11 elements.")
#     alpha = y0_ket[0]
#     rho00, rho01, rho10, rho11, rho22, rho21, rho12, rho20, rho02 = y0_ket[2:]
#     rho_3level = np.array([
#         [rho00, rho01, rho02],
#         [rho10, rho11, rho12],
#         [rho20, rho21, rho22]
#     ], dtype=complex)

#     l1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
#     l2 = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex)
#     l3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex)
#     l4 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex)
#     l5 = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=complex)
#     l6 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex)
#     l7 = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex)
#     l8 = (1/np.sqrt(3)) * np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=complex)
#     lambdas = [l1, l2, l3, l4, l5, l6, l7, l8]

#     Sigma0 = np.zeros((10, 10))
#     for i in range(8):
#         for j in range(8):
#             Li, Lj = lambdas[i], lambdas[j]
#             anticommutator = Li @ Lj + Lj @ Li
#             exp_val = np.trace(rho_3level @ anticommutator)
#             Sigma0[i, j] = np.real(exp_val) / 2
            
#     re_a, im_a = np.real(alpha), np.imag(alpha)
#     Sigma0[8, 8] = 2 * re_a**2 + 0.5
#     Sigma0[9, 9] = 2 * im_a**2 + 0.5
#     Sigma0[8, 9] = Sigma0[9, 8] = 2 * re_a * im_a
    
#     m_x = np.array([np.real(np.trace(rho_3level @ L)) for L in lambdas])
#     m_q = np.sqrt(2) * re_a
#     m_p = np.sqrt(2) * im_a
    
#     for i in range(8):
#         Sigma0[i, 8] = Sigma0[8, i] = m_x[i] * m_q
#         Sigma0[i, 9] = Sigma0[9, i] = m_x[i] * m_p

#     return Sigma0

def get_initial_covariance_matrix(y0_ket):
    import numpy as np
    alpha = y0_ket[0]
    rho00, rho01, rho10, rho11, rho22, rho21, rho12, rho20, rho02 = y0_ket[2:]
    rho = np.array([[rho00, rho01, rho02],
                    [rho10, rho11, rho12],
                    [rho20, rho21, rho22]], dtype=complex)

    # Gell-Mann:
    l1 = np.array([[0,1,0],[1,0,0],[0,0,0]],complex)
    l2 = np.array([[0,-1j,0],[1j,0,0],[0,0,0]],complex)
    l3 = np.array([[1,0,0],[0,-1,0],[0,0,0]],complex)
    l4 = np.array([[0,0,1],[0,0,0],[1,0,0]],complex)
    l5 = np.array([[0,0,-1j],[0,0,0],[1j,0,0]],complex)
    l6 = np.array([[0,0,0],[0,0,1],[0,1,0]],complex)
    l7 = np.array([[0,0,0],[0,0,-1j],[0,1j,0]],complex)
    l8 = (1/np.sqrt(3))*np.array([[1,0,0],[0,1,0],[0,0,-2]],complex)
    lams = [l1,l2,l3,l4,l5,l6,l7,l8]

    # <λ_a>
    m = np.array([np.trace(rho@L) for L in lams], dtype=complex).real

    S = np.zeros((10,10), float)
    # Spin covariances = 1/2⟨{λa,λb}⟩ - ⟨λa⟩⟨λb⟩
    for a in range(8):
        for b in range(8):
            val = np.trace(rho@(lams[a]@lams[b] + lams[b]@lams[a]))
            S[a,b] = 0.5*val.real - m[a]*m[b]

    # Boson block (coherent state):
    S[8,8] = 0.5
    S[9,9] = 0.5
    # No initial atom-field correlations:
    # S[a,8]=S[a,9]=S[8,a]=S[9,a]=0

    return S
def build_symplectic_matrix(populations):
    """
    Creates a 10x10 symplectic matrix for ONE set of population values.
    """
    if len(populations) != 8:
        raise ValueError("The 'populations' vector must have exactly 8 elements.")
    f = np.zeros((8, 8, 8))
    f[0, 1, 2] = 1; 
    f[0, 3, 6] = 0.5; 
    f[0, 4, 5] = -0.5
    f[1, 3, 5] = 0.5; 
    f[1, 4, 6] = 0.5
    f[2, 3, 4] = 0.5; 
    f[2, 5, 6] = -0.5
    f[3, 4, 7] = np.sqrt(3) / 2; 
    f[5, 6, 7] = np.sqrt(3) / 2
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
    # 2. s(t) = commutator form (NOT a symplectic transform)
    m_at_t = Y_at_t[:10]
    gellman_values = m_at_t[2:]
    s_np = build_commutator_form_s_2f(gellman_values)
    s_sym = sp.Matrix(s_np)

    print("\nKommutator-Form s(t):")
    pprint(s_sym)
    check_commutator_form(s_np, gellman_values)

    # Heisenberg/RS-Check
    M = sigma_np + 0.5j*s_np
    eigM_min = np.min(np.linalg.eigvalsh(M))
    print("\nKleinster Eigenwert von M(t)=Σ+i/2 s:", eigM_min)
    return sigma_sym, s_sym



def is_PDS(x):
    if np.all(np.linalg.eigvals(x) > 0):
        return True
    else:
        return np.linalg.eigenvals(x)
    


def su3_f_tensor_standard():
    """Standard SU(3) f_{abc} for Tr[λ_a λ_b]=2 δ_ab and [λ_a,λ_b]=2i f_{abc} λ_c."""
    import numpy as np
    f = np.zeros((8,8,8), dtype=float)
    # Base triples (ascending):
    f[0,1,2] = +1.0            # f_123
    f[0,3,6] = +0.5            # f_147
    f[0,4,5] = -0.5            # f_156  (SIGN FIX)
    f[1,3,5] = +0.5            # f_246  (SIGN FIX)
    f[1,4,6] = +0.5            # f_257
    f[2,3,4] = +0.5            # f_345
    f[2,5,6] = -0.5            # f_367
    f[3,4,7] = np.sqrt(3)/2.0  # f_458
    f[5,6,7] = np.sqrt(3)/2.0  # f_678
    # total antisymmetry:
    for a in range(8):
        for b in range(8):
            for c in range(8):
                if a<b<c and f[a,b,c]!=0:
                    val = f[a,b,c]
                    f[b,a,c] = -val; f[a,c,b] = -val; f[c,a,b] =  val
                    f[c,b,a] = -val; f[b,c,a] =  val
    return f

def check_commutator_form(s, m_vec, tol=1e-10):
    import numpy as np
    ok = True
    if not np.allclose(s, -s.T, atol=tol):
        print("s ist nicht schiefsymmetrisch."); ok=False
    qp = s[8:10,8:10]
    if not np.allclose(qp, np.array([[0,1],[-1,0]]), atol=tol):
        print("Q,P-Block ist nicht kanonisch."); ok=False
    m1,m2,m3 = m_vec[0], m_vec[1], m_vec[2]
    target = np.array([[0,   2*m3, -2*m2],
                       [-2*m3, 0,   2*m1],
                       [2*m2, -2*m1, 0]])
    if not np.allclose(s[0:3,0:3], target, atol=tol):
        print("SU(2)-Unterblock (λ1..λ3) stimmt nicht.")
        print("Erwartet:\n", target, "\nIst:\n", s[0:3,0:3]); ok=False
    if ok: print("s-Form OK (schief, QP ok, SU(2)-Unterblock ok).")

def build_commutator_form_s_2f(m_vec):
    """
    Build the 10x10 commutator form s with s_ab = 2 * sum_c f_{abc} m_c.
    Order: indices 0..7 -> λ1..λ8, indices 8..9 -> Q,P.
    """
    import numpy as np
    assert len(m_vec)==8, "m_vec must be the 8 singles <λ1..λ8>."
    f = su3_f_tensor_standard()
    s = np.zeros((10,10), dtype=float)
    # Spin block: s_ab = 2 f_{abc} m_c
    for a in range(8):
        for b in range(8):
            s[a,b] = 2.0 * np.dot(f[a,b,:], m_vec)
    # Bosonic block
    s[8,9] = +1.0
    s[9,8] = -1.0
    return s

def build_s_form_timeseries_2f(sol):
    """Time series of the commutator form s(t) with s_ab=2 f_{abc} m_c."""
    s_times = []
    m_times = sol.y[:10, :]
    for i in range(m_times.shape[1]):
        gellman = m_times[2:, i]     # λ1..λ8
        s_times.append(build_commutator_form_s_2f(gellman))
    return s_times

import numpy as np
from scipy.linalg import eig

# =========================================
# Constants
# =========================================

# --- Minimal, robust Schur-based canonicalization (drop-in) ---

import numpy as np
from scipy.linalg import schur


import numpy as np
from typing import Dict, List
from scipy.linalg import schur
import numpy as np
import sympy as sp
from sympy import Matrix, pprint



def find_R_and_J_for_skew(
    S: np.ndarray,
    tol: float = 1e-12,
    orient_tol: float = 1e-10,
    clean_tol: float = 1e-12,
    snap_unit_tol: float = 1e-9
):
    """
    Canonicalize a real skew-symmetric matrix S to block-diagonal form and
    enforce the unit blocks to be exactly [[0, 1], [-1, 0]].

    Steps:
      1) Project to skew-part for numerical safety: S <- 0.5*(S - S^T)
      2) Real Schur: S = Q T Q^T, set R = Q^T so that RsRT = R S R^T is block-diagonal
      3) Per 2x2 block, apply an orthogonal ±1 diagonal 'E' so that the *superdiagonal* is >= 0
      4) Build a positive block scaling J (per 2x2 block: (1/sqrt(sigma)) * I_2), with sigma = |(RsRT)[i,i+1]|
      5) JR = J @ R, Ju = JR @ S @ JR.T  → each 2x2 block is ~ [[0, 1], [-1, 0]]
         (Optionally snap values very close to 0 or ±1.)

    Returns
    -------
    Ju : np.ndarray
        Canonical unit-block form with all 2x2 blocks equal to [[0, 1], [-1, 0]] (within tolerance).
    R  : np.ndarray
        Orthogonal transform so that R S R^T is block-diagonal, after sign-fixing (includes ±1 flips).
    J  : np.ndarray
        Positive block-diagonal scaling that normalizes magnitudes to 1.

    Notes
    -----
    * Sign indeterminacy inside each 2x2 block is resolved by a diagonal ±1 matrix E (orthogonal),
      chosen so the (i,i+1) entry becomes nonnegative. J stays positive definite (no signs).
    * Degenerate blocks (sigma ~ 0) are left unscaled and their sign is not forced.
    """
    # --- 0) Input & skew projection ---
    S = np.array(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("S must be a real square matrix.")
    S = 0.5 * (S - S.T)
    n = S.shape[0]

    # --- 1) Real Schur: S = Q T Q^T, use R = Q^T so RsRT is (quasi-)block diag
    T, Q = schur(S, output='real')
    R = Q.T
    RsRT = R @ S @ R.T  # numerically cleaner than using T directly

    # --- 2) Enforce orientation: positive superdiagonal in every 2x2 block
    e = np.ones(n)  # diagonal of E (±1)
    i = 0
    while i < n:
        is_2x2 = (i < n-1) and (abs(RsRT[i+1, i]) > orient_tol or abs(RsRT[i, i+1]) > orient_tol)
        if is_2x2:
            # Use the larger magnitude between super- and subdiagonal to decide the sign robustly
            a = RsRT[i, i+1]
            b = RsRT[i+1, i]
            val = a if abs(a) >= abs(b) else -b  # ideal skew: b ~ -a, so this is consistent
            if val < 0:
                # Flip second basis vector in this block to make superdiagonal >= 0
                e[i+1] *= -1.0
            i += 2
        else:
            i += 1

    # Apply E (orthogonal ±1 diag) to fix signs in RsRT and absorb it into R
    E = np.diag(e)
    if not np.allclose(E, np.eye(n)):
        R = E @ R
        RsRT = E @ RsRT @ E  # keeps skew-symmetry and makes (i,i+1) >= 0

    # --- 3) Positive scaling to unit blocks
    J = np.eye(n)
    i = 0
    while i < n:
        is_2x2 = (i < n-1) and (abs(RsRT[i, i+1]) > orient_tol or abs(RsRT[i+1, i]) > orient_tol)
        if is_2x2:
            sigma = abs(RsRT[i, i+1])  # superdiagonal is now nonnegative
            if sigma > tol:
                s = 1.0 / np.sqrt(sigma)
                J[i:i+2, i:i+2] = s * np.eye(2)
            i += 2
        else:
            i += 1

    # --- 4) Compose and clean
    JR = J @ R
    Ju = JR @ S @ JR.T

    # Optional clean-up: tiny numbers to 0
    if clean_tol is not None:
        Ju[np.abs(Ju) < clean_tol] = 0.0
        RsRT[np.abs(RsRT) < clean_tol] = 0.0
        R[np.abs(R) < clean_tol] = 0.0
        J[np.abs(J) < clean_tol] = 0.0

    # Optional: snap entries very close to ±1 in each 2x2 block to exactly ±1 (cosmetic, helps prints)
    # if snap_unit_tol is not None:
    #     i = 0
    #     while i < n:
    #         if i < n-1:
    #             a = Ju[i, i+1]
    #             b = Ju[i+1, i]
    #             if a > 0 and abs(a - 1.0) <= snap_unit_tol and abs(b + 1.0) <= snap_unit_tol:
    #                 Ju[i, i+1] = 1.0
    #                 Ju[i+1, i] = -1.0
    #             # zero the tiny diagonals explicitly
    #             if abs(Ju[i, i]) <= snap_unit_tol:   Ju[i, i] = 0.0
    #             if abs(Ju[i+1, i+1]) <= snap_unit_tol: Ju[i+1, i+1] = 0.0
    #             i += 2
    #         else:
    #             i += 1

    return Ju, R, J
    

# --- timeseries_JRsRJ bleibt gleich, ruft nur obige Funktion auf ---
ETA = np.array([[0.0, 1.0], [-1.0, 0.0]])
import numpy as np

def threshold_matrix(A, tol=1e-6):
    """
    Set all entries of A with absolute value < tol to zero.
    """
    A = np.array(A, dtype=float)   # make sure it's a numpy array
    A[np.abs(A) < tol] = 0.0
    return A

def timeseries_JRsRJ(s_timeseries: list,
                     zero_tol: float = 1e-10,
                     clean_tol: float = 1e-12):
    JRsRJ_ts, Js_ts, Rs_ts = [], [], []

    for s in s_timeseries:

        spin = s[:8, :8]

        spin_can, R8, J8 = find_R_and_J_for_skew(spin
        )

        s_can_10 = np.zeros((10, 10), dtype=np.float64)
        s_can_10[:8, :8] = spin_can
        s_can_10[8:10, 8:10] = ETA
        s_can_10[np.abs(s_can_10) < clean_tol] = 0.0

        J10 = np.zeros((10, 10), dtype=np.float64)
        R10 = np.zeros((10, 10), dtype=np.float64)
        J10[:8, :8] = J8
        J10[8:10, 8:10] = np.eye(2)
        R10[:8, :8] = R8
        R10[8:10, 8:10] = np.eye(2)

        ortho_err10 = np.linalg.norm(R10.T @ R10 - np.eye(10), ord=np.inf)
        if ortho_err10 > 1e-10:
            print(f"⚠️ Warnung: ||R(10)^T R(10) − I||_∞ = {ortho_err10:.2e}")
        s_can_10=threshold_matrix(s_can_10)
        J10=threshold_matrix(J10)
        R10=threshold_matrix(R10)
        JRsRJ_ts.append(s_can_10)
        Js_ts.append(J10)
        Rs_ts.append(R10)

    return JRsRJ_ts, Js_ts, Rs_ts

import numpy as np

def find_unit_blocks_from_s(s, tol_pair=1e-9, tol_off=1e-12):
    """
    Finde 2x2-Einheitsblöcke in einer kanonischen skew-Matrix s (8x8 oder 10x10).
    Gibt eine sortierte Liste disjunkter Paare (i,j) mit i<j zurück,
    bei denen s[i,j]≈+1, s[j,i]≈-1 und sonstige Kopplungen in den Zeilen/Spalten klein sind.
    """
    s = np.asarray(s, float)
    n = s.shape[0]
    used = np.zeros(n, dtype=bool)
    pairs = []
    for i in range(n):
        if used[i]: 
            continue
        row = s[i, :].copy(); row[i] = 0.0
        j = int(np.argmax(np.abs(row)))
        a, b = s[i, j], s[j, i]
        if (np.abs(a-1.0) <= tol_pair and np.abs(b+1.0) <= tol_pair):
            # check off-couplings small
            others = [k for k in range(n) if k not in (i,j)]
            if (np.all(np.abs(s[i, others]) <= tol_off) and
                np.all(np.abs(s[j, others]) <= tol_off) and
                not used[j]):
                i2, j2 = (i, j) if i < j else (j, i)
                used[i2] = used[j2] = True
                pairs.append((i2, j2))
    pairs = sorted(set(pairs))
    return pairs

def build_permutation_to_match_pairs(pairs_src, pairs_tgt, n):
    """
    Baue eine Permutationsmatrix P (n×n), die die 2x2-Paare von pairs_src
    auf die Ziel-Paare pairs_tgt abbildet. Beide sind Listen disjunkter (i,j) mit i<j.
    Nicht benutzte Indizes bleiben (stabil) in ihrer Reihenfolge.
    """
    if len(pairs_src) != len(pairs_tgt):
        raise ValueError("Unterschiedliche Anzahl an 2x2-Blöcken – Matching nicht möglich.")
    # freie Quell-/Zielindizes
    used_src = set([i for p in pairs_src for i in p])
    used_tgt = set([i for p in pairs_tgt for i in p])
    free_src = [i for i in range(n) if i not in used_src]
    free_tgt = [i for i in range(n) if i not in used_tgt]

    mapping = {}
    # 1) mappe Blockpaare in gleicher Reihenfolge
    for (isrc,jsrc),(itgt,jtgt) in zip(pairs_src, pairs_tgt):
        mapping[isrc] = itgt
        mapping[jsrc] = jtgt
    # 2) mappe die restlichen 1×1-Indizes stabil
    for i, t in zip(free_src, free_tgt):
        mapping[i] = t

    # baue P so, dass x_new = P^T x_alt (d.h. A_new = P^T A_alt P)
    P = np.zeros((n,n))
    for i_old, i_new in mapping.items():
        P[i_old, i_new] = 1.0
    return P

def align_series_to_reference(JRsRJ_ts, Js_ts, Rs_ts, spin_dim=8):
    """
    Richte pro Zeitschritt (s_can, J, R) so aus, dass die 2x2-Blöcke der Spin-Teile
    dieselben Index-Paare wie bei t=0 haben.
    Wir permutieren NUR die Spin-8×8-Ecken; Q,P (letzte 2) bleiben unverändert.
    """
    aligned_s, aligned_J, aligned_R = [], [], []

    s0 = JRsRJ_ts[0]
    # Referenzpaare aus Spin-Block (0..spin_dim-1)
    ref_pairs = [(i,j) for (i,j) in find_unit_blocks_from_s(s0[:spin_dim,:spin_dim])]
    # Fallback: wenn keine Paare erkannt – nichts tun
    if not ref_pairs:
        return JRsRJ_ts, Js_ts, Rs_ts

    for s_can, J10, R10 in zip(JRsRJ_ts, Js_ts, Rs_ts):
        spin = s_can[:spin_dim,:spin_dim]
        cur_pairs = [(i,j) for (i,j) in find_unit_blocks_from_s(spin)]
        # baue Permutation P8, erweitere auf 10×10
        P8 = build_permutation_to_match_pairs(cur_pairs, ref_pairs, spin_dim)
        P10 = np.eye(10); P10[:spin_dim,:spin_dim] = P8

        s_al = P10.T @ s_can @ P10
        J_al = P10.T @ J10 @ P10
        R_al = P10.T @ R10         # beachte: R wirkt links auf Koordinatenbasis

        aligned_s.append(s_al)
        aligned_J.append(J_al)
        aligned_R.append(R_al)

    return aligned_s, aligned_J, aligned_R














# ========================================================================
# HAUPTSKRIPT
# ========================================================================

if __name__ == "__main__":
    # 1. SETUP: Parameter und Anfangsbedingungen
    print("1. System wird eingerichtet...")
    g0, Delta1, Delta2, V, Gamma, Omega, kappa, eta = sp.symbols("g0 Delta1 Delta2 V Gamma Omega kappa eta")
    Omega_val = 8#8.0  # laser drive 1->2
    Gamma_val = 2  #atom decay 1->0
    
    Delta1_val=1  #detuning from 1 
    Delta2_val=1 #detuning from 2
    #############
    g0_val=1 #cavity coupling s
    eta_val=1 # cavity drive 
    kappa_val=1 #cavity decay
    #V_val =-Delta2/2*((Omega_val*kappa_val/(4*eta_val*g0))**2+1)
    #V_val =-1/2*(4+1)
    V_val =-6  #interaction potential
    

    t_span = (0.0, 500.0)
    rtol_val=1e-12
    atol_val=1e-14
    y0_ket = np.array([0+0j, 0+0j, 1+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j])
    #y0_ket = np.array([0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 1+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j])
    m0 = convert_state(y0_ket)
    Sigma0 = get_initial_covariance_matrix(y0_ket)









    numeric_params = { 
        g0:     g0_val, Delta1: Delta1_val, Delta2: Delta2_val, V: V_val, Gamma: Gamma_val, 
        Omega: Omega_val, kappa:   kappa_val, eta: eta_val
    }
    params = {
        'g0':    g0_val, 'kappa':   kappa_val, 'gamma': g0_val, 'Gamma': Gamma_val,
        'Omega': Omega_val, 'delta1': Delta1_val, 'delta2':Delta2_val,
        'eta': eta_val, 'V': V_val
    }
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


    #pprint(Q_sym)
    #pprint(P_sym)
    #pprint(G_sym)




    plots=False
    solve = True
   # 4. LÖSEN
    if solve:
        print("3. Differentialgleichungssystem wird gelöst...")
        
        t_eval = np.linspace(*t_span, 1001)
        Y0 = np.concatenate([np.real(m0), Sigma0.flatten()])
        sol = solve_ivp(
            fun=lambda t, y: rhs_combined(t, y, params, g_func, w_func),
            t_span=t_span, y0=Y0, t_eval=t_eval, method='RK45',rtol=rtol_val,atol=atol_val
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
            #max_real_eigenvalues_G.append(np.max(np.real(eigenvalues_G)))

        # --- Berechnung für M(t) und Sigma(t) ---
        #s_timeseries = build_symplectic_matrix_ts(sol)
        s_timeseries = build_s_form_timeseries_2f(sol)
        min_eigenvalue_trajectory_M = []
        min_eigenvalues_Sigma = []

        for i in range(len(sol.t)):
            sigma_t = sol.y[10:, i].reshape((10, 10))
            min_eigenvalues_Sigma.append(np.min(np.linalg.eigvalsh(sigma_t)))
            s_t = s_timeseries[i]
            M_t = sigma_t + (1j / 2) * s_t
            min_eigenvalue_trajectory_M.append(np.min(np.linalg.eigvalsh(M_t)))
        
        # --- Kombinierter Plot für alle vier Verläufe ---
        def compute_G_W_spectra(sol, g_func, w_func):
            """Return per-time-step spectra: eigvals(G_t) and eigvalsh(W_t)."""
            G_eigs_all = []   # list of arrays (complex) for each t
            W_eigs_all = []   # list of arrays (real, sorted) for each t

            for i in range(len(sol.t)):
                m_t = sol.y[:10, i]
                # reorder: your G/W expect [λ1..λ8, Q, P] i.e. m[2:] + m[:2]
                m_reordered = np.concatenate([m_t[2:], m_t[:2]])

                G_t = g_func(*m_reordered)
                W_t = w_func(*m_reordered)

                # G is generally non-symmetric: use eigvals (complex)
                G_eigs = np.linalg.eigvals(G_t)
                # W is symmetric (diffusion): use eigvalsh (real, sorted)
                W_eigs = np.linalg.eigvalsh(W_t)

                G_eigs_all.append(G_eigs)
                W_eigs_all.append(W_eigs)

            return G_eigs_all, W_eigs_all

        # ... inside your `if solve:` block, AFTER `sol = solve_ivp(...)` and BEFORE plotting:

        # 4a) Compute time series of spectra
        G_eigs_all, W_eigs_all = compute_G_W_spectra(sol, g_func, w_func)

        # 4b) Build the requested trajectories:
        #     - max real part among eigenvalues of G(t)
        #     - min eigenvalue of W(t)
        max_real_eigs_G_over_time = [np.max(np.real(ev)) for ev in G_eigs_all]
        min_eigs_W_over_time      = [np.min(ev) for ev in W_eigs_all]

        # 4c) Print summary stats (German console output)
        idx_max_G   = int(np.argmax(max_real_eigs_G_over_time))
        idx_min_W   = int(np.argmin(min_eigs_W_over_time))

       # --- Berechnung für G(t) und W(t) ---
        min_eigenvalues_W = []
        max_real_eigenvalues_G = []

        for i in range(len(sol.t)):
            m_t = sol.y[:10, i]
            m_reordered = np.concatenate([m_t[2:], m_t[:2]])

            # W ist hermitesch → eigvalsh
            W_t = w_func(*m_reordered)
            min_eigenvalues_W.append(np.min(np.linalg.eigvalsh(W_t)).real)

            # G ist i.A. nicht hermitesch → eigvals, dann Realteil
            G_t = g_func(*m_reordered)
            eigenvalues_G = np.linalg.eigvals(G_t)
            max_real_eigenvalues_G.append(np.max(np.real(eigenvalues_G)))

        # --- Berechnung für M(t) und Sigma(t) ---
        s_timeseries = build_s_form_timeseries_2f(sol)
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

        # Hier jetzt auch G und W mitplotten:
        plt.plot(sol.t, max_real_eigenvalues_G, label='Größter Realteil Eig(G(t))', color='green', linestyle='-.')
        plt.plot(sol.t, min_eigenvalues_W, label='Kleinster Eig(W(t))', color='darkcyan', linestyle='--')

        plt.plot(sol.t, min_eigenvalues_Sigma, label=r'Kleinster Eig($\Sigma(t)$)', color='orange')
        plt.plot(sol.t, min_eigenvalue_trajectory_M, label=r'Kleinster Eig(M(t) = $\Sigma + 0.5is$)', color='purple', linewidth=2)

        plt.axhline(0, color='red', linestyle=':', linewidth=2, label='Referenzlinie (y=0)')
        plt.title('Zeitentwicklung der relevanten Eigenwerte')
        plt.xlabel('Zeit')
        plt.ylabel('Wert des Eigenwerts / Realteils')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if plots:
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
        if plots:
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
        if plots:
            plt.show()

        # print("\n\n" + "="*70)
        # print("DETAILLIERTE MATRIX-ANALYSE FÜR EINEN ZEITPUNKT")
        # print("="*70)
        # # Überprüfe die Matrizen am Ende der Simulation
        # get_and_check_matrices_at_time(t_target=sol.t[-1], sol=sol)





        import numpy as np
        from itertools import combinations
        from typing import List, Tuple, Dict, Any

        # ============================================================
        # 0) Utility: Block finder on canonical skew-symmetric s (10x10)
        # ============================================================

        def find_canonical_blocks_from_s(
            s: np.ndarray,
            tol_pair: float = 1e-10,
            tol_off: float = 1e-12
        ) -> Tuple[List[Tuple[int,int]], List[int]]:
            """
            Detect 2x2 canonical blocks in real skew-symmetric s (shape 10x10, order [λ1..λ8,Q,P]).
            A canonical block (i,j) satisfies:
            - |s[i,j]| ≈ 1 within tol_pair, and s[j,i] ≈ -s[i,j]
            - all other couplings in rows/cols i,j are ≲ tol_off
            Orientation: we return i<j and enforce s[i,j] > 0 by swapping if needed.

            Returns:
            blocks : list of 0-based pairs (i,j) with i<j
            singles: list of 0-based indices not assigned to any 2x2 block
            """
            s = np.asarray(s, dtype=float)
            if s.shape != (10, 10):
                raise ValueError("s must be 10x10.")
            used = np.zeros(10, dtype=bool)
            blocks: List[Tuple[int,int]] = []

            for i in range(10):
                if used[i]:
                    continue
                row = s[i, :].copy()
                row[i] = 0.0
                j = int(np.argmax(np.abs(row)))
                val = row[j]
                # Pair candidate?
                if np.abs(val) >= (1.0 - tol_pair) and np.allclose(s[j, i], -val, atol=tol_pair):
                    # Check off-couplings
                    others_i = np.delete(np.arange(10), [i, j])
                    others_j = np.delete(np.arange(10), [i, j])
                    if (np.all(np.abs(s[i, others_i]) <= tol_off) and
                        np.all(np.abs(s[j, others_j]) <= tol_off) and
                        (not used[j])):
                        # Orient so that s[i,j] > 0
                        if val < 0:
                            i, j = j, i
                        used[i] = used[j] = True
                        blocks.append((min(i, j), max(i, j)))

            singles = [k for k in range(10) if not used[k]]
            return blocks, singles


        # ============================================================
        # 1) Check time invariance of canonical s(t) = J R^T s R J
        # ============================================================

        def is_time_invariant_s(
            s_canonical_ts: List[np.ndarray],
            atol: float = 1e-12
        ) -> Tuple[bool, float, int]:
            """
            Check if all s_can(t) are identical (within atol) to s_can(0).
            Returns:
            invariant : bool
            max_dev   : float (max ∞-norm deviation across time)
            argmax_t  : int   (time index with max deviation)
            """
            if len(s_canonical_ts) == 0:
                raise ValueError("s_canonical_ts is empty.")
            s0 = np.asarray(s_canonical_ts[0], dtype=float)
            max_dev = 0.0
            argmax_t = 0
            for t in range(1, len(s_canonical_ts)):
                diff = np.asarray(s_canonical_ts[t], float) - s0
                dev = np.max(np.abs(diff))
                if dev > max_dev:
                    max_dev = dev
                    argmax_t = t
            invariant = (max_dev <= atol)
            print("— Prüfe Zeitinvarianz von s (kanonischer Frame) —")
            print(f"Maximale Abweichung (∞-Norm): {max_dev:.3e}")
            print("Ergebnis:", "Zeitlich konstant (innerhalb Toleranz)" if invariant else "NICHT konstant")
            return invariant, max_dev, argmax_t


        # ============================================================
        # 2) Rotate Sigma(t) into (J,R)-frame: Sigma_rot = J R^T Sigma R J
        # ============================================================

        def rotate_sigma_timeseries(Js: List[np.ndarray],
                                    Rs: List[np.ndarray],
                                    Sigma_ts: List[np.ndarray]) -> List[np.ndarray]:
            """
            For each t, compute Sigma_rot(t) = J @ R.T @ Sigma @ R @ J.
            Assumes 10x10 shapes for all.
            """
            if not (len(Js) == len(Rs) == len(Sigma_ts)):
                raise ValueError("Lengths of Js, Rs, Sigma_ts must match.")
            out = []
            for J, R, Sigma in zip(Js, Rs, Sigma_ts):
                # Symmetrize numerically for stability
                Sigma = 0.5*(Sigma + Sigma.T)
                Sigma_rot = J @ (R.T @ (Sigma @ (R @ J)))
                out.append(np.real_if_close(Sigma_rot, tol=1000))
            return out


        # ============================================================
        # 3) Build 4x4 submatrix from two 2x2 blocks
        # ============================================================

        def build_4x4_from_two_blocks(Sigma_rot: np.ndarray,
                                    blockA: Tuple[int,int],
                                    blockB: Tuple[int,int]) -> np.ndarray:
            """
            Given Sigma_rot (10x10) and two 2x2 blocks (i,j), (k,l) with i<j, k<l,
            return the 4x4 submatrix in order [i,j,k,l]:
                [ A  B ]
                [ C  D ]
            """
            i, j = sorted(blockA)
            k, l = sorted(blockB)
            idx = [i, j, k, l]
            sub4 = Sigma_rot[np.ix_(idx, idx)]
            return np.real_if_close(sub4, tol=1000)


        def split_4x4_blocks(sub4: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            """
            Split 4x4 into four 2x2 blocks (A,B,C,D) assuming order [i,j,k,l]:
                sub4 = [[A, B],
                        [C, D]]
            """
            if sub4.shape != (4, 4):
                raise ValueError("sub4 must be 4x4.")
            A = sub4[0:2, 0:2]
            B = sub4[0:2, 2:4]
            C = sub4[2:4, 0:2]
            D = sub4[2:4, 2:4]
            return A, B, C, D


        # ============================================================
        # 4) Analyze all 4x4 combos over all times (fixed blocks)
        # ============================================================

        def analyze_4x4_timeseries(
            Sigma_rot_ts: List[np.ndarray],
            fixed_blocks: List[Tuple[int,int]],
            label_prefix: str = "pair",
            atol_BeqCT: float = 1e-12
        ) -> List[Dict[str, Any]]:
            """
            For each time and for ALL combinations of the fixed 2x2 blocks:
            - build sub4 = [[A,B],[C,D]]
            - eigvals_2M: eigenvalues of (2 * sub4)
            - B_eq_Ct: check B == C.T within atol_BeqCT
            - if B_eq_Ct: compute det(2A), det(2B), det(2C), det(2*sub4)

            Returns list (per time) of dicts: label -> record
            label: f"{label_prefix}_(i+1,j+1)_x_(k+1,l+1)"
            record keys:
                "blocks"     : ((i,j),(k,l))          # 0-based indices used
                "eigvals_2M" : np.ndarray shape (4,)  # eigenvalues of 2*sub4
                "B_eq_Ct"    : bool
                "det_2A"     : float or None
                "det_2B"     : float or None
                "det_2C"     : float or None
                "det_2M"     : float or None         # det of 2*sub4
            """
            # sanitize fixed blocks
            fixed_blocks = [tuple(sorted(b)) for b in fixed_blocks]
            fixed_blocks = sorted(list(set(fixed_blocks)))

            results_per_time: List[Dict[str, Any]] = []
            for t, Sigma_rot in enumerate(Sigma_rot_ts):
                out_t: Dict[str, Any] = {}
                if len(fixed_blocks) < 2:
                    results_per_time.append(out_t)
                    continue

                for (ba, bb) in combinations(fixed_blocks, 2):
                    (i, j), (k, l) = ba, bb
                    label = f"{label_prefix}_({i+1},{j+1})_x_({k+1},{l+1})"

                    sub4 = build_4x4_from_two_blocks(Sigma_rot, ba, bb)
                    A, B, C, D = split_4x4_blocks(sub4)

                    eigvals_2M = np.linalg.eigvals(2.0 * sub4)
                    B_eq_Ct = np.allclose(B, C.T, atol=atol_BeqCT)

                    det_2A = det_2B = det_2C = det_2M = None
                    if B_eq_Ct:
                        det_2A = float(np.linalg.det(2.0 * A))
                        det_2B = float(np.linalg.det(2.0 * B))
                        det_2C = float(np.linalg.det(2.0 * C))
                        det_2M = float(np.linalg.det(2.0 * sub4))

                    out_t[label] = {
                        "blocks": ((i, j), (k, l)),
                        "eigvals_2M": eigvals_2M,
                        "B_eq_Ct": bool(B_eq_Ct),
                        "det_2A": det_2A,
                        "det_2B": det_2B,
                        "det_2C": det_2C,
                        "det_2M": det_2M,
                    }
                results_per_time.append(out_t)
            return results_per_time


        # ============================================================
        # 5) OPTIONAL convenience: run the whole pipeline once
        # ============================================================

        def run_block_analysis_pipeline(
            JRsRJ_ts: List[np.ndarray],
            Js: List[np.ndarray],
            Rs: List[np.ndarray],
            Sigma_ts: List[np.ndarray],
            s_const_atol: float = 1e-12,
            tol_pair: float = 1e-10,
            tol_off: float = 1e-12,
            atol_BeqCT: float = 1e-12,
            label_prefix: str = "pair"
        ) -> Dict[str, Any]:
            """
            Full pipeline:
            - check time invariance of canonical s
            - detect fixed 2x2 blocks from JRsRJ[0]
            - rotate Sigma_ts -> Sigma_rot_ts
            - analyze all 4x4 combinations for all times

            Returns:
            {
                "s_invariant": bool,
                "s_max_dev"  : float,
                "s_argmax_t" : int,
                "fixed_blocks": List[(i,j)],
                "analysis_ts" : List[Dict[label -> result]]
            }
            """
            # 1) s invariance
            s_invariant, s_max_dev, s_argmax_t = is_time_invariant_s(JRsRJ_ts, atol=s_const_atol)

            # 2) fixed blocks from the first time
            blocks_ref, singles_ref = find_canonical_blocks_from_s(JRsRJ_ts[0], tol_pair=tol_pair, tol_off=tol_off)
            print("Fixe 2×2-Blöcke (1-basiert):", [(i+1, j+1) for (i,j) in blocks_ref])
            if singles_ref:
                print("Isolierte Indizes (1-basiert):", [k+1 for k in singles_ref])

            # 3) rotate all Sigma
            Sigma_rot_ts = rotate_sigma_timeseries(Js, Rs, Sigma_ts)

            # 4) analyze all 4x4s
            analysis_ts = analyze_4x4_timeseries(
                Sigma_rot_ts,
                fixed_blocks=blocks_ref,
                label_prefix=label_prefix,
                atol_BeqCT=atol_BeqCT
            )

            return {
                "s_invariant": s_invariant,
                "s_max_dev": s_max_dev,
                "s_argmax_t": s_argmax_t,
                "fixed_blocks": blocks_ref,
                "analysis_ts": analysis_ts
            }


        # ============================================================
        # =============== USAGE (plug into your code) ================
        # ============================================================
        # Annahmen (aus deinem Projekt vorhanden):
        s_timeseries = build_s_form_timeseries_2f(sol)            # 10x10 s(t)
        JRsRJ, Js, Rs = timeseries_JRsRJ(s_timeseries)            # kanonisches s(t), und zugehörige J,R (padded 10x10)
        Sigma_ts = [sol.y[10:, i].reshape(10,10) for i in range(sol.t.size)]  # Roh-Σ(t)

        import numpy as np

        
        #JRsRJ_aligned, Js_aligned, Rs_aligned = align_series_to_reference(JRsRJ, Js, Rs, spin_dim=8)
        #for s in JRsRJ[0:150]:
        #print("Hewllow")
        #JRsRJ = JRsRJ[4:]

        JRsRJ[4:]
        Js[4:]
        Sigma_ts[4:]
        Rs[4:]
        Sigma_rot_ts = []
        for index in range(0,len(Sigma_ts)):
            Sigma_rot_ts.append(Js[index]@Rs[index].T@Sigma_ts[i]@Rs[index]@Js[index])
        #pprint(Matrix(Sigma_rot_ts[2]))    
        def plot_sigma_diagonals(Sigma_rot_ts, times, title="Diagonalelemente von Σ(t) im kanonischen Frame"):
            """
            Plot all diagonal elements of the rotated covariance matrices vs time.

            Parameters
            ----------
            Sigma_rot_ts : list of np.ndarray
                Zeitreihe der rotierten Kovarianzmatrizen Σ_rot(t), jede 10x10.
            times : array-like
                Zeitpunkte, gleiche Länge wie Sigma_rot_ts.
            title : str
                Titel für den Plot.
            """
            nsteps = len(Sigma_rot_ts)
            if nsteps != len(times):
                raise ValueError("Länge von Sigma_rot_ts und times muss übereinstimmen.")
            
            # Stack diagonals
            diagonals = np.array([np.diag(Sigma) for Sigma in Sigma_rot_ts])  # shape (T,10)

            plt.figure(figsize=(12, 6))
            for i in range(diagonals.shape[1]):
                plt.plot(times, diagonals[:, i], label=f"Σ[{i+1},{i+1}]")
            plt.xlabel("Zeit")
            plt.ylabel("Diagonalelemente Σ_ii(t)")
            plt.title(title)
            plt.grid(True)
            plt.legend(ncol=2, fontsize=9)
            plt.tight_layout()
            plt.show()
        plot_sigma_diagonals(Sigma_rot_ts, sol.t)