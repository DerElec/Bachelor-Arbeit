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
from scipy.interpolate import interp1d
import numpy as np
# Disable pretty-printer line wrapping and widen column budget
sp.init_printing(use_unicode=True, wrap_line=False, num_columns=300)


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

def rhs_singlets_x(t, x, params):
    """
    Time derivative for the 10-vector [Q, P, x1..x8] (singlets only).
    It reconstructs the full ket state, applies the ket-ODE, and projects back.

    State conventions:
      rho = 1/3 * I + 1/2 * sum_a x_a * lambda_a
      Q = (a + a†)/sqrt(2),  P = (a - a†)/(i*sqrt(2))

    Params keys (all complex allowed): 'kappa','gamma','Gamma','Omega','delta1','delta2','eta','V'
    """
    x1, x2, x3, x4, x5, x6, x7, x8,Q,P = x

    # Reconstruct ladder operators (do NOT conjugate here)
    a  = (Q + 1j*P)/np.sqrt(2)
    ad = (Q - 1j*P)/np.sqrt(2)

    # Reconstruct density-matrix entries from x's
    rho00 = 1/3 + 0.5*( x3 + x8/np.sqrt(3) )
    rho11 = 1/3 + 0.5*(-x3 + x8/np.sqrt(3))
    rho22 = 1 - rho00 - rho11

    rho01 = (x1 + 1j*x2)/2
    rho10 = (x1 - 1j*x2)/2
    rho02 = (x4 + 1j*x5)/2
    rho20 = (x4 - 1j*x5)/2
    rho12 = (x6 + 1j*x7)/2
    rho21 = (x6 - 1j*x7)/2

    # Unpack parameters
    kappa = params['kappa']; gamma = params['gamma']; Gamma = params['Gamma']
    Omega = params['Omega']; delta1 = params['delta1']; delta2 = params['delta2']
    eta   = params['eta'];   V     = params['V']

    # --- Ket-ODE (your model, kept verbatim but with explicit a† equation) ---
    da_dt        = -kappa/2 * a - 1j*(gamma*rho01) + eta
    da_dagger_dt = -kappa/2 * ad + 1j*(gamma*rho10) + np.conj(eta)  # explicit; no conj(da_dt)

    d00 = Gamma*rho11 + 1j*gamma*(rho10*a - rho01*ad)
    d01 = -Gamma/2*rho01 + 1j*(-delta1*rho01 + gamma*(rho11*a - rho00*a) - Omega/2*rho02)
    d10 = -Gamma/2*rho10 - 1j*(-delta1*rho10 - gamma*(rho11*ad - rho00*ad) + Omega/2*rho20)  # conj(d01) if params real

    d11 = -Gamma*rho11 + 1j*gamma*(rho01*ad - rho10*a) + 1j*(Omega/2)*(rho21 - rho12)
    d22 = 1j*(Omega/2)*(rho12 - rho21)

    d21 = -Gamma/2*rho21 + 1j*(delta2*rho21 - delta1*rho21 - gamma*rho20*a + (Omega/2)*(rho11 - rho22) + 2*V*rho21*rho22)
    d12 = -Gamma/2*rho12 - 1j*(delta2*rho12 - delta1*rho12 - gamma*rho02*ad + (Omega/2)*(rho11 - rho22) + 2*V*rho12*rho22)

    d02 = 1j*(-delta2*rho02 - Omega/2*rho01 - 2*V*rho02*rho22 + gamma*rho12*a)
    d20 = -1j*(-delta2*rho20 - Omega/2*rho10 - 2*V*rho20*rho22 + gamma*rho21*ad)

    # --- Project back to x-space ---
    dx1 = d01 + d10
    dx2 = -1j*d01 + 1j*d10
    dx3 = d00 - d11
    dx4 = d02 + d20
    dx5 = -1j*d02 + 1j*d20
    dx6 = d12 + d21
    dx7 = -1j*d12 + 1j*d21
    dx8 = (d00 + d11 - 2*d22)/np.sqrt(3)

    dQ = (da_dt + da_dagger_dt)/np.sqrt(2)
    dP = (da_dt - da_dagger_dt)/(1j*np.sqrt(2))

    return np.array([dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8,dQ, dP], dtype=complex)
# ========================================================================
# HAUPTSKRIPT
# ========================================================================


if __name__ == "__main__":
    # 1. SETUP: Parameter und Anfangsbedingungen
    print("1. System wird eingerichtet...")
    g0, Delta1, Delta2, V, Gamma, Omega, kappa, eta = sp.symbols("g0 Delta1 Delta2 V Gamma Omega kappa eta")
    Omega_val =8#8.0  # laser drive 1->2
    Gamma_val = 1  #atom decay 1->0
    V_val =-6#-6.0  #interaction potentiall
    Delta1_val=1  #detuning from 1 
    Delta2_val=1 #detuning from 2
    #############
    g0_val=1#cavity coupling 
    eta_val=1 # cavity drive 
    kappa_val=1#cavity decay
    T_end = 200.0         # total simulation time (adjust as needed)
    N_eval = int(T_end*100)
    y0_ket = np.array([0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 1+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j])
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
    #a a^dagger ket00, ket01, ket10, ket11, ket22, ket21, ket12, ket20, ket02 
    # 2. SYMBOLIK: Umwandlung in schnelle numerische Funktionen
    print("2. Symbolische Matrizen werden in numerische Funktionen umgewandelt...")
    m_syms = sp.symbols('m1:11')
    G_num_params, _, _, _, _, _, _, W_num_params, _, _, _ = covar_everything.get_important_matricies(numeric_params)
    print("Alle Parameter wurden erfolgreich ersetzt.")
    g_func = sp.lambdify(m_syms, G_num_params, 'numpy')
    w_func = sp.lambdify(m_syms, W_num_params, 'numpy')
    # 3. DGL: Definition der kombinierten RHS-Funktion

    #G_sym, sDs_sym, Z_sym, P_sym, Q_sym,sE_sym, Z_prime_sym, W_sym, Sigma_dt_sym, Sigma_sym, K_sym = covar_everything.get_important_matricies(numeric_params)
    G_sym, sDs_sym, Z_sym, P_sym, Q_sym,sE_sym, Z_prime_sym, W_sym, Sigma_dt_sym, Sigma_sym, K_sym = covar_everything.get_important_matricies_symbol()
    #dsigmadt=Sigma_sym@G_sym.T+G_sym@Sigma_sym+ W_sym
    dsigma_dt=Sigma_sym@G_sym.T+G_sym@Sigma_sym+W_sym
    pprint(G_sym)
    #pprint(W_sym)
    #print(dsigma_dt[0,0])
    print(Sigma_dt_sym[1,2])
    print(Sigma_dt_sym[2,1])
    #print(Sigma0[0,0])
    #initial state makes sense, differential equation aswell, next up check singlets
    # ============================================================
    # ZEITENTWICKLUNG: Singlets + Kovarianzmatrix  (läuft direkt nach Deinem Code)
    # ============================================================

    print("3. DGLs werden aufgebaut und integriert...")

    # --- Helper: extract initial means (x1..x8,Q,P) from y0_ket ------------------
    def initial_means_from_y0(y0_vec: np.ndarray) -> np.ndarray:
        """
        Build [x1..x8, Q, P] from y0_ket:
        rho entries are ordered as: rho00, rho01, rho10, rho11, rho22, rho21, rho12, rho20, rho02 = y0[2:]
        and a, a† = y0[0], y0[1].
        Relations used:
        rho01 = (x1 + i x2)/2,   rho02 = (x4 + i x5)/2,   rho12 = (x6 + i x7)/2
        rho00 - rho11 = x3
        (rho00 + rho11 - 2 rho22) / sqrt(3) = x8
        a = (Q + i P)/sqrt(2),   a† = (Q - i P)/sqrt(2)
        """
        a, adag = y0_vec[0], y0_vec[1]
        rho00, rho01, rho10, rho11, rho22, rho21, rho12, rho20, rho02 = y0_vec[2:]

        # Gell-Mann means x1..x8 (these combinations are guaranteed real for Hermitian rho)
        x1 = (rho01 + rho10)
        x2 = (-1j * rho01 + 1j * rho10)
        x3 = (rho00 - rho11)
        x4 = (rho02 + rho20)
        x5 = (-1j * rho02 + 1j * rho20)
        x6 = (rho12 + rho21)
        x7 = (-1j * rho12 + 1j * rho21)
        x8 = (rho00 + rho11 - 2.0 * rho22) / np.sqrt(3.0)

        # Bosonic quadratures
        Q = (a + adag) / np.sqrt(2.0)
        P = (a - adag) / (1j * np.sqrt(2.0))

        x = np.array([x1, x2, x3, x4, x5, x6, x7, x8, Q, P], dtype=complex)
        # Enforce reals; tiny numerical imaginary parts may appear from inputs
        return np.real_if_close(x, tol=1e-12)

    # Initial mean vector and covariance
    x0 = initial_means_from_y0(y0_ket)                 # shape (10,)
    Sigma0 = np.array(Sigma0, dtype=float)             # 10x10, already built above
    Sigma0 = 0.5 * (Sigma0 + Sigma0.T)                 # ensure symmetry

    # --- Evaluate G(m), W(m) safely as 10x10 floats --------------------------------
    def eval_G_W(m_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # m_vec is [x1..x8, Q, P] (length 10)
        Gm = np.asarray(g_func(*m_vec), dtype=float)
        Wm = np.asarray(w_func(*m_vec), dtype=float)
        Gm = Gm.reshape(10, 10)
        Wm = Wm.reshape(10, 10)
        return Gm, Wm

    # --- Combined RHS for solve_ivp -------------------------------------------------
    def rhs_combined(t: float, y: np.ndarray) -> np.ndarray:
        """
        y = [ x(10), vec(Sigma)(100) ]
        Returns concatenated derivative: [ dx, dSigma_flat ]
        """
        x = y[:10]
        Sigma = y[10:].reshape(10, 10)

        # Means dynamics (ensure real output)
        dx = np.real_if_close(rhs_singlets_x(t, x, params), tol=1e-12)
        dx = np.asarray(dx, dtype=float)

        # Covariance dynamics with current G(m), W(m)
        Gm, Wm = eval_G_W(x)
        dSigma = Sigma @ Gm.T + Gm @ Sigma + Wm

        # Keep symmetry numerically
        dSigma = 0.5 * (dSigma + dSigma.T)

        return np.concatenate([dx, dSigma.reshape(-1)])

    # --- Time grid and integration --------------------------------------------------
    # Basis B (covar_everything): [x1,x2,x3,x4,x5,x6,x7,x8,Q,P]
    # Basis A (integration here): [Q,P,x1,x2,x3,x4,x5,x6,x7,x8]
    perm_B2A = np.array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])  # x_A = x_B[perm_B2A]
    perm_A2B = np.array([2, 3, 4, 5, 6, 7, 8, 9, 0, 1])  # x_B = x_A[perm_A2B]

    def reorder_mat_B2A(M: np.ndarray) -> np.ndarray:
        """Reorder 10x10 matrix from B-basis to A-basis."""
        M = np.asarray(M)
        return M[np.ix_(perm_B2A, perm_B2A)]

    def reorder_cov_B2A(S: np.ndarray) -> np.ndarray:
        """Reorder 10x10 covariance from B-basis to A-basis (Σ_A = Π Σ_B Πᵀ)."""
        S = np.asarray(S)
        return S[np.ix_(perm_B2A, perm_B2A)]

    # ----- Initial means in A-basis (use your convert_state) ---------------------
    # convert_state(y0_ket) from your code returns [Q,P,x1..x8]
    x0_A = np.asarray(convert_state(y0_ket), dtype=complex)
    x0_A = np.real_if_close(x0_A, tol=1e-12).astype(float)

    # ----- Initial Sigma: your get_initial_covariance_matrix built it in B-basis --
    # bring it to A-basis for consistent integration
    Sigma0_B = np.array(Sigma0, dtype=float)  # from your earlier code
    Sigma0_A = reorder_cov_B2A(0.5*(Sigma0_B + Sigma0_B.T))  # ensure symmetric

    # ----- Helper: evaluate G(m),W(m) and reorder to A-basis ---------------------
    def eval_G_W_A(xA: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate G,W at means in A-basis by:
        1) map xA -> xB,
        2) call g_func/w_func,
        3) reorder matrices to A-basis.
        """
        mB = xA[perm_A2B]  # map to B-basis
        GB = np.asarray(g_func(*mB), dtype=float).reshape(10, 10)
        WB = np.asarray(w_func(*mB), dtype=float).reshape(10, 10)
        GA = reorder_mat_B2A(GB)
        WA = reorder_mat_B2A(WB)
        return GA, WA

    # ----- Pack/unpack upper triangular of Sigma (55 vars) -----------------------
    _TRIU_IDX = np.triu_indices(10)

    def pack_upper(S: np.ndarray) -> np.ndarray:
        """Pack symmetric 10x10 matrix into upper-triangular vector (length 55)."""
        return S[_TRIU_IDX]

    def unpack_upper(v: np.ndarray) -> np.ndarray:
        """Unpack upper-triangular vector (length 55) into symmetric 10x10 matrix."""
        S = np.zeros((10, 10), dtype=float)
        S[_TRIU_IDX] = v
        S[(_TRIU_IDX[1], _TRIU_IDX[0])] = v
        return S

    # ----- Combined RHS in A-basis (means + packed Sigma) ------------------------
    def rhs_combined_packed_A(t: float, y: np.ndarray) -> np.ndarray:
        """
        y = [ xA(10), sigma_up(55) ], where xA = [Q,P,x1..x8].
        Means from your rhs_gellmann_qp_from_x; covariance via dΣ = GΣ + ΣGᵀ + W.
        """
        xA = y[:10]
        sigma_up = y[10:]
        SigmaA = unpack_upper(sigma_up)

        # Means dynamics (use your correct RHS; enforce real)
        dxA = np.real_if_close(rhs_gellmann_qp_from_x(t, xA, params), tol=1e-12)
        dxA = np.asarray(dxA, dtype=float)

        # Covariance dynamics with reordered G,W
        GA, WA = eval_G_W_A(xA)
        dSigmaA = GA @ SigmaA + SigmaA @ GA.T + WA
        dSigmaA = 0.5 * (dSigmaA + dSigmaA.T)  # keep symmetry

        return np.concatenate([dxA, pack_upper(dSigmaA)])

    # ----- Time grid + integration (Radau for stiffness) -------------------------
    print("Integration (Radau) startet ...")
    T_end = 20.0 if 'T_end' not in globals() else T_end
    N_eval = 2001 if 'N_eval' not in globals() else N_eval
    t_span = (0.0, T_end)
    t_eval = np.linspace(t_span[0], t_span[1], N_eval)

    y0_packed = np.concatenate([x0_A, pack_upper(Sigma0_A)])

    sol = solve_ivp(rhs_combined_packed_A, t_span, y0_packed, t_eval=t_eval,
                    method="Radau", rtol=1e-7, atol=1e-9)

    print(f"Integration beendet. success={sol.success}, nfev={sol.nfev}, "
        f"njev={getattr(sol,'njev','NA')}, nlu={getattr(sol,'nlu','NA')}")
    if not sol.success:
        print("WARNUNG:", sol.message)

    # ----- Unpack solution --------------------------------------------------------
    t = sol.t
    X_t = sol.y[:10, :].T                       # shape (N,10), order [Q,P,x1..x8]
    Sigma_t = np.empty((t.size, 10, 10), dtype=float)
    sigma_up_over_time = sol.y[10:, :].T
    for k in range(t.size):
        S_k = unpack_upper(sigma_up_over_time[k])
        Sigma_t[k] = 0.5 * (S_k + S_k.T)

    # ----- Symplectic matrix S (only Q,P at indices 0,1 in A-basis) --------------
    S = np.zeros((10, 10), dtype=float)
    S[0, 1] =  1.0
    S[1, 0] = -1.0

    # ----- Minimal eigenvalues over time -----------------------------------------
    lam_min_Sigma = np.empty_like(t)
    lam_min_Q     = np.empty_like(t)
    for k in range(t.size):
        Sig = Sigma_t[k]
        lam_min_Sigma[k] = np.min(np.linalg.eigvalsh(Sig))
        Qmat = Sig + 0.5j * S
        lam_min_Q[k] = np.min(np.linalg.eigvalsh(Qmat)).real

    # ============================================================
    # PLOTTING
    # ============================================================

    # 1) Singlets (Q,P,x1..x8) ----------------------------------------------------
    labels_x = ["Q", "P"] + [f"x{i}" for i in range(1, 9)]
    plt.figure(figsize=(9, 5))
    for i in range(10):
        plt.plot(t, X_t[:, i], label=labels_x[i])
    plt.xlabel("Zeit")
    plt.ylabel("Mittelwerte")
    plt.title("Singlets (Q, P, x1..x8) über der Zeit")
    plt.legend(ncol=2, fontsize=8, frameon=False)
    plt.tight_layout()

    # 2) Alle Kovarianzmatrix-Elemente --------------------------------------------
    plt.figure(figsize=(9, 5))
    for i in range(10):
        for j in range(10):
            lw = 1.5 if i == j else 0.8
            alpha = 1.0 if i == j else 0.15
            lbl = rf"$\Sigma_{{{i+1},{j+1}}}$" if i == j else None
            plt.plot(t, Sigma_t[:, i, j], label=lbl, linewidth=lw, alpha=alpha)
    plt.xlabel("Zeit")
    plt.ylabel(r"$\Sigma_{ij}$")
    plt.title("Alle Kovarianzmatrix-Elemente über der Zeit")
    plt.legend(fontsize=8, frameon=False)
    plt.tight_layout()

    # 3) Minimaler Eigenwert von Σ und Σ + i/2 S ----------------------------------
    plt.figure(figsize=(9, 5))
    plt.plot(t, lam_min_Sigma, label=r"$\lambda_{\min}(\Sigma)$", linewidth=1.8)
    plt.plot(t, lam_min_Q,     label=r"$\lambda_{\min}(\Sigma + 0.5i S)$", linewidth=1.8)
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Zeit")
    plt.ylabel("Minimaler Eigenwert")
    plt.title("Minimaler Eigenwert von Σ und Σ + i/2 S")
    plt.legend(frameon=False)
    plt.tight_layout()

    # 4) Populations ρ00, ρ11, ρ22 + Summe ----------------------------------------
    # From x3 and x8 in A-basis: x3 = X_t[:, 2+2], x8 = X_t[:, 2+7]
    x3 = X_t[:, 4]   # positions: [Q(0),P(1),x1(2),x2(3),x3(4),...,x8(9)]
    x8 = X_t[:, 9]
    rho00 = 1/3 + 0.5*( x3 + x8/np.sqrt(3) )
    rho11 = 1/3 + 0.5*( -x3 + x8/np.sqrt(3) )
    rho22 = 1.0 - rho00 - rho11
    rho_sum = rho00 + rho11 + rho22

    plt.figure(figsize=(9, 5))
    plt.plot(t, rho00.real, label=r"$\rho_{00}$")
    plt.plot(t, rho11.real, label=r"$\rho_{11}$")
    plt.plot(t, rho22.real, label=r"$\rho_{22}$")
    plt.plot(t, rho_sum.real, "--", label=r"$\rho_{00}+\rho_{11}+\rho_{22}$", linewidth=1.2)
    plt.axhline(1.0, linestyle=":", linewidth=1.0)
    plt.xlabel("Zeit")
    plt.ylabel("Populationen")
    plt.title(r"Populations $\rho_{00},\rho_{11},\rho_{22}$ und Summe")
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()

    plt.show()
    print("Alle Plots fertig.")