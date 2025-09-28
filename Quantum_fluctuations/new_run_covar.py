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
    Omega_val =0#8.0  # laser drive 1->2
    Gamma_val = 0  #atom decay 1->0
    V_val =0#-6.0  #interaction potential
    Delta1_val=0  #detuning from 1 
    Delta2_val=0 #detuning from 2
    #############
    g0_val=1#cavity coupling 
    eta_val=0 # cavity drive 
    kappa_val=1#cavity decay
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
    y0_ket = np.array([1+0j, 0+0j, 1+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j])
    #y0_ket = np.array([0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 1+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j])
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
    T_end = 20.0         # total simulation time (adjust as needed)
    N_eval = 2001        # number of time samples
    t_span = (0.0, T_end)
    t_eval = np.linspace(t_span[0], t_span[1], N_eval)

    y0 = np.concatenate([x0, Sigma0.reshape(-1)])
    sol = solve_ivp(rhs_combined, t_span, y0, t_eval=t_eval,
                    method="RK45", rtol=1e-8, atol=1e-10)

    if not sol.success:
        print("WARNUNG: Integration nicht erfolgreich:", sol.message)
    else:
        print("Integration erfolgreich abgeschlossen.")

    # --- Unpack solution ------------------------------------------------------------
    t = sol.t
    X_t = sol.y[:10, :].T                                   # shape (N, 10)
    Sigma_t = sol.y[10:, :].T.reshape(-1, 10, 10)           # shape (N, 10, 10)
    # Enforce symmetry in outputs (guard against tiny drifts)
    Sigma_t = 0.5 * (Sigma_t + np.transpose(Sigma_t, (0, 2, 1)))

    # --- Symplectic matrix S (only for Q,P block at indices 8,9) -------------------
    S = np.zeros((10, 10), dtype=float)
    S[8, 9] = 1.0
    S[9, 8] = -1.0

    # --- Minimal eigenvalues over time ---------------------------------------------
    lam_min_Sigma = np.empty_like(t)
    lam_min_Q     = np.empty_like(t)

    for k in range(t.size):
        Sig = 0.5 * (Sigma_t[k] + Sigma_t[k].T)            # real-symmetric
        lam_min_Sigma[k] = np.min(np.linalg.eigvalsh(Sig))

        Qmat = Sig + 0.5j * S                              # complex-Hermitian
        # Numerical guard: eigvalsh returns real for Hermitian; take .real to be explicit.
        lam_min_Q[k] = np.min(np.linalg.eigvalsh(Qmat)).real

    # ============================================================
    # PLOTTING
    # ============================================================
    labels_x = [f"x{i}" for i in range(1, 9)] + ["Q", "P"]

    # 1) Singlets (x1..x8,Q,P)
    plt.figure(figsize=(9, 5))
    for i in range(10):
        plt.plot(t, X_t[:, i], label=labels_x[i])
    plt.xlabel("Zeit")
    plt.ylabel("Mittelwerte")
    plt.title("Singlets und Feld-Quadraturen über der Zeit")
    plt.legend(ncol=2, fontsize=8, frameon=False)
    plt.tight_layout()

    # 2) Alle Kovarianzmatrix-Elemente
    plt.figure(figsize=(9, 5))
    for i in range(10):
        for j in range(10):
            # Diagonalen deutlich, Off-Diagonalen transparent
            lw = 1.5 if i == j else 0.8
            alpha = 1.0 if i == j else 0.15
            # Nur Diagonale beschriften, um Legendenflut zu vermeiden
            lbl = rf"$\Sigma_{{{i+1},{j+1}}}$" if i == j else None
            plt.plot(t, Sigma_t[:, i, j], label=lbl, linewidth=lw, alpha=alpha)
    plt.xlabel("Zeit")
    plt.ylabel(r"$\Sigma_{ij}$")
    plt.title("Alle Kovarianzmatrix-Elemente über der Zeit")
    plt.legend(fontsize=8, frameon=False)
    plt.tight_layout()

    # 3) Minimale Eigenwerte: Σ und Σ + i/2 S
    plt.figure(figsize=(9, 5))
    plt.plot(t, lam_min_Sigma, label=r"$\lambda_{\min}(\Sigma)$", linewidth=1.8)
    plt.plot(t, lam_min_Q,     label=r"$\lambda_{\min}(\Sigma + 0.5i S)$", linewidth=1.8)
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Zeit")
    plt.ylabel("Minimaler Eigenwert")
    plt.title("Minimaler Eigenwert von Σ und Σ + i/2 S über der Zeit")
    plt.legend(frameon=False)
    plt.tight_layout()

    plt.show()
    
    # ============================================================
    # 4) Populations rho00, rho11, rho22 aus Gell-Mann-Matrizen + Plot
    #     (füge diesen Block nach der Integration ein; t und X_t werden verwendet)
    # ============================================================

    print("4. Populations werden aus den Gell-Mann-Matrizen rekonstruiert und geplottet...")

    # -- Predefine Gell-Mann matrices (3x3) once ------------------
    l1 = np.array([[0,1,0],[1,0,0],[0,0,0]],complex)
    l2 = np.array([[0,-1j,0],[1j,0,0],[0,0,0]],complex)
    l3 = np.array([[1,0,0],[0,-1,0],[0,0,0]],complex)
    l4 = np.array([[0,0,1],[0,0,0],[1,0,0]],complex)
    l5 = np.array([[0,0,-1j],[0,0,0],[1j,0,0]],complex)
    l6 = np.array([[0,0,0],[0,0,1],[0,1,0]],complex)
    l7 = np.array([[0,0,0],[0,0,-1j],[0,1j,0]],complex)
    l8 = (1/np.sqrt(3))*np.array([[1,0,0],[0,1,0],[0,0,-2]],complex)
    LAMBDAS = [l1,l2,l3,l4,l5,l6,l7,l8]

    def populations_from_x_with_lambdas(x8_vec: np.ndarray) -> Tuple[float, float, float]:
        """
        Reconstruct populations from Gell-Mann expansion:
        rho = I/3 + 1/2 * sum_{a=1}^8 x_a * lambda_a
        Input: x8_vec = [x1,..,x8] (shape (8,))
        Returns: (rho00, rho11, rho22) as floats
        """
        rho = (np.eye(3, dtype=complex) / 3.0)
        for a in range(8):
            xa = float(np.real_if_close(x8_vec[a], tol=1e-12))  # means are real
            rho = rho + 0.5 * xa * LAMBDAS[a]
        # Take real parts of diagonal (Hermitian by construction)
        return rho[0,0].real, rho[1,1].real, rho[2,2].real

    # Vectorized over time (loop is fine for N~1e3-1e4)
    N = t.size
    rho00 = np.empty(N)
    rho11 = np.empty(N)
    rho22 = np.empty(N)
    for k in range(N):
        # X_t[k, :8] holds [x1..x8]
        r00, r11, r22 = populations_from_x_with_lambdas(X_t[k, :8])
        rho00[k], rho11[k], rho22[k] = r00, r11, r22

    rho_sum = rho00 + rho11 + rho22

    # Optional: also compute via closed-form formulas for a quick internal consistency check
    x3 = X_t[:, 2]
    x8 = X_t[:, 7]
    rho00_formula = 1/3 + 0.5*( x3 + x8/np.sqrt(3) )
    rho11_formula = 1/3 + 0.5*( -x3 + x8/np.sqrt(3) )
    rho22_formula = 1.0 - rho00_formula - rho11_formula
    max_dev = np.max(np.abs(rho00 - rho00_formula)) + np.max(np.abs(rho11 - rho11_formula)) + np.max(np.abs(rho22 - rho22_formula))
    print(f"   Konsistenz-Check (Formel vs. Lambda-Rekonstruktion), max. Abweichung: {max_dev:.3e}")

    # Sanity checks for probability simplex
    min_val = min(rho00.min(), rho11.min(), rho22.min())
    max_val = max(rho00.max(), rho11.max(), rho22.max())
    sum_dev = np.max(np.abs(rho_sum - 1.0))
    print(f"   Wertebereich Populations: min={min_val:.6f}, max={max_val:.6f}, max|rho00+rho11+rho22-1|={sum_dev:.3e}")

    # -- Plot populations and their sum -------------------------------------------
    plt.figure(figsize=(9, 5))
    plt.plot(t, rho00, label=r"$\rho_{00}$")
    plt.plot(t, rho11, label=r"$\rho_{11}$")
    plt.plot(t, rho22, label=r"$\rho_{22}$")
    plt.plot(t, rho_sum, label=r"$\rho_{00}+\rho_{11}+\rho_{22}$", linestyle="--", linewidth=1.2)
    plt.axhline(1.0, linestyle=":", linewidth=1.0)  # visual guide for normalization
    plt.xlabel("Zeit")
    plt.ylabel("Populationen")
    plt.title(r"Rekonstruierte Populations $\rho_{00},\rho_{11},\rho_{22}$ und Summe")
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()

    plt.show()
    print("Populations-Plot fertig.")
    print("Plots fertig.")
