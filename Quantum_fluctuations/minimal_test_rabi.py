# Python code (comments in English; console/output in German)

import numpy as np
from numpy.linalg import eigvalsh
from scipy.integrate import solve_ivp

# ---------------------------
# SU(3) utilities
# ---------------------------

def gellmann_matrices():
    """Return standard 8 Gell-Mann matrices as complex numpy arrays (Tr λa λb = 2 δab)."""
    I = 1j
    l1 = np.array([[0,1,0],[1,0,0],[0,0,0]],dtype=complex)
    l2 = np.array([[0,-I,0],[I,0,0],[0,0,0]],dtype=complex)
    l3 = np.array([[1,0,0],[0,-1,0],[0,0,0]],dtype=complex)
    l4 = np.array([[0,0,1],[0,0,0],[1,0,0]],dtype=complex)
    l5 = np.array([[0,0,-I],[0,0,0],[I,0,0]],dtype=complex)
    l6 = np.array([[0,0,0],[0,0,1],[0,1,0]],dtype=complex)
    l7 = np.array([[0,0,0],[0,0,-I],[0,I,0]],dtype=complex)
    l8 = (1/np.sqrt(3))*np.array([[1,0,0],[0,1,0],[0,0,-2]],dtype=complex)
    return [l1,l2,l3,l4,l5,l6,l7,l8]

# Non-zero structure constants f_{abc} in 0-based indexing (a,b,c in {0..7}) and full antisymmetrization
def su3_f_full():
    """Build fully antisymmetric SU(3) structure constants f_{abc} with standard normalization."""
    from math import sqrt
    base = {
        (0,1,2): 1.0,             # f_123 = 1
        (0,3,6): 0.5,             # f_1,4,7  = +1/2
        (0,4,5): -0.5,            # f_1,5,6  = -1/2
        (1,3,5): 0.5,             # f_2,4,6  = +1/2
        (1,4,6): 0.5,             # f_2,5,7  = +1/2
        (2,3,4): 0.5,             # f_3,4,5  = +1/2
        (2,5,6): -0.5,            # f_3,6,7  = -1/2
        (3,4,7): sqrt(3)/2,       # f_4,5,8  = +√3/2
        (5,6,7): sqrt(3)/2,       # f_6,7,8  = +√3/2
    }
    # antisymmetrize
    f = {}
    def set_perm(sign, i,j,k,val):
        f[(i,j,k)] = sign*val
    for (i,j,k), val in base.items():
        # even permutations
        set_perm(+1, i,j,k, val)
        set_perm(+1, j,k,i, val)
        set_perm(+1, k,i,j, val)
        # odd permutations
        set_perm(-1, i,k,j, val)
        set_perm(-1, k,j,i, val)
        set_perm(-1, j,i,k, val)
    return f

def anticommutator_covariance(rho, lams):
    """Compute Sigma_ab = 1/2 Tr(rho {λa,λb}) - <λa><λb> for a,b=1..8 from a 3x3 density matrix rho."""
    m = np.array([np.trace(rho @ L) for L in lams], dtype=complex).real
    S = np.zeros((8,8), float)
    for a in range(8):
        for b in range(8):
            anti = lams[a] @ lams[b] + lams[b] @ lams[a]
            val = 0.5*np.trace(rho @ anti)
            S[a,b] = (val.real - m[a]*m[b])
    return S, m

# ---------------------------
# Dynamics under H = (Ω/2) λ6  (here λ6 is index 5 in 0-based)
# ---------------------------

def build_G(omega, f, gen_index=5):
    """Adjoint generator matrix G_{ab} = -Ω f_{(gen),a,b} (Hamiltonian flow for means)"""
    G = np.zeros((8,8), float)
    for a in range(8):
        for b in range(8):
            G[a,b] = -omega * f.get((gen_index,a,b), 0.0)
    return G

def evolve_m_and_Sigma(t_span, t_eval, m0, Sigma0, G):
    """Integrate m(t) and Sigma(t) with dm/dt = G m, dSigma/dt = G Sigma + Sigma G^T (unitary part only)."""
    # Flatten Sigma
    def rhs(t, y):
        m = y[:8]
        S = y[8:].reshape(8,8)
        dm = G @ m
        dS = G @ S + S @ G.T
        return np.concatenate([dm, dS.ravel()])
    y0 = np.concatenate([m0, Sigma0.ravel()])
    sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval, rtol=1e-9, atol=1e-12, method="RK45")
    Ms = sol.y[:8,:].T              # shape (T,8)
    Sigmas = sol.y[8:,:].T.reshape(len(t_eval), 8, 8)
    return sol.t, Ms, Sigmas

def s_matrix_from_m(m, f):
    """Build s_ab = 2 * sum_c f_{a,b,c} m_c (commutators of fluctuation operators)."""
    s = np.zeros((8,8), float)
    for a in range(8):
        for b in range(8):
            acc = 0.0
            for c in range(8):
                acc += 2.0 * f.get((a,b,c), 0.0) * m[c]
            s[a,b] = acc
    return s

def is_psd(matrix, tol=1e-10):
    """Check Hermitian PSD: smallest eigenvalue >= -tol."""
    # Ensure Hermitian symmetrization
    H = 0.5*(matrix + matrix.T.conj())
    evals = eigvalsh(H)
    return evals.min(), np.all(evals >= -tol)

def rs_condition_ok(Sigma, s, tol=1e-10):
    """Check Robertson–Schrödinger condition: Sigma + (i/2) s  >= 0 (as Hermitian matrix)."""
    M = Sigma.astype(complex) + 0.5j * s
    evals = eigvalsh(0.5*(M + M.T.conj()))
    return evals.min(), np.all(evals >= -tol)

# ---------------------------
# Example setup
# ---------------------------

# Parameters
Omega = 1.0      # Rabi frequency
t0, t_end = 0.0, 20.0
t_eval = np.linspace(t0, t_end, 401)

# Initial atomic qutrit state rho (pure state |1>): |1> = (0,1,0)^T  (basis order |0>,|1>,|2>)
e1 = np.array([0,1,0], dtype=complex)
rho0 = np.outer(e1, e1.conj())

# Build Gell-Mann, initial Sigma and m from rho0
lams = gellmann_matrices()
Sigma0, m0 = anticommutator_covariance(rho0, lams)

# Structure constants and generator
f = su3_f_full()
G = build_G(Omega, f, gen_index=5)  # gen_index=5 corresponds to λ6 (couples levels 1<->2)

# Evolve
ts, Ms, Sigmas = evolve_m_and_Sigma((t0, t_end), t_eval, m0, Sigma0, G)

# ---------------------------
# Checks: PSD(Σ) and RS(Σ, s(m(t))) for all times
# ---------------------------

min_eval_Sigma = []
ok_psd_Sigma = []
min_eval_RS = []
ok_rs = []

for t_idx in range(len(ts)):
    S = Sigmas[t_idx]
    m = Ms[t_idx]
    s = s_matrix_from_m(m, f)
    meS, okS = is_psd(S, tol=1e-10)
    meR, okR = rs_condition_ok(S, s, tol=1e-10)
    min_eval_Sigma.append(meS)
    ok_psd_Sigma.append(okS)
    min_eval_RS.append(meR)
    ok_rs.append(okR)

# Summary (German console output)
print("=== Simulation abgeschlossen ===")
print(f"Zeitschritte: {len(ts)}, Zeitintervall: [{ts[0]:.3f}, {ts[-1]:.3f}]")
print("\n-- PSD-Check für Σ(t) --")
print(f"Minimales Eigenwert-Minimum über alle Zeiten: {np.min(min_eval_Sigma): .3e}")
print(f"Ist Σ(t) für alle Zeiten PSD (innerhalb Toleranz)? -> {np.all(ok_psd_Sigma)}")

print("\n-- Robertson–Schrödinger (RS) --")
print(f"Minimales Eigenwert-Minimum von M(t)=Σ+i/2 s über alle Zeiten: {np.min(min_eval_RS): .3e}")
print(f"Erfüllt RS-Bedingung für alle Zeiten (innerhalb Toleranz)? -> {np.all(ok_rs)}")

# Optional: report first violations if any
if not np.all(ok_psd_Sigma):
    idxs = [i for i,v in enumerate(ok_psd_Sigma) if not v][:5]
    print("\nAchtung: Σ(t) nicht PSD bei den ersten Zeiten (Index, t, minEV):")
    for i in idxs:
        print(f"  i={i}, t={ts[i]:.4f}, minEV={min_eval_Sigma[i]: .3e}")
if not np.all(ok_rs):
    idxs = [i for i,v in enumerate(ok_rs) if not v][:5]
    print("\nAchtung: RS-Bedingung verletzt bei den ersten Zeiten (Index, t, minEV):")
    for i in idxs:
        print(f"  i={i}, t={ts[i]:.4f}, minEV={min_eval_RS[i]: .3e}")
