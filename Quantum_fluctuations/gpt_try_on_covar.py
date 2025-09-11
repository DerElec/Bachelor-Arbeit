# -*- coding: utf-8 -*-
"""
Coupled simulation of singlets (Q,P,x1..x8) and pair correlators m_i m_j.
- Comments in English
- Console/plot texts in German
"""

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ---------- 1) Import your symbolic Sigma_dt from covar_everything ----------
import covar_everything as covar

# Physical/symbolic parameters used by your covar_everything module
g0, Delta1, V, gamma, Omega, kappa, eta, Delta2 = sp.symbols("g0 Delta1 V gamma Omega kappa eta Delta2")

numeric_params = {
    g0:     1,
    Delta1: 1,
    Delta2: 1,
    V:      -4.0,
    gamma:  1,
    Omega:  8,
    kappa:  1,
    eta:    1
}

# Pull important matrices; we need Sigma_dt
G, sDs, Z, Pm, Qm, Z_prime, W, Sigma_dt, Sigma, K = covar.get_important_matricies(numeric_params)

# ---------- 2) Helper: symbols and machinery to build RHS for pair correlators ----------
def build_pair_symbol_matrix(n: int):
    """Create an n×n matrix of symbols M with entries m{i}m{j} (upper triangle used)."""
    return sp.Matrix(n, n, lambda i, j: sp.symbols(f"m{i+1}m{j+1}"))

def list_pairs(M, symmetric_pairs: bool = True):
    """Order pairs consistently; by default take upper-triangular including diagonal."""
    n = M.shape[0]
    pairs = []
    if symmetric_pairs:
        for i in range(n):
            for j in range(i, n):
                pairs.append((i, j, M[i, j]))
    else:
        for i in range(n):
            for j in range(n):
                pairs.append((i, j, M[i, j]))
    return pairs

def build_pair_equations(Sigma_dt: sp.Matrix, M: sp.Matrix, symmetric_pairs: bool = True):
    """Build element-wise equations: d(m_im_j)/dt = Sigma_dt[i,j]."""
    assert Sigma_dt.shape == M.shape, "Sigma_dt and M must have same shape"
    eqs, rhs_list = [], []
    for (i, j, sym_ij) in list_pairs(M, symmetric_pairs=symmetric_pairs):
        d_ij = sp.symbols(f"d{sym_ij.name}_dt")  # e.g. dm3m7_dt
        rhs = sp.simplify(Sigma_dt[i, j])
        eqs.append(sp.Eq(d_ij, rhs))
        rhs_list.append(rhs)
    return eqs, rhs_list

def pack_state_symbols(mP, M: sp.Matrix, symmetric_pairs: bool = True):
    """
    Create consistent ordering for the full state vector y:
      y = [m1..mn, then m1m1, m1m2, ..., mnmn (upper triangle)]
    """
    y_syms = list(mP)
    for (_, _, sym_ij) in list_pairs(M, symmetric_pairs=symmetric_pairs):
        y_syms.append(sym_ij)
    return y_syms

def build_rhs_lambdify(rhs_pairs_list, y_syms, extra_params=()):
    """
    Lambdify the pair RHS list to a NumPy-callable function f(z0,...,zK,*extra_params).
    The z's correspond to y_syms in order.
    """
    z = sp.symbols(f'z0:{len(y_syms)}')
    subs_map = {sym: z[k] for k, sym in enumerate(y_syms)}
    rhs_z = [expr.xreplace(subs_map) for expr in rhs_pairs_list]
    args = list(z) + list(extra_params)
    f = sp.lambdify(args, rhs_z, modules='numpy')
    return f

def enforce_upper_triangle(M):
    """Replace lower-triangle symbols m{j}m{i} by m{i}m{j} for a consistent unique representation."""
    n = M.shape[0]
    sub_map = {}
    for i in range(n):
        for j in range(i+1, n):
            upper = M[i, j]                  # m{i+1}m{j+1}
            lower = sp.symbols(f"m{j+1}m{i+1}")
            sub_map[lower] = upper
    return sub_map

def pairs_index_map(M, symmetric_pairs=True):
    """Build a dict mapping (i,j) with i<=j to the flat index in the pairs slice."""
    idx = {}
    pairs = list_pairs(M, symmetric_pairs=symmetric_pairs)
    for k, (i, j, _) in enumerate(pairs):
        idx[(i, j)] = k
    return idx

# Clean Sigma_dt by enforcing m_j m_i -> m_i m_j
sub_map = enforce_upper_triangle(Sigma_dt)
Sigma_dt_clean = Sigma_dt.xreplace(sub_map)

# Prepare symbols and RHS for the pair system
n = 10  # number of singlets m1..m10  (Q=m9, P=m10)
mP = sp.symbols('m1:11')  # m1,...,m10
M = build_pair_symbol_matrix(n)
pair_eqs, rhs_pairs_list = build_pair_equations(Sigma_dt_clean, M, symmetric_pairs=True)
y_syms = pack_state_symbols(mP, M, symmetric_pairs=True)

# NOTE: assume Sigma_dt depends on (gamma, kappa); extend tuple if needed
rhs_pairs_func = build_rhs_lambdify(rhs_pairs_list, y_syms, extra_params=(gamma, kappa))

# ---------- 3) Your ket/x dynamics (singlets) ----------
def rhs_gellmann_qp_from_ket(t, y, params):
    """Compute derivatives for y = [a, ad, ρ00, ρ01, ρ10, ρ11, ρ22, ρ21, ρ12, ρ20, ρ02]."""
    a, a_dagger = y[0], y[1]
    ket00, ket01, ket10, ket11, ket22, ket21, ket12, ket20, ket02 = y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10]
    κ, γ, Γ, Ω, δ1, δ2, η, Vval = (params[k] for k in ('kappa','gamma','Gamma','Omega','delta1','delta2','eta','V'))

    da_dt        = -κ/2 * a - 1j*(γ*ket01) + η
    da_dagger_dt = np.conj(da_dt)

    d00 = Γ*ket11 + 1j*γ*(ket10*a - ket01*a_dagger)
    d01 = -Γ/2*ket01 + 1j*(-δ1*ket01 + γ*(ket11*a - ket00*a) - Ω/2*ket02)
    d10 = np.conj(d01)

    d11 = -Γ*ket11 + 1j*γ*(ket01*a_dagger - ket10*a) + 1j*(Ω/2)*(ket21 - ket12)
    d22 =  1j*(Ω/2)*(ket12 - ket21)

    d21 = -Γ/2*ket21 + 1j*(δ2*ket21 - δ1*ket21 - γ*ket20*a + (Ω/2)*(ket11 - ket22) + 2*Vval*ket21*ket22)
    d12 = np.conj(d21)

    d02 =  1j*(-δ2*ket02 - Ω/2*ket01 - 2*Vval*ket02*ket22 + γ*ket12*a)
    d20 = np.conj(d02)

    return np.array([da_dt, da_dagger_dt, d00, d01, d10, d11, d22, d21, d12, d20, d02], dtype=complex)

def convert_state(y):
    """
    If len(y)==11, project to 10-vector [Q,P,x1..x8].
    If len(y)==10, reconstruct to 11-vector [a,ad,ρ00,ρ01,ρ10,ρ11,ρ22,ρ21,ρ12,ρ20,ρ02].
    """
    if len(y) == 11:
        a, ad = y[0], y[1]
        rho00, rho01, rho10, rho11, rho22, rho21, rho12, rho20, rho02 = y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10]
        x1 = rho01 + rho10
        x2 = -1j*(rho01 - rho10)
        x3 = rho00 - rho11
        x4 = rho02 + rho20
        x5 = -1j*(rho02 - rho20)
        x6 = rho12 + rho21
        x7 = -1j*(rho12 - rho21)
        x8 = (rho00 + rho11 - 2*rho22)/np.sqrt(3)
        Q  = (a + ad)/np.sqrt(2)
        P  = (a - ad)/(1j*np.sqrt(2))
        return np.array([Q, P, x1, x2, x3, x4, x5, x6, x7, x8], dtype=complex)

    elif len(y) == 10:
        Q, P = y[0], y[1]
        x1, x2, x3, x4, x5, x6, x7, x8 = y[2:]
        a  = (Q + 1j*P)/np.sqrt(2)
        ad = np.conj(a)
        rho00 = 1/3 + 0.5*( x3 + x8/np.sqrt(3) )
        rho11 = 1/3 + 0.5*(-x3 + x8/np.sqrt(3))
        rho22 = 1 - rho00 - rho11
        rho01 = (x1 + 1j*x2)/2
        rho02 = (x4 + 1j*x5)/2
        rho12 = (x6 + 1j*x7)/2
        rho10, rho20, rho21 = np.conj(rho01), np.conj(rho02), np.conj(rho12)
        return np.array([a, ad, rho00, rho01, rho10, rho11, rho22, rho21, rho12, rho20, rho02], dtype=complex)

    else:
        raise ValueError(f"Länge von y muss 10 oder 11 sein, got {len(y)}")

def rhs_gellmann_qp_from_x(t, x, params):
    """Compute d[Q,P,x1..x8]/dt by converting to ket, calling rhs, then projecting."""
    y_full = convert_state(x)
    dy = rhs_gellmann_qp_from_ket(t, y_full, params)
    da_dt, da_dagger_dt = dy[0], dy[1]
    d00, d01, d10, d11 = dy[2], dy[3], dy[4], dy[5]
    d22, d21, d12, d20, d02 = dy[6], dy[7], dy[8], dy[9], dy[10]

    dx1 =  d01 + d10
    dx2 = -1j*d01 + 1j*d10
    dx3 =  d00 - d11
    dx4 =  d02 + d20
    dx5 = -1j*d02 + 1j*d20
    dx6 =  d12 + d21
    dx7 = -1j*d12 + 1j*d21
    dx8 = (d00 + d11 - 2*d22)/np.sqrt(3)
    dQ  = (da_dt + da_dagger_dt)/np.sqrt(2)
    dP  = (da_dt - da_dagger_dt)/(1j*np.sqrt(2))
    return np.array([dQ, dP, dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8], dtype=complex)

# ---------- 4) Coupled RHS for [singlets; pairs] ----------
def build_coupled_rhs(rhs_pairs_func, pairs_idx_map, params):
    """
    Return a function F(t, y_full) that computes:
      y_full = [m1..m10, then pairs in upper-tri order]
      dy_full/dt = [dx/dt, d(pairs)/dt]
    """
    gamma_val = params['gamma']
    kappa_val = params['kappa']

    def F(t, y):
        # y[0:10] are the singlets in order m1..m10 with convention:
        #   m1..m8  == x1..x8
        #   m9 == Q, m10 == P
        x = y[:10]
        dx = rhs_gellmann_qp_from_x(t, x, params)

        # d(pairs)/dt from lambdified Sigma_dt; expects all y symbols separately plus (gamma,kappa)
        d_pairs = np.array(rhs_pairs_func(*y, gamma_val, kappa_val), dtype=complex)
        return np.concatenate([dx, d_pairs])

    return F

# ---------- 5) Simulation setup ----------
if __name__ == "__main__":
    # ODE parameters (adjust as needed)
    params = dict(
        kappa=1.0, gamma=1.0, Gamma=1.0,
        Omega=8.0, delta1=1.0, delta2=1.0,
        eta=1.0, V=-8.0
    )

    # Initial ket state: rho00=1, others 0, a=0
    y0_ket = np.array([0+0j, 0+0j, 1+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j], dtype=complex)
    # Project to singlets x = [Q,P,x1..x8]  (m9=Q, m10=P)
    x0 = convert_state(y0_ket)  # length 10, complex

    # Build pair index map and initial pairs (outer product, upper triangle order)
    pmap = pairs_index_map(M, symmetric_pairs=True)
    num_pairs = len(pmap)  # should be n(n+1)/2 = 55 for n=10
    pairs0 = np.zeros(num_pairs, dtype=complex)
    for (i, j), k in pmap.items():
        pairs0[k] = x0[i] * x0[j]  # uncorrelated initial products

    # Full initial state: [singlets; pairs]
    y0_full = np.concatenate([x0, pairs0])

    # Time span
    t0, t_end = 0.0, 500.0   # kürzer als 2000 für schnellere Runs; anpassbar
    t_span = (t0, t_end)

    # Build coupled RHS
    F = build_coupled_rhs(rhs_pairs_func, pmap, params)

    print("Starte gekoppelte Integration von Singlets und Paar-Korrelationen ...")
    sol = solve_ivp(F, t_span, y0_full, method='RK45', atol=1e-8, rtol=1e-8)

    if not sol.success:
        print("Achtung: Integrator meldet kein 'success'. Nachricht:", sol.message)

    # ---------- 6) Extract and plot diagonal correlations m_a m_a ----------
    print("Plotte Diagonal-Korrelationen m_am_a (a=1..10) über der Zeit ...")
    t_vals = sol.t
    Y = sol.y  # shape (10 + 55, len(t))

    # Helper to fetch pair component for (i,i)
    def get_series_diag(a_index_zero_based):
        """Return time series for m_a m_a with a_index in [0..9]."""
        k = pmap[(a_index_zero_based, a_index_zero_based)]  # index within pairs slice
        return Y[10 + k, :]  # pairs slice starts at offset 10

    # Prepare all 10 diagonal series
    diag_series = [np.real(get_series_diag(a)) for a in range(10)]

    # Plot real parts of diagonals m_am_a
    plt.figure()
    for a in range(10):
        label = f"m{a+1}m{a+1}" if a < 8 else ("Q·Q" if a == 8 else "P·P")
        plt.plot(t_vals, diag_series[a], label=label)
    plt.xlabel("Zeit")
    plt.ylabel("Re(m_am_a)")
    plt.title("Diagonale Paar-Korrelationen Re(m_am_a)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

    print("Fertig. (Hinweis: Q=m9, P=m10; die Plots zeigen Realteile der Diagonalkorrelationen.)")
