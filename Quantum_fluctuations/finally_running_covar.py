# -*- coding: utf-8 -*-
# Console text: German; Code & comments: English

import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
import sympy as sp

# ---------------------------------------------------------------------
# 0) Your model-specific import: must provide Sigma_dt via sympy.Matrix
# ---------------------------------------------------------------------
import covar_everything as covar


# ============================================================
# A) Singles dynamics (your provided functions, kept consistent)
# ============================================================

def rhs_gellmann_qp_from_ket(t, y, params):
    """Compute derivatives for y = [a, ad, ρ00, ρ01, ρ10, ρ11, ρ22, ρ21, ρ12, ρ20, ρ02]."""
    # unpack state
    a, a_dagger = y[0], y[1]
    ket00, ket01, ket10, ket11, ket22, ket21, ket12, ket20, ket02 = y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10]
    # unpack parameters
    κ, γ, Γ, Ω, δ1, δ2, η, V = (params[k] for k in ('kappa','gamma','Gamma','Omega','delta1','delta2','eta','V'))

    # ladder dynamics
    da_dt        = -κ/2 * a - 1j*(γ*ket01) + η
    da_dagger_dt = np.conj(da_dt)

    # density-matrix amplitudes
    d00 = Γ*ket11 + 1j*γ*(ket10*a - ket01*a_dagger)
    d01 = -Γ/2*ket01 + 1j*(-δ1*ket01 + γ*(ket11*a - ket00*a) - Ω/2*ket02)
    d10 = np.conj(d01)

    d11 = -Γ*ket11 + 1j*γ*(ket01*a_dagger - ket10*a) + 1j*(Ω/2)*(ket21 - ket12)
    d22 = 1j*(Ω/2)*(ket12 - ket21)

    d21 = -Γ/2*ket21 + 1j*(δ2*ket21 - δ1*ket21 - γ*ket20*a + (Ω/2)*(ket11 - ket22) + 2*V*ket21*ket22)
    d12 = np.conj(d21)

    d02 = 1j*(-δ2*ket02 - Ω/2*ket01 - 2*V*ket02*ket22 + γ*ket12*a)
    d20 = np.conj(d02)

    return np.array([da_dt, da_dagger_dt, d00, d01, d10, d11, d22, d21, d12, d20, d02], dtype=complex)


def convert_state(y):
    """
    If len(y)==11, project to 10-vector [Q,P,x1..x8].
    If len(y)==10, reconstruct to 11-vector [a,ad,ρ00,ρ01,ρ10,ρ11,ρ22,ρ21,ρ12,ρ20,ρ02].
    """
    if len(y) == 11:
        # full → projected
        a, ad = y[0], y[1]
        rho00, rho01, rho10, rho11, rho22, rho21, rho12, rho02, rho20 = y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[10], y[9]
        # Gell-Mann x's
        x1 = rho01 + rho10
        x2 = -1j*(rho01 - rho10)
        x3 = rho00 - rho11
        x4 = rho02 + rho20
        x5 = -1j*(rho02 - rho20)
        x6 = rho12 + rho21
        x7 = -1j*(rho12 - rho21)
        x8 = (rho00 + rho11 - 2*rho22)/np.sqrt(3)
        # quadratures
        Q = (a + ad)/np.sqrt(2)
        P = (a - ad)/(1j*np.sqrt(2))
        return np.array([Q, P, x1, x2, x3, x4, x5, x6, x7, x8], dtype=complex)

    elif len(y) == 10:
        # projected → full
        Q, P = y[0], y[1]
        x1, x2, x3, x4, x5, x6, x7, x8 = y[2:]
        # ladder
        a  = (Q + 1j*P)/np.sqrt(2)
        ad = np.conj(a)
        # density diag
        rho00 = 1/3 + 0.5*( x3 + x8/np.sqrt(3) )
        rho11 = 1/3 + 0.5*(-x3 + x8/np.sqrt(3))
        rho22 = 1 - rho00 - rho11
        # off-diags
        rho01 = (x1 + 1j*x2)/2
        rho02 = (x4 + 1j*x5)/2
        rho12 = (x6 + 1j*x7)/2
        rho10, rho20, rho21 = np.conj(rho01), np.conj(rho02), np.conj(rho12)
        return np.array([a, ad, rho00, rho01, rho10, rho11, rho22, rho21, rho12, rho20, rho02], dtype=complex)

    else:
        raise ValueError(f"Länge von y muss 10 oder 11 sein, erhalten: {len(y)}")


def rhs_gellmann_qp_from_x(t, x, params):
    """Compute d[Q,P,x1..x8]/dt by converting to ket, calling rhs, then projecting."""
    y_full = convert_state(x)
    dy = rhs_gellmann_qp_from_ket(t, y_full, params)
    # unpack
    da_dt, da_dagger_dt = dy[0], dy[1]
    d00, d01, d10, d11 = dy[2], dy[3], dy[4], dy[5]
    d22, d21, d12, d20, d02 = dy[6], dy[7], dy[8], dy[9], dy[10]
    # compute dx's
    dx1 =  d01 + d10
    dx2 = -1j*d01 + 1j*d10
    dx3 =  d00 - d11
    dx4 =  d02 + d20
    dx5 = -1j*d02 + 1j*d20
    dx6 =  d12 + d21
    dx7 = -1j*d12 + 1j*d21
    dx8 = (d00 + d11 - 2*d22)/np.sqrt(3)
    # quadrature derivatives
    dQ = (da_dt + da_dagger_dt)/np.sqrt(2)
    dP = (da_dt - da_dagger_dt)/(1j*np.sqrt(2))
    return np.array([dQ, dP, dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8], dtype=complex)


# ============================================================
# B) Pair system tools (build from Sigma_dt)
# ============================================================

def make_m_symbols(n=10):
    """Return (m1,...,mn) as sympy symbols."""
    return sp.symbols(f"m1:{n+1}")

def build_pair_symbol_matrix(n: int):
    """Create the symbol matrix M with entries m{i}m{j}."""
    return sp.Matrix(n, n, lambda i, j: sp.symbols(f"m{i+1}m{j+1}"))

def list_pairs(M, symmetric_pairs: bool = True):
    """List (i,j,sym) in lexicographic order; if symmetric, use upper triangle including diagonal."""
    n = M.shape[0]
    if symmetric_pairs:
        return [(i, j, M[i, j]) for i in range(n) for j in range(i, n)]
    else:
        return [(i, j, M[i, j]) for i in range(n) for j in range(n)]

def enforce_upper_triangle_map(M: sp.Matrix):
    """Build substitution map that replaces lower-triangle symbols m{j}m{i} by m{i}m{j}."""
    n = M.shape[0]
    sub_map = {}
    for i in range(n):
        for j in range(i+1, n):
            upper = M[i, j]                     # m{i+1}m{j+1}
            lower = sp.symbols(f"m{j+1}m{i+1}") # m{j+1}m{i+1}
            sub_map[lower] = upper
    return sub_map

def build_pair_equations(Sigma_dt: sp.Matrix, M: sp.Matrix, symmetric_pairs: bool = True):
    """Create element-wise ODE equations for pairs: dm{i}m{j}_dt = Sigma_dt[i,j]."""
    assert Sigma_dt.shape == M.shape, "Sigma_dt and M must have same shape"
    pairs = list_pairs(M, symmetric_pairs=symmetric_pairs)
    eqs, rhs_list = [], []
    for (i, j, sym_ij) in pairs:
        d_ij = sp.symbols(f"d{sym_ij.name}_dt")
        rhs  = sp.simplify(Sigma_dt[i, j])
        eqs.append(sp.Eq(d_ij, rhs))
        rhs_list.append(rhs)
    return eqs, rhs_list, pairs

def pack_state_symbols_with_order(singles_order_syms, M: sp.Matrix, symmetric_pairs: bool = True):
    """State order = [singles in the given order, then pairs (upper triangle)]."""
    y_syms = list(singles_order_syms)
    for _, _, sym_ij in list_pairs(M, symmetric_pairs=symmetric_pairs):
        y_syms.append(sym_ij)
    return y_syms

def build_rhs_lambdify(rhs_pairs_list, y_syms, extra_params=()):
    """
    Lambdify RHS of pairs, expecting arguments in the exact order of y_syms
    (singles first, then pairs).
    """
    z = sp.symbols(f'z0:{len(y_syms)}')
    subs_map = {sym: z[k] for k, sym in enumerate(y_syms)}
    rhs_z = [expr.xreplace(subs_map) for expr in rhs_pairs_list]
    args = list(z) + list(extra_params)
    return sp.lambdify(args, rhs_z, modules='numpy')

def build_combined_system_with_order(Sigma_dt_raw: sp.Matrix,
                                     singles_order_syms,   # list of sympy symbols, length 10
                                     symmetric_pairs=True,
                                     extra_params=()):
    """
    singles_order_syms defines how (m1..m10) map to your actual singles vector [Q,P,x1..x8].
    Example (your case): [m9, m10, m1, m2, m3, m4, m5, m6, m7, m8]
    """
    n_singles = len(singles_order_syms)
    assert Sigma_dt_raw.shape == (n_singles, n_singles), "Sigma_dt shape must match #singles"

    # Pair symbol matrix (canonical m-symbol space)
    M = build_pair_symbol_matrix(n_singles)

    # Map lower triangle -> upper triangle symbols inside Sigma_dt
    sub_map = enforce_upper_triangle_map(M)
    Sigma_dt_clean = Sigma_dt_raw.xreplace(sub_map)

    # Equations/RHS for pairs
    pair_eqs, rhs_pairs_list, pairs_order = build_pair_equations(Sigma_dt_clean, M, symmetric_pairs=symmetric_pairs)

    # Build y_syms in the requested singles order, then pairs
    y_syms = pack_state_symbols_with_order(singles_order_syms, M, symmetric_pairs=symmetric_pairs)

    # Lambdify RHS; the mapping to z0..zN follows y_syms (thus your singles order)
    rhs_pairs_func = build_rhs_lambdify(rhs_pairs_list, y_syms, extra_params=extra_params)

    return dict(
        M=M,
        Sigma_dt_clean=Sigma_dt_clean,
        state_symbols=y_syms,
        rhs_pairs_func=rhs_pairs_func,
        pairs_order=pairs_order,
        pair_eqs=pair_eqs
    )


# ============================================================
# C) Combined RHS for integration
# ============================================================

def combined_rhs(t, y, params, sys):
    """
    y is concatenation:
      y[:10]  -> singles [Q,P,x1..x8]  (order forced via singles_order_syms)
      y[10:]  -> pairs in the same order as sys['pairs_order'] (upper triangle)
    Returns dy/dt with the same structure.
    """
    # 1) singles
    x = y[:10]
    dx = rhs_gellmann_qp_from_x(t, x, params)

    # 2) pairs
    dy_pairs = np.asarray(sys['rhs_pairs_func'](*y), dtype=complex)

    # concat
    return np.concatenate([dx, dy_pairs])


# ============================================================
# D) Example run
# ============================================================

if __name__ == "__main__":
    # ---- Physical/numerical parameters for singles RHS
    params = dict(
        kappa=1.0, gamma=1.0, Gamma=1.0,
        Omega=8.0, delta1=1.0, delta2=1.0,
        eta=1.0, V=-8.0
    )

    # ---- Build Sigma_dt from your covar module (substitute to numeric if needed)
    # Define symbols exactly as covar uses them:
    g0, Delta1, Vsym, gamma_sym, Omega_sym, kappa_sym, eta_sym, Delta2 = sp.symbols("g0 Delta1 V gamma Omega kappa eta Delta2")
    numeric_params = {
        g0:     1,
        Delta1: 1,
        Delta2: 1,
        Vsym:  -4.0,
        gamma_sym: 1,
        Omega_sym: 8,
        kappa_sym: 1,
        eta_sym:   1
    }

    # Get matrices (as in your example)
    G, sDs, Z, Pm, Qm, Z_prime, W, Sigma_dt_raw, Sigma, K = covar.get_important_matricies(numeric_params)
    # If Sigma_dt_raw still contains symbols, ensure substitution (idempotent if already numeric):
    Sigma_dt_raw = sp.Matrix(Sigma_dt_raw).xreplace(numeric_params)

    # ---- Define singles order to respect your convention: m9 ≡ Q, m10 ≡ P
    m_syms = make_m_symbols(10)  # (m1,..,m10)
    # singles_order maps to [Q,P,x1..x8] ≡ [m9, m10, m1, m2, ..., m8]
    singles_order = [m_syms[8], m_syms[9], *m_syms[0:8]]

    # ---- Build combined (singles + pairs) system
    sys = build_combined_system_with_order(Sigma_dt_raw, singles_order_syms=singles_order,
                                           symmetric_pairs=True, extra_params=())
    n_pairs = len(sys['pairs_order'])
    n_tot   = 10 + n_pairs

    # ---- Sanity check: RHS symbols of pairs ⊆ state symbols
    rhs_free = set().union(*[eq.rhs.free_symbols for eq in sys['pair_eqs']])
    state_set = set(sys['state_symbols'])
    missing = rhs_free - state_set
    if len(missing) > 0:
        print("WARNUNG: RHS enthält Symbole, die nicht im Zustandsvektor sind:", missing)

    # ---- Initial conditions
    # Singles: start in |0> population (ρ00=1)
    y0_full_ket = np.array([0+0j, 0+0j, 1+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j])
    x0 = convert_state(y0_full_ket)   # shape (10,), [Q,P,x1..x8]

    # Pairs (upper triangle): zero init (change if you have analytic covariances)
    pairs0 = np.zeros(n_pairs, dtype=complex)

    # Concatenate to global initial state
    y0 = np.concatenate([x0, pairs0])

    # ---- Integrate
    t_span = (0.0, 500.0)
    print(f"Starte Integration… (Dim={n_tot}, Paare={n_pairs})")
    sol = solve_ivp(lambda t, y: combined_rhs(t, y, params, sys),
                    t_span, y0, method='RK45', atol=1e-8, rtol=1e-8)
    print(f"Fertig. Schritte: {len(sol.t)}")

    # ---- Reconstruct populations from singles
    t_vals = sol.t
    rho00 = np.empty_like(t_vals, dtype=float)
    rho11 = np.empty_like(t_vals, dtype=float)
    rho22 = np.empty_like(t_vals, dtype=float)
    for i, yi in enumerate(sol.y.T):
        Q, P, x1, x2, x3, x4, x5, x6, x7, x8 = yi[:10]
        sum00_11 = (2 + np.sqrt(3)*x8)/3
        rho00[i] = np.real((sum00_11 + x3)/2)
        rho11[i] = np.real((sum00_11 - x3)/2)
        rho22[i] = 1 - rho00[i] - rho11[i]

    # ---- Plot populations
    plt.figure()
    plt.plot(t_vals, rho00, label=r'$\rho_{00}$')
    plt.plot(t_vals, rho11, label=r'$\rho_{11}$')
    plt.plot(t_vals, rho22, label=r'$\rho_{22}$')
    plt.plot(t_vals, rho00+rho11+rho22, '--', label='Spur')
    plt.xlabel('Zeit')
    plt.ylabel('Population')
    plt.legend(loc='best')
    plt.title('Populationsdynamik (Singles)')
    plt.tight_layout()

    # ---- Plot one covariance entry, e.g., m3m7 (i=2, j=6)
    target_ij = (2, 6)   # m3m7
    idx_pairs = None
    for k, (i, j, sym_ij) in enumerate(sys['pairs_order']):
        if (i, j) == target_ij:
            idx_pairs = k
            break
    if idx_pairs is not None:
        y_idx = 10 + idx_pairs
        m3m7_traj = sol.y[y_idx, :]
        plt.figure()
        plt.plot(t_vals, np.real(m3m7_traj), label='Re(m3m7)')
        plt.plot(t_vals, np.imag(m3m7_traj), '--', label='Im(m3m7)')
        plt.xlabel('Zeit')
        plt.ylabel('m3m7')
        plt.legend(loc='best')
        plt.title('Ausgewählte Kovarianz (m3m7)')
        plt.tight_layout()

    plt.show()
