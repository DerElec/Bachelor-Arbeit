# Python code (comments in English; console/output in German)

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import run_script as singlets
from run_script import convert_state,rhs_gellmann_qp_from_ket,rhs_gellmann_qp_from_x
import matplotlib.pyplot as plt
import covar_everything as covar
import symplectic_matrix as symplect
from typing import Sequence, Optional, Tuple, Dict
import re

def _nameval_from_numeric_params(numeric_params: dict, m_vec: np.ndarray | None = None) -> dict[str, float]:
    """
    Build a name->float map from numeric_params (keys are sympy.Symbols)
    and optionally from the current singles vector m = [x1..x8,Q,P].
    """
    nameval: dict[str, float] = {}
    # symbols from numeric_params
    for sym, val in numeric_params.items():
        try:
            nameval[sym.name] = float(val)
        except Exception:
            nameval[sym.name] = float(sp.N(val))
    # current singles m1..m10 (if provided)
    if m_vec is not None:
        m_vec = np.asarray(m_vec).reshape(-1)
        for i in range(min(10, len(m_vec))):
            # we nehmen den Realteil – Kovarianz/Drift ist in deiner Pipeline reell
            nameval[f"m{i+1}"] = float(np.real(m_vec[i]))
    return nameval

def _matrix_to_numeric(M, nameval: dict[str, float]) -> np.ndarray:
    """
    Convert a (possibly symbolic) matrix M to a numeric numpy array by simultaneous name-based substitution.
    - Supports SymPy matrices with free symbols like Delta1, Delta2, m1..m10, etc.
    - Raises a helpful error if some names could not be substituted.
    """
    if isinstance(M, sp.MatrixBase):
        free: set[sp.Symbol] = set(M.free_symbols)
        if free:
            # map every free symbol by name
            subs_pairs = []
            missing = []
            for s in free:
                v = nameval.get(s.name, None)
                if v is None:
                    missing.append(s.name)
                else:
                    subs_pairs.append((s, v))
            if missing:
                raise ValueError(f"Fehlende Substitutionen für Symbole: {sorted(set(missing))}")
            M_num = M.subs(subs_pairs, simultaneous=True)
        else:
            M_num = M
        M_num = sp.N(M_num)
        return np.array(M_num.tolist(), dtype=float)
    else:
        # assume already numeric-like
        return np.array(M, dtype=float)

def sym_min_eig_num(M, nameval: dict[str, float]) -> float:
    """Smallest eigenvalue of the symmetric real part of M after numeric substitution."""
    Mn = _matrix_to_numeric(M, nameval)
    Ms = 0.5 * (Mn + Mn.T)
    return float(np.linalg.eigvalsh(Ms).min())

def print_block_psd_report(G, sDs, Z, sE, W, nameval: dict[str, float]):
    print("\n=== PSD-Report der Bausteine ===")
    print("min λ( sym(sDs) ) =", f"{sym_min_eig_num(sDs, nameval): .3e}")
    print("min λ( sym(Z)   ) =", f"{sym_min_eig_num(Z,   nameval): .3e}")
    print("min λ( sym(sE)  ) =", f"{sym_min_eig_num(sE,  nameval): .3e}")
    print("min λ( sym(W)   ) =", f"{sym_min_eig_num(W,   nameval): .3e}", "  <-- aktuelles 'D'? sollte ≥ 0 sein")


def su3_f_tensor():
    f = {}
    base = {
        (0,1,2): 1.0,
        (0,3,6): 0.5,  (1,3,5): 0.5,  (1,4,6): -0.5,
        (2,3,4): 0.5,  (2,5,6): 0.5,
        (3,4,7): np.sqrt(3)/2.0,
        (5,6,7): -np.sqrt(3)/2.0,
    }
    f_full = np.zeros((8,8,8), dtype=float)
    def set_trip(a,b,c,val):
        f_full[a,b,c] = val
        f_full[b,a,c] = -val
    for (a,b,c), val in base.items():
        set_trip(a,b,c,val)
    return f_full

F_TENSOR = su3_f_tensor()
# Python code (comments in English; console/output in German)
import numpy as np
import matplotlib.pyplot as plt

def plot_sigma_tilde_diagonals(Sigma_tilde, t_eff, names=None, indices=None, ylim=None):
    """
    Plot trajectories of diagonal entries Σ̃_ii(t).
    - Sigma_tilde: array (nt, 10, 10)
    - t_eff: time vector (nt,)
    - names: list of 10 labels; default ['x1',...,'x8','Q','P']
    - indices: list of i to plot (0-based). Default = all 0..9
    - ylim: tuple (ymin, ymax) or None
    """
    nt, n, _ = Sigma_tilde.shape
    if names is None:
        names = [f"x{i+1}" for i in range(8)] + ["Q", "P"]
    if indices is None:
        indices = list(range(n))

    plt.figure(figsize=(7.2, 4.4))
    for i in indices:
        yi = Sigma_tilde[:, i, i]
        # Σ̃ sollte reell-symmetrisch sein; falls doch komplex, Realteil plotten
        plt.plot(t_eff, np.real(yi), label=f"{names[i]}{names[i]}")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Zeit")
    plt.ylabel("Σ̃$_{ii}$")
    plt.title("Trajektorien der Diagonalen von Σ̃(t)")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_sigma_tilde_pairs(Sigma_tilde, t_eff, pairs, names=None):
    """
    Plot trajectories of selected off-diagonals Σ̃_ij(t) (real & imag if present).
    - pairs: list of (i,j) with 0-based indices
    - names: labels for 10 components; default ['x1',...,'x8','Q','P']
    """
    if names is None:
        names = [f"x{i+1}" for i in range(8)] + ["Q", "P"]

    plt.figure(figsize=(7.2, 4.4))
    for (i, j) in pairs:
        yij = Sigma_tilde[:, i, j]
        lab = f"{names[i]}{names[j]}"
        plt.plot(t_eff, np.real(yij), label=f"Re[{lab}]")
        if np.any(np.abs(np.imag(yij)) > 0):
            plt.plot(t_eff, np.imag(yij), "--", label=f"Im[{lab}]")
    plt.xlabel("Zeit")
    plt.ylabel("Σ̃$_{ij}$")
    plt.title("Ausgewählte Off-Diagonalen von Σ̃(t)")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()

def C_from_x_fast(x8):
    xc = np.real(np.asarray(x8, dtype=float))
    C_real = 2.0 * np.einsum('abc,c->ab', F_TENSOR, xc)
    return 1j * C_real  # purely imaginary
def S10_from_C8(C8):
    """Embed 8×8 C8 in 10×10 und füge den kanonischen QP-Block ein."""
    S10 = np.zeros((10, 10), dtype=complex)
    S10[:8, :8] = C8
    S10[8, 9] = 1.0
    S10[9, 8] = -1.0
    return S10
def compute_JRSRJ_fast(sol, step=1):
    t = sol.t
    idxs = np.arange(0, t.size, step, dtype=int)
    nt_eff = idxs.size
    JRSRJ = np.zeros((nt_eff, 10, 10), dtype=complex)

    for k_eff, k in enumerate(idxs):
        x8 = sol.y[0:8, k]
        C8 = C_from_x_fast(x8)
        final_C, _, _ = symplect.transform_complex_S(C8)
        JRSRJ[k_eff] = symplect.expand_to_10x10_sym(1j * final_C)
    return {"t": t[idxs], "JRSRJ": JRSRJ}

def print_JRSRJ_first_last(series):
    t = series["t"]
    JRSRJ = series["JRSRJ"]

    print("\n===== Erste JRSRJ (t = {:.6f}) =====".format(t[0]))
    sp.pprint(sp.Matrix(JRSRJ[0]))

    print("\n===== Letzte JRSRJ (t = {:.6f}) =====".format(t[-1]))
    sp.pprint(sp.Matrix(JRSRJ[-1]))

# 4) Main fast routine: compute S,J,R timeseries (optionally sampled)
def compute_SJR_fast(sol, step=1, compute_JR=True):
    t = sol.t
    idxs = np.arange(0, t.size, step, dtype=int)
    nt_eff = idxs.size

    S10 = np.zeros((nt_eff, 10, 10), dtype=complex)
    J   = np.zeros_like(S10)
    R   = np.zeros_like(S10)
    JRSRJ = np.zeros_like(S10)

    for k_eff, k in enumerate(idxs):
        x8 = sol.y[0:8, k]         # m1..m8
        C8 = C_from_x_fast(x8)     # 8x8 purely imaginary
        S10[k_eff] = S10_from_C8(C8)

        if compute_JR:
            # Your routine expects purely imaginary input of size 8x8
            # -> transform on the 8x8 C8
            final_C, J_small, R_small = symplect.transform_complex_S(C8)
            # expand to 10x10 (pad identity on last two dims)
            J[k_eff]     = symplect.expand_to_10x10(J_small)
            R[k_eff]     = symplect.expand_to_10x10(R_small)
            JRSRJ[k_eff] = symplect.expand_to_10x10_sym(1j * final_C)
        else:
            J[k_eff]     = np.eye(10, dtype=complex)
            R[k_eff]     = np.eye(10, dtype=complex)
            JRSRJ[k_eff] = np.nan

    return {"t": t[idxs], "idxs": idxs, "S10": S10, "J": J, "R": R, "JRSRJ": JRSRJ}

# 5) Print first & last (compact numeric printing, much faster than SymPy pprint)
def print_S_first_last(series):
    np.set_printoptions(precision=4, suppress=True)
    t = series["t"]
    S10 = series["S10"]

    print("\n===== Erste S(10x10) (numerisch)  t = {:.6f} =====".format(t[0]))
    print(S10[0])

    print("\n===== Letzte S(10x10) (numerisch) t = {:.6f} =====".format(t[-1]))
    print(S10[-1])

    # Optional: also J and R
    print("\n-- Erste J --"); print(series["J"][0])
    print("\n-- Erste R --"); print(series["R"][0])

# =====================================================================
# === (A) Your singles dynamics as given (kept verbatim / adapter) ====
# =====================================================================

# -- Your functions must be defined/imported here ----------------------
# rhs_gellmann_qp_from_ket(t, y11, params) -> 11 complex derivs
# convert_state(y) : 11<->10 projection between [a,ad,rho..] and [Q,P,x1..x8]
# rhs_gellmann_qp_from_x(t, x10, params) -> [dQ,dP,dx1..dx8]

# (Paste your definitions here, omitted for brevity in this snippet)
# ---------------------------------------------------------------------
def expand_to_10x10_sympy_with_QP(C8_sym: sp.Matrix) -> sp.Matrix:
    """Embed an 8x8 SymPy matrix into 10x10 and add canonical QP block [[0,1],[-1,0]]."""
    if not isinstance(C8_sym, sp.MatrixBase) or C8_sym.shape != (8, 8):
        raise ValueError("Erwarte eine 8x8 SymPy-Matrix.")
    zero_8x2 = sp.zeros(8, 2)
    zero_2x8 = sp.zeros(2, 8)
    # QP block
    QP = sp.Matrix([[0, 1],
                    [-1, 0]])
    top    = C8_sym.row_join(zero_8x2)
    bottom = zero_2x8.row_join(QP)
    return top.col_join(bottom)  # 10x10 SymPy

def compute_symplectic_series_sympy(sol):
    """
    Aus solve_ivp-Lösung 'sol' (y: [m1..m8, m9(Q), m10(P), …]) berechne:
      - S8_sym_list[k]  = get_S_matrix_gellman(x1..x8)  (8x8, SymPy)
      - S10_sym_list[k] = expand_to_10x10_sympy_with_QP(S8_sym_list[k])  (10x10, SymPy)
    Gibt (S8_sym_list, S10_sym_list) zurück.
    """
    nt = sol.y.shape[1]
    S8_sym_list  = []
    S10_sym_list = []

    for k in range(nt):
        # Singles in m-Ordnung: m1..m8 = x1..x8
        x1_to_x8 = sol.y[0:8, k]
        # Falls komplex: meist sind Re(x) für die SU(3)-Kommutatoren physikalisch gemeint.
        # Du kannst bei Bedarf .real durch die volle Zahl ersetzen.
        x_vals = [sp.N(complex(val)) for val in x1_to_x8]

        # 1) 8x8 SymPy-Matrix C(x) = get_S_matrix_gellman(x)
        C8_sym = symplect.get_S_matrix_gellman(x_vals)  # bereits SymPy-Matrix
        # 2) auf 10x10 mit QP-Block einbetten
        S10_sym = expand_to_10x10_sympy_with_QP(C8_sym)

        S8_sym_list.append(C8_sym)
        S10_sym_list.append(S10_sym)

    return S8_sym_list, S10_sym_list
def _pair_index_map(pairs_order):
    """Build a dict: (i,j) -> local pair index k (0-based within the pairs block)."""
    return { (i, j): k for k, (i, j, _) in enumerate(pairs_order) }

def plot_diagonal_fluctuations(idx, sol, names=None):
    """
    Plot all diagonal pair entries m_i m_i over time (i=1..n).
    - idx: result of prepare_system_for_solve_ivp(...)
    - sol: solve_ivp solution object
    - names: optional list of pretty labels for m_i (length n). If None -> 'm1'..'mN'
    """
    t_vals = sol.t
    pairs_order = idx["pairs_order"]
    pair_start  = idx["pairs_slice"].start
    n_singles   = idx["singles_slice"].stop - idx["singles_slice"].start
    n           = n_singles  # number of m_i

    if names is None:
        names = [f"m{i+1}" for i in range(n)]

    pij2k = _pair_index_map(pairs_order)

    # Gather trajectories for each diagonal (i,i)
    plt.figure(figsize=(8, 5))
    for i in range(n):
        # Upper-triangle contains (i,i) by construction
        k_local = pij2k[(i, i)]                 # local index within pairs block
        row = pair_start + k_local              # global row index in sol.y
        traj = sol.y[row, :]                    # trajectory of m_i m_i

        # Plot real part; if imag exists, also plot it dashed
        label_base = f"{names[i]}{names[i]}"
        plt.plot(t_vals, np.real(traj), label=f"Re[{label_base}]")
        if np.any(np.abs(np.imag(traj)) > 0):
            plt.plot(t_vals, np.imag(traj), "--", label=f"Im[{label_base}]")

    plt.xlabel("Zeit")
    plt.ylabel("Fluktuation (m_i m_i)")
    plt.title("Diagonale Kovarianzen (Varianzen) m_im_i")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()


def singles_rhs_from_m_order(t, m_singles, params):
    """
    Adapter from our m-order [x1..x8, Q, P] to your rhs_gellmann_qp_from_x (which expects [Q,P,x1..x8]).
    Returns derivatives in m-order: [dx1..dx8, dQ, dP].
    """
    # unpack m-singles: m1..m8 = x1..x8, m9=Q, m10=P
    x1_to_x8 = m_singles[0:8]
    Q = m_singles[8]
    P = m_singles[9]

    # Build x-vector in your expected order: [Q, P, x1..x8]
    x_vec = np.array([Q, P, *x1_to_x8], dtype=complex)

    # Compute [dQ, dP, dx1..dx8]
    dQ_dP_dx = rhs_gellmann_qp_from_x(t, x_vec, params)

    # Map back to m-order: [dx1..dx8, dQ, dP]
    dQ = dQ_dP_dx[0]
    dP = dQ_dP_dx[1]
    dx = dQ_dP_dx[2:10]  # dx1..dx8

    return np.array([*dx, dQ, dP], dtype=complex)


# =====================================================================
# === (B) Covariance (pairs) pipeline from Sigma_dt  ===================
# =====================================================================

def build_pair_symbol_matrix(n: int):
    """Create symbol matrix M with entries m{i}m{j}."""
    return sp.Matrix(n, n, lambda i, j: sp.symbols(f"m{i+1}m{j+1}"))

def list_pairs(M, symmetric_pairs: bool = True):
    """Ordered (i,j,sym_ij); upper triangle incl. diag if symmetric_pairs=True."""
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
    """Make ODEs d(m_i m_j)/dt = Sigma_dt[i,j]."""
    assert Sigma_dt.shape == M.shape, "Sigma_dt and M must have same shape"
    eqs, rhs_list = [], []
    pairs = list_pairs(M, symmetric_pairs=symmetric_pairs)
    for (i, j, sym_ij) in pairs:
        d_ij = sp.symbols(f"d{sym_ij.name}_dt")
        rhs = sp.simplify(Sigma_dt[i, j])
        eqs.append(sp.Eq(d_ij, rhs))
        rhs_list.append(rhs)
    return eqs, rhs_list, pairs

def pack_state_symbols(mP_syms, M: sp.Matrix, symmetric_pairs: bool = True):
    """State symbols vectorization: [m singles..., selected pairs...]"""
    y_syms = list(mP_syms)
    pairs = list_pairs(M, symmetric_pairs=symmetric_pairs)
    for (_, _, sym_ij) in pairs:
        y_syms.append(sym_ij)
    return y_syms, pairs

def build_rhs_lambdify(rhs_pairs_list, y_syms, extra_params=()):
    """Lambdify pair-derivatives as NumPy callable."""
    z = sp.symbols(f'z0:{len(y_syms)}')
    subs_map = {sym: z[k] for k, sym in enumerate(y_syms)}
    rhs_z = [sp.simplify(expr.xreplace(subs_map)) for expr in rhs_pairs_list]
    args = list(z) + list(extra_params)
    f = sp.lambdify(args, rhs_z, modules='numpy')
    return f

def enforce_upper_triangle(M):
    """Replace lower m{j}m{i} by upper m{i}m{j} so Sigma_dt uses a unique naming."""
    n = M.shape[0]
    sub_map = {}
    for i in range(n):
        for j in range(i+1, n):
            upper = M[i, j]
            lower = sp.symbols(f"m{j+1}m{i+1}")
            sub_map[lower] = upper
    return sub_map

def prepare_system_for_solve_ivp(mP_syms, Sigma_dt, symmetric_pairs=True, extra_params=()):
    """
    Returns:
      idx: dict with slices and ordering
      rhs_pairs_func(*z, *params) -> d/dt of pairs in our pair-order
    """
    n = len(mP_syms)
    assert Sigma_dt.shape == (n, n), "Sigma_dt must be n x n with n=len(mP_syms)"

    M = build_pair_symbol_matrix(n)
    Sigma_dt = Sigma_dt.xreplace(enforce_upper_triangle(M))

    pair_eqs, rhs_pairs_list, pairs_order = build_pair_equations(Sigma_dt, M, symmetric_pairs=symmetric_pairs)
    state_symbols, pairs_order = pack_state_symbols(mP_syms, M, symmetric_pairs=symmetric_pairs)
    rhs_pairs_func = build_rhs_lambdify_forgiving(
    rhs_pairs_list, state_symbols, pairs_order, extra_params=extra_params
)

    singles_len = len(mP_syms)
    pairs_len = len(state_symbols) - singles_len

    idx = {
        "singles_slice": slice(0, singles_len),
        "pairs_slice": slice(singles_len, singles_len + pairs_len),
        "pairs_order": pairs_order,
        "state_symbols": state_symbols
    }
    return idx, rhs_pairs_func

# =====================================================================
# === (C) Build combined RHS for solve_ivp =============================
# =====================================================================

def normalize_sigma_symbols_in(expr, M):
    """
    Replace ANY symbol of the form m{i}m{j} and m{j}m{i} in 'expr'
    by the UNIQUE symbol M[i-1,j-1] (upper-triangle representative).
    """
    sub = {}
    for i in range(n):
        for j in range(n):
            s_ij = sp.symbols(f"m{i+1}m{j+1}")
            s_ji = sp.symbols(f"m{j+1}m{i+1}")
            rep = M[min(i,j), max(i,j)]  # always use upper-triangle representative
            sub[s_ij] = rep
            sub[s_ji] = rep
    return expr.xreplace(sub)

def make_combined_rhs(idx, rhs_pairs_func, params_dict, extra_param_values_tuple=()):
    """
    Returns fun(t,y) -> concatenated derivatives for singles+pairs.
    - Singles part uses your rhs via 'singles_rhs_from_m_order'.
    - Pairs part uses lambdified Sigma_dt (expects all state entries separately).
    """
    s_slice = idx["singles_slice"]
    p_slice = idx["pairs_slice"]

    def fun(t, y):
        # Ensure complex dtype for singles consistency
        y = y.astype(complex, copy=False)

        # (1) Singles dynamics in m-order
        y_singles = y[s_slice]
        dy_singles = singles_rhs_from_m_order(t, y_singles, params_dict)  # complex (10,)

        # (2) Pair dynamics via lambdify (works with real or complex; cast to complex)
        args = tuple(y.tolist()) + tuple(extra_param_values_tuple)
        dy_pairs = np.asarray(rhs_pairs_func(*args))
        # enforce complex to match solve_ivp dtype for the whole state
        dy_pairs = dy_pairs.astype(complex, copy=False)

        return np.concatenate([dy_singles, dy_pairs])

    return fun


def build_rhs_lambdify_forgiving(rhs_pairs_list, y_syms, pairs_order, extra_params=()):
    """
    Robust lambdify that does *name-based* substitution of ANY symbols appearing in rhs_pairs_list.
    - Singles:  m1..m10     -> z[idx_of_single]
    - Pairs:    m{i}m{j}    -> z[idx_of_pair for (min(i,j),max(i,j)) in pairs_order]
    This ignores identity/assumptions of symbols in Sigma_dt.
    """
    import re
    z = sp.symbols(f"z0:{len(y_syms)}")       # dummy args matching y_syms length

    singles_len = len(y_syms) - len(pairs_order)  # = 10 for our case
    pair_idx = { (i, j): k for k, (i, j, _) in enumerate(pairs_order) }

    re_single = re.compile(r"^m(\d+)$")
    re_pair   = re.compile(r"^m(\d+)m(\d+)$")

    def symbol_to_z(s: sp.Symbol):
        name = s.name
        m = re_single.match(name)
        if m:
            i = int(m.group(1))       # 1..10
            return z[i - 1]           # singles are first (0..9)
        m = re_pair.match(name)
        if m:
            i = int(m.group(1)) - 1   # 0-based
            j = int(m.group(2)) - 1
            ii, jj = (i, j) if i <= j else (j, i)
            k_local = pair_idx[(ii, jj)]      # 0..n_pairs-1
            return z[singles_len + k_local]   # global index in y
        # leave physical parameters (gamma, kappa, ...) untouched
        return s

    # Build substitution map from ALL symbols we see in the expressions
    subs_map = {}
    for expr in rhs_pairs_list:
        for s in expr.atoms(sp.Symbol):
            mapped = symbol_to_z(s)
            if mapped is not s:
                subs_map[s] = mapped

    # Apply simultaneous substitution -> z-forms
    rhs_z = [sp.simplify(expr.subs(subs_map, simultaneous=True)) for expr in rhs_pairs_list]

    # Build callable: (z0.., *extra_params) -> RHS
    args = list(z) + list(extra_params)
    return sp.lambdify(args, rhs_z, modules='numpy')



def initial_covariance_from_state(m0_singles, atom_scale=1.0, boson_scale=1.0):
    """
    Build Σ(0) for m = [x1..x8, Q, P] assuming:
      - Atomic |0> state: Var(x1,x2,x4,x5)=1, Var(x3,x6,x7,x8)=0 (scaled by atom_scale)
      - Bosonic vacuum: Var(Q)=Var(P)=1/2 (scaled by boson_scale), Cov(Q,P)=0
      - No cross-correlations at t=0.
    Returns Σ as a (10x10) np.ndarray (complex dtype).
    """
    Σ = np.zeros((10,10), dtype=complex)

    # Atomic variances (x1..x8)
    var_x = np.zeros(8, dtype=float)
    var_x[[0,1,3,4]] = 1.0  # x1,x2,x4,x5
    var_x *= atom_scale

    for i in range(8):
        Σ[i, i] = var_x[i]

    # Bosonic block (Q=m9 index 8, P=m10 index 9)
    Σ[8, 8] = 0.5 * boson_scale   # Var(Q)
    Σ[9, 9] = 0.5 * boson_scale   # Var(P)
    Σ[8, 9] = 0.0                 # Cov(Q,P) = 0
    Σ[9, 8] = 0.0

    return Σ

def pack_upper_triangle_from_covariance(Σ, pairs_order):
    """
    Given a covariance matrix Σ for m[0..9] in m-order [x1..x8,Q,P],
    produce the vector of upper-triangle entries in the same order as pairs_order.
    """
    m0_pairs = []
    for (i, j, _sym) in pairs_order:
        m0_pairs.append(Σ[i, j])
    return np.asarray(m0_pairs, dtype=complex)
import numpy as np
import sympy as sp
from typing import Dict, Tuple

def su3_structure_constants() -> Dict[Tuple[int,int,int], float]:
    """Return non-zero SU(3) structure constants f_{abc} with a,b,c in {1..8}.
    Convention: [λ_a, λ_b] = 2i ∑_c f_{abc} λ_c  (Gell-Mann basis).
    Only independent positive entries are listed; full antisymmetry is applied in code.
    """
    f = {}
    # Standard SU(3) non-zero f_abc > 0:
    base = {
        (1,2,3): 1.0,
        (1,4,7): 0.5, (1,5,6): 0.5,
        (2,4,6): 0.5, (2,5,7): 0.5,
        (3,4,5): 0.5, (3,6,7): 0.5,
        (4,5,8): np.sqrt(3)/2.0,
        (6,7,8): np.sqrt(3)/2.0,
    }
    # Fill with antisymmetry f_{abc} = +val and antisymmetric in a<->b, cyclic with sign:
    def set_f(a,b,c,val):
        f[(a,b,c)] = val
        f[(b,a,c)] = -val
    for (a,b,c), val in base.items():
        set_f(a,b,c,val)
        # also fill permutations that follow from Jacobi/antisym? Not needed:
        # we only ever use f_{a b c} with (a,b) explicit and sum over c.
    return f

def omega_from_state(x1_to_x8: np.ndarray) -> np.ndarray:
    """Build 10x10 Omega for a single time point given x1..x8 (expectation values).
       Ordering: [x1..x8, Q, P].
    """
    f = su3_structure_constants()
    Omega = np.zeros((10,10), dtype=float)

    # SU(3) block 0..7
    for a in range(1,9):          # 1..8
        for b in range(1,9):
            s = 0.0
            for c in range(1,9):
                s += f.get((a,b,c), 0.0) * x1_to_x8[c-1].real  # <x_c> assumed real; adjust if needed
            Omega[a-1, b-1] = 2.0 * s

    # Q,P block
    Omega[8,9] = 1.0
    Omega[9,8] = -1.0

    # Cross blocks are zero
    return Omega

def omega_timeseries_from_solution(sol) -> np.ndarray:
    """Compute Omega(t_k) for all time points in solve_ivp solution 'sol'.
       'sol.y' assumed packed as [m1..m8, m9(Q), m10(P), pairs...].
       Returns array of shape (nt, 10, 10).
    """
    nt = sol.y.shape[1]
    Omegas = np.zeros((nt, 10, 10), dtype=float)
    for k in range(nt):
        x1_to_x8 = sol.y[0:8, k]  # m1..m8 = x1..x8
        Omegas[k] = omega_from_state(x1_to_x8)
    return Omegas

def check_omega_properties(Omega: np.ndarray) -> Dict[str, float]:
    """Simple diagnostics for a single Omega: antisymmetry error, rank."""
    antisymm_err = np.linalg.norm(Omega + Omega.T)
    rank = np.linalg.matrix_rank(Omega)
    return {"antisymmetry_norm": float(antisymm_err), "rank": float(rank)}
# ============================================================
# Transform Σ(t) -> Σ̃(t) = J^T R^T Σ R J  and plot heatmaps
# Requires:
#   - 'sol' from solve_ivp with state packing [m1..m8, m9(Q), m10(P), pairs...]
#   - 'idx' from prepare_system_for_solve_ivp (has pairs_order, slices)
#   - 'SJR' from compute_SJR_fast(sol, step=1, compute_JR=True)
#       with keys: "t", "idxs", "J", "R"
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

def reconstruct_sigma_series(sol, idx, take_real=True):
    """Reconstruct Σ(t_k) (10x10) from the packed pair entries in sol.y using idx['pairs_order']."""
    n = idx["singles_slice"].stop - idx["singles_slice"].start  # should be 10
    pair_start = idx["pairs_slice"].start
    pairs_order = idx["pairs_order"]
    nt = sol.y.shape[1]

    Sigmas = np.zeros((nt, n, n), dtype=complex)
    # fill upper triangle from stored pair order
    for k_loc, (i, j, _sym) in enumerate(pairs_order):
        row = pair_start + k_loc
        traj = sol.y[row, :]                       # length nt
        Sigmas[:, i, j] = traj
        Sigmas[:, j, i] = np.conjugate(traj)       # ensure Hermitian symmetry
    # Optionally enforce real symmetric covariance
    if take_real:
        Sigmas = np.real(Sigmas)
    return Sigmas

def transform_sigma_series(Sigmas, SJR):
    """Apply Σ̃ = J^T R^T Σ R J for all sampled times in SJR (uses SJR['idxs'] mapping)."""
    idxs = SJR["idxs"]                # indices into full time grid used for JR computation
    J = SJR["J"]                      # shape (nt_eff, 10, 10)
    R = SJR["R"]
    nt_eff = idxs.size
    n = Sigmas.shape[1]
    Sigma_tilde = np.zeros((nt_eff, n, n), dtype=Sigmas.dtype)
    for k_eff, k in enumerate(idxs):
        Σ = Sigmas[k]                 # (10,10)
        Jk = J[k_eff]
        Rk = R[k_eff]
        # Σ̃ = J^T R^T Σ R J
        Sigma_tilde[k_eff] = Jk.T @ Rk.T @ Σ @ Rk @ Jk
        # enforce symmetry numerically
        Sigma_tilde[k_eff] = 0.5 * (Sigma_tilde[k_eff] + Sigma_tilde[k_eff].T)
    return Sigma_tilde

def plot_sigma_tilde_heatmaps(Sigma_tilde, t_eff, which=("first","last")):
    """Plot heatmaps of Σ̃ at selected times (first/last)."""
    choices = []
    if "first" in which: choices.append((0, "Erste"))
    if "last"  in which: choices.append((-1, "Letzte"))

    for idx, label in choices:
        idx_eff = idx if idx >= 0 else Sigma_tilde.shape[0] + idx
        Σt = Sigma_tilde[idx_eff]
        plt.figure(figsize=(5.0, 4.5))
        plt.imshow(Σt, origin="lower", interpolation="nearest")
        plt.colorbar(label="Σ̃-Eintragswert")
        plt.title(f"{label} Σ̃ (t = {t_eff[idx_eff]:.6f})")
        plt.xlabel("Index j")
        plt.ylabel("Index i")
        plt.tight_layout()
    plt.show()
def check_psd_series(Sigmas: np.ndarray, t: np.ndarray, name: str = "Σ", tol: float = 1e-12, do_plot: bool = True):
    """
    Check positive semidefiniteness for a time series of covariance matrices.
    Uses eigvalsh of the symmetrized real part: S = (Σ + Σ^T)/2, then eigs = eigvalsh(Re(S)).
    Prints summary and (optionally) plots min eigenvalue vs time.
    """
    nt = Sigmas.shape[0]
    min_eigs = np.empty(nt, dtype=float)
    bad = []
    for k in range(nt):
        S = 0.5 * (Sigmas[k] + Sigmas[k].T)
        S = np.real(S)  # covariance should be real-symmetric in our pipeline
        vals = np.linalg.eigvalsh(S)
        m = float(vals.min())
        min_eigs[k] = m
        if m < -tol:
            bad.append(k)

    print(f"[PSD-Check] {name}: min(λ_min)={min_eigs.min():.3e}, max(λ_min)={min_eigs.max():.3e}, "
          f"Anzahl Verletzungen (λ_min < -{tol:g}): {len(bad)} / {nt}")
    if len(bad) > 0:
        first_bad = bad[0]
        print(f"  ↳ Erste Verletzung bei t={t[first_bad]:.6g} (λ_min={min_eigs[first_bad]:.3e})")

    # if do_plot:
    #     plt.figure(figsize=(6.4, 3.6))
    #     plt.plot(t, min_eigs, label="λ_min(Σ)")
    #     plt.axhline(0.0, linestyle="--", color="gray", linewidth=1)
    #     plt.xlabel("Zeit"); plt.ylabel("kleinster Eigenwert")
    #     plt.title(f"PSD-Check: kleinster Eigenwert von {name}(t)")
    #     plt.legend(); plt.tight_layout(); plt.show()

    return min_eigs, bad


# ---------- BACK-TRANSFORM (Gell-Mann -> 'rho'-nahe Ebene) ----------
def A_m_to_rho() -> np.ndarray:
    """
    Linear map A (10x10) that converts m = [x1..x8,Q,P] to
    y = [Re ρ01, Im ρ01, Re ρ02, Im ρ02, Re ρ12, Im ρ12, ρ00, ρ11, Q, P].
    Note: ρ22 = 1 - ρ00 - ρ11 (affin), which does not enter the covariance (constants drop out).
    Covariance transforms as Σ_y = A Σ_m A^T.
    """
    A = np.zeros((10, 10), dtype=float)
    # x1..x8 at cols 0..7; Q,P at cols 8,9
    # coherences
    A[0, 0] = 0.5  # Re ρ01 = x1/2
    A[1, 1] = 0.5  # Im ρ01 = x2/2
    A[2, 3] = 0.5  # Re ρ02 = x4/2
    A[3, 4] = 0.5  # Im ρ02 = x5/2
    A[4, 5] = 0.5  # Re ρ12 = x6/2
    A[5, 6] = 0.5  # Im ρ12 = x7/2
    # populations (linear part only; constant offsets vanish in covariances)
    # ρ00 = 1/3 + (√3/6) x8 + (1/2) x3
    # ρ11 = 1/3 + (√3/6) x8 - (1/2) x3
    A[6, 2] = 0.5           # x3
    A[6, 7] = np.sqrt(3)/6  # x8
    A[7, 2] = -0.5          # x3
    A[7, 7] = np.sqrt(3)/6  # x8
    # field quadratures
    A[8, 8] = 1.0           # Q
    A[9, 9] = 1.0           # P
    return A
# ---------- Forensic diagnostics on A/D assembly ----------
def sym_min_eig(M):
    """Smallest eigenvalue of the symmetric real part of M."""
    M = np.array(sp.Matrix(M), dtype=float) if isinstance(M, sp.MatrixBase) else np.array(M, dtype=float)
    Ms = 0.5*(M+M.T)
    return float(np.linalg.eigvalsh(Ms).min())

def print_block_psd_report(G, sDs, Z, sE, W):
    print("\n=== PSD-Report der Bausteine ===")
    print("min λ( sym(sDs) ) =", f"{sym_min_eig(sDs): .3e}")
    print("min λ( sym(Z)   ) =", f"{sym_min_eig(Z): .3e}")
    print("min λ( sym(sE)  ) =", f"{sym_min_eig(sE): .3e}")
    print("min λ( sym(W)   ) =", f"{sym_min_eig(W): .3e}", "  <-- aktuelles D, sollte ≥ 0 sein")
    # Spot: sE wirkt bei dir wie Dämpfung auf Q,P → gehört in A, nicht in D
    if sym_min_eig(W) < 0:
        print("Hinweis: W ist nicht PSD. Prüfe, ob sE in D gelandet ist (falsch).")

def split_D_correctly(sDs, Z):
    """Construct a 'physically sane' diffusion: D = sym(sDs + Z)."""
    Zs = 0.5*(Z + Z.T)
    return sDs + Zs

def reassemble_dynamics(G, sDs, Z, sE):
    """
    Recommended assembly:
      A_eff = G + A_extra    (antisymmetric bits, damping etc.)
      D_eff = sym(sDs + Z)   (must be PSD)
    Here we move sE into A (friction) and use only sym(Z) in D.
    """
    A_eff = G + sE          # sE = -κ/2 on Q,P → drift (Friction)
    D_eff = split_D_correctly(sDs, Z)  # PSD candidate
    return A_eff, D_eff

def sigma_rho_series_from_sigma(Sigmas_m: np.ndarray) -> np.ndarray:
    """
    Build Σ_rho(t) = A Σ_m(t) A^T for all times, where m = [x1..x8,Q,P] and
    rho basis y = [Re ρ01, Im ρ01, Re ρ02, Im ρ02, Re ρ12, Im ρ12, ρ00, ρ11, Q, P].
    """
    A = A_m_to_rho()
    # vectorized: Σ_rho(t) = A Σ_m(t) A^T
    return np.einsum("ij,tjk,lk->til", A, Sigmas_m, A)
def psd_culprit_probe(Sigmas, A_eff, D_eff, t, k_idx):
    """Project dΣ/dt contributions onto the eigenvector of λ_min(Σ)."""
    Σ = Sigmas[k_idx]
    # eigenvector of smallest eigenvalue (symmetric real part)
    evals, evecs = np.linalg.eigh(0.5*(Σ+Σ.T))
    v = evecs[:, np.argmin(evals)]
    # contributions (approx; Σ assumed symmetric here)
    drift_contrib = float(v.T @ (A_eff @ Σ + Σ @ A_eff.T) @ v)
    diff_contrib  = float(v.T @ (0.5*(D_eff+D_eff.T)) @ v)
    print(f"[t={t[k_idx]:.6g}] v^T (AΣ+ΣA^T) v = {drift_contrib: .3e} ;  v^T D v = {diff_contrib: .3e}")
    return drift_contrib, diff_contrib
def is_psd_numeric(M, tol=1e-10):
    """Quick PSD test (numeric): all eigenvalues >= -tol."""
    # Use symmetric part to be safe against rounding
    w = np.linalg.eigvalsh(0.5*(M+M.T))
    return w.min() >= -tol, w.min()

def project_to_psd(M, tol=1e-12):
    """Project symmetric matrix to nearest PSD (eigvalue clipping)."""
    S = 0.5*(M+M.T)
    w, V = np.linalg.eigh(S)
    w_clipped = np.maximum(w, 0.0)
    return (V * w_clipped) @ V.T

def symmetrize(M):
    """Force exact symmetry."""
    return 0.5*(M+M.T)

def check_hurwitz(A):
    """Return True if all eigenvalues of A have negative real part."""
    lam = np.linalg.eigvals(A)
    return np.all(np.real(lam) < 0.0), lam

# Example diagnostic after each integration step:
def check_symmetry(M):
    if M.equals(M.T):
        return "symmetric"
    elif M.equals(-M.T):
        return "antisymmetric"
    else:
        return "neither"
def plot_sigma_rho_pairs(Sigma_rho: np.ndarray, t: np.ndarray, pairs: Sequence[Tuple[int,int]], names: Optional[Sequence[str]] = None):
    """
    Plot trajectories of selected off-diagonals in the rho-near basis.
    Default names: ['Reρ01','Imρ01','Reρ02','Imρ02','Reρ12','Imρ12','ρ00','ρ11','Q','P']
    """
    if names is None:
        names = ["Reρ01","Imρ01","Reρ02","Imρ02","Reρ12","Imρ12","ρ00","ρ11","Q","P"]
    plt.figure(figsize=(7.2, 4.4))
    for (i, j) in pairs:
        yij = Sigma_rho[:, i, j]
        lab = f"{names[i]},{names[j]}"
        plt.plot(t, np.real(yij), label=f"Re[{lab}]")
        if np.any(np.abs(np.imag(yij)) > 0):
            plt.plot(t, np.imag(yij), "--", label=f"Im[{lab}]")
    plt.xlabel("Zeit"); plt.ylabel("Σ$_{ij}$ (rho-Basis)")
    plt.title("Kovarianzen in der rho-nahen Ebene")
    plt.legend(ncol=2, fontsize=8); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    print("Baue kombiniertes Singles+Kovarianz-System …")
    n = 10
    mP_syms = sp.symbols('m1:11')  # m1..m8 = x1..x8, m9=Q, m10=P
    g0, Delta1, V, gamma, Omega, kappa, eta, Delta2 = sp.symbols("g0 Delta1 V gamma Omega kappa eta Delta2")
    V_val=-1/2*((8/(4))**2+1)
    numeric_params = { g0:1, Delta1:1, Delta2:1, V:V_val, gamma:2, Omega:8, kappa:1, eta:1 }

    G,sDs,Z,P,Q,Z_prime,W,Sigma_dt,Sigma,K=covar.get_important_matricies(numeric_params)
    
    G_sym,sDs_sym,Z_sym,P_sym,Q_sym,Z_prime_sym,W_sym,Sigma_dt_sym,Sigma_sym,K_sym=covar.get_important_matricies_symbol()


    # (A) We assume Sigma_dt already has the correct symbolic form and ordering
    Sigma_dt_clean = Sigma_dt
    # --- Build initial singles state (uncomment your block or use this minimal one) ---
    # Your original:
    y0_ket = np.array([0+0j, 0+0j, 1+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j])
    x0 = convert_state(y0_ket)                   # [Q, P, x1..x8]
    m0_singles = np.array([*x0[2:10], x0[0], x0[1]], dtype=complex)  # [x1..x8, Q, P]

    # --- Build a name->value map for substituting symbols in W ---
    nameval = _nameval_from_numeric_params(numeric_params, m_vec=m0_singles)

    # --- Evaluate W at the initial state (use the symbolic W if available) ---
    # Prefer W_sym here (pure symbolic form). If you only have numeric W, use that.
    W_num = _matrix_to_numeric(W_sym, nameval)   # -> numpy.float64 array

    # --- Symmetrize (just in case) and PSD test ---
    W_sympart = 0.5 * (W_num + W_num.T)
    w = np.linalg.eigvalsh(W_sympart)
    ok = (w.min() >= -1e-12)

    print("=== PSD-Test von W am Initialzustand ===")
    print("min λ( sym(W) ) =", f"{w.min(): .3e}")
    print("PSD ?", "JA" if ok else "NEIN")
