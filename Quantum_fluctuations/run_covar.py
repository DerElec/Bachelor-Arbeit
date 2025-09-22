# Python code (comments in English; console/output in German)

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import run_script as singlets
from run_script import convert_state, rhs_gellmann_qp_from_ket, rhs_gellmann_qp_from_x
import matplotlib.pyplot as plt
from typing import Sequence, Optional, Tuple, Dict

import covar_everything as covar
import symplectic_matrix as symplect


# ============================================================
# === Symplectic form s from singles in m-order = [x1..x8,Q,P]
# ============================================================


def su3_f_tensor() -> np.ndarray:
    """
    Return f[a,b,c] for SU(3) (0-based for λ1..λ8).
    Only ascending base triples are set; total antisymmetry fills the rest.
    """
    f = np.zeros((8, 8, 8), dtype=float)
    pos = [
        (0,1,2, 1.0),                        # f_{123} = 1
        (0,3,6, 0.5), (1,3,5, 0.5), (1,4,6, 0.5),
        (2,3,4, 0.5), (2,5,6, 0.5),
        (3,4,7, np.sqrt(3)/2.0), (5,6,7, np.sqrt(3)/2.0),
        # (1,5,6) i.e. 0-based (0,4,5) is already implied by antisymmetry via the above;
        # add explicitly if you want to be maximally explicit:
        (0,4,5, 0.5),
    ]

    for (i,j,k,val) in pos:
        f[i,j,k] = val

    # enforce total antisymmetry over permutations of (i,j,k)
    from itertools import permutations
    def parity(base, perm):
        # parity by bubble-sort count
        a,b,c = base
        x,y,z = perm
        arr = [a,b,c]
        swaps = 0
        if arr[0] != x:
            j = arr.index(x); arr[0],arr[j] = arr[j],arr[0]; swaps += 1
        if arr[1] != y:
            j = arr.index(y); arr[1],arr[j] = arr[j],arr[1]; swaps += 1
        return -1 if swaps % 2 else 1

    for (i,j,k,val) in pos:
        base = (i,j,k)
        for p in set(permutations(base, 3)):
            f[p] = parity(base, p) * val
    return f


# Python code (comments in English; console/output in German)

import numpy as np

# ---------- SU(3) helpers: standard Gell-Mann set and d_{abc} ----------
def _gellmann_matrices():
    """Return the 8 standard Gell-Mann matrices λ_a with Tr(λ_a λ_b) = 2 δ_ab."""
    λ1 = np.array([[0,1,0],[1,0,0],[0,0,0]], dtype=float)
    λ2 = np.array([[0,-1j,0],[1j,0,0],[0,0,0]], dtype=complex)
    λ3 = np.array([[1,0,0],[0,-1,0],[0,0,0]], dtype=float)
    λ4 = np.array([[0,0,1],[0,0,0],[1,0,0]], dtype=float)
    λ5 = np.array([[0,0,-1j],[0,0,0],[1j,0,0]], dtype=complex)
    λ6 = np.array([[0,0,0],[0,0,1],[0,1,0]], dtype=float)
    λ7 = np.array([[0,0,0],[0,0,-1j],[0,1j,0]], dtype=complex)
    λ8 = (1/np.sqrt(3)) * np.array([[1,0,0],[0,1,0],[0,0,-2]], dtype=float)
    mats = [λ1, λ2, λ3.astype(complex), λ4.astype(complex),
            λ5, λ6.astype(complex), λ7, λ8.astype(complex)]
    return mats

def _su3_d_tensor():
    """
    Compute d_{abc} from definition:
        d_{abc} = (1/4) Tr({λ_a, λ_b} λ_c), with Tr(λ_a λ_b)=2 δ_ab.
    Returns an (8,8,8) real array.
    """
    λ = _gellmann_matrices()
    d = np.zeros((8,8,8), dtype=float)
    for a in range(8):
        for b in range(8):
            ab = λ[a] @ λ[b] + λ[b] @ λ[a]  # anticommutator
            for c in range(8):
                val = np.trace(ab @ λ[c]) / 4.0
                d[a,b,c] = float(np.real_if_close(val))
    return d

# ---------- Field covariance builders on (Q,P) ----------
def _field_covariance_qp(spec: dict | None = None, *, qp_comm: float = 1.0) -> np.ndarray:
    """
    Build 2x2 covariance for (Q,P).
    Conventions: [Q,P] = i * qp_comm  -> vacuum variances = qp_comm/2.
    Supported spec["type"]: 'vacuum'/'coherent', 'thermal', 'squeezed', 'sigma'.
    """
    if spec is None:
        spec = {"type": "vacuum"}
    t = str(spec.get("type", "vacuum")).lower()

    if t in ("vacuum", "coherent"):
        s = 0.5 * qp_comm
        return np.array([[s, 0.0],[0.0, s]], dtype=float)

    if t == "thermal":
        n_th = float(spec.get("n_th", 0.0))
        s = (n_th + 0.5) * qp_comm
        return np.array([[s, 0.0],[0.0, s]], dtype=float)

    if t == "squeezed":
        r = float(spec.get("r", 0.0))     # squeeze parameter
        phi = float(spec.get("phi", 0.0)) # angle
        dq = 0.5 * qp_comm * np.exp(-2.0 * r)
        dp = 0.5 * qp_comm * np.exp(+2.0 * r)
        R = np.array([[np.cos(phi), -np.sin(phi)],
                      [np.sin(phi),  np.cos(phi)]], dtype=float)
        D = np.diag([dq, dp])
        return R @ D @ R.T

    if t == "sigma":
        Σqp = np.array(spec["Sigma"], dtype=float)
        return 0.5 * (Σqp + Σqp.T)

    raise ValueError(f"Unknown field spec type: {t}")

# ---------- Main builder: Σ(0) from singles in m-order ----------
def build_initial_sigma_from_state(
    m0_singles: np.ndarray,
    *,
    generator: str = "lambda",    # 'lambda' for λ_a, or 't' for t_a = λ_a/2
    qp_comm: float = 1.0,         # [Q,P] = i * qp_comm
    field_spec: dict | None = None,
    cross_atom_field: np.ndarray | None = None
) -> np.ndarray:
    """
    Build Σ(0) for m-order = [x1..x8, Q, P] using exact SU(3) identities.

    Parameters
    ----------
    m0_singles : array_like, shape (10,)
        [<λ1>,...,<λ8>, <Q>, <P>] if generator='lambda';
        If generator='t', pass [<t1>,...,<t8>, <Q>, <P>] with t_a=λ_a/2.
    generator : {'lambda','t'}
        Normalization of your atomic expectation values.
    qp_comm : float
        Canonical commutator factor: [Q,P] = i * qp_comm.
    field_spec : dict or None
        Covariance on (Q,P). See _field_covariance_qp().
    cross_atom_field : array_like or None
        Optional 8x2 cross-covariances between atomic (1..8) and (Q,P). Default None->zeros.

    Returns
    -------
    Σ0 : (10,10) ndarray (float, symmetric)
    """
    m0 = np.asarray(m0_singles, dtype=float)
    if m0.shape[0] != 10:
        raise ValueError("m0_singles must have length 10: [x1..x8, Q, P].")

    x = m0[:8]  # atomic means
    Σ_qp = _field_covariance_qp(field_spec, qp_comm=qp_comm)  # 2x2

    d = _su3_d_tensor()  # 8x8x8
    if generator.lower() == "lambda":
        # Identity: λ_a λ_b = (2/3) δ_ab I + (d_{abc} + i f_{abc}) λ_c
        # -> 1/2 <{λ_a,λ_b}> = (2/3) δ_ab + sum_c d_{abc} <λ_c>
        base = (2.0/3.0) * np.eye(8)
        add  = np.tensordot(d, x, axes=([2],[0]))   # (8,8)
        Σ_atom = base + add - np.outer(x, x)
    elif generator.lower() == "t":
        # For t_a = λ_a/2:
        # -> 1/2 <{t_a,t_b}> = (1/6) δ_ab + (1/2) sum_c d_{abc} <t_c>
        base = (1.0/6.0) * np.eye(8)
        add  = 0.5 * np.tensordot(d, x, axes=([2],[0]))
        Σ_atom = base + add - np.outer(x, x)
    else:
        raise ValueError("generator must be 'lambda' or 't'.")

    Σ0 = np.zeros((10,10), dtype=float)
    Σ0[:8,:8] = 0.5 * (Σ_atom + Σ_atom.T)

    # Cross atom-field (8x2)
    if cross_atom_field is not None:
        CAF = np.asarray(cross_atom_field, dtype=float)
        if CAF.shape != (8,2):
            raise ValueError("cross_atom_field must have shape (8,2).")
        Σ0[:8, 8:10] = CAF
        Σ0[8:10, :8] = CAF.T

    # Field block
    Σ0[8:10, 8:10] = Σ_qp

    # Final symmetry enforcement
    Σ0 = 0.5 * (Σ0 + Σ0.T)
    return Σ0

# ---------- Matrix check: Hermitian & PSD ----------
def check_hermitian_psd(M, *, herm_tol=1e-10, psd_tol=1e-12, return_eigs=False, verbose=True):
    """
    Check if matrix M is Hermitian and Positive Semi-Definite (PSD).

    - Hermiticity via max|M - M^†|.
    - PSD via min eigenvalue of the Hermitian part H=(M+M^†)/2.

    Returns dict with flags and diagnostics; optionally eigenvalues.
    """
    A = np.array(M, dtype=complex, copy=False)

    antiherm = A - A.conjugate().T
    max_anti = float(np.max(np.abs(antiherm)))
    is_herm = (max_anti <= herm_tol)

    H = 0.5 * (A + A.conjugate().T)
    H = 0.5 * (H + H.conjugate().T)  # enforce exact Hermitian symmetry
    evals = np.linalg.eigvalsh(H)
    min_eig = float(np.min(evals.real))
    is_psd = (min_eig >= -psd_tol)

    if verbose:
        print("=== Matrix-Check: Hermitesch & PSD ===")
        print(f"max|M - M^†| = {max_anti:.3e}  ⇒ Hermitesch: {is_herm} (Toleranz {herm_tol:g})")
        print(f"min λ(H)     = {min_eig:.3e}  ⇒ PSD:        {is_psd} (Toleranz {psd_tol:g})")

    out = {
        "is_hermitian": is_herm,
        "is_psd": is_psd,
        "max_antiherm_norm": max_anti,
        "min_eig": min_eig,
    }
    if return_eigs:
        out["eigvals"] = evals
    return out

def symplectic_from_singles(singles: np.ndarray, *, scale_su3: float = 2.0) -> np.ndarray:
    """
    Build the real 10x10 symplectic form s = -i<[F_a,F_b]> for F=[λ1..λ8, Q, P].
      - singles[0..7] are <λ1>.. <λ8> (Gell-Mann expectations),
      - singles[8], singles[9] are <Q>, <P> (not used in the SU(3) block).
    Conventions:
      [λ_i, λ_j] = 2i f_{ijk} λ_k  -> s_ij = 2 f_{ijk} <λ_k> = scale_su3 * sum_k f_{ijk} * x_k
      [Q,P] = i                    -> s_QP = +1, s_PQ = -1
      Cross (λ_i with Q,P) = 0
    Returns: s (10x10 real, skew-symmetric).
    """
    singles = np.asarray(singles, dtype=float)
    if singles.shape[0] < 10:
        raise ValueError("Expected singles of length >= 10: [x1..x8, Q, P].")

    x = singles[:8]  # <λ_k>
    f = su3_f_tensor()

    # SU(3) 8x8 block: s_ij = scale_su3 * sum_k f_{ijk} x_k
    C = np.zeros((8, 8), dtype=float)
    for i in range(8):
        C[i, :] = (scale_su3 * (f[i, :, :] @ x)).astype(float)
    # ensure skew-symmetry numerically
    C = 0.5 * (C - C.T)

    # QP block (2x2)
    B = np.array([[0.0, 1.0],
                  [-1.0, 0.0]], dtype=float)

    # Assemble 10x10
    S = np.zeros((10, 10), dtype=float)
    S[:8, :8] = C
    S[8:, 8:] = B
    return S


def symplectic_series_from_solution(sol, idx, *, scale_su3: float = 2.0) -> np.ndarray:
    """
    Build s(t_k) for all time points of 'sol' using singles(t_k) and the commutator definition.
    - idx['singles_slice'] must select the first 10 entries [x1..x8,Q,P] from state y.
    Returns: array with shape (nt, 10, 10), real and skew-symmetric.
    """
    sl = idx["singles_slice"]
    Y = sol.y.T  # shape (nt, nstate)
    singles_series = Y[:, sl]  # (nt, 10)

    nt = singles_series.shape[0]
    out = np.zeros((nt, 10, 10), dtype=float)
    for k in range(nt):
        out[k] = symplectic_from_singles(singles_series[k], scale_su3=scale_su3)
    # guard against roundoff
    out = 0.5 * (out - np.transpose(out, (0, 2, 1)))
    return out


# ============================================================
# === Pairs system: symbolic assembly and lambdify (forgiving)
# ============================================================

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

def enforce_upper_triangle(M):
    """Replace lower m{j}m{i} by upper m{i}m{j} so Sigma_dt uses a unique naming."""
    n = M.shape[0]
    sub_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            upper = M[i, j]
            lower = sp.symbols(f"m{j+1}m{i+1}")
            sub_map[lower] = upper
    return sub_map

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
    return sp.lambdify(args, rhs_z, modules="numpy")

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


# ============================================================
# === Singles RHS adapter (m-order <-> your expected order)
# ============================================================

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


# ============================================================
# === Covariance reconstruction & transforms
# ============================================================

def initial_covariance_from_state(m0_singles, atom_scale=1.0, boson_scale=1.0):
    """
    Build Σ(0) for m = [x1..x8, Q, P] assuming:
      - Atomic |0> state: Var(x1,x2,x4,x5)=1, Var(x3,x6,x7,x8)=0 (scaled by atom_scale)
      - Bosonic vacuum: Var(Q)=Var(P)=1/2 (scaled by boson_scale), Cov(Q,P)=0
      - No cross-correlations at t=0.
    Returns Σ as a (10x10) np.ndarray (complex dtype).
    """
    Σ = np.zeros((10, 10), dtype=complex)

    # Atomic variances (x1..x8)
    var_x = np.zeros(8, dtype=float)
    var_x[[0, 1, 3, 4]] = 1.0  # x1,x2,x4,x5
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


# ============================================================
# === R, J from real s; transform Σ with them
# ============================================================

def compute_RJ_series(sol, idx, *, step: int = 1):
    """
    Compute (R_k, J_k) for sampled time indices using the 8x8 SU(3) block of the real symplectic s(t).
    Pipeline:
        s(t)    : real, skew-symmetric 10x10 from symplectic_series_from_solution
        C8(t)   : = s(t)[:8,:8]
        (C8_can, J8, R8) = transform_complex_S(1j*C8)   # canonicalization in complex form
        J10, R10: expand to 10x10 by padding identity on the last 2x2 (Q,P)
        s_tilde : J10^T R10^T s R10 J10    (real)
    Returns:
        dict with keys: "t", "idxs", "s", "J", "R", "s_tilde"
    """
    # 1) real s(t) series, shape (nt,10,10)
    S_all = symplectic_series_from_solution(sol, idx)  # real & skew
    t = sol.t
    nt = len(t)
    idxs = np.arange(0, nt, step, dtype=int)

    # preallocate
    J_ser  = np.zeros((len(idxs), 10, 10), dtype=float)
    R_ser  = np.zeros((len(idxs), 10, 10), dtype=float)
    s_tser = np.zeros((len(idxs), 10, 10), dtype=float)

    for kk, k in enumerate(idxs):
        s_k = S_all[k]
        C8  = s_k[:8, :8]                     # real skew (SU(3) block)
        # Canonicalize on 1j*C8 (routine expects purely imaginary input):
        C8_can, J8, R8 = symplect.transform_complex_S(1j * C8)
        # Expand to 10x10 by padding identity on Q,P
        J10 = symplect.expand_to_10x10(J8)
        R10 = symplect.expand_to_10x10(R8)
        # Store real versions
        J_ser[kk] = np.real_if_close(J10).astype(float)
        R_ser[kk] = np.real_if_close(R10).astype(float)
        # Build s_tilde = J^T R^T s R J  (remains real)
        s_t = J_ser[kk].T @ R_ser[kk].T @ s_k @ R_ser[kk] @ J_ser[kk]
        # enforce skew-symmetry numerically
        s_t = 0.5 * (s_t - s_t.T)
        s_tser[kk] = s_t

    return {"t": t[idxs], "idxs": idxs, "s": S_all, "J": J_ser, "R": R_ser, "s_tilde": s_tser}


def transform_sigma_with_RJ(Sigmas: np.ndarray, RJ_series: dict) -> np.ndarray:
    """
    Transform the covariance series with R,J from compute_RJ_series:
        Σ̃_k = J_k^T R_k^T Σ(t_k) R_k J_k
    Sigmas shape: (nt, 10, 10)   (full time grid)
    RJ_series["idxs"] selects the sampled times where R,J exist.
    Returns:
        Sigma_tilde (nsamp, 10, 10) aligned with RJ_series["t"].
    """
    idxs = RJ_series["idxs"]
    Jser = RJ_series["J"]
    Rser = RJ_series["R"]

    nsamp = len(idxs)
    out = np.zeros((nsamp, 10, 10), dtype=float)
    for kk, k in enumerate(idxs):
        Sg = 0.5 * (Sigmas[k] + Sigmas[k].T)          # guard symmetry
        T  = Rser[kk] @ Jser[kk]
        out[kk] = (T.T @ Sg @ T)
        out[kk] = 0.5 * (out[kk] + out[kk].T)         # enforce symmetry
    return out


def populations_from_x(x_series: np.ndarray):
    """
    Compute populations (rho00, rho11, rho22) from x1..x8 time series.
    x_series: array shape (nt, 8) with columns [x1..x8]
    Returns: rho00, rho11, rho22 each shape (nt,)
    """
    # Take real part to avoid tiny imaginary round-off
    x = np.real(np.asarray(x_series))
    x3 = x[:, 2]   # x3 corresponds to <lambda_3>
    x8 = x[:, 7]   # x8 corresponds to <lambda_8>

    # Standard qutrit formulas (see note above)
    rho00 = 1.0/3.0 + 0.5*( x3 + x8/np.sqrt(3.0) )
    rho11 = 1.0/3.0 + 0.5*( -x3 + x8/np.sqrt(3.0) )
    rho22 = 1.0/3.0 + 0.5*( -2.0*x8/np.sqrt(3.0) )
    return rho00, rho11, rho22


def plot_populations_from_sol(sol, idx, title_suffix: str = ""):
    """
    Extract x1..x8 from sol/idx (m-order singles = [x1..x8,Q,P]) and plot populations over time.
    """
    sl = idx["singles_slice"]
    Y = np.asarray(sol.y).T                                 # (nt, nstate)
    singles = Y[:, sl]                                      # (nt, 10)
    x_series = singles[:, :8]                               # (nt, 8)

    rho00, rho11, rho22 = populations_from_x(x_series)

    plt.figure(figsize=(7.5, 4.5))
    plt.plot(sol.t, rho00, label=r"$\rho_{00}$")
    plt.plot(sol.t, rho11, label=r"$\rho_{11}$")
    plt.plot(sol.t, rho22, label=r"$\rho_{22}$")
    plt.xlabel("t")
    plt.ylabel("Population")
    plt.title("Populationen ρ₀₀, ρ₁₁, ρ₂₂ " + title_suffix)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print("Populationen geplottet.")
    return rho00, rho11, rho22





# ---------- Convenience wrapper to do both ----------
def plot_populations_and_sigma(sol, idx, only_upper_sigma: bool = False):
    """
    Convenience wrapper: plot populations and then the full Σ time-series grid.
    """
    _ = plot_populations_from_sol(sol, idx)



# Python code (comments in English; console/output in German)

import numpy as np
import matplotlib.pyplot as plt




# ====================== 2) Σ so plotten wie s(t) ======================

def plot_sigma_entries_over_time(sol, idx, entries=((0,0),(1,1),(2,2),(8,8),(9,9)) , take_real=True):
    """
    Plot selected Σ_ij(t) entries vs time (same style as s-plot).
    Default entries: a few diagonals incl. Q and P.
    """
    Sigmas = reconstruct_sigma_series(sol, idx, take_real=take_real)  # (nt,10,10)
    t = sol.t
    plt.figure(figsize=(7.5, 4.5))
    for (i,j) in entries:
        y = np.real(Sigmas[:, i, j]) if take_real else Sigmas[:, i, j]
        plt.plot(t, y, label=f"Σ[{i},{j}]")
    plt.xlabel("t"); plt.ylabel("Σ_{ij}(t)")
    plt.title("Ausgewählte Einträge der Kovarianzmatrix")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    print("Ausgewählte Σ(t)-Einträge geplottet.")

def heatmap_sigma_snapshots(sol, idx, times=(0.0, 1.0, None), vmin=None, vmax=None, take_real=True):
    """
    Heatmaps of Σ(t) at selected times. times may include None -> use final time.
    """
    Sigmas = reconstruct_sigma_series(sol, idx, take_real=take_real)
    t = sol.t
    times_eff = []
    for tau in times:
        if tau is None:
            times_eff.append(t[-1])
        else:
            times_eff.append(tau)

    ncols = len(times_eff)
    plt.figure(figsize=(4.5*ncols, 4.0))
    for c, tau in enumerate(times_eff, 1):
        k = int(np.argmin(np.abs(t - tau)))
        plt.subplot(1, ncols, c)
        M = np.real(Sigmas[k]) if take_real else Sigmas[k]
        im = plt.imshow(M, origin="lower", vmin=vmin, vmax=vmax, aspect="equal")
        plt.title(f"Σ @ t≈{t[k]:.3g}")
        plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    print("Heatmaps von Σ(t) erstellt.")



# --- Diagnostics: check skew-symmetry and canonical QP block --------------------
def report_s_properties(S_all, *, check_qp: bool = True):
    """
    Print basic diagnostics for s(t):
    - maximal skew-symmetry deviation
    - minimal singular value (rank check hint)
    - optional check of the (Q,P) block signs
    """
    nt = S_all.shape[0]

    # Skew-symmetry deviation over all times
    devs = np.linalg.norm(S_all + np.transpose(S_all, (0,2,1)), ord=np.inf, axis=(1,2))
    print(f"Skewsymmetric-Abweichung max über alle Zeiten: {np.max(devs):.3e}")

    # Singular values at a few sample times (rank hints)
    sample_ids = np.linspace(0, nt-1, num=min(5, nt), dtype=int)
    mins = []
    for k in sample_ids:
        sv = np.linalg.svd(S_all[k], compute_uv=False)
        mins.append(np.min(sv))
    print("Minimale Singulärwerte an Stichprobenzeiten:",
          ", ".join(f"{v:.3e}" for v in mins))

    # Optional: check canonical QP block (indices 8,9 in m-order)
    if check_qp:
        qp = S_all[:, 8:10, 8:10]
        qp_det = np.linalg.det(qp)
        print(f"(Q,P)-Block: Mittelwert det(s_QP(t)) = {np.mean(qp_det):.6f}  (sollte ~1.0 sein)")
        # Quick sign check of s_QP = [[0, +1],[-1, 0]]
        s_qp_01 = np.mean(qp[:, 0, 1])
        s_qp_10 = np.mean(qp[:, 1, 0])
        print(f"(Q,P)-Block: Mittelwert s[Q,P] = {s_qp_01:.6f} (sollte +1), s[P,Q] = {s_qp_10:.6f} (sollte -1)")


# --- Pretty print for selected time steps --------------------------------------
def print_s_at_times(S_all, t, times_to_show=()):
    """
    Pretty-print s(t_k) at selected physical times.
    times_to_show: iterable of floats; picks nearest time indices.
    """
    if not times_to_show:
        return
    for tau in times_to_show:
        k = int(np.argmin(np.abs(t - tau)))
        print(f"\ns(t) bei t≈{t[k]:.6g}:")
        # print compact with aligned columns
        with np.printoptions(precision=4, suppress=True, linewidth=140):
            print(S_all[k])


# --- Optional: plot a few matrix elements vs time ------------------------------
def plot_s_entries_over_time(S_all, t, entries=((8,9),(9,8),(0,1),(1,2))):
    """
    Quick time-series plots of selected s_{ij}(t) entries.
    """
    plt.figure(figsize=(7.5, 4.5))
    for (i,j) in entries:
        plt.plot(t, S_all[:, i, j], label=f"s[{i},{j}]")
    plt.xlabel("t")
    plt.ylabel("s^{αβ}(t)")
    plt.title("Ausgewählte Einträge der symplektischen Matrix")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    print("Ausgewählte Einträge von s(t) geplottet.")


# --- Save to disk --------------------------------------------------------------
def save_s_series(S_all, path_npz="s_series.npz", key="s_series"):
    """
    Save the full time series to disk for later analysis.
    """
    np.savez_compressed(path_npz, **{key: S_all})
    print(f"s(t)-Serie gespeichert nach: {path_npz}  (Key='{key}')")


# ============================================================
# === Main: assemble, integrate, transform, and inspect
# ============================================================
# Python code (comments in English; console/output in German)

import numpy as np
import matplotlib.pyplot as plt

# ---------- Numerics helpers ----------
def symmetrize_hermitian(M):
    """Make a matrix Hermitian numerically: (M + M^†)/2."""
    return 0.5 * (M + M.conjugate().T)

def min_eigvals_hermitian_series(series):
    """
    Compute eigenvalues for a time series of Hermitian matrices.
    series: array (nt, n, n)
    Returns:
        lam_min: shape (nt,) minimal eigenvalue at each time
        lam_all: list of 1D arrays of eigenvalues per time (ascending)
    """
    nt = series.shape[0]
    lam_min = np.empty(nt, dtype=float)
    lam_all = []
    for k in range(nt):
        Hk = symmetrize_hermitian(series[k])
        # eigvalsh is for Hermitian; returns ascending real eigenvalues
        evals = np.linalg.eigvalsh(Hk)
        lam_all.append(evals)
        lam_min[k] = np.min(evals.real)
    return lam_min, lam_all

def is_psd_from_min(lam_min, tol=1e-10):
    """Return boolean mask for PSD (min eigenvalue >= -tol)."""
    return lam_min >= -tol

# ---------- Core checks ----------
def check_psd_sigma_and_uncertainty(sol, idx, *, tol=1e-10, take_real_sigma=True, scale_su3=2.0,
                                    make_plots=True):
    """
    For all time steps, test:
      (i)  Σ(t) PSD
      (ii) Σ(t) + (i/2) s(t) PSD    (Robertson-Schrödinger uncertainty)
    Uses numerically reconstructed Σ(t) and s(t).
    """
    # 1) Reconstruct Σ(t) in m-order [x1..x8, Q, P]
    Sigmas = reconstruct_sigma_series(sol, idx, take_real=take_real_sigma)  # (nt,10,10)
    # Ensure (real-)symmetric or Hermitian
    for k in range(Sigmas.shape[0]):
        Sigmas[k] = symmetrize_hermitian(Sigmas[k])

    # 2) Compute s(t) from singles (real skew-symmetric)
    S_all = symplectic_series_from_solution(sol, idx, scale_su3=scale_su3)  # (nt,10,10)
    # Enforce skew-symmetry
    S_all = 0.5 * (S_all - np.transpose(S_all, (0,2,1)))

    # 3) Check Σ(t) PSD
    lam_min_Sigma, _ = min_eigvals_hermitian_series(Sigmas)
    psd_Sigma = is_psd_from_min(lam_min_Sigma, tol=tol)

    # 4) Check Σ + i/2 s PSD (this is Hermitian because i*s is Hermitian)
    nt = Sigmas.shape[0]
    Sigma_plus_is_over_2 = np.empty_like(Sigmas, dtype=complex)
    for k in range(nt):
        Sigma_plus_is_over_2[k] = symmetrize_hermitian(Sigmas[k] + 0.5j * S_all[k])

    lam_min_unc, _ = min_eigvals_hermitian_series(Sigma_plus_is_over_2)
    psd_unc = is_psd_from_min(lam_min_unc, tol=tol)

    # 5) Console report
    print("=== PSD-Checks über alle Zeiten ===")
    print(f"Σ + i/2 s:  min(λ_min) = {np.min(lam_min_unc):.3e},  "
      f"verletzte Zeitpunkte = {np.count_nonzero(~psd_unc)}")

    # Show first few violating indices (if any)
    bad_sigma = np.where(~psd_Sigma)[0]
    bad_unc   = np.where(~psd_unc)[0]
    if bad_sigma.size > 0:
        k = bad_sigma[:10]
        print("Erste verletzte Indizes für Σ PSD:", k, " (Zeiten ~", sol.t[k], ")")
    if bad_unc.size > 0:
        k = bad_unc[:10]
        print("Erste verletzte Indizes für Σ + i/2 s PSD:", k, " (Zeiten ~", sol.t[k], ")")

    # 6) Optional plots of minimal eigenvalue vs time
    if make_plots:
        plt.figure(figsize=(7.5, 4.5))
        plt.plot(sol.t, lam_min_Sigma, label="min λ(Σ)")
        plt.plot(sol.t, lam_min_unc, label="min λ(Σ + i/2·s)")
        plt.axhline(0.0, linestyle="--")
        plt.xlabel("t")
        plt.ylabel("Minimaler Eigenwert")
        plt.title("PSD-Tests über die Zeit")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        print("Plots der minimalen Eigenwerte erstellt.")

    return {
        "lam_min_Sigma": lam_min_Sigma,
        "lam_min_unc": lam_min_unc,
        "psd_Sigma_mask": psd_Sigma,
        "psd_unc_mask": psd_unc,
        "S_all": S_all,
        "Sigmas": Sigmas
    }
import numpy as np

def build_C_full(f_tensor, boson_factor=1.0):
    """
    Build commutator matrix C for 8 SU(3) fluctuation operators + q,p bosonic mode.

    f_tensor[a,b,c] : structure constants f_{abc} (antisymmetric in a,b)
    boson_factor    : kappa with [q,p] = i*kappa (usually 1 or 2)

    Returns:
      C : (10x10) numpy array
    """
    dim_spin = 8
    J_spin = np.zeros((dim_spin, dim_spin))
    # build spin commutator block with factor 2
    for a in range(dim_spin):
        for b in range(dim_spin):
            for c in range(dim_spin):
                J_spin[a, b] += 2 * f_tensor[a, b, c]  # factor 2 included

    # bosonic commutator block
    J_boson = np.array([[0, boson_factor], [-boson_factor, 0]])

    # assemble block-diagonal C
    C = np.block([
        [J_spin,               np.zeros((dim_spin, 2))],
        [np.zeros((2, dim_spin)), J_boson]
    ])

    return C




def _build_nameval_from_numeric_params(numeric_params: dict) -> dict[str, float]:
    """Make a name->float map from a dict keyed by SymPy Symbols (or strings)."""
    nameval: dict[str, float] = {}
    for k, v in numeric_params.items():
        if isinstance(k, sp.Symbol):
            name = k.name
        else:
            name = str(k)
        try:
            nameval[name] = float(v)
        except Exception:
            nameval[name] = float(sp.N(v))
    return nameval

def numeric_matrix_from_sym(
    M_sym: sp.Matrix,
    subs_primary: dict[sp.Symbol, complex] | None = None,
    numeric_params: dict | None = None,
    default_zero: bool = True,
) -> np.ndarray:
    """
    Substitute (1) explicit symbol->value map (e.g. m1..m10) and (2) numeric_params.
    Also try name-based substitution for any remaining symbols.
    If symbols still remain and default_zero=True, set them to 0 with a warning.
    Returns a dense complex numpy array.
    """
    subs_primary = subs_primary or {}
    numeric_params = numeric_params or {}

    # First pass: identity-based substitution
    M_tmp = sp.Matrix(M_sym).subs(subs_primary, simultaneous=True)

    # Build name-based map from numeric_params
    nameval = _build_nameval_from_numeric_params(numeric_params)

    # Collect remaining symbols
    remaining = sorted(list(M_tmp.free_symbols), key=lambda s: s.name)

    # Second pass: name-based substitution (match by s.name)
    name_based_pairs = {}
    for s in remaining:
        if s.name in nameval:
            name_based_pairs[s] = nameval[s.name]

    if name_based_pairs:
        M_tmp = M_tmp.subs(name_based_pairs, simultaneous=True)

    # Check again which symbols remain
    remaining2 = sorted(list(M_tmp.free_symbols), key=lambda s: s.name)
    if remaining2:
        if default_zero:
            print("Warnung: Unbelegte Symbole gefunden und auf 0 gesetzt:", [s.name for s in remaining2])
            zero_map = {s: 0.0 for s in remaining2}
            M_tmp = M_tmp.subs(zero_map, simultaneous=True)
        else:
            raise ValueError(f"Unbelegte Symbole in Matrix: {[s.name for s in remaining2]}")

    # Ensure numeric value (evaluate) and convert to numpy
    M_eval = sp.N(M_tmp)  # evalf
    return np.array(M_eval.tolist(), dtype=complex)
    #


def check_hermitian_psd(M, *, herm_tol=1e-10, psd_tol=1e-12, return_eigs=False, verbose=True):
    """
    Check if matrix M is Hermitian and Positive Semi-Definite (PSD).

    Parameters
    ----------
    M : array_like
        Input matrix (n x n). Real or complex.
    herm_tol : float
        Tolerance for Hermiticity: max|M - M^†| <= herm_tol -> Hermitian.
    psd_tol : float
        Tolerance for PSD: min_eig(H) >= -psd_tol (with H = (M+M^†)/2).
    return_eigs : bool
        If True, also return all eigenvalues (ascending).
    verbose : bool
        If True, print a short German report.

    Returns
    -------
    result : dict
        {
          "is_hermitian": bool,
          "is_psd": bool,
          "max_antiherm_norm": float,   # max|M - M^†|
          "min_eig": float,             # minimal eigenvalue of Hermitian part
          "eigvals": np.ndarray (optional)
        }
    """
    A = np.array(M, dtype=complex, copy=False)

    # Hermitian deviation (use elementwise max-norm)
    antiherm = A - A.conjugate().T
    max_anti = np.max(np.abs(antiherm))

    is_herm = max_anti <= herm_tol

    # Symmetrize to guard tiny non-Hermitian noise before PSD test
    H = 0.5 * (A + A.conjugate().T)
    # For numerical stability, ensure exact Hermitian symmetry numerically
    H = 0.5 * (H + H.conjugate().T)

    # Hermitian eigenvalues (ascending, real for Hermitian)
    eigvals = np.linalg.eigvalsh(H)
    min_eig = float(np.min(eigvals.real))
    is_psd = min_eig >= -psd_tol

    if verbose:
        print("=== Matrix-Check: Hermitesch & PSD ===")
        print(f"max|M - M^†| = {max_anti:.3e}  ⇒ Hermitesch: {is_herm} (Toleranz {herm_tol:g})")
        print(f"min λ(H)     = {min_eig:.3e}  ⇒ PSD:        {is_psd} (Toleranz {psd_tol:g})")

    out = {
        "is_hermitian": bool(is_herm),
        "is_psd": bool(is_psd),
        "max_antiherm_norm": float(max_anti),
        "min_eig": min_eig,
    }
    if return_eigs:
        out["eigvals"] = eigvals
    return out


def plot_sigma_and_check_psd(
    sol,
    idx,
    *,
    take_real: bool = True,
    tol: float = 1e-10,
    also_uncertainty: bool = True,
    entries=((0, 0), (1, 1), (2, 2), (8, 8), (9, 9)),
    heatmap_times=(0.0, None),  # None -> use final time
    save_prefix: str | None = None,
    show_plots: bool = True,
):
    """
    Reconstruct Σ(t) for all time steps, check PSD, and make diagnostic plots.

    Parameters
    ----------
    sol : scipy.integrate.OdeResult
        Solution returned by solve_ivp (expects .t and .y).
    idx : dict
        Indexing dictionary returned by prepare_system_for_solve_ivp (must contain
        'singles_slice', 'pairs_slice', and 'pairs_order').
    take_real : bool
        If True, drop tiny imaginary parts of Σ and enforce a real symmetric covariance.
    tol : float
        PSD tolerance: min_eig >= -tol counts as PSD.
    also_uncertainty : bool
        If True, also check PSD of Σ + i/2·s(t) (Robertson–Schrödinger inequality).
    entries : iterable[(i,j)]
        Selected Σ_ij(t) entries to plot as time series.
    heatmap_times : iterable[float|None]
        Times to snapshot Σ(t) as heatmaps; None means the final time.
    save_prefix : str | None
        If set, save plots to files using this prefix (e.g., "out/sigma_psd").
    show_plots : bool
        If True, call plt.show() at the end.

    Returns
    -------
    results : dict
        {
          "Sigmas": (nt,10,10) ndarray (Hermitian-symmetrized),
          "lam_min_Sigma": (nt,) minimal eigenvalue of Σ(t),
          "psd_Sigma_mask": (nt,) boolean mask,
          "lam_min_unc": (nt,)  # only if also_uncertainty
          "psd_unc_mask": (nt,) # only if also_uncertainty
        }
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # --- Local, collision-safe reconstruction (avoids the stub variant) -----
    def _reconstruct_sigma_series_local(sol, idx, take_real=True):
        """Rebuild Σ(t_k) from packed pairs in sol.y using idx['pairs_order']."""
        n = idx["singles_slice"].stop - idx["singles_slice"].start  # should be 10
        pair_start = idx["pairs_slice"].start
        pairs_order = idx["pairs_order"]
        nt = sol.y.shape[1]

        Sig = np.zeros((nt, n, n), dtype=complex)
        for k_loc, (i, j, _sym) in enumerate(pairs_order):
            row = pair_start + k_loc
            traj = sol.y[row, :]               # length nt
            Sig[:, i, j] = traj
            Sig[:, j, i] = np.conjugate(traj)  # ensure Hermitian symmetry

        if take_real:
            Sig = np.real(Sig)
        return Sig

    # ---------- 1) Σ(t): reconstruct & enforce Hermitian ----------
    Sigmas = _reconstruct_sigma_series_local(sol, idx, take_real=take_real)
    # Use the helper already present in this module:
    #   - symmetrize_hermitian(M)
    #   - min_eigvals_hermitian_series(series)
    for k in range(Sigmas.shape[0]):
        Sigmas[k] = symmetrize_hermitian(Sigmas[k])  # (M + M^†)/2

    lam_min_Sigma, _ = min_eigvals_hermitian_series(Sigmas)
    psd_Sigma = is_psd_from_min(lam_min_Sigma, tol=tol)

    # ---------- 2) Optional: uncertainty test Σ + i/2·s ----------
    lam_min_unc = None
    psd_unc = None
    if also_uncertainty:
        # s(t) from singles using the helper in this module
        S_all = symplectic_series_from_solution(sol, idx)   # real, skew-symmetric
        S_all = 0.5 * (S_all - np.transpose(S_all, (0, 2, 1)))  # guard skew
        Sigma_plus_is_over_2 = np.empty_like(Sigmas, dtype=complex)
        for k in range(Sigmas.shape[0]):
            # i*s is Hermitian; Σ + i/2·s is Hermitian -> eigenvalues real
            Sigma_plus_is_over_2[k] = symmetrize_hermitian(Sigmas[k] + 0.5j * S_all[k])
        lam_min_unc, _ = min_eigvals_hermitian_series(Sigma_plus_is_over_2)
        psd_unc = is_psd_from_min(lam_min_unc, tol=tol)

    # ---------- 3) Console report (German) ----------
    print("=== PSD-Prüfung über alle Zeitschritte ===")
    print(f"Σ(t): min(λ_min) = {np.min(lam_min_Sigma):.3e},  "
          f"Anzahl Verletzungen = {np.count_nonzero(~psd_Sigma)}")
    if also_uncertainty:
        print(f"Σ(t) + i/2·s(t): min(λ_min) = {np.min(lam_min_unc):.3e},  "
              f"Anzahl Verletzungen = {np.count_nonzero(~psd_unc)}")

    bad_sigma = np.where(~psd_Sigma)[0]
    if bad_sigma.size > 0:
        k = bad_sigma[:10]
        print("Erste verletzte Indizes für Σ PSD:", k, "  (Zeiten ≈", sol.t[k], ")")
    if also_uncertainty and (psd_unc is not None):
        bad_unc = np.where(~psd_unc)[0]
        if bad_unc.size > 0:
            k = bad_unc[:10]
            print("Erste verletzte Indizes für Σ + i/2·s PSD:", k, "  (Zeiten ≈", sol.t[k], ")")

    # ---------- 4) Plots ----------
    # (a) minimal eigenvalues vs time
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(sol.t, lam_min_Sigma, label="min λ(Σ)")
    if also_uncertainty:
        plt.plot(sol.t, lam_min_unc, label="min λ(Σ + i/2·s)")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("t")
    plt.ylabel("Minimaler Eigenwert")
    plt.title("PSD-Tests über die Zeit")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_min_eigs.png", dpi=150)
        print(f"Plot gespeichert: {save_prefix}_min_eigs.png")

    # (b) selected Σ_ij(t) entries
    if entries:
        plt.figure(figsize=(7.5, 4.5))
        for (i, j) in entries:
            y = np.real(Sigmas[:, i, j]) if np.iscomplexobj(Sigmas) else Sigmas[:, i, j]
            plt.plot(sol.t, y, label=f"Σ[{i},{j}]")
        plt.xlabel("t")
        plt.ylabel("Σ_{ij}(t)")
        plt.title("Ausgewählte Einträge der Kovarianzmatrix")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        if save_prefix:
            plt.savefig(f"{save_prefix}_entries.png", dpi=150)
            print(f"Plot gespeichert: {save_prefix}_entries.png")

    # (c) heatmap snapshots at requested times
    if heatmap_times:
        times_eff = []
        for tau in heatmap_times:
            times_eff.append(sol.t[-1] if tau is None else float(tau))
        ncols = len(times_eff)
        plt.figure(figsize=(4.5 * ncols, 4.0))
        for c, tau in enumerate(times_eff, 1):
            k = int(np.argmin(np.abs(sol.t - tau)))
            M = np.real(Sigmas[k]) if np.iscomplexobj(Sigmas) else Sigmas[k]
            plt.subplot(1, ncols, c)
            im = plt.imshow(M, origin="lower", aspect="equal")
            plt.title(f"Σ @ t≈{sol.t[k]:.3g}")
            plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        if save_prefix:
            plt.savefig(f"{save_prefix}_heatmaps.png", dpi=150)
            print(f"Plot gespeichert: {save_prefix}_heatmaps.png")

    if show_plots:
        plt.show()

    results = {
        "Sigmas": Sigmas,
        "lam_min_Sigma": lam_min_Sigma,
        "psd_Sigma_mask": psd_Sigma,
    }
    if also_uncertainty:
        results.update({
            "lam_min_unc": lam_min_unc,
            "psd_unc_mask": psd_unc
        })
    return results


def gellmann_matrices():
    """Return the 8 standard Gell-Mann matrices λ_a with Tr(λ_a λ_b) = 2 δ_ab."""
    l1 = np.array([[0,1,0],[1,0,0],[0,0,0]], dtype=float)
    l2 = np.array([[0,0,-1],[0,0,0],[ -1,0,0]], dtype=float)  # will set sign via i later
    l2 = np.array([[0,-1,0],[1,0,0],[0,0,0]], dtype=float) * 0  # placeholder to avoid confusion
    # Build explicitly to avoid mistakes (use standard definitions)
    λ1 = np.array([[0,1,0],[1,0,0],[0,0,0]], dtype=float)
    λ2 = np.array([[0,-1,0],[1,0,0],[0,0,0]], dtype=float) * 0  # not used; we construct via complex, then take real part
    λ2 = np.array([[0,-1j,0],[1j,0,0],[0,0,0]], dtype=complex)
    λ3 = np.array([[1,0,0],[0,-1,0],[0,0,0]], dtype=float)
    λ4 = np.array([[0,0,1],[0,0,0],[1,0,0]], dtype=float)
    λ5 = np.array([[0,0,-1j],[0,0,0],[1j,0,0]], dtype=complex)
    λ6 = np.array([[0,0,0],[0,0,1],[0,1,0]], dtype=float)
    λ7 = np.array([[0,0,0],[0,0,-1j],[0,1j,0]], dtype=complex)
    λ8 = (1/np.sqrt(3)) * np.array([[1,0,0],[0,1,0],[0,0,-2]], dtype=float)
    # Ensure dtype=complex for all, to handle λ2, λ5, λ7 correctly
    mats = [λ1, λ2, λ3.astype(complex), λ4.astype(complex), λ5, λ6.astype(complex), λ7, λ8.astype(complex)]
    return mats

def su3_d_tensor():
    """
    Compute d_{abc} via definition:
        d_{abc} = (1/4) Tr({λ_a, λ_b} λ_c)   with Tr(λ_a λ_b)=2 δ_ab
    Returns d with shape (8,8,8), dtype=float (purely real).
    """
    λ = gellmann_matrices()
    d = np.zeros((8,8,8), dtype=float)
    for a in range(8):
        for b in range(8):
            # anticommutator
            ab = λ[a] @ λ[b] + λ[b] @ λ[a]
            for c in range(8):
                val = np.trace(ab @ λ[c]) / 4.0
                # val should be real; discard tiny imaginary noise
                d[a,b,c] = float(np.real_if_close(val))
    return d







# ========================== ADD/REPLACE: robust initial Σ and s ==========================
import numpy as np

def _su3_structure_constants_from_lambdas(lams):
    """Compute f_{abc} and d_{abc} via trace identities for given generators lams."""
    f = np.zeros((8,8,8), dtype=float)
    d = np.zeros((8,8,8), dtype=float)
    for a in range(8):
        for b in range(8):
            comm = lams[a] @ lams[b] - lams[b] @ lams[a]
            anti = lams[a] @ lams[b] + lams[b] @ lams[a]
            for c in range(8):
                f[a,b,c] = np.real(np.trace(comm @ lams[c])/(4j))
                d[a,b,c] = np.real(np.trace(anti @ lams[c])/4.0)
    return f, d

def _gellmann_standard():
    """Return standard Gell-Mann matrices λ_a with Tr(λa λb)=2 δab."""
    l = []
    l.append(np.array([[0,1,0],[1,0,0],[0,0,0]], dtype=complex))                      # λ1
    l.append(np.array([[0,-1j,0],[1j,0,0],[0,0,0]], dtype=complex))                   # λ2
    l.append(np.array([[1,0,0],[0,-1,0],[0,0,0]], dtype=complex))                     # λ3
    l.append(np.array([[0,0,1],[0,0,0],[1,0,0]], dtype=complex))                      # λ4
    l.append(np.array([[0,0,-1j],[0,0,0],[1j,0,0]], dtype=complex))                   # λ5
    l.append(np.array([[0,0,0],[0,0,1],[0,1,0]], dtype=complex))                      # λ6
    l.append(np.array([[0,0,0],[0,0,-1j],[0,1j,0]], dtype=complex))                   # λ7
    l.append((1/np.sqrt(3))*np.array([[1,0,0],[0,1,0],[0,0,-2]], dtype=complex))      # λ8
    return l

def sigma_atom_from_m(m8, d_tensor, *, generator='lambda'):
    """
    Atomic block Σ^{(λ)}(0) from singles m (length 8).
    For λ:  Σ_ab = 2/3 δ_ab + d_{abc} m_c - m_a m_b
    For t=λ/2: Σ_ab = 1/6 δ_ab + 1/2 d_{abc} m_c - m_a m_b
    """
    m = np.asarray(m8, dtype=float)
    if generator.lower() == 'lambda':
        base = (2.0/3.0) * np.eye(8)
        add  = np.tensordot(d_tensor, m, axes=(2,0))
    else:  # 't'
        base = (1.0/6.0) * np.eye(8)
        add  = 0.5 * np.tensordot(d_tensor, m, axes=(2,0))
    Sigma = base + add - np.outer(m, m)
    return 0.5*(Sigma+Sigma.T)

def sigma_qp_from_spec(*, qp_comm=1.0, kind='coherent', n_th=0.0, r=0.0, phi=0.0, Sigma=None):
    """
    Build 2x2 field covariance:
      coherent/vacuum: diag(qp_comm/2, qp_comm/2)
      thermal:        (n_th+1/2)*qp_comm * I
      squeezed:       rotate(diag(e^{-2r}, e^{+2r})) scaled by qp_comm/2
      Sigma:          use provided 2x2 matrix
    """
    if kind in ('coherent','vacuum'):
        s = 0.5*qp_comm
        return np.array([[s,0.0],[0.0,s]], dtype=float)
    if kind=='thermal':
        s = (n_th+0.5)*qp_comm
        return np.array([[s,0.0],[0.0,s]], dtype=float)
    if kind=='squeezed':
        dq = 0.5*qp_comm*np.exp(-2.0*r)
        dp = 0.5*qp_comm*np.exp(+2.0*r)
        R  = np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]], dtype=float)
        return R @ np.diag([dq,dp]) @ R.T
    if kind=='Sigma' and Sigma is not None:
        M = np.array(Sigma, dtype=float);  return 0.5*(M+M.T)
    raise ValueError("Unknown field kind.")

def build_initial_sigma_and_s_from_rho_y0(
    y0_ket: np.ndarray,
    *,
    generator='lambda',
    qp_comm=1.0,
    field_kind='coherent',
    field_kwargs=None
):
    """
    Full pipeline from rho-basis initial state to Σ(0) and s(0).
    y0_ket = [da, da†, rho00, rho01, rho10, rho11, rho22, rho21, rho12, rho20, rho02]
    """
    # 1) Map rho-basis -> x=[Q,P,x1..x8] using your helper
    x0 = convert_state(y0_ket)  # complex array

    # 2) Singles in m-order [x1..x8, Q, P]
    m0_singles = np.array([*x0[2:10], x0[0], x0[1]], dtype=complex)
    m8 = np.real(m0_singles[:8])

    # 3) Atomic Σ via d-tensor (computed from exact λs)
    lams = _gellmann_standard()
    _f, d = _su3_structure_constants_from_lambdas(lams)
    Sigma_atom = sigma_atom_from_m(m8, d, generator=generator)

    # 4) Field Σ
    field_kwargs = field_kwargs or {}
    Sigma_qp = sigma_qp_from_spec(qp_comm=qp_comm, kind=field_kind, **field_kwargs)

    # 5) Assemble Σ(0)
    Sigma0 = np.zeros((10,10), dtype=float)
    Sigma0[:8,:8] = Sigma_atom
    Sigma0[8:,8:] = Sigma_qp

    # 6) s(0) from singles (your su3 f-tensor + canonical QP)
    #    We reuse your robust builder so normalization is consistent with your code.
    s0 = symplectic_from_singles(np.real(m0_singles), scale_su3=2.0)  # uses su3_f_tensor() internally

    return m0_singles, Sigma0, s0

# ========================== MAIN DRIVER (ρ-basis → solve → checks) ==========================
def run_qf_from_rho_initial(
    y0_ket: np.ndarray,
    t_span=(0.0, 50.0),
    nt=2001,
    *,
    params_singles: dict,
    qp_comm=1.0,
    field_kind='coherent',
    field_kwargs=None,
    ode_rtol=1e-8,
    ode_atol=1e-8
):
    """
    End-to-end: rho-initial -> m-order singles/pairs ICs -> integrate -> PSD checks each step.
    """
    # (A) Build Σ(0), s(0) and singles m(0)
    m0_singles, Sigma0, s0 = build_initial_sigma_and_s_from_rho_y0(
        y0_ket, generator='lambda', qp_comm=qp_comm,
        field_kind=field_kind, field_kwargs=field_kwargs
    )

    # (B) Prepare RHS for pairs system (Σ̇ = ... from your symbolic Sigma_dt)
    #     (we assume you've already built Sigma_dt and the idx/rhs func earlier)
    # -> if not yet available, do it here (using your existing helpers)
    #     idx, rhs_pairs_func = prepare_system_for_solve_ivp(mP_syms, Sigma_dt, ...)
    # (we rely on idx, rhs_pairs_func existing in outer scope)

    # (C) Pack initial state: singles + upper-triangle(Σ0)
    m0_pairs = pack_upper_triangle_from_covariance(Sigma0, idx["pairs_order"])
    y0 = np.concatenate([m0_singles, m0_pairs])

    # (D) Combined RHS (singles via your rhs_gellmann_qp_from_x, pairs via lambdified Σ̇)
    fun = make_combined_rhs(idx, rhs_pairs_func, params_dict=params_singles, extra_param_values_tuple=())

    # (E) Integrate
    t_eval = np.linspace(t_span[0], t_span[1], nt)
    sol = solve_ivp(fun, t_span, y0, t_eval=t_eval, method="RK45", rtol=ode_rtol, atol=ode_atol)

    # (F) Reconstruct Σ(t) and compute s(t) at each step; check Robertson–Schrödinger
    results = check_psd_sigma_and_uncertainty(
        sol, idx, tol=1e-10, take_real_sigma=True, scale_su3=2.0, make_plots=True
    )
    return sol, results





def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

if __name__ == "__main__":
    print("Baue kombiniertes Singles+Kovarianz-System …")
    regular_plots = True

    # (1) Symbols, sizes
    t_span = (0.0, 120.0)
    t_eval = np.linspace(*t_span, 2001)
    n = 10
    mP_syms = sp.symbols('m1:11')  # m1..m8 = x1..x8, m9=Q, m10=P

    g0, Delta1, Delta2, V, Gamma, Omega, kappa, eta = sp.symbols("g0 Delta1 Delta2 V Gamma Omega kappa eta")
    Omega_val = 8
    Gamma_val = 2
    V_val = -0.5 * ((Omega_val / 4)**2 + 1)
    numeric_params = { g0: 1, Delta1: 1, Delta2: 1, V: V_val, Gamma: Gamma_val, Omega: Omega_val, kappa: 1, eta: 1 }

    # Pull matrices from your covar module (numeric + symbolic, if needed later)
    G, sDs, Z, P, Q,sE, Z_prime, W, Sigma_dt, Sigma, K = covar.get_important_matricies(numeric_params)
    #G_sym, sDs_sym, Z_sym, P_sym, Q_sym,sE_sym, Z_prime_sym, W_sym, Sigma_dt_sym, Sigma_sym, K_sym = covar.get_important_matricies_symbol()
    
    #sp.pprint(sp.Matrix(G_sym))
    #sp.pprint(sp.Matrix(Q_sym))
    #sp.pprint(sp.Matrix(Z_sym))
    #sp.pprint(sp.Matrix(G_sym))

    #sp.pprint(sDs_sym)
    # # Example usage:
    # # Suppose you have f_tensor (8x8x8 antisymmetric array) already constructed
    # f_tensor=su3_f_tensor()
    # mP_syms = sp.symbols('m1:11')                            # m1..m10
    # y0_ket = np.array([0+0j, 0+0j, 1+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j])
    # x0 = convert_state(y0_ket)   
    # m0_singles = np.array([*x0[2:10], x0[0], x0[1]], dtype=complex) 
    # subs_m = { sym: complex(val) for sym, val in zip(mP_syms, m0_singles) }
    # C = build_C_full(f_tensor, boson_factor=2)
    # G0 = numeric_matrix_from_sym(G_sym, subs_primary=subs_m, numeric_params=numeric_params, default_zero=True)
    # W0 = numeric_matrix_from_sym(W_sym, subs_primary=subs_m, numeric_params=numeric_params, default_zero=True)

    # # --- 4) Kommutator-Matrix C bauen (zustandsunabhängig) -------------------------

    # f_tensor = su3_f_tensor()
    # C = build_C_full(f_tensor, boson_factor=2.0)  # [Q,P] with [Q,P] = i*2

    # # --- 5) check(m(0)) berechnen und drucken -------------------------------------

    # check0 = W0 + 1j/2.0 * (C - G0 @ C @ G0.T)

    # print("\ncheck-Matrix am Anfang (mit Initialzustand eingesetzt):")
    # with np.printoptions(precision=6, suppress=True, linewidth=160):
    #     sp.pprint(sp.Matrix(check0))
    #     res = check_hermitian_psd(check0, herm_tol=1e-12, psd_tol=1e-12, return_eigs=True)

    

    # (2) Prepare pair system
    Sigma_dt_clean = Sigma_dt
    sp.simplify(Sigma_dt_clean)





    idx, rhs_pairs_func = prepare_system_for_solve_ivp(
        mP_syms=mP_syms,
        Sigma_dt=Sigma_dt_clean,
        symmetric_pairs=True,
        extra_params=()
    )

    # (3) Singles physical parameters (for your singles RHS)
    params = dict(
        g0=1, kappa=1.0, gamma=1.0, Gamma=Gamma_val,
        Omega=Omega_val, delta1=1.0, delta2=1.0,
        eta=1.0, V=V_val
    )

    # (4) Combined RHS
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
            dy_pairs = np.asarray(rhs_pairs_func(*args)).astype(complex, copy=False)

            return np.concatenate([dy_singles, dy_pairs])

        return fun

    fun = make_combined_rhs(idx, rhs_pairs_func, params_dict=params, extra_param_values_tuple=())

    # (5) Initial conditions
    # 11-d ket IC -> x-space -> m-order [x1..x8, Q, P]
    #y0_ket = np.array([0+0j, 0+0j, 1+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j])
    #[da_dt, da_dagger_dt, d00, d01, d10, d11, d22, d21, d12, d20, d02]
  # ===== Example: rho00=0.8, rho11=0.0, rho22=0.2; displaced coherent cavity (Q,P) means =====
    # y0_ket = [da, da†, ρ00, ρ01, ρ10, ρ11, ρ22, ρ21, ρ12, ρ20, ρ02]
    y0_ket = np.array([
        0+0j, 0+0j,          # da, da† (not used for Σ(0); convert_state handles mapping)
        1+0j, 0+0j, 0+0j,  # ρ00, ρ01, ρ10
        0.0+0j,              # ρ11
        0.0+0j, 0+0j, 0+0j,  # ρ22, ρ21, ρ12
        0+0j, 0+0j           # ρ20, ρ02
    ])

    # Singles-Parameter für dein rhs (passe an dein System an)
    params_singles = dict(g0=1.0, kappa=1.0, gamma=1.0, Gamma=2.0,
                        Omega=8.0, delta1=1.0, delta2=1.0, eta=1.0, V=-0.5*((8/4)**2+1))

    # Feld-Varianzen (koherent/vakuum → Var(Q)=Var(P)=qp_comm/2), hier zusätzlich Mittelwerte in convert_state
    sol, res = run_qf_from_rho_initial(
        y0_ket,
        t_span=(0.0, 120.0),
        nt=2001,
        params_singles=params_singles,
        qp_comm=1.0,                 # [Q,P] = i
        field_kind='coherent',       # or 'thermal', 'squeezed', 'Sigma'
        field_kwargs=None,
        ode_rtol=1e-8, ode_atol=1e-8
    )





















#     # Covariance IC Σ(0)
#     Σ0 = initial_covariance_from_state(m0_singles, atom_scale=1.0, boson_scale=1.0)
#     m0_pairs = pack_upper_triangle_from_covariance(Σ0, idx["pairs_order"])

#     y0 = np.concatenate([m0_singles, m0_pairs])

#     # (6) Smoke-test: rhs_pairs_func must accept a flat state -> returns correct length
#     y_len = len(idx["state_symbols"])
#     try:
#         _ = rhs_pairs_func(*([0.0] * y_len))
#         print("Lambdify-Test OK – Länge z-Ausgabe =", len(_))
#     except Exception as e:
#         print("Lambdify-Test FEHLER:", repr(e))

#     # (7) Numerical RHS check at t=0 (optional)
#     print("\n===== Numerischer RHS-Check bei t=0 =====")
#     dy_singles0 = singles_rhs_from_m_order(0.0, y0[idx["singles_slice"]], params)
#     dy_pairs0   = np.asarray(rhs_pairs_func(*y0.tolist()))
#     print("Singles @0:", dy_singles0)

#     # (8) Integrate
#     sol = solve_ivp(fun, t_span, y0, t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-8)

#     plot_populations_and_sigma(sol, idx, only_upper_sigma=False)
#     res = plot_sigma_and_check_psd(
#     sol, idx,
#     take_real=True,
#     tol=1e-10,
#     also_uncertainty=True,
#     entries=((0,0),(1,1),(2,2),(8,8),(9,9)),
#     heatmap_times=(sol.t[0], sol.t[len(sol.t)//2], None),  # start, mid, final
#     save_prefix=None,  # or e.g. "out/sigma_psd"
#     show_plots=True
# )
    
