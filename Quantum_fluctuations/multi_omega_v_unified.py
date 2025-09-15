
# -*- coding: utf-8 -*-
"""
Unified-params multi-(Omega, V) parallel runner (complete, self-contained helpers).

- Code & comments: English
- Console/output text: German
- No plotting.
- Results written to one HDF5 file, one group per (Omega, V).

This version **unifies** the previously separate "params" (singles RHS) and
"numeric_params" (symbolic Sigma_dt substitution) into a single dataclass `PhysParams`.
We expose two derived dicts when needed:
  - `to_covar_dict()`  → keys exactly matching the symbolic names in Sigma_dt (e.g. 'Delta1')
  - `to_singles_dict()`→ keys expected by your singles RHS (e.g. 'delta1')

External modules expected (from your project):
  - run_script.convert_state, run_script.rhs_gellmann_qp_from_x
  - covar_everything (as 'covar') providing get_important_matricies_symbol()
  - symplectic_matrix (as 'symplect') providing:
        transform_complex_S(C8), expand_to_10x10, expand_to_10x10_sym
"""

from __future__ import annotations

import numpy as np
import sympy as sp
from typing import Iterable, Optional, Dict, Tuple, List
from dataclasses import dataclass, asdict, replace
from itertools import product
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from scipy.integrate import solve_ivp
import h5py

# ---- External project modules ----
import covar_everything as covar
import symplectic_matrix as symplect
from run_script import convert_state, rhs_gellmann_qp_from_x


# =====================================================================
# ======================== Unified parameters =========================
# =====================================================================

@dataclass(frozen=True)
class PhysParams:
    """
    Unified physical parameters for both:
      - Sigma_dt symbolic substitution (expects exact symbol names like 'Delta1'),
      - singles RHS (expects 'delta1', 'delta2', etc.).
    Notes:
      * 'Gamma' appears only in singles RHS.
      * 'g0','Delta1','Delta2','gamma','kappa','eta','Omega','V' are used in Sigma_dt.
      * For singles RHS we map Delta1→delta1, Delta2→delta2.
    """
    g0: float = 1.0
    Delta1: float = 1.0
    Delta2: float = 1.0
    gamma: float = 1.0   # choose a consistent default; override as needed
    kappa: float = 1.0
    eta: float = 1.0
    Omega: float = 8.0
    V: float = -1.0
    Gamma: float = 2.0   # singles-only

    def to_covar_dict(self) -> Dict[str, float]:
        """Mapping for Sigma_dt substitution (names must match symbolic names)."""
        return dict(
            g0=float(self.g0),
            Delta1=float(self.Delta1),
            Delta2=float(self.Delta2),
            gamma=float(self.gamma),
            kappa=float(self.kappa),
            eta=float(self.eta),
            Omega=float(self.Omega),
            V=float(self.V),
        )

    def to_singles_dict(self) -> Dict[str, float]:
        """Mapping for rhs_gellmann_qp_from_x (expected lowercase deltas)."""
        return dict(
            kappa=float(self.kappa),
            gamma=float(self.gamma),
            Gamma=float(self.Gamma),
            Omega=float(self.Omega),
            delta1=float(self.Delta1),
            delta2=float(self.Delta2),
            eta=float(self.eta),
            V=float(self.V),
        )

    def with_OV(self, Omega: float, V: float) -> "PhysParams":
        """Return a copy with Omega and V overridden (for grid runs)."""
        return replace(self, Omega=float(Omega), V=float(V))


@dataclass
class SolveConfig:
    """Numerics/physics configuration for grid runs (uses unified PhysParams)."""
    params: PhysParams = PhysParams()
    # Integration settings
    t_min: float = 0.0
    t_max: float = 5000.0
    num_points: int = 2001
    rtol: float = 1e-8
    atol: float = 1e-8
    method: str = "RK45"


# =====================================================================
# =============== SU(3) tensor & symplectic helpers ===================
# =====================================================================

def su3_f_tensor() -> np.ndarray:
    """Return antisymmetric f_{abc} (0-based indices a,b,c in 0..7)."""
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

def C_from_x_fast(x8: np.ndarray) -> np.ndarray:
    """Build 8x8 purely imaginary C matrix from x[0..7] via SU(3) structure constants."""
    xc = np.real(np.asarray(x8, dtype=float))
    C_real = 2.0 * np.einsum('abc,c->ab', F_TENSOR, xc)
    return 1j * C_real  # purely imaginary

def compute_SJR_fast(sol, step: int = 1, compute_JR: bool = True) -> Dict[str, np.ndarray]:
    """
    Compute S(10x10) (with QP block), and J,R, and JRSRJ from the singles trajectory in 'sol'.
    Expects sol.y packed as [m1..m8, m9(Q), m10(P), pairs...].
    """
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

        # embed 8x8 block into 10x10 with canonical QP
        S10[k_eff, :8, :8] = C8
        S10[k_eff, 8, 9] = 1.0
        S10[k_eff, 9, 8] = -1.0

        if compute_JR:
            final_C, J_small, R_small = symplect.transform_complex_S(C8)
            J[k_eff]     = symplect.expand_to_10x10(J_small)
            R[k_eff]     = symplect.expand_to_10x10(R_small)
            JRSRJ[k_eff] = symplect.expand_to_10x10_sym(1j * final_C)
        else:
            J[k_eff]     = np.eye(10, dtype=complex)
            R[k_eff]     = np.eye(10, dtype=complex)
            JRSRJ[k_eff] = np.nan

    return {"t": t[idxs], "idxs": idxs, "S10": S10, "J": J, "R": R, "JRSRJ": JRSRJ}


# =====================================================================
# ==================== Singles & pairs ODE pipeline ===================
# =====================================================================

def build_pair_symbol_matrix(n: int) -> sp.Matrix:
    """Create symbol matrix M with entries m{i}m{j} (1-based naming)."""
    return sp.Matrix(n, n, lambda i, j: sp.symbols(f"m{i+1}m{j+1}"))

def list_pairs(M: sp.Matrix, symmetric_pairs: bool = True):
    """Return ordered (i,j,sym_ij) over upper triangle incl. diagonal if symmetric_pairs=True."""
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

def enforce_upper_triangle(M: sp.Matrix):
    """Replace lower m{j}m{i} by upper m{i}m{j} so Sigma_dt uses a unique naming."""
    n = M.shape[0]
    sub_map = {}
    for i in range(n):
        for j in range(i+1, n):
            upper = M[i, j]
            lower = sp.symbols(f"m{j+1}m{i+1}")
            sub_map[lower] = upper
    return sub_map

def build_rhs_lambdify_forgiving(rhs_pairs_list, y_syms, pairs_order, extra_params=()):
    """
    Robust lambdify that does name-based substitution of ANY symbols appearing in rhs_pairs_list.
    - Singles:  m1..m10     -> z[idx_of_single]
    - Pairs:    m{i}m{j}    -> z[idx_of_pair for (min(i,j),max(i,j)) in pairs_order]
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
        return s

    # Build substitution map from ALL symbols seen in the expressions
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

def singles_rhs_from_m_order(t, m_singles, params):
    """
    Adapter from m-order [x1..x8, Q, P] to rhs_gellmann_qp_from_x (expects [Q,P,x1..x8]).
    Returns derivatives in m-order: [dx1..dx8, dQ, dP].
    """
    x1_to_x8 = m_singles[0:8]
    Q = m_singles[8]
    P = m_singles[9]
    x_vec = np.array([Q, P, *x1_to_x8], dtype=complex)
    dQ_dP_dx = rhs_gellmann_qp_from_x(t, x_vec, params)
    dQ = dQ_dP_dx[0]
    dP = dQ_dP_dx[1]
    dx = dQ_dP_dx[2:10]  # dx1..dx8
    return np.array([*dx, dQ, dP], dtype=complex)


# =====================================================================
# ======================= Covariance utilities ========================
# =====================================================================

def initial_covariance_from_state(m0_singles, atom_scale=1.0, boson_scale=1.0):
    """
    Build Σ(0) for m = [x1..x8, Q, P] assuming:
      - Atomic |0> state: Var(x1,x2,x4,x5)=1, Var(x3,x6,x7,x8)=0 (scaled by atom_scale)
      - Bosonic vacuum: Var(Q)=Var(P)=1/2 (scaled by boson_scale)
      - No cross-correlations at t=0.
    Returns Σ as a (10x10) complex array (real-symmetric content).
    """
    Σ = np.zeros((10,10), dtype=complex)
    var_x = np.zeros(8, dtype=float)
    var_x[[0,1,3,4]] = 1.0  # x1,x2,x4,x5
    var_x *= atom_scale
    for i in range(8):
        Σ[i, i] = var_x[i]
    Σ[8, 8] = 0.5 * boson_scale   # Var(Q)
    Σ[9, 9] = 0.5 * boson_scale   # Var(P)
    Σ[8, 9] = 0.0
    Σ[9, 8] = 0.0
    return Σ

def pack_upper_triangle_from_covariance(Σ, pairs_order) -> np.ndarray:
    """Pick the upper-triangle entries of Σ in the same order as pairs_order."""
    m0_pairs = []
    for (i, j, _sym) in pairs_order:
        m0_pairs.append(Σ[i, j])
    return np.asarray(m0_pairs, dtype=complex)

def reconstruct_sigma_series(sol, idx, take_real: bool = True) -> np.ndarray:
    """
    Reconstruct Σ(t_k) (10x10) from the packed pair entries in sol.y using idx['pairs_order'].
    Returns array of shape (nt,10,10). When take_real=True, returns real-symmetric covariances.
    """
    n = idx["singles_slice"].stop - idx["singles_slice"].start  # should be 10
    pair_start = idx["pairs_slice"].start
    pairs_order = idx["pairs_order"]
    nt = sol.y.shape[1]

    Sigmas = np.zeros((nt, n, n), dtype=complex)
    for k_loc, (i, j, _sym) in enumerate(pairs_order):
        row = pair_start + k_loc
        traj = sol.y[row, :]                       # length nt
        Sigmas[:, i, j] = traj
        Sigmas[:, j, i] = np.conjugate(traj)       # ensure Hermitian symmetry
    if take_real:
        Sigmas = np.real(Sigmas)
    return Sigmas


# =====================================================================
# ====================== Runner and I/O to HDF5 =======================
# =====================================================================

def substitute_params_in_matrix(M_sym: sp.Matrix, name_to_val: Dict[str, float]) -> sp.Matrix:
    """Substitute known physical parameter symbols by name, leaving m_i symbols intact."""
    subs_pairs = []
    for s in M_sym.free_symbols:
        val = name_to_val.get(s.name, None)
        if val is not None:
            subs_pairs.append((s, float(val)))
    if subs_pairs:
        return sp.simplify(M_sym.subs(subs_pairs, simultaneous=True))
    return M_sym

@dataclass
class RunResult:
    """Data container for a single (Omega, V) run."""
    Omega: float
    V: float
    t: np.ndarray
    singlets: np.ndarray            # (nt, 10), complex
    diag_sigma: np.ndarray          # (nt, 10), float
    corr_xi_Q: np.ndarray           # (nt, 8), float
    corr_xi_P: np.ndarray           # (nt, 8), float
    corr_x1to3_x7: np.ndarray       # (nt, 3), float
    corr_x1to3_x8: np.ndarray       # (nt, 3), float
    J: np.ndarray                   # (nt, 10, 10), complex
    R: np.ndarray                   # (nt, 10, 10), complex
    solver_success: bool
    solver_message: str

def _run_single_case(omega_val: float, v_val: float, cfg: SolveConfig) -> RunResult:
    """Run one (Omega, V) case and return all requested arrays."""
    # ---- Unified params (override Ω,V for this run) ----
    p = cfg.params.with_OV(omega_val, v_val)

    # --- Sigma_dt with symbolic params substituted ---
    _, _, _, _, _, _, _, Sigma_dt_sym, _, _ = covar.get_important_matricies_symbol()
    Sigma_dt_param = substitute_params_in_matrix(Sigma_dt_sym, p.to_covar_dict())

    # --- Prepare pair system lambdified RHS ---
    mP_syms = sp.symbols('m1:11')  # m1..m10; order [x1..x8, Q, P]
    idx, rhs_pairs_func = prepare_system_for_solve_ivp(
        mP_syms=mP_syms,
        Sigma_dt=Sigma_dt_param,
        symmetric_pairs=True,
        extra_params=()
    )

    # --- Singles RHS params from unified dataclass ---
    singles_params = p.to_singles_dict()

    # --- Initial conditions ---
    y0_ket = np.array([0+0j, 0+0j, 1+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j])
    x0 = convert_state(y0_ket)  # [Q,P,x1..x8]
    m0_singles = np.array([*x0[2:10], x0[0], x0[1]], dtype=complex)

    Σ0 = initial_covariance_from_state(m0_singles, atom_scale=1.0, boson_scale=1.0)
    m0_pairs = pack_upper_triangle_from_covariance(Σ0, idx["pairs_order"])
    y0 = np.concatenate([m0_singles, m0_pairs])

    # --- Build combined RHS ---
    fun = make_combined_rhs(idx, rhs_pairs_func, params_dict=singles_params, extra_param_values_tuple=())

    # --- Time grid ---
    t_eval = np.linspace(cfg.t_min, cfg.t_max, cfg.num_points)
    t_span = (cfg.t_min, cfg.t_max)

    # --- Solve ---
    sol = solve_ivp(fun, t_span, y0, t_eval=t_eval, method=cfg.method, rtol=cfg.rtol, atol=cfg.atol)

    if not sol.success:
        nt = len(t_eval)
        def cshape(*sh): return np.full(sh, np.nan, dtype=np.complex128)
        def rshape(*sh): return np.full(sh, np.nan, dtype=np.float64)
        return RunResult(
            Omega=float(omega_val), V=float(v_val), t=t_eval,
            singlets=cshape(nt, 10),
            diag_sigma=rshape(nt, 10),
            corr_xi_Q=rshape(nt, 8),
            corr_xi_P=rshape(nt, 8),
            corr_x1to3_x7=rshape(nt, 3),
            corr_x1to3_x8=rshape(nt, 3),
            J=cshape(nt, 10, 10),
            R=cshape(nt, 10, 10),
            solver_success=False,
            solver_message=str(sol.message)
        )

    # --- Reconstruct Σ(t), extract requested entries ---
    Sigmas = reconstruct_sigma_series(sol, idx, take_real=True)  # (nt,10,10), real
    diag_sigma = np.diagonal(Sigmas, axis1=1, axis2=2)  # (nt, 10)
    corr_xi_Q = Sigmas[:, 0:8, 8]     # (nt, 8)
    corr_xi_P = Sigmas[:, 0:8, 9]     # (nt, 8)
    corr_x1to3_x7 = Sigmas[:, 0:3, 6]  # (nt, 3)
    corr_x1to3_x8 = Sigmas[:, 0:3, 7]  # (nt, 3)

    # --- J, R per time step ---
    SJR = compute_SJR_fast(sol, step=1, compute_JR=True)
    J = SJR["J"]        # (nt,10,10) complex
    R = SJR["R"]        # (nt,10,10) complex
    if J.shape[0] != sol.t.size:
        raise RuntimeError("Längen-Mismatch zwischen J/R und t-Gitter.")

    s_slice = idx["singles_slice"]
    singlets = sol.y[s_slice, :].T  # (nt, 10) complex

    return RunResult(
        Omega=float(omega_val),
        V=float(v_val),
        t=sol.t.copy(),
        singlets=singlets.astype(np.complex128, copy=False),
        diag_sigma=diag_sigma.astype(np.float64, copy=False),
        corr_xi_Q=corr_xi_Q.astype(np.float64, copy=False),
        corr_xi_P=corr_xi_P.astype(np.float64, copy=False),
        corr_x1to3_x7=corr_x1to3_x7.astype(np.float64, copy=False),
        corr_x1to3_x8=corr_x1to3_x8.astype(np.float64, copy=False),
        J=J.astype(np.complex128, copy=False),
        R=R.astype(np.complex128, copy=False),
        solver_success=True,
        solver_message=str(sol.message)
    )

def make_combined_rhs(idx, rhs_pairs_func, params_dict, extra_param_values_tuple=()):
    """
    Returns fun(t,y) -> concatenated derivatives for singles+pairs.
    - Singles part uses your rhs via 'singles_rhs_from_m_order'.
    - Pairs part uses lambdified Sigma_dt (expects all state entries separately).
    """
    s_slice = idx["singles_slice"]

    def fun(t, y):
        y = y.astype(complex, copy=False)
        y_singles = y[s_slice]
        dy_singles = singles_rhs_from_m_order(t, y_singles, params_dict)  # complex (10,)
        args = tuple(y.tolist()) + tuple(extra_param_values_tuple)
        dy_pairs = np.asarray(rhs_pairs_func(*args)).astype(complex, copy=False)
        return np.concatenate([dy_singles, dy_pairs])

    return fun


# ---------------- HDF5 I/O ----------------

def _safe_group_name(omega_val: float, v_val: float) -> str:
    """Create a readable and HDF5-safe group name for a (Omega, V) run."""
    def fmt(x: float) -> str:
        s = f"{x:.6g}"
        s = s.replace("+", "").replace("-", "m").replace(".", "p")
        return s
    return f"Omega_{fmt(omega_val)}__V_{fmt(v_val)}"

def _write_dataset(g, name: str, data: np.ndarray):
    """Write dataset with mild compression and chunking."""
    g.create_dataset(name, data=data, compression="gzip", compression_opts=4, shuffle=True)

def save_results_to_h5(filepath: str, results: List[RunResult], cfg: SolveConfig):
    """Write all runs into one HDF5 file."""
    with h5py.File(filepath, "w") as h5:
        # File-level metadata
        h5.attrs["note"] = "Results of multi-(Omega,V) grid. All arrays time-major (nt, ...)."
        h5.attrs["singlets_order"] = "[x1..x8, Q, P]"
        h5.attrs["created_by"] = "multi_omega_v_unified.py"

        # Store unified params (base values) as attributes (stringified dict)
        h5.attrs["params_base"] = str(asdict(cfg.params))

        # Integration settings
        h5.attrs["t_min"] = float(cfg.t_min)
        h5.attrs["t_max"] = float(cfg.t_max)
        h5.attrs["num_points"] = int(cfg.num_points)
        h5.attrs["rtol"] = float(cfg.rtol)
        h5.attrs["atol"] = float(cfg.atol)
        h5.attrs["method"] = str(cfg.method)

        root = h5.create_group("runs")
        for r in results:
            grp = root.create_group(_safe_group_name(r.Omega, r.V))
            grp.attrs["Omega"] = r.Omega
            grp.attrs["V"] = r.V
            grp.attrs["solver_success"] = bool(r.solver_success)
            grp.attrs["solver_message"] = r.solver_message

            _write_dataset(grp, "t", r.t.astype(np.float64, copy=False))
            _write_dataset(grp, "singlets", r.singlets)              # complex
            _write_dataset(grp, "diag_sigma", r.diag_sigma)          # real
            _write_dataset(grp, "corr_xi_Q", r.corr_xi_Q)            # real
            _write_dataset(grp, "corr_xi_P", r.corr_xi_P)            # real
            _write_dataset(grp, "corr_x1to3_x7", r.corr_x1to3_x7)    # real
            _write_dataset(grp, "corr_x1to3_x8", r.corr_x1to3_x8)    # real
            _write_dataset(grp, "J", r.J)                            # complex
            _write_dataset(grp, "R", r.R)                            # complex


# ---------------- Public API ----------------

def run_grid(
    omega_values: Iterable[float],
    v_values: Iterable[float],
    outfile: str,
    cfg: Optional[SolveConfig] = None,
    max_workers: Optional[int] = None,
    use_threads: bool = False,
) -> str:
    """
    Run all (Omega, V) combinations in parallel and save to 'outfile' (HDF5).

    Parameters
    ----------
    omega_values : Iterable[float]
        Values for Omega to sweep.
    v_values : Iterable[float]
        Values for V to sweep.
    outfile : str
        Path to HDF5 output file.
    cfg : SolveConfig, optional
        Numerical/physical configuration (contains unified PhysParams).
    max_workers : int, optional
        Degree of parallelism. If None, the executors' default is used.
    use_threads : bool, default False
        If True, use threads. For CPU-bound workloads, processes are recommended.

    Returns
    -------
    str
        The absolute path to the written HDF5 file.
    """
    if cfg is None:
        cfg = SolveConfig()

    combos = [(float(om), float(vv)) for om, vv in product(omega_values, v_values)]
    if len(combos) == 0:
        print("Keine Kombinationen gefunden – nichts zu tun.")
        return outfile

    print(f"Starte {len(combos)} Läufe im Parallelbetrieb …")

    executor_cls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    results: List[RunResult] = []
    errors: List[Tuple[Tuple[float, float], str]] = []

    def _job(args):
        om, vv, cf = args
        return _run_single_case(om, vv, cf)

    with executor_cls(max_workers=max_workers) as ex:
        fut2key = {ex.submit(_job, (om, vv, cfg)): (om, vv) for om, vv in combos}
        for fut in as_completed(fut2key):
            om, vv = fut2key[fut]
            try:
                res = fut.result()
                results.append(res)
                status = "OK" if res.solver_success else "FEHLER"
                print(f"Fertig: Omega={om}, V={vv} → {status}")
            except Exception as e:
                msg = repr(e)
                errors.append(((om, vv), msg))
                print(f"Fehler bei Omega={om}, V={vv}: {msg}")

    results.sort(key=lambda r: (r.Omega, r.V))
    save_results_to_h5(outfile, results, cfg)

    if errors:
        print(f"Abgeschlossen mit {len(errors)} Fehler(n). Details sind im Konsolen-Log oben.")
    else:
        print("Alle Läufe erfolgreich abgeschlossen.")

    return outfile


# ---------------- Example main (does not auto-run grid) ----------------

if __name__ == "__main__":
    print("Dies ist nur ein Beispiel – passe Werte nach Bedarf an und rufe run_grid(...) auf.")
    # Example (commented out to avoid accidental long runs):
    # cfg = SolveConfig(params=PhysParams(gamma=1.0, Gamma=2.0, kappa=1.0, eta=1.0))
    # outpath = "grid_results_unified.h5"
    # run_grid(omega_values=[8.0, 12.0], v_values=[-2.0, -4.0], outfile=outpath, cfg=cfg, max_workers=None, use_threads=False)
