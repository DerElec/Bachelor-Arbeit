
# -*- coding: utf-8 -*-
"""
Unified-params multi-(Omega, V) parallel runner (FIXED for ProcessPool pickling):
- Moves the job function to module scope to avoid "Can't get local object 'run_grid.<locals>._job'".
- Code & comments in English; console/output in German.
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

# ======================== Unified parameters =========================
@dataclass(frozen=True)
class PhysParams:
    g0: float = 1.0
    Delta1: float = 1.0
    Delta2: float = 1.0
    gamma: float = 1.0
    kappa: float = 1.0
    eta: float = 1.0
    Omega: float = 8.0
    V: float = -1.0
    Gamma: float = 2.0

    def to_covar_dict(self) -> Dict[str, float]:
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
        return replace(self, Omega=float(Omega), V=float(V))

@dataclass
class SolveConfig:
    params: PhysParams = PhysParams()
    t_min: float = 0.0
    t_max: float = 5000.0
    num_points: int = 2001
    rtol: float = 1e-8
    atol: float = 1e-8
    method: str = "RK45"

# =============== SU(3) tensor & symplectic helpers ===================
def su3_f_tensor() -> np.ndarray:
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
    xc = np.real(np.asarray(x8, dtype=float))
    C_real = 2.0 * np.einsum('abc,c->ab', F_TENSOR, xc)
    return 1j * C_real

def compute_SJR_fast(sol, step: int = 1, compute_JR: bool = True) -> Dict[str, np.ndarray]:
    t = sol.t
    idxs = np.arange(0, t.size, step, dtype=int)
    nt_eff = idxs.size
    S10 = np.zeros((nt_eff, 10, 10), dtype=complex)
    J   = np.zeros_like(S10)
    R   = np.zeros_like(S10)
    JRSRJ = np.zeros_like(S10)
    for k_eff, k in enumerate(idxs):
        x8 = sol.y[0:8, k]
        C8 = C_from_x_fast(x8)
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

# ==================== Singles & pairs ODE pipeline ===================
def build_pair_symbol_matrix(n: int) -> sp.Matrix:
    return sp.Matrix(n, n, lambda i, j: sp.symbols(f"m{i+1}m{j+1}"))

def list_pairs(M: sp.Matrix, symmetric_pairs: bool = True):
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
    y_syms = list(mP_syms)
    pairs = list_pairs(M, symmetric_pairs=symmetric_pairs)
    for (_, _, sym_ij) in pairs:
        y_syms.append(sym_ij)
    return y_syms, pairs

def enforce_upper_triangle(M: sp.Matrix):
    n = M.shape[0]
    sub_map = {}
    for i in range(n):
        for j in range(i+1, n):
            upper = M[i, j]
            lower = sp.symbols(f"m{j+1}m{i+1}")
            sub_map[lower] = upper
    return sub_map

def build_rhs_lambdify_forgiving(rhs_pairs_list, y_syms, pairs_order, extra_params=()):
    import re
    z = sp.symbols(f"z0:{len(y_syms)}")
    singles_len = len(y_syms) - len(pairs_order)
    pair_idx = { (i, j): k for k, (i, j, _) in enumerate(pairs_order) }
    re_single = re.compile(r"^m(\d+)$")
    re_pair   = re.compile(r"^m(\d+)m(\d+)$")
    def symbol_to_z(s: sp.Symbol):
        name = s.name
        m = re_single.match(name)
        if m:
            i = int(m.group(1));  return z[i - 1]
        m = re_pair.match(name)
        if m:
            i = int(m.group(1)) - 1; j = int(m.group(2)) - 1
            ii, jj = (i, j) if i <= j else (j, i)
            k_local = pair_idx[(ii, jj)]
            return z[singles_len + k_local]
        return s
    subs_map = {}
    for expr in rhs_pairs_list:
        for s in expr.atoms(sp.Symbol):
            mapped = symbol_to_z(s)
            if mapped is not s:
                subs_map[s] = mapped
    rhs_z = [sp.simplify(expr.subs(subs_map, simultaneous=True)) for expr in rhs_pairs_list]
    args = list(z) + list(extra_params)
    return sp.lambdify(args, rhs_z, modules='numpy')

def prepare_system_for_solve_ivp(mP_syms, Sigma_dt, symmetric_pairs=True, extra_params=()):
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
    x1_to_x8 = m_singles[0:8]; Q = m_singles[8]; P = m_singles[9]
    x_vec = np.array([Q, P, *x1_to_x8], dtype=complex)
    dQ_dP_dx = rhs_gellmann_qp_from_x(t, x_vec, params)
    dQ = dQ_dP_dx[0]; dP = dQ_dP_dx[1]; dx = dQ_dP_dx[2:10]
    return np.array([*dx, dQ, dP], dtype=complex)

# ======================= Covariance utilities ========================
def initial_covariance_from_state(m0_singles, atom_scale=1.0, boson_scale=1.0):
    Σ = np.zeros((10,10), dtype=complex)
    var_x = np.zeros(8, dtype=float); var_x[[0,1,3,4]] = 1.0; var_x *= atom_scale
    for i in range(8): Σ[i, i] = var_x[i]
    Σ[8, 8] = 0.5 * boson_scale; Σ[9, 9] = 0.5 * boson_scale
    Σ[8, 9] = 0.0; Σ[9, 8] = 0.0
    return Σ

def pack_upper_triangle_from_covariance(Σ, pairs_order) -> np.ndarray:
    m0_pairs = []
    for (i, j, _sym) in pairs_order: m0_pairs.append(Σ[i, j])
    return np.asarray(m0_pairs, dtype=complex)

def reconstruct_sigma_series(sol, idx, take_real: bool = True) -> np.ndarray:
    n = idx["singles_slice"].stop - idx["singles_slice"].start
    pair_start = idx["pairs_slice"].start
    pairs_order = idx["pairs_order"]
    nt = sol.y.shape[1]
    Sigmas = np.zeros((nt, n, n), dtype=complex)
    for k_loc, (i, j, _sym) in enumerate(pairs_order):
        row = pair_start + k_loc
        traj = sol.y[row, :]
        Sigmas[:, i, j] = traj
        Sigmas[:, j, i] = np.conjugate(traj)
    if take_real: Sigmas = np.real(Sigmas)
    return Sigmas

# ====================== Runner and I/O to HDF5 =======================
def substitute_params_in_matrix(M_sym: sp.Matrix, name_to_val: Dict[str, float]) -> sp.Matrix:
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
    Omega: float; V: float; t: np.ndarray
    singlets: np.ndarray; diag_sigma: np.ndarray
    corr_xi_Q: np.ndarray; corr_xi_P: np.ndarray
    corr_x1to3_x7: np.ndarray; corr_x1to3_x8: np.ndarray
    J: np.ndarray; R: np.ndarray
    solver_success: bool; solver_message: str

def _run_single_case(omega_val: float, v_val: float, cfg: SolveConfig) -> RunResult:
    p = cfg.params.with_OV(omega_val, v_val)
    _, _, _, _, _, _, _, Sigma_dt_sym, _, _ = covar.get_important_matricies_symbol()
    Sigma_dt_param = substitute_params_in_matrix(Sigma_dt_sym, p.to_covar_dict())
    mP_syms = sp.symbols('m1:11')
    idx, rhs_pairs_func = prepare_system_for_solve_ivp(mP_syms=mP_syms, Sigma_dt=Sigma_dt_param,
                                                       symmetric_pairs=True, extra_params=())
    singles_params = p.to_singles_dict()
    y0_ket = np.array([0+0j, 0+0j, 1+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j, 0+0j])
    x0 = convert_state(y0_ket)
    m0_singles = np.array([*x0[2:10], x0[0], x0[1]], dtype=complex)
    Σ0 = initial_covariance_from_state(m0_singles, atom_scale=1.0, boson_scale=1.0)
    m0_pairs = pack_upper_triangle_from_covariance(Σ0, idx["pairs_order"])
    y0 = np.concatenate([m0_singles, m0_pairs])
    t_eval = np.linspace(cfg.t_min, cfg.t_max, cfg.num_points)
    sol = solve_ivp(
        make_combined_rhs(idx, rhs_pairs_func, params_dict=singles_params, extra_param_values_tuple=()),
        (cfg.t_min, cfg.t_max), y0, t_eval=t_eval, method=cfg.method, rtol=cfg.rtol, atol=cfg.atol
    )
    if not sol.success:
        nt = len(t_eval)
        def cshape(*sh): return np.full(sh, np.nan, dtype=np.complex128)
        def rshape(*sh): return np.full(sh, np.nan, dtype=np.float64)
        return RunResult(float(omega_val), float(v_val), t_eval,
                         cshape(nt,10), rshape(nt,10), rshape(nt,8), rshape(nt,8),
                         rshape(nt,3), rshape(nt,3), cshape(nt,10,10), cshape(nt,10,10),
                         False, str(sol.message))
    Sigmas = reconstruct_sigma_series(sol, idx, take_real=True)
    diag_sigma = np.diagonal(Sigmas, axis1=1, axis2=2)
    corr_xi_Q = Sigmas[:, 0:8, 8]; corr_xi_P = Sigmas[:, 0:8, 9]
    corr_x1to3_x7 = Sigmas[:, 0:3, 6]; corr_x1to3_x8 = Sigmas[:, 0:3, 7]
    SJR = compute_SJR_fast(sol, step=1, compute_JR=True)
    J = SJR["J"]; R = SJR["R"]
    singlets = sol.y[idx["singles_slice"], :].T
    return RunResult(float(omega_val), float(v_val), sol.t.copy(),
                     singlets.astype(np.complex128, copy=False),
                     diag_sigma.astype(np.float64, copy=False),
                     corr_xi_Q.astype(np.float64, copy=False),
                     corr_xi_P.astype(np.float64, copy=False),
                     corr_x1to3_x7.astype(np.float64, copy=False),
                     corr_x1to3_x8.astype(np.float64, copy=False),
                     J.astype(np.complex128, copy=False), R.astype(np.complex128, copy=False),
                     True, str(sol.message))

def make_combined_rhs(idx, rhs_pairs_func, params_dict, extra_param_values_tuple=()):
    s_slice = idx["singles_slice"]
    def fun(t, y):
        y = y.astype(complex, copy=False)
        dy_singles = singles_rhs_from_m_order(t, y[s_slice], params_dict)
        dy_pairs = np.asarray(rhs_pairs_func(*tuple(y.tolist()))).astype(complex, copy=False)
        return np.concatenate([dy_singles, dy_pairs])
    return fun

def _safe_group_name(omega_val: float, v_val: float) -> str:
    def fmt(x: float) -> str:
        s = f"{x:.6g}"; s = s.replace("+","").replace("-","m").replace(".","p")
        return s
    return f"Omega_{fmt(omega_val)}__V_{fmt(v_val)}"

def _write_dataset(g, name: str, data: np.ndarray):
    g.create_dataset(name, data=data, compression="gzip", compression_opts=4, shuffle=True)

def save_results_to_h5(filepath: str, results: List[RunResult], cfg) -> None:
    with h5py.File(filepath, "w") as h5:
        h5.attrs["note"] = "Results of multi-(Omega,V) grid. All arrays time-major (nt, ...)."
        h5.attrs["singlets_order"] = "[x1..x8, Q, P]"
        h5.attrs["created_by"] = "multi_omega_v_unified_fixed.py"
        h5.attrs["params_base"] = str(asdict(cfg.params))
        h5.attrs["t_min"] = float(cfg.t_min); h5.attrs["t_max"] = float(cfg.t_max)
        h5.attrs["num_points"] = int(cfg.num_points)
        h5.attrs["rtol"] = float(cfg.rtol); h5.attrs["atol"] = float(cfg.atol)
        h5.attrs["method"] = str(cfg.method)
        root = h5.create_group("runs")
        for r in results:
            grp = root.create_group(_safe_group_name(r.Omega, r.V))
            grp.attrs["Omega"] = r.Omega; grp.attrs["V"] = r.V
            grp.attrs["solver_success"] = bool(r.solver_success)
            grp.attrs["solver_message"] = r.solver_message
            _write_dataset(grp, "t", r.t.astype(np.float64, copy=False))
            _write_dataset(grp, "singlets", r.singlets)
            _write_dataset(grp, "diag_sigma", r.diag_sigma)
            _write_dataset(grp, "corr_xi_Q", r.corr_xi_Q)
            _write_dataset(grp, "corr_xi_P", r.corr_xi_P)
            _write_dataset(grp, "corr_x1to3_x7", r.corr_x1to3_x7)
            _write_dataset(grp, "corr_x1to3_x8", r.corr_x1to3_x8)
            _write_dataset(grp, "J", r.J); _write_dataset(grp, "R", r.R)

# ---- MODULE-LEVEL job function to make ProcessPool picklable ----
def _job_entry(om: float, vv: float, cfg) -> RunResult:
    return _run_single_case(om, vv, cfg)

def run_grid(
    omega_values: Iterable[float],
    v_values: Iterable[float],
    outfile: str,
    cfg: Optional[SolveConfig] = None,
    max_workers: Optional[int] = None,
    use_threads: bool = False,
) -> str:
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
    with executor_cls(max_workers=max_workers) as ex:
        fut2key = {ex.submit(_job_entry, om, vv, cfg): (om, vv) for (om, vv) in combos}
        for fut in as_completed(fut2key):
            om, vv = fut2key[fut]
            try:
                res = fut.result()
                results.append(res)
                status = "OK" if res.solver_success else "FEHLER"
                print(f"Fertig: Omega={om}, V={vv} → {status}")
            except Exception as e:
                msg = repr(e); errors.append(((om, vv), msg))
                print(f"Fehler bei Omega={om}, V={vv}: {msg}")
    results.sort(key=lambda r: (r.Omega, r.V))
    save_results_to_h5(outfile, results, cfg)
    if errors:
        print(f"Abgeschlossen mit {len(errors)} Fehler(n). Details sind im Konsolen-Log oben.")
    else:
        print("Alle Läufe erfolgreich abgeschlossen.")
    return outfile

if __name__ == "__main__":
    # Minimal smoke-test configuration; adjust or remove
    print("Beispiel: bitte run_grid(...) mit deinem Grid aufrufen.")
