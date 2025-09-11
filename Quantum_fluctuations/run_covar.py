import covar_everything as covar 
import numpy as np
import sympy as sp
g0, Delta1, V, gamma, Omega, kappa, eta, Delta2 = sp.symbols("g0 Delta1 V gamma Omega kappa eta Delta2")
numeric_params = {
    g0:     1,
    Delta1: 1,
    Delta2: 1,
    V:      -4.0,
    gamma:  1,
    Omega:  8,
    kappa:  1,
    eta: 1

}
lambda_values= [1,5, 0, 5, 5, 5, 8, 4]
G,sDs,Z,P,Q,Z_prime,W,Sigma_dt,Sigma,K=covar.get_important_matricies(numeric_params)


JRsRJ,J,R=covar.get_traffo(lambda_values)
#sp.pprint(sp.Matrix(JRsRJ))
#print("hellow")
#sp.pprint(sp.simplify(sp.Matrix(Z_prime/2+Z_prime.T/2)))
mP = sp.symbols('m1:11')
#sp.pprint(Sigma_dt[5,5])

#sp.pprint(Sigma_dt)
#calculate initial state for fluctuations

def build_pair_symbol_matrix(n: int):
    """Create the symbol matrix M with entries m{i}m{j} following your naming scheme."""
    M = sp.Matrix(n, n, lambda i, j: sp.symbols(f"m{i+1}m{j+1}"))
    return M

def list_pairs(M, symmetric_pairs: bool = True):
    """Return a list of (i,j,sym) entries in a consistent order for state vector packing."""
    n = M.shape[0]
    pairs = []
    if symmetric_pairs:
        for i in range(n):
            for j in range(i, n):  # upper triangle incl. diagonal
                pairs.append((i, j, M[i, j]))
    else:
        for i in range(n):
            for j in range(n):
                pairs.append((i, j, M[i, j]))
    return pairs

def build_pair_equations(Sigma_dt: sp.Matrix, M: sp.Matrix, symmetric_pairs: bool = True):
    """
    Create element-wise ODE equations for pairs:
      dm{i}m{j}_dt = Sigma_dt[i,j]
    Returns a list of sympy.Eq and a matching RHS list in the same order as 'pairs'.
    """
    assert Sigma_dt.shape == M.shape, "Sigma_dt and M must have same shape"
    eqs = []
    rhs_list = []
    pairs = list_pairs(M, symmetric_pairs=symmetric_pairs)

    for (i, j, sym_ij) in pairs:
        d_ij = sp.symbols(f"d{sym_ij.name}_dt")  # e.g. dm1m3_dt
        eqs.append(sp.Eq(d_ij, sp.simplify(Sigma_dt[i, j])))
        rhs_list.append(sp.simplify(Sigma_dt[i, j]))
    return eqs, rhs_list, pairs

def pack_state_symbols(mP, M: sp.Matrix, symmetric_pairs: bool = True):
    """
    Build a flat state vector symbol list y = [m1,...,mn, then pairs in chosen order].
    Returns y_syms (list) and a mapping dict {symbol_in_formula -> symbol_in_y}.
    """
    y_syms = list(mP)  # singles first (m1..mn)
    mapping = {}

    # Singles map to themselves
    for s in mP:
        mapping[s] = s

    # Pairs next
    pairs = list_pairs(M, symmetric_pairs=symmetric_pairs)
    for (_, _, sym_ij) in pairs:
        y_syms.append(sym_ij)
        mapping[sym_ij] = sym_ij

    return y_syms, mapping

def build_rhs_lambdify(rhs_pairs_list, y_syms, extra_params=()):
    """
    Create a NumPy-callable RHS function for the pair part:
       f(y, *params) -> dy_pairs
    where y packs [m singles..., M pairs...].
    If your Sigma_dt depends on parameters (e.g. gamma, kappa), pass them in 'extra_params'.
    """
    # Build a dummy symbol vector z0: z0,...,z{len(y_syms)-1} to lambdify compactly
    z = sp.symbols(f'z0:{len(y_syms)}')
    # Map each original state symbol to z[k]
    subs_map = {sym: z[k] for k, sym in enumerate(y_syms)}
    # Substitute into RHS
    rhs_z = [expr.xreplace(subs_map) for expr in rhs_pairs_list]

    # Lambdify: arguments = (z0, z1, ..., zN-1, *extra_params)
    args = list(z) + list(extra_params)
    f = sp.lambdify(args, rhs_z, modules='numpy')
    return f


def prepare_system_from_Sigma_dt(mP, Sigma_dt, symmetric_pairs=True, extra_params=()):
    """
    Full preparation pipeline:
      - build M symbols matching Sigma_dt size
      - create element-wise equations for pairs
      - create consistent state packing (singles then pairs)
      - build a RHS function (NumPy) for the pair derivatives
    Returns dict with:
      'M', 'pair_eqs', 'pairs_order', 'state_symbols', 'rhs_pairs_func'
    """
    n = len(mP)
    assert Sigma_dt.shape == (n, n), "Sigma_dt must be n×n with n=len(mP)"

    M = build_pair_symbol_matrix(n)
    pair_eqs, rhs_pairs_list, pairs = build_pair_equations(Sigma_dt, M, symmetric_pairs=symmetric_pairs)
    y_syms, _ = pack_state_symbols(mP, M, symmetric_pairs=symmetric_pairs)
    rhs_pairs_func = build_rhs_lambdify(rhs_pairs_list, y_syms, extra_params=extra_params)

    return {
        "M": M,
        "pair_eqs": pair_eqs,           # list of sympy.Eq: d(mimj)/dt = ...
        "pairs_order": pairs,           # ordering of pairs used in state packing
        "state_symbols": y_syms,        # [m1..mn, then pairs...]
        "rhs_pairs_func": rhs_pairs_func  # callable: f(*y, *params) -> d(pairs)/dt
    }

def enforce_upper_triangle(M):
    """
    Build a substitution map that replaces lower-triangle symbols m{j}m{i} by m{i}m{j}.
    """
    n = M.shape[0]
    sub_map = {}
    for i in range(n):
        for j in range(i+1, n):
            upper = M[i, j]           # m{i+1}m{j+1}
            lower = sp.symbols(f"m{j+1}m{i+1}")  # m{j+1}m{i+1}
            sub_map[lower] = upper
    return sub_map
# Python (comments in English; console/output in German)
import sympy as sp

def eq_index_by_symbol(system):
    """Map LHS symbols (e.g. dm3m7_dt) and RHS pair symbols (e.g. m3m7) to equations."""
    idx = {}
    for eq in system["pair_eqs"]:
        lhs = eq.lhs                          # e.g. dm3m7_dt
        rhs = eq.rhs
        idx[lhs] = eq
        # Also map from the corresponding pair symbol name if present in LHS name
        # Example: dm3m7_dt  --> key "m3m7"
        name = str(lhs)
        if name.startswith("d") and name.endswith("_dt"):
            core = name[1:-3]                 # drop leading 'd' and trailing '_dt'
            idx[core] = eq
    return idx

def eq_index_by_ij(system):
    """Map (i,j) tuple from pairs_order to equation."""
    pairs = system["pairs_order"]             # list of (i, j, m_im_j)
    eqs   = system["pair_eqs"]
    return { (i,j): eq for ( (i,j,_), eq ) in zip(pairs, eqs) }

def show_equation(system, key):
    """
    key can be:
      - a sympy.Symbol on the LHS (e.g., sp.symbols('dm3m7_dt'))
      - a string 'm3m7' (pair symbol core)
      - a (i,j) tuple (0-based indices)
    Prints the matching equation.
    """
    # Try by LHS symbol / 'mimj' string

    idx_sym = eq_index_by_symbol(system)
    if isinstance(key, sp.Basic) or isinstance(key, str):
        eq = idx_sym.get(key, None)
        if eq is not None:
            sp.pprint(eq)    # pretty print in console
            return eq

    # Try by (i,j)
    if isinstance(key, tuple) and len(key) == 2:
        idx_ij = eq_index_by_ij(system)
        eq = idx_ij.get(key, None)
        if eq is not None:
            sp.pprint(eq)
            return eq

    print("Keine Gleichung gefunden für:", key)
    return None

def equations_in_row(system, i):
    """All equations for row i (0-based) according to pairs_order (useful for symmetric upper-triangle)."""
    out = []
    for (ii, jj, _), eq in zip(system["pairs_order"], system["pair_eqs"]):
        if ii == i:
            out.append(eq)
    return out

def equations_with_symbol(system, sym_substring="m3"):
    """
    Filter equations whose RHS involves any symbol containing 'sym_substring'
    (e.g. 'm3', 'm10', 'm3m7').
    """
    res = []
    for eq in system["pair_eqs"]:
        # string-based filter is simple & robust
        if sym_substring in str(eq.rhs):
            res.append(eq)
    return res


sub_map = enforce_upper_triangle(Sigma_dt)
Sigma_dt_clean = Sigma_dt.xreplace(sub_map)
#system = prepare_system_from_Sigma_dt(mP, Sigma_dt, symmetric_pairs=True, extra_params=(gamma, kappa))
system = prepare_system_from_Sigma_dt(mP, Sigma_dt_clean, symmetric_pairs=True, extra_params=(gamma, kappa))
y_syms = system["state_symbols"]
f_pairs = system["rhs_pairs_func"]

# show_equation(system, (2,6))
# print("---------------------")
# sp.pprint(Sigma_dt[2,6])