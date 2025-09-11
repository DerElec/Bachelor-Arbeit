# Python 3.x / SymPy
import sympy as sp

def build_gellmann():
    """
    Return the 8 standard Gell-Mann matrices (3x3) with tr(λ_a λ_b) = 2 δ_ab.
    """
    I = sp.I
    sqrt3 = sp.sqrt(3)

    lambdas = [
        sp.Matrix([[0, 1, 0],
                   [1, 0, 0],
                   [0, 0, 0]]),
        sp.Matrix([[0, -I, 0],
                   [I,  0, 0],
                   [0,  0, 0]]),
        sp.Matrix([[1,  0, 0],
                   [0, -1, 0],
                   [0,  0, 0]]),
        sp.Matrix([[0, 0, 1],
                   [0, 0, 0],
                   [1, 0, 0]]),
        sp.Matrix([[0,  0, -I],
                   [0,  0,  0],
                   [I,  0,  0]]),
        sp.Matrix([[0, 0, 0],
                   [0, 0, 1],
                   [0, 1, 0]]),
        sp.Matrix([[0,  0,  0],
                   [0,  0, -I],
                   [0,  I,  0]]),
        (1/sqrt3)*sp.Matrix([[ 1, 0, 0],
                             [ 0, 1, 0],
                             [ 0, 0,-2]])
    ]
    return lambdas


def structure_constants_from_lambdas(lambdas):
    """
    Compute SU(3) structure constants f^{abc} and d^{abc} from the Gell-Mann matrices.

    Conventions:
      [λ_a, λ_b] = 2i f^{abc} λ_c
      {λ_a, λ_b} = (4/3) δ_{ab} I + 2 d^{abc} λ_c
    Trace formulas used:
      f^{abc} = (1 / (4 i)) tr([λ_a, λ_b] λ_c)
      d^{abc} = (1 / 4)     tr({λ_a, λ_b} λ_c)
    """
    PHYS_DIM = 8
    I = sp.I

    f = [[[0]*PHYS_DIM for _ in range(PHYS_DIM)] for __ in range(PHYS_DIM)]
    d = [[[0]*PHYS_DIM for _ in range(PHYS_DIM)] for __ in range(PHYS_DIM)]

    for a in range(PHYS_DIM):
        for b in range(PHYS_DIM):
            comm   = lambdas[a]*lambdas[b] - lambdas[b]*lambdas[a]
            anticom= lambdas[a]*lambdas[b] + lambdas[b]*lambdas[a]
            for c in range(PHYS_DIM):
                f_val = sp.simplify(sp.trace(comm * lambdas[c])/(4*I))
                d_val = sp.simplify(sp.trace(anticom * lambdas[c])/4)
                f[a][b][c] = f_val
                d[a][b][c] = d_val
    return f, d


def compute_R_from_f_d(f_arr, d_arr, gamma=None, simplify_entries=False):
    """
    Build the 8x8 diffusion matrix R_ab using *numeric* structure constants f,d
    (as nested Python lists with SymPy numbers), following:

      R_{ab} = γ ∑_{c,d} [ (2/3) δ_{cd} + ∑_{k} m_k ( d^{cdk} + i f^{cdk} ) ]
                        * ( f^{1ac} - i f^{2ac} ) * ( f^{1bd} + i f^{2bd} )

    Parameters
    ----------
    f_arr, d_arr : list[list[list]]
        8x8x8 arrays with SymPy numbers (structure constants).
    gamma : sympy.Symbol or None
        If None, a fresh symbol 'gamma' is created.
    simplify_entries : bool
        If True, apply `sympy.simplify` elementwise (may be slow).

    Returns
    -------
    R : sympy.Matrix (8x8)
    syms : dict with
        'gamma'  : the gamma symbol,
        'm'      : tuple (m1,...,m8),
        'f', 'd' : the input arrays (for convenience).
    """
    I = sp.I
    PHYS_DIM = 8
    if gamma is None:
        gamma = sp.symbols('gamma', real=True)
    m = sp.symbols('m1:9', complex=True)  # (m1,...,m8)

    R = sp.MutableDenseMatrix(PHYS_DIM, PHYS_DIM, [0]*(PHYS_DIM*PHYS_DIM))

    # Physics-style 1-based indices in the formula -> adjust by -1 in Python lists.
    for a in range(1, PHYS_DIM+1):
        for b in range(1, PHYS_DIM+1):
            expr = 0
            for c in range(1, PHYS_DIM+1):
                U_ac = f_arr[0][a-1][c-1] - I * f_arr[1][a-1][c-1]      # f^{1ac} - i f^{2ac}
                for d in range(1, PHYS_DIM+1):
                    V_bd = f_arr[0][b-1][d-1] + I * f_arr[1][b-1][d-1]  # f^{1bd} + i f^{2bd}

                    # T_{cd} = (2/3) δ_{cd} + Σ_k m_k (d^{cdk} + i f^{cdk})
                    delta_cd = 1 if c == d else 0
                    T_cd = sp.Rational(2, 3)*delta_cd
                    T_cd += sum(m[k-1]*(d_arr[c-1][d-1][k-1] + I*f_arr[c-1][d-1][k-1])
                                for k in range(1, PHYS_DIM+1))

                    expr += T_cd * U_ac * V_bd

            R[a-1, b-1] = gamma * (sp.simplify(expr) if simplify_entries else expr)

    return sp.Matrix(R), {'gamma': gamma, 'm': m, 'f': f_arr, 'd': d_arr}


def compute_R_su3(gamma=None, simplify_entries=False):
    """
    Convenience wrapper:
      - builds Gell-Mann matrices,
      - computes numeric f,d,
      - returns R_ab.

    Returns
    -------
    R : sympy.Matrix (8x8)
    data : dict with 'gamma','m','f','d','lambdas'
    """
    lambdas = build_gellmann()
    f_arr, d_arr = structure_constants_from_lambdas(lambdas)
    R, syms = compute_R_from_f_d(f_arr, d_arr, gamma=gamma, simplify_entries=simplify_entries)
    syms.update({'lambdas': lambdas})
    return R, syms


# --- Example usage (uncomment to test) ---
if __name__ == "__main__":
    R, syms = compute_R_su3(gamma=None, simplify_entries=False)
    sp.pprint(sp.simplify((R+R.T)/2))
