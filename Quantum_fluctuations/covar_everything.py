import sympy as sp
import structured_all_matricies as struc_matr
import symbolic_gellman_dgl as sgdgl
import sympy as sp
import symplectic_matrix as symplect
import numpy as np
import skew_matrix_example as skew

def input_numerics(numeric_params, *matrices):
    """
    Substitute all non-'m' symbols in multiple SymPy matrices.

    Parameters
    ----------
    numeric_params : dict
        Mapping from SymPy Symbol objects to numerical values.
        Must NOT contain the 'm...' parameters that should remain symbolic.
    *matrices : sympy.Matrix
        One or more SymPy matrices to be processed.

    Returns
    -------
    list[sympy.Matrix]
        A list of matrices with the requested substitutions applied,
        in the same order in which the matrices were passed in.
    """

    # --- 1) Collect every free symbol that appears in any matrix -------------
    all_syms = set().union(*(M.free_symbols for M in matrices))

    # --- 2) Build the substitution dictionary once ---------------------------
    subs_dict = {
        sym: numeric_params[sym]           # take the numeric value
        for sym in all_syms
        if not sym.name.startswith("m")    # skip 'm…' parameters
           and sym in numeric_params       # skip symbols without a given value
    }

    # --- 3) Apply substitutions to every matrix ------------------------------
    return [M.subs(subs_dict) for M in matrices]


def replace_lambda_q_p(*objects):
    """
    Substitute lambda1–lambda8 → m1–m8, q → m9, and p → m10
    in one or more SymPy expressions / matrices.

    Parameters
    ----------
    *objects : sympy.Expr | sympy.Matrix
        Any number of SymPy expressions or matrices.

    Returns
    -------
    sympy.Expr | sympy.Matrix | list
        If exactly one object is provided, the substituted object is returned.
        Otherwise, a list with all substituted objects (in the original order)
        is returned.
    """

    # --- build substitution dictionary once -------------------------------
    lambdas = sp.symbols("lambda1:9")   # (lambda1, ..., lambda8)
    ms      = sp.symbols("m1:9")        # (m1, ..., m8)
    q, p    = sp.symbols("q p")
    m9, m10 = sp.symbols("m9 m10")

    subs_dict = {lam: m for lam, m in zip(lambdas, ms)}
    subs_dict.update({q: m9, p: m10})

    # --- apply substitutions ----------------------------------------------
    replaced = [obj.subs(subs_dict) for obj in objects]

    # --- return single object or list --------------------------------------
    return replaced[0] if len(replaced) == 1 else replaced








# s=transform_complex_S(s0)
# sp.pprint(s)
def get_traffo(lambda_vals):
    C_num_sympy = skew.get_C_matrix(lambda_vals)
    C_complex = np.array(C_num_sympy.tolist(), dtype=complex)

    # Rufe die finale Funktion auf
    final_C_real, J, R = skew.transform_complex_C(C_complex)


    final_C_complex = final_C_real * 1j
    return final_C_complex,J,R

def get_important_matricies(numeric_params):
    #dRtsRdt, R0,R,st,s_st,s_bt,s0,s_s0,s_b0,D_SS, D_SB, D_BS, D_BB,D_simplified,B,dxdt=sgdgl.create_all()
    G,sDs,Z,P,Q,Z_prime,W=struc_matr.run_all()
    # --- 1. Define non-commuting symbols for the operators ----------
    x_ops = sp.symbols('m1:9', commutative=False)  # x1 … x8
    q, p = sp.symbols('m9 m10', commutative=False)    # q, p
    F_ops = list(x_ops) + [q, p]                   # full vector F

    # --- 2. Build the 10 × 10 symbolic matrix K --------------------
    n = len(F_ops)
    # Helper: create a unique symbol ⟨A B⟩ for every pair (A,B)
    def second_moment_symbol(A, B):
        return sp.Symbol(f'{A}{B}', real=True)   # expectation value ⟨A B⟩

    K = sp.Matrix([[second_moment_symbol(F_ops[i], F_ops[j])
                    for j in range(n)] for i in range(n)])

    # --- 3. Obtain the symmetrised covariance matrix Σ -------------
    Sigma = (K + K.T) / 2



    term1= G @ Sigma
    term2=Sigma @ G.T
    term3=W
    Sigma_dt=term1+term2+term3

    #  alternativ: symbols_in_M = M.atoms(sp.Symbol)


    return input_numerics(numeric_params,G,sDs,Z,P,Q,Z_prime,W,Sigma_dt,Sigma,K)




def get_important_matricies_symbol():
    #dRtsRdt, R0,R,st,s_st,s_bt,s0,s_s0,s_b0,D_SS, D_SB, D_BS, D_BB,D_simplified,B,dxdt=sgdgl.create_all()
    G,sDs,Z,P,Q,Z_prime,W=struc_matr.run_all()
    # --- 1. Define non-commuting symbols for the operators ----------
    x_ops = sp.symbols('m1:9', commutative=False)  # x1 … x8
    q, p = sp.symbols('m9 m10', commutative=False)    # q, p
    F_ops = list(x_ops) + [q, p]                   # full vector F

    # --- 2. Build the 10 × 10 symbolic matrix K --------------------
    n = len(F_ops)
    # Helper: create a unique symbol ⟨A B⟩ for every pair (A,B)
    def second_moment_symbol(A, B):
        return sp.Symbol(f'{A}{B}', real=True)   # expectation value ⟨A B⟩

    K = sp.Matrix([[second_moment_symbol(F_ops[i], F_ops[j])
                    for j in range(n)] for i in range(n)])

    # --- 3. Obtain the symmetrised covariance matrix Σ -------------
    Sigma = (K + K.T) / 2



    term1= G @ Sigma
    term2=Sigma @ G.T
    term3=W
    Sigma_dt=term1+term2+term3

    #  alternativ: symbols_in_M = M.atoms(sp.Symbol)


    return G,sDs,Z,P,Q,Z_prime,W,Sigma_dt,Sigma,K