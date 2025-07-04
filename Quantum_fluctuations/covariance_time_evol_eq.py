import sympy as sp
import structured_all_matricies as struc_matr
import symbolic_gellman_dgl as sgdgl
import sympy as sp
import symplectic_matrix as symplect
import numpy as np




dRtsRdt, R0,R,st,s_st,s_bt,s0,s_s0,s_b0,D_SS, D_SB, D_BS, D_BB,D,dxdt=sgdgl.create_all()
G,sDs,Z,P,Q,Z_prime,W=struc_matr.run_all()


# --- 1. Define non-commuting symbols for the operators ----------
x_ops = sp.symbols('x1:9', commutative=False)  # x1 … x8
q, p = sp.symbols('q p', commutative=False)    # q, p
F_ops = list(x_ops) + [q, p]                   # full vector F

# --- 2. Build the 10 × 10 symbolic matrix K --------------------
n = len(F_ops)
# Helper: create a unique symbol ⟨A B⟩ for every pair (A,B)
def second_moment_symbol(A, B):
    return sp.Symbol(f'<{A}{B}>', real=True)   # expectation value ⟨A B⟩

K = sp.Matrix([[second_moment_symbol(F_ops[i], F_ops[j])
                for j in range(n)] for i in range(n)])

# --- 3. Obtain the symmetrised covariance matrix Σ -------------
Sigma = (K + K.T) / 2


# print("Raw second-moment matrix K:")
# sp.pprint(K)
# print("\nSymmetrised covariance matrix Σ:")
# sp.pprint(Sigma)

term1=G*Sigma
term2=Sigma*G.T
term3=W
Sigma_dt=term1+term2+term3
#sp.pprint(Sigma_dt)

#if Sigma_dt.is_symmetric():
#   print("sym")

#sp.pprint(R.T*Sigma*R)

sp.pprint(D)