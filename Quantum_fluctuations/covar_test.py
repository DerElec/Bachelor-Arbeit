# Python code (comments in English; console/output in German)

import sympy as sp

# --- define symbols ---
rho00, rho01, rho10, rho11, rho22, rho21, rho12, rho20, rho02 = sp.symbols(
    'rho00 rho01 rho10 rho11 rho22 rho21 rho12 rho20 rho02', complex=True
)

# density matrix (3x3 Hermitian but here symbolic)
rho = sp.Matrix([
    [rho00, rho01, rho02],
    [rho10, rho11, rho12],
    [rho20, rho21, rho22]
])

# --- Gell-Mann matrices ---
l1 = sp.Matrix([[0,1,0],[1,0,0],[0,0,0]])
l2 = sp.Matrix([[0,-sp.I,0],[sp.I,0,0],[0,0,0]])
l3 = sp.Matrix([[1,0,0],[0,-1,0],[0,0,0]])
l4 = sp.Matrix([[0,0,1],[0,0,0],[1,0,0]])
l5 = sp.Matrix([[0,0,-sp.I],[0,0,0],[sp.I,0,0]])
l6 = sp.Matrix([[0,0,0],[0,0,1],[0,1,0]])
l7 = sp.Matrix([[0,0,0],[0,0,-sp.I],[0,sp.I,0]])
l8 = (1/sp.sqrt(3))*sp.Matrix([[1,0,0],[0,1,0],[0,0,-2]])
lams = [l1,l2,l3,l4,l5,l6,l7,l8]

# --- expectation values <λ_a> ---
m = [sp.simplify(sp.trace(rho*L)) for L in lams]

# --- covariance matrix Σ (8x8 spin block) ---
Sigma = sp.MutableDenseMatrix(8,8,lambda i,j: 0)

for a in range(8):
    for b in range(8):
        anticom = lams[a]*lams[b] + lams[b]*lams[a]
        exp_ab = sp.trace(rho*anticom) / 2
        Sigma[a,b] = sp.simplify(exp_ab - m[a]*m[b])

sp.pprint(Sigma)
