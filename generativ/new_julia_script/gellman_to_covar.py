import sympy as sp

###############################################################################
# Part I: Gemeinsame Definitionen für Gell-Mann Matrizen und f-Konstanten (für Q und W)
###############################################################################

# Define the imaginary unit and common symbolic parameters.
I = sp.I  
gamma, kappa, Gamma = sp.symbols('gamma kappa Gamma')  

# For Q- und W-Berechnungen: define m-vector with 8 components (m1,...,m8)
mQ = sp.symbols('m1:9')  # m1, m2, ..., m8

# Define the 8 Gell-Mann matrices for SU(3) (gemeinsam für Q und W)
lambda1 = sp.Matrix([[0, 1, 0],
                     [1, 0, 0],
                     [0, 0, 0]])
lambda2 = sp.Matrix([[0, -I, 0],
                     [I,  0, 0],
                     [0,  0, 0]])
lambda3 = sp.Matrix([[1,  0, 0],
                     [0, -1, 0],
                     [0,  0, 0]])
lambda4 = sp.Matrix([[0, 0, 1],
                     [0, 0, 0],
                     [1, 0, 0]])
lambda5 = sp.Matrix([[0, 0, -I],
                     [0, 0,  0],
                     [I, 0,  0]])
lambda6 = sp.Matrix([[0, 0, 0],
                     [0, 0, 1],
                     [0, 1, 0]])
lambda7 = sp.Matrix([[0, 0, 0],
                     [0, 0, -I],
                     [0, I, 0]])
lambda8 = (1/sp.sqrt(3)) * sp.Matrix([[1, 0, 0],
                                        [0, 1, 0],
                                        [0, 0, -2]])

lambda_matrices = [lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7, lambda8]

# Compute the generators T = lambda/2.
T = [mat/2 for mat in lambda_matrices]

# Define function to compute the structure constant f^{abc}:
def compute_structure_constant(a, b, c):
    # a, b, c are 1-indexed.
    Ta = T[a-1]
    Tb = T[b-1]
    Tc = T[c-1]
    commutator = Ta * Tb - Tb * Ta
    trace_val = sp.trace(commutator * Tc)
    f_val = -2 * I * trace_val
    return sp.simplify(f_val)

# Precompute f^{abc} for indices 1,...,8.
f_constants = {}
for a in range(1, 9):
    for b in range(1, 9):
        for c in range(1, 9):
            f_constants[(a, b, c)] = compute_structure_constant(a, b, c)

def f(a, b, c):
    """Return precomputed f^{abc} (1-indexed)."""
    return f_constants.get((a, b, c), 0)

###############################################################################
# Part II: Berechnung der Matrix W (10x10)
###############################################################################
# W wird definiert als: W = Z + M_sDs
# Dabei ist Z die symmetrisierte s-Matrix und M_sDs wird mittels KroneckerDelta konstruiert.

def compute_s_matrix():
    s = sp.Matrix(10, 10, lambda i, j: 0)
    for a in range(1, 9):
        for b in range(1, 9):
            expr = 0
            # Sum over indices c and d = 1,...,8.
            for c in range(1, 9):
                for d in range(1, 9):
                    term1 = f(1, a, c) * f(b, 1, d)
                    term2 = I * f(1, a, c) * f(b, 2, d)
                    term3 = - I * f(2, a, c) * f(b, 1, d)
                    term4 = f(2, a, c) * f(b, 2, d)
                    expr += gamma * mQ[c-1] * mQ[d-1] * (term1 + term2 + term3 + term4)
            s[a-1, b-1] = sp.simplify(expr)
    return s

s_matrix = compute_s_matrix()
Z = (s_matrix + s_matrix.T) / 2

# Compute additional sDs term (using KroneckerDelta).
p, q = 10, 9  # p -> index 10, q -> index 9.
M_sDs = sp.Matrix(10, 10, lambda i, j: 0)
for a in range(1, 11):
    for c in range(1, 11):
        delta_term = sp.KroneckerDelta(a, p) * sp.KroneckerDelta(c, p) + \
                     sp.KroneckerDelta(a, q) * sp.KroneckerDelta(c, q)
        M_sDs[a-1, c-1] = sp.simplify(kappa/sp.Integer(8) * delta_term)

# Final W-Matrix:
W = sp.simplify(Z + M_sDs)

###############################################################################
# Part III: Berechnung der Matrix Q (10x10) aus m_{α k} (mittels f und d)
###############################################################################
def compute_d_constant(a, b, c):
    la = lambda_matrices[a-1]
    lb = lambda_matrices[b-1]
    lc = lambda_matrices[c-1]
    anticommutator = la * lb + lb * la
    trace_val = sp.trace(anticommutator * lc)
    d_val = sp.Rational(1, 4) * trace_val
    return sp.simplify(d_val)

d_constants = {}
for a in range(1, 9):
    for b in range(1, 9):
        for c in range(1, 9):
            d_constants[(a, b, c)] = compute_d_constant(a, b, c)

def d_const(a, b, c):
    return d_constants.get((a, b, c), 0)

def compute_Q_matrix():
    m_matrix = sp.Matrix(10, 10, lambda i, j: 0)
    for alpha in range(1, 9):
        for k in range(1, 9):
            expr = 0
            for c in range(1, 9):
                term1 = I * f(alpha, 1, c) * (d_const(1, c, k) + I * f(1, c, k))
                term2 = I * f(alpha, 1, c) * (d_const(2, c, k) + I * f(2, c, k))
                term3 = - (d_const(1, c, k) + I * f(1, c, k)) * f(alpha, 2, c)
                term4 =   (d_const(2, c, k) + I * f(2, c, k)) * f(alpha, 2, c)
                term5 = f(1, alpha, c) * (d_const(c, 1, k) + I * f(c, 1, k))
                term6 = I * f(1, alpha, c) * (d_const(c, 2, k) + I * f(c, 2, k))
                term7 = - I * f(2, alpha, c) * (d_const(c, 1, k) + I * f(c, 1, k))
                term8 = f(2, alpha, c) * (d_const(c, 2, k) + I * f(c, 2, k))
                expr += term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8
            m_matrix[alpha-1, k-1] = sp.simplify((Gamma/sp.Integer(4)) * expr)
    return m_matrix

Q = sp.simplify(compute_Q_matrix())

###############################################################################
# Part IV: Berechnung der Matrix P (10x10) gemäß separater Formel
###############################################################################
# Hier werden eigene Symbole mP (10 Komponenten) und F (Feldvariablen) verwendet.
mP = sp.symbols('m1:11')  # m1, ..., m10
F = sp.symbols('F1:11')   # F1, ..., F10

# Zusätzliche Konstanten:
g0, V_const = sp.symbols('g0 V')
Delta1, Omega, N, eta = sp.symbols('Delta1 Omega N eta')

# h_{μν}-Komponenten:
h9_list = [g0] + [0]*9       # h9[0]=g0, sonst 0.
h10_list = [0, -g0] + [0]*8   # h10[1]=-g0, sonst 0.
h88 = V_const/sp.Integer(3)  # h₈,₈ = V/3

# ω-Vektor (10 Einträge; nur Indizes 1,...,8 relevant im linearen Term):
omega_vec = [0,
             0,
            -Delta1,
             0,
             0,
             Omega,
             0,
            -V_const/(3*sp.sqrt(3)),
             0,
             2*sp.sqrt(N)*eta]

# Eigene Gell-Mann Matrizen für P:
lambda1_P = sp.Matrix([[0, 1, 0],
                       [1, 0, 0],
                       [0, 0, 0]])
lambda2_P = sp.Matrix([[0, -sp.I, 0],
                       [sp.I, 0, 0],
                       [0, 0, 0]])
lambda3_P = sp.Matrix([[1, 0, 0],
                       [0, -1, 0],
                       [0, 0, 0]])
lambda4_P = sp.Matrix([[0, 0, 1],
                       [0, 0, 0],
                       [1, 0, 0]])
lambda5_P = sp.Matrix([[0, 0, -sp.I],
                       [0, 0, 0],
                       [sp.I, 0, 0]])
lambda6_P = sp.Matrix([[0, 0, 0],
                       [0, 0, 1],
                       [0, 1, 0]])
lambda7_P = sp.Matrix([[0, 0, 0],
                       [0, 0, -sp.I],
                       [0, sp.I, 0]])
lambda8_P = sp.Matrix([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, -2]])/sp.sqrt(3)

lambda_matrices_P = [lambda1_P, lambda2_P, lambda3_P, lambda4_P,
                     lambda5_P, lambda6_P, lambda7_P, lambda8_P]

# Für die f-Konstanten in P (modifizierte Definition):
def compute_f_P(a, b, c):
    la = lambda_matrices_P[a-1]
    lb = lambda_matrices_P[b-1]
    lc = lambda_matrices_P[c-1]
    commutator = la * lb - lb * la
    f_val = sp.trace(commutator * lc)
    return sp.simplify(f_val / (4 * sp.I))

def f_constant_P(g, a, c):
    if g > 8 or a > 8 or c > 8:
        return 0
    return compute_f_P(g, a, c)

def explicit_f_P(g, a, c):
    val = f_constant_P(g, a, c)
    # If nonzero, return a symbol of the form f^{g_a_c}
    if val != 0:
        return sp.Symbol(f"f^{{{g}_{a}_{c}}}")
    else:
        return 0

# Helper: Projection for F-Terme.
def proj(i, gamma_idx):
    return F[i-1] if i == gamma_idx else 0

# Build matrix P_{a,γ} (10x10) according to the given formula.
P = sp.zeros(10, 10)
for a in range(1, 11):          # a = 1,...,10
    for gamma_idx in range(1, 11):  # γ = 1,...,10
        coeff = 0
        # --- Cavity-Atom Term für μ = 9 (gamma_fix = 1, h9_list[0] = g0) ---
        if gamma_idx == 9:
            sumA = 0
            for c in range(1, 9):
                sumA += explicit_f_P(1, a, c) * mP[c-1]
            coeff += - h9_list[0] * 2 * sumA * proj(9, 9)
        if gamma_idx <= 8:
            coeff += - h9_list[0] * 2 * mP[8] * explicit_f_P(1, a, gamma_idx) * proj(gamma_idx, gamma_idx)
        if gamma_idx == 1:
            coeff += - h9_list[0] * (sp.Rational(1,2) if a == 10 else 0) * proj(1, 1)
        
        # --- Cavity-Atom Term für μ = 10 (gamma_fix = 2, h10_list[1] = -g0) ---
        if gamma_idx == 10:
            sumB = 0
            for c in range(1, 9):
                sumB += explicit_f_P(2, a, c) * mP[c-1]
            coeff += - h10_list[1] * 2 * sumB * proj(10, 10)
        if gamma_idx <= 8:
            coeff += - h10_list[1] * 2 * mP[9] * explicit_f_P(2, a, gamma_idx) * proj(gamma_idx, gamma_idx)
        if gamma_idx == 2:
            coeff += h10_list[1] * (sp.Rational(1,2) if a == 9 else 0) * proj(2, 2)
        
        # --- Atom-Atom Term ---
        if gamma_idx == 8:
            sumC = explicit_f_P(8, a, gamma_idx) * mP[gamma_idx]
            coeff += - h88 * 4 * sumC * proj(8, 8)
        if gamma_idx <= 8:
            coeff += - h88 * 4 * mP[7] * explicit_f_P(8, a, gamma_idx) * proj(gamma_idx, gamma_idx)
        
        # --- Linearer Term ---
        if gamma_idx <= 8:
            for i in range(1, 9):
                coeff += -2 * omega_vec[i-1] * explicit_f_P(i, a, gamma_idx) * proj(gamma_idx, gamma_idx)
        
        P[a-1, gamma_idx-1] = sp.factor(coeff) * F[gamma_idx-1]

###############################################################################
# Part V: Gesamte Matrix G = P + Q
###############################################################################
G = sp.simplify(P + Q)

###############################################################################
# Substitution der F-Terme: Setze alle F_i auf 1.
subsF = {F[i]: 1 for i in range(10)}
G_noF = sp.simplify(G.subs(subsF))

###############################################################################
# Neue Substitution: Ersetze die explizit in P verwendeten f-Symbole (z.B. f^{g_a_c}) 
# durch die berechneten f-Werte (f_constant_P), sofern ungleich 0.
subs_f = {}
for g in range(1, 9):
    for a in range(1, 9):   # Nur für a, c = 1,...,8, da sonst f_constant_P 0 liefert.
        for c in range(1, 9):
            val = f_constant_P(g, a, c)
            if val != 0:
                symbol = sp.Symbol(f"f^{{{g}_{a}_{c}}}")
                subs_f[symbol] = val

G_substituted = sp.simplify(G_noF.subs(subs_f))

###############################################################################
# Final Output: Print matrix G (ohne F-Terme und mit eingesetzten f-Termen) und W.
###############################################################################
print("Die Matrix G = P + Q (F-Terme und f-Terme substituiert) lautet:")
sp.pprint(G_substituted)
print("\nLaTeX Code der Matrix G (F- und f-Terme ersetzt):")
print(sp.latex(G_substituted))

print("\n--------------------------------------------------\n")

print("Die Matrix W lautet:")
sp.pprint(W)
print("\nLaTeX Code der Matrix W:")
print(sp.latex(W))
