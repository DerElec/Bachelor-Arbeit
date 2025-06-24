import sympy as sp

# Define the symbolic variable kappa
kappa = sp.symbols('kappa')

# Create a 10x10 zero matrix using sympy
M = sp.Matrix(10, 10, lambda i, j: 0)

# Define the indices (using 1-indexing in the mathematical Darstellung)
p = 10  # corresponds to index 10
q = 9   # corresponds to index 9

# Fill the matrix according to the formula:
# M_{ac} = -kappa/4 ( delta_{a,p} delta_{c,p} + delta_{a,q} delta_{c,q} )
for a in range(1, 11):  # a = 1, ..., 10
    for c in range(1, 11):  # c = 1, ..., 10
        # Use sympy's KroneckerDelta to represent the delta functions.
        M[a - 1, c - 1] = kappa/sp.Integer(8) * (sp.KroneckerDelta(a, p)*sp.KroneckerDelta(c, p) +
                                                    sp.KroneckerDelta(a, q)*sp.KroneckerDelta(c, q))

# Optional: simplify the matrix (wenn nötig)
sDs = sp.simplify(M)

# Ausgabe der Matrix und ihres LaTeX-Codes
print("Die 10x10 Matrix M lautet:")
sp.pprint(sDs)

#latex_M = sp.latex(sDs)
#print("\nLaTeX Code der Matrix M:")
#print(latex_M)
